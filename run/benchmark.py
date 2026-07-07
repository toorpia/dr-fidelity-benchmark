"""Experiment driver / single CLI entry point for the DR fidelity benchmark.

Generates synthetic data with known ground truth, adds SNR-controlled noise, embeds it with each DR
method over R seeds (deterministic methods once), computes the exact distance-fidelity metrics
(vs-ambient and vs-truth), aggregates over seeds (median + bootstrap 95% CI + std), quantifies
run-to-run stability (Procrustes), saves embeddings + tables + figures, and records the environment.

Reproducibility: CPU + single thread by default; seeds BOTH numpy and torch before every embed; same
CLI args reproduce identical numbers. Run e.g.::

    python run/benchmark.py --dataset all --methods all --seeds 3 --dim 768 --n 1000 --snr inf 4 1

See README for the full methodology.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _early_arg(flag, default=None):
    """Read an arg value from sys.argv before heavy imports (needed for CUDA env setup)."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


# --- environment must be configured BEFORE importing torch / method libs ---
_DEVICE = _early_arg("--device", "cpu")
if _DEVICE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""      # force CPU for FP determinism
os.environ.setdefault("TQDM_DISABLE", "1")        # silence PCC/other progress bars
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
torch.set_num_threads(1)

import pandas as pd  # noqa: E402

from methods import SkipMethod, default_methods, get_method  # noqa: E402
from metrics.compute import compute_all  # noqa: E402
from metrics.distances import DEFAULT_CUTOFFS  # noqa: E402
from run import aggregate, figures  # noqa: E402
from run.environment import write_environment  # noqa: E402
from synth import add_noise, list_datasets, make_dataset  # noqa: E402


def parse_snr(tokens):
    out = []
    for t in tokens:
        t = str(t).lower()
        out.append(float("inf") if t in ("inf", "clean", "none") else float(t))
    return out


def noise_rng(data_seed: int, snr: float) -> np.random.Generator:
    key = 10 ** 9 if not np.isfinite(snr) else int(round(snr * 1000))
    return np.random.default_rng(np.random.SeedSequence([int(data_seed), key]))


def set_seeds(seed: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def merge_with_existing(new_df: "pd.DataFrame", path: Path, key_cols) -> "pd.DataFrame":
    """Merge freshly-computed rows into an existing results table (if any).

    Rows of the existing table whose ``key_cols`` combination was re-run now are REPLACED by the new
    rows; every other existing row is kept verbatim. This lets a partial run (e.g. ``--dataset
    outliers``) extend ``results/`` without clobbering the committed results of the other datasets.
    Aggregation is deterministic (fixed bootstrap seed), so re-aggregating the kept rows reproduces
    their published numbers exactly.
    """
    if new_df.empty or not path.exists():
        return new_df
    # round_trip: pandas' default float parser is off by 1 ulp on some values, which would
    # perturb the kept (not re-run) rows on rewrite -- kept rows must survive byte-identically
    old = pd.read_csv(path, float_precision="round_trip")
    reran = set(map(tuple, new_df[key_cols].drop_duplicates().itertuples(index=False)))
    keep = old[[t not in reran for t in old[key_cols].itertuples(index=False, name=None)]]
    return pd.concat([keep, new_df], ignore_index=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DR fidelity benchmark (distance-preservation focus)")
    p.add_argument("--dataset", default="all", help="comma list or 'all'")
    p.add_argument("--methods", default="all", help="comma list or 'all'")
    p.add_argument("--seeds", type=int, default=3, help="R independent seeds for stochastic methods")
    p.add_argument("--dim", type=int, default=768, help="ambient dimension D")
    p.add_argument("--n", type=int, default=1000, help="number of points N")
    p.add_argument("--snr", nargs="+", default=["inf"], help="SNR sweep (power ratio); 'inf'=clean (default).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures"))
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--no-per-point", action="store_true", help="skip the secondary per-point Shepard variant")
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    datasets = list_datasets() if args.dataset == "all" else args.dataset.split(",")
    methods = default_methods() if args.methods == "all" else args.methods.split(",")
    snrs = parse_snr(args.snr)
    include_pp = not args.no_per_point
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    emb_root = out_dir / "embeddings"
    figdir = Path(args.figdir)

    print(f"datasets={datasets} methods={methods} seeds={args.seeds} N={args.n} D={args.dim} "
          f"snr={snrs} device={args.device} per_point={include_pp}")

    per_run_rows, stability_frames, skipped = [], [], set()

    for dataset in datasets:
        base = make_dataset(dataset, n=args.n, d=args.dim, seed=args.data_seed)
        X_truth = base["truth_coords"]
        ds_figdir = figdir / dataset; ds_figdir.mkdir(parents=True, exist_ok=True)
        for snr in snrs:
            X = add_noise(base["clean"], snr, noise_rng(args.data_seed, snr))
            snr_lab = figures.snr_label(snr)
            tag = f"n{args.n}_d{args.dim}_snr{snr_lab}"
            emb_by_method = {}
            print(f"\n=== {dataset}  SNR={snr_lab} ===")
            for method in methods:
                if method in skipped:
                    continue
                m = get_method(method)
                seeds = range(args.seeds) if m.stochastic else [0]
                context = {"root": str(ROOT), "dataset": dataset, "snr": snr, "tag": tag}
                emb_by_method[method] = {}
                for seed in seeds:
                    set_seeds(seed)
                    try:
                        Y = m.embed(X, seed=seed, device=args.device, context=context)
                    except SkipMethod as e:
                        print(f"  [skip] {method}: {e}")
                        skipped.add(method); emb_by_method.pop(method, None)
                        break
                    except Exception as e:  # noqa: BLE001 - one bad seed must not abort the run
                        print(f"  [warn] {method} seed{seed} failed: {type(e).__name__}: {e}")
                        continue
                    # archive embedding
                    ep = emb_root / dataset / f"snr{snr_lab}" / method / f"seed{seed}.npy"
                    ep.parent.mkdir(parents=True, exist_ok=True)
                    np.save(ep, Y)
                    emb_by_method[method][seed] = Y
                    row = dict(dataset=dataset, snr=snr, method=method, seed=seed,
                               stochastic=m.stochastic, n=args.n, d=args.dim)
                    row.update(compute_all(X, Y, X_truth, cutoffs=DEFAULT_CUTOFFS,
                                           include_per_point=include_pp, labels=base["labels"],
                                           outlier_idx=base.get("outlier_idx"),
                                           outlier_dir=base.get("outlier_dir"),
                                           population=base.get("population")))
                    per_run_rows.append(row)
                if emb_by_method.get(method):
                    print(f"  {method:9s} runs={len(emb_by_method[method])}")

            # stability for this config
            stab_in = {(dataset, snr, mth): sm for mth, sm in emb_by_method.items() if len(sm) >= 2}
            if stab_in:
                df_run = pd.DataFrame(per_run_rows)
                stability_frames.append(aggregate.compute_stability(stab_in, df_run))

            # explanatory figures (depend only on the high-D features X, no embedding needed)
            if not args.no_figures:
                try:
                    figures.distance_distribution(X, dataset, snr, ds_figdir)
                    figures.recall_bias_figure(X, dataset, snr, ds_figdir)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] explanatory figure failed: {e}")

            # figures for this config (representative = lowest available seed)
            if not args.no_figures and emb_by_method:
                rep = {mth: sm[min(sm)] for mth, sm in emb_by_method.items() if sm}
                methods_present = [mm for mm in methods if mm in rep]
                try:
                    figures.shepard_scatter(X, rep, dataset, snr, ds_figdir)
                    # the population gallery (majority/minority markers) supersedes the plain
                    # embeddings panel on the populations dataset
                    if base.get("population") is None:
                        figures.embedding_panels(rep, base["color_value"], base["color_name"],
                                                 dataset, snr, ds_figdir)
                    if base.get("outlier_idx") is not None:
                        figures.outlier_gallery(rep, base["labels"], base["outlier_idx"],
                                                dataset, snr, ds_figdir,
                                                outlier_dir=base.get("outlier_dir"))
                    if base.get("population") is not None:
                        figures.population_gallery(rep, base["labels"], base["population"],
                                                   dataset, snr, ds_figdir)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] figure (scatter/embeddings) failed: {e}")

    # ---- aggregate + persist (merging into existing tables so partial runs don't clobber) ----
    # method-aware merge key: a method-subset re-run (e.g. ``--methods toorPIA``) replaces only
    # that method's rows and keeps every other method verbatim. Keyed WITHOUT seed: seeds always
    # run from 0, so a re-run owns all of a method's rows -- this also purges stale seeds when a
    # method's seed structure changes (e.g. a stochastic method becoming deterministic)
    per_run = merge_with_existing(pd.DataFrame(per_run_rows), out_dir / "metrics_per_run.csv",
                                  key_cols=["dataset", "snr", "method"])
    per_run.to_csv(out_dir / "metrics_per_run.csv", index=False)
    agg = aggregate.aggregate_runs(per_run, n_boot=args.bootstrap)
    agg.to_csv(out_dir / "metrics_aggregated.csv", index=False)
    if stability_frames:
        stab = pd.concat(stability_frames, ignore_index=True).drop_duplicates(
            subset=["dataset", "snr", "method"], keep="last")
        stab = merge_with_existing(stab, out_dir / "stability.csv",
                                   key_cols=["dataset", "snr", "method"])
        stab.to_csv(out_dir / "stability.csv", index=False)

    # curve + heatmap figures from the aggregated table
    if not args.no_figures and len(agg):
        for dataset in datasets:
            ds_figdir = figdir / dataset
            present = [m for m in methods if ((agg.dataset == dataset) & (agg.method == m)).any()]
            for snr in snrs:
                try:
                    # on outliers/populations the membership-pair-ρ scatter is the placement
                    # reading; the plain near-vs-global scatter is redundant there
                    if dataset == "outliers":
                        figures.outlier_score_scatter(agg, dataset, snr, present, ds_figdir)
                    elif dataset == "populations":
                        figures.population_score_scatter(agg, dataset, snr, present, ds_figdir)
                    else:
                        figures.score_scatter(agg, dataset, snr, present, ds_figdir)
                    figures.summary_heatmap(agg, dataset, snr, present, ds_figdir)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] aggregate figure failed ({dataset} snr={snr}): {e}")

    write_environment(ROOT / "ENVIRONMENT.md")
    print(f"\nDONE. per-run rows={len(per_run)}  aggregated rows={len(agg)}  "
          f"skipped methods={sorted(skipped) or 'none'}")
    print(f"  tables -> {out_dir}/  figures -> {figdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
