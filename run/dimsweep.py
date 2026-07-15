"""Curse-of-dimensionality sweep on the ``noise_dims`` dataset (3 tight clusters + noise dims).

For each total dimensionality D, generate the noise-dims dataset (3 tight clusters in 3 standardized
signal columns + D-3 standard-normal noise columns), embed every method over R seeds, and record the
2-D label-separation scores (kNN accuracy / silhouette) that headline the supplement -- the full
distance-band metric set (vs truth and vs ambient) is also computed into the per-run CSV, but is
deliberately not headlined: a global rho is only meaningful paired with its near band, and neither
reads cleanly against this probe's noise-dominated ambient. Quantified replication of the notebook
experiment ``01_hi_dimensional_data_anlaysis.ipynb``.

    python run/dimsweep.py --dims 6 40 80 200 300 400 500 --methods all --seeds 3 --n 1000
    # toorPIA-only extensions -- embedding arm to D=768, csvform to D=768:
    python run/dimsweep.py --dims 6 40 80 200 300 400 500 768 --methods toorPIA --toorpia-endpoint embedding
    python run/dimsweep.py --dims 768 --methods toorPIA --seeds 3 --n 1000
    # extreme-D extension: 5 noise realizations x 3 method seeds (realization-sensitive regime)
    python run/dimsweep.py --dims 2000 6000 --methods toorPIA --seeds 3 --n 1000 --data-seed 42 0 1 2 3

toorPIA appears in this probe through BOTH of its endpoints, in two stages: (1)
``--toorpia-endpoint embedding`` (rows labeled ``toorPIA-embedding``; separate
``dimsweep_embedding_*.csv`` tables) probes the raw-geometry ``basemap_embedding`` endpoint the
main benchmark uses -- the same-footing comparison against the six generic methods; (2) the
default ``--toorpia-endpoint csvform`` probes ``basemap_csvform``, toorPIA's DEFAULT
CSV-data-analysis pipeline (per-item preprocessing, exposes ``random_seed``; committed caches
under ``external_embeddings/noise_dims/toorpia/fit/``) -- the endpoint a practitioner uses on raw
tabular data, whose tolerance to unknown numbers of noise dimensions is the supplement's headline.
``--figures-only`` rebuilds the section figures from the merged CSVs and the archived seed-0
embeddings (both endpoints + the ambient raw-feature kNN baseline) without embedding anything.

Results MERGE into an existing ``dimsweep_per_run.csv`` by (dim, data_seed, method, seed), so
partial runs (like the toorPIA-only extension above) extend the sweep without clobbering it. Unlike
``run/sweep.py``, a ``SkipMethod`` here skips only the current dim (no permanent blacklist):
high-D toorPIA API calls are the likeliest to fail transiently, and every successful embedding is
cached under ``external_embeddings/noise_dims/`` so a re-run is cheap.
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
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


if _early_arg("--device", "cpu") == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
torch.set_num_threads(1)
import pandas as pd  # noqa: E402

from methods import SkipMethod, default_methods, get_method  # noqa: E402
from metrics.compute import compute_all  # noqa: E402
from metrics.label_separation import knn_label_accuracy, silhouette_by_label  # noqa: E402
from run.aggregate import bootstrap_median_ci  # noqa: E402
from run import figures  # noqa: E402
from synth.noise_dims import make_noise_dims  # noqa: E402


def set_seeds(seed):
    np.random.seed(int(seed)); torch.manual_seed(int(seed))


def toorpia_csvform_embed(X, seed, tag):
    """toorPIA via its default CSV-analysis pipeline (``basemap_csvform``), cache-first.

    Cache mode is ``fit`` -- the committed caches of this probe (the pipeline behind
    ``fit_transform``/``basemap_csvform`` is the same engine; spot-verified at D=768:
    pdist rho 0.998, kNN accuracy 0.975 vs 0.976). Explicit per-column float type/weight
    options replicate the client's DataFrame auto-detection (server-side auto-detection
    can misread a headerless numeric column as a date).
    """
    from methods.external import external_path, load_external, save_external
    cached = load_external(ROOT, "noise_dims", "toorpia", "fit", tag, seed)
    if cached is not None:
        return cached
    if not os.environ.get("TOORPIA_API_KEY"):
        raise SkipMethod(
            f"no cached toorPIA embedding at "
            f"{external_path(ROOT, 'noise_dims', 'toorpia', 'fit', tag, seed)} and "
            "TOORPIA_API_KEY is unset")
    import tempfile

    from toorpia import toorPIA

    d = X.shape[1]
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        path = f.name
    pd.DataFrame(np.ascontiguousarray(X, dtype=np.float64),
                 columns=[f"d{i}" for i in range(d)]).to_csv(path, index=False)
    try:
        res = toorPIA().basemap_csvform(
            path,
            weight_option_str=",".join(f"{i + 1}:1" for i in range(d)),
            type_option_str=",".join(f"{i + 1}:float" for i in range(d)),
            random_seed=int(seed), vector_normalization=False)
    finally:
        os.unlink(path)
    if res is None:
        raise SkipMethod("basemap_csvform returned None (auth / transport / server failure)")
    Y = np.asarray(res["xyData"], dtype=np.float64)
    if Y.shape != (X.shape[0], 2):
        raise SkipMethod(f"basemap_csvform returned shape {Y.shape}, expected {(X.shape[0], 2)}")
    save_external(Y, ROOT, "noise_dims", "toorpia", "fit", tag, seed)
    return Y


DISPLAY = {"toorPIA": "toorPIA (basemap_csvform)",
           "toorPIA-embedding": "toorPIA (basemap_embedding)"}
GRID_DIMS = (6, 40, 200, 300, 400, 500, 768)


def build_figures(out_dir, figdir, knn_key, args):
    """Rebuild the section figures from the merged CSVs + archived seed-0 embeddings.

    Combines the csvform sweep (``dimsweep_*.csv``) with the embedding-endpoint arm
    (``dimsweep_embedding_*.csv``, if present) and the ambient raw-feature kNN baseline
    (recomputed deterministically; 5 noise realizations at the extreme dims, matching how those
    cells aggregate). Grid panels load the archived ``seed0`` embeddings, so a partial re-run
    still redraws the full figures.
    """
    agg = pd.read_csv(out_dir / "dimsweep_aggregated.csv")
    emb_p = out_dir / "dimsweep_embedding_aggregated.csv"
    if emb_p.exists():
        agg = pd.concat([agg, pd.read_csv(emb_p)], ignore_index=True)
    agg = agg.replace({"method": DISPLAY})
    order = ([m for m in default_methods() if m != "toorPIA"]
             + [DISPLAY["toorPIA-embedding"], DISPLAY["toorPIA"]])
    present = [m for m in order if (agg.method == m).any()]

    ambient = {}
    for d in sorted(int(v) for v in agg.dim.unique()):
        dss = [42, 0, 1, 2, 3] if d >= 1500 else [42]
        vals = []
        for ds in dss:
            b = make_noise_dims(n=args.n, d=d, seed=ds, n_principal=args.n_principal,
                                cluster_std=args.cluster_std)
            vals.append(knn_label_accuracy(b["clean"], b["labels"], k=args.knn_k))
        ambient[d] = float(np.median(vals))

    figures.noise_dims_endpoint_curve(agg, present, ambient, figdir, knn_key=knn_key,
                                      k=args.knn_k, chance=1.0 / args.n_principal)

    grid = {}
    for m in present:
        src = {v: k for k, v in DISPLAY.items()}.get(m, m)   # display name -> archive dir name
        for d in GRID_DIMS:
            p = out_dir / "embeddings" / "noise_dims" / f"dim{d}" / src / "seed0.npy"
            if p.exists():
                grid.setdefault(m, {})[d] = np.load(p)
    labels_ref = make_noise_dims(n=args.n, d=args.n_principal + 3, seed=42,
                                 n_principal=args.n_principal,
                                 cluster_std=args.cluster_std)["labels"]
    grid_dims = sorted({d for g in grid.values() for d in g})
    if grid and len(grid_dims) >= 3:
        figures.dims_grid(grid, labels_ref, [m for m in present if m in grid], grid_dims, figdir)
    print(f"figures rebuilt -> {figdir}/ (methods: {present})")


def main(argv=None):
    p = argparse.ArgumentParser(description="dimensionality sweep on the noise-dims dataset")
    p.add_argument("--dims", nargs="+", type=int,
                   default=[6, 40, 80, 200, 300, 400, 500])
    p.add_argument("--methods", default="all")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--n-principal", type=int, default=3)
    p.add_argument("--cluster-std", type=float, default=0.005)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-seed", nargs="+", type=int, default=[42],
                   help="one or more NOISE-REALIZATION seeds; multiple seeds probe realization "
                        "sensitivity (used at the extreme-D breaking regime)")
    p.add_argument("--toorpia-endpoint", choices=["csvform", "embedding"], default="csvform",
                   help="which toorPIA endpoint the probe drives: basemap_csvform (default; the "
                        "CSV-analysis pipeline, cache mode 'fit', rows labeled 'toorPIA') or "
                        "basemap_embedding (raw-geometry comparison; rows labeled "
                        "'toorPIA-embedding' in separate dimsweep_embedding_*.csv tables)")
    p.add_argument("--figures-only", action="store_true",
                   help="rebuild the section figures from the existing CSVs and archived seed-0 "
                        "embeddings; embeds nothing")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures" / "noise_dims"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    methods = default_methods() if args.methods == "all" else args.methods.split(",")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    knn_key = f"knn_acc_k{args.knn_k}"
    emb_mode = args.toorpia_endpoint == "embedding"
    stem = "dimsweep_embedding" if emb_mode else "dimsweep"

    if args.figures_only:
        build_figures(out_dir, figdir, knn_key, args)
        return 0

    print(f"dimsweep dims={args.dims} methods={methods} seeds={args.seeds} "
          f"N={args.n} n_principal={args.n_principal} cluster_std={args.cluster_std:g} "
          f"toorpia_endpoint={args.toorpia_endpoint}")

    rows = []
    for dim in args.dims:
        for ds in args.data_seed:
            base = make_noise_dims(n=args.n, d=dim, seed=ds,
                                   n_principal=args.n_principal, cluster_std=args.cluster_std)
            X, X_truth, labels = base["clean"], base["truth_coords"], base["labels"]
            print(f"\n=== D={dim}  ({dim - args.n_principal} noise dims)  data_seed={ds} ===",
                  flush=True)
            for method in methods:
                m = get_method(method)
                # toorPIA/csvform exposes random_seed (stochastic); the embedding endpoint and
                # the registry wrapper are deterministic
                use_csvform = method == "toorPIA" and args.toorpia_endpoint == "csvform"
                label = "toorPIA-embedding" if (method == "toorPIA" and emb_mode) else method
                stochastic = True if use_csvform else m.stochastic
                seeds = range(args.seeds) if stochastic else [0]
                # data seed 42 keeps the historical tag so the committed toorPIA caches stay valid
                tag = f"n{args.n}_dim{dim}" if ds == 42 else f"n{args.n}_dim{dim}_ds{ds}"
                ctx = {"root": str(ROOT), "dataset": "noise_dims", "snr": float("inf"),
                       "tag": tag}
                k = 0
                for seed in seeds:
                    set_seeds(seed)
                    try:
                        Y = (toorpia_csvform_embed(X, seed, tag) if use_csvform
                             else m.embed(X, seed=seed, device=args.device, context=ctx))
                    except SkipMethod as e:
                        print(f"  [skip] {method} dim{dim}: {e}"); break
                    except Exception as e:  # noqa: BLE001
                        print(f"  [warn] {method} dim{dim} ds{ds} seed{seed}: "
                              f"{type(e).__name__}: {e}"); continue
                    sub = f"dim{dim}" if ds == 42 else f"dim{dim}_ds{ds}"
                    emb_dir = out_dir / "embeddings" / "noise_dims" / sub / label
                    emb_dir.mkdir(parents=True, exist_ok=True)
                    np.save(emb_dir / f"seed{seed}.npy", Y)
                    row = dict(dim=dim, n=args.n, data_seed=ds, method=label, seed=seed,
                               stochastic=stochastic)
                    row.update(compute_all(X, Y, X_truth, include_per_point=False, labels=labels))
                    row[knn_key] = knn_label_accuracy(Y, labels, k=args.knn_k)
                    row["silhouette_2d"] = silhouette_by_label(Y, labels)
                    rows.append(row); k += 1
                if k:
                    print(f"  {method:9s} runs={k}", flush=True)

    per_run = pd.DataFrame(rows)
    # merge into an existing table: rows whose (dim, data_seed, method, seed) was re-run now are
    # replaced, everything else is kept verbatim (same contract as run/benchmark.py) -- this lets a
    # partial run (e.g. toorPIA-only at an extra dimension) extend the committed sweep
    per_run_path = out_dir / f"{stem}_per_run.csv"
    if len(per_run) and per_run_path.exists():
        old = pd.read_csv(per_run_path, float_precision="round_trip")
        if "data_seed" not in old.columns:      # tables written before the multi-realization column
            old["data_seed"] = 42
        key = ["dim", "data_seed", "method", "seed"]
        reran = set(map(tuple, per_run[key].drop_duplicates().itertuples(index=False)))
        keep = old[[t not in reran for t in old[key].itertuples(index=False, name=None)]]
        per_run = pd.concat([keep, per_run], ignore_index=True)
        per_run = per_run.sort_values(key).reset_index(drop=True)
    if not len(per_run):
        # every requested cell was skipped (e.g. API unavailable): writing now would clobber
        # the committed tables with empty ones -- leave them untouched
        print("\nDONE. rows=0 (all cells skipped; tables left untouched)")
        return 1
    per_run.to_csv(per_run_path, index=False)

    # aggregate: median + bootstrap 95% CI per (method, dim, metric)
    sweep_metrics = ["shepard_p100__vs_truth", "shepard_p20__vs_truth", "full_shepard",
                     "shepard_p5__vs_ambient", knn_key, "silhouette_2d",
                     "tight_over_compression", "recall_k15"]
    agg = []
    if len(per_run):
        for (method, dim), g in per_run.groupby(["method", "dim"]):
            for metric in sweep_metrics:
                if metric not in g:
                    continue
                vals = g[metric].to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                if not vals.size:
                    continue
                lo, hi = bootstrap_median_ci(vals)
                agg.append(dict(method=method, dim=dim, metric=metric,
                                median=float(np.median(vals)), ci_lo=lo, ci_hi=hi,
                                n_runs=int(vals.size)))
    agg = pd.DataFrame(agg)
    agg.to_csv(out_dir / f"{stem}_aggregated.csv", index=False)

    if not args.no_figures and len(per_run):
        build_figures(out_dir, figdir, knn_key, args)

    if len(per_run):
        pivot = per_run.pivot_table(index="method", columns="dim", values=knn_key, aggfunc="median")
        print(f"\n{knn_key} (chance = 1/{args.n_principal}), median over seeds:")
        print(pivot.round(3).to_string())
    print(f"\nDONE. rows={len(per_run)}  -> {out_dir}/dimsweep_*.csv ; figures -> {figdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
