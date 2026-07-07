"""Single-knob sweeps: outlier_factor on `outliers`, minority_frac on `populations`.

For each knob value, generate the dataset, embed every method over R seeds, and record the sweep
metrics. Modes:

* ``--sweep outlier_factor`` -- (CLEAN data) global Shepard rho and the anomaly-pair-restricted
  Shepard rho vs the outlier separation factor of the `outliers` dataset.
* ``--sweep minority_frac`` -- (SNR=1, matching the report's canonical noise level) the
  population-membership rho family vs the minority population's share of the points, on the
  imbalanced two-population dataset.

    python run/sweep.py --sweep outlier_factor --outlier-factor 1.5 2 3 5 8 --methods all --seeds 3
    python run/sweep.py --sweep minority_frac --minority-frac 0.5 0.25 0.1 0.05 --methods all --seeds 3
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
from run.aggregate import bootstrap_median_ci  # noqa: E402
from run import figures  # noqa: E402
from synth.outliers import make_outliers  # noqa: E402
from synth.populations import make_populations  # noqa: E402

# per-mode configuration: knob column, dataset factory, sweep metrics, output stem, figure, figdir
MODES = {
    "outlier_factor": dict(
        metrics=["full_shepard", "shepard_p5__vs_ambient", "outlier_shepard__vs_ambient",
                 "pair_angle_2d_max"],
        stem="sweep_outliers", figdir="outliers", ctx_dataset="outliers_sweep", tag="of"),
    "minority_frac": dict(
        metrics=["full_shepard", "shepard_p5__vs_ambient", "minority_shepard__vs_ambient",
                 "within_majority_shepard__vs_ambient", "within_minority_shepard__vs_ambient",
                 "cross_population_shepard__vs_ambient"],
        stem="sweep_populations", figdir="populations", ctx_dataset="populations_sweep", tag="mf"),
}


def set_seeds(seed):
    np.random.seed(int(seed)); torch.manual_seed(int(seed))


def main(argv=None):
    p = argparse.ArgumentParser(description="single-knob sweep (outlier_factor | minority_frac)")
    p.add_argument("--sweep", choices=list(MODES), required=True)
    p.add_argument("--outlier-factor", nargs="+", type=float, default=[1.5, 2, 3, 5, 8])
    p.add_argument("--minority-frac", nargs="+", type=float, default=[0.5, 0.25, 0.1, 0.05])
    p.add_argument("--methods", default="all")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--n-clusters", type=int, default=None,
                   help="bulk cluster count (default: 5)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=None)
    args = p.parse_args(argv)

    mode = MODES[args.sweep]
    knob = args.sweep
    values = {"outlier_factor": args.outlier_factor,
              "minority_frac": args.minority_frac}[knob]
    methods = default_methods() if args.methods == "all" else args.methods.split(",")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir) if args.figdir else ROOT / "figures" / mode["figdir"]
    figdir.mkdir(parents=True, exist_ok=True)
    print(f"sweep {knob}={values} methods={methods} seeds={args.seeds} N={args.n} D={args.dim}")

    rows, skipped = [], set()
    for v in values:
        if knob == "outlier_factor":
            base = make_outliers(n=args.n, d=args.dim, seed=args.data_seed,
                                 n_clusters=args.n_clusters or 5, outlier_factor=v)
        else:
            base = make_populations(n=args.n, d=args.dim, seed=args.data_seed,
                                    n_clusters=args.n_clusters or 5, minority_frac=v)
        X, X_truth = base["clean"], base["truth_coords"]
        if knob == "minority_frac":                       # this sweep runs at the report's SNR=1
            from run.benchmark import noise_rng
            from synth import add_noise
            X = add_noise(base["clean"], 1.0, noise_rng(args.data_seed, 1.0))
        print(f"\n=== {knob}={v:g}  (intra_nn={base['intra_nn']:.3f}) ===")
        emb_seed0 = {}
        for method in methods:
            if method in skipped:
                continue
            m = get_method(method)
            seeds = range(args.seeds) if m.stochastic else [0]
            ctx = {"root": str(ROOT), "dataset": mode["ctx_dataset"], "snr": float("inf"),
                   "tag": f"{mode['tag']}{v:g}"}
            k = 0
            for seed in seeds:
                set_seeds(seed)
                try:
                    Y = m.embed(X, seed=seed, device=args.device, context=ctx)
                except SkipMethod as e:
                    print(f"  [skip] {method}: {e}"); skipped.add(method); break
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] {method} {knob}{v:g} seed{seed}: {type(e).__name__}: {e}"); continue
                row = {knob: v, "method": method, "seed": seed, "stochastic": m.stochastic}
                row.update(compute_all(X, Y, X_truth, include_per_point=False,
                                       labels=base["labels"],
                                       outlier_idx=base.get("outlier_idx"),
                                       outlier_dir=base.get("outlier_dir"),
                                       population=base.get("population")))
                rows.append(row); k += 1
                if seed == 0:
                    emb_seed0[method] = Y
            if k:
                print(f"  {method:9s} runs={k}")
        # one gallery per minority fraction (seed-0 embeddings): the sweep's per-point view
        if knob == "minority_frac" and emb_seed0:
            pct = int(round(v * 100))
            figures.population_gallery(
                emb_seed0, base["labels"], base["population"],
                f"populations {100 - pct}% vs {pct}%", 1.0, figdir,
                fname=f"population_gallery_mf{v:g}.png")

    # method-aware merge (same contract as run/benchmark.py): a method-subset re-run replaces only
    # that method's rows -- keyed without seed so a changed seed structure purges stale seeds
    from run.benchmark import merge_with_existing
    per_run = merge_with_existing(pd.DataFrame(rows), out_dir / f"{mode['stem']}_per_run.csv",
                                  key_cols=[knob, "method"])
    per_run.to_csv(out_dir / f"{mode['stem']}_per_run.csv", index=False)

    # aggregate: median + bootstrap 95% CI per (method, knob value, metric)
    agg = []
    for (method, v), g in per_run.groupby(["method", knob]):
        for metric in mode["metrics"]:
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if not vals.size:
                continue
            lo, hi = bootstrap_median_ci(vals)
            agg.append({"method": method, knob: v, "metric": metric,
                        "median": float(np.median(vals)), "ci_lo": lo, "ci_hi": hi,
                        "n_runs": int(vals.size)})
    agg = pd.DataFrame(agg)
    agg.to_csv(out_dir / f"{mode['stem']}_aggregated.csv", index=False)

    # plot every method present in the MERGED table (not just the ones run this time), so a
    # method-subset re-run still redraws the full curve
    present = [m for m in default_methods() if (agg.method == m).any()]
    if knob == "outlier_factor":
        figures.sweep_outliers_curve(agg, present, figdir)
    else:
        figures.populations_sweep_curve(agg, present, figdir)
    print(f"\nDONE. sweep rows={len(per_run)}  -> {out_dir}/{mode['stem']}_*.csv ; figure -> {figdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
