"""Dynamic-range sweep on the `clusters` dataset.

For each dynamic_range, generate the high-dim `clusters` dataset (CLEAN), embed every method over R
seeds, and record the near-band (`p5`) and global (`full`) Shepard rho. The output curve traces how
each method's near and global distance fidelity varies as the inter- to intra-cluster ratio grows.

    python run/sweep.py --dynamic-range 2 5 10 20 50 --methods all --seeds 20
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
from synth.clusters import make_clusters  # noqa: E402

SWEEP_METRICS = ["shepard_p5__vs_ambient", "shepard_p10__vs_ambient", "full_shepard",
                 "shepard_p100__vs_truth"]


def set_seeds(seed):
    np.random.seed(int(seed)); torch.manual_seed(int(seed))


def main(argv=None):
    p = argparse.ArgumentParser(description="dynamic-range sweep on the clusters dataset")
    p.add_argument("--dynamic-range", nargs="+", type=float, default=[2, 5, 10, 20, 50])
    p.add_argument("--methods", default="all")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--n-clusters", type=int, default=7)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures" / "clusters"))
    args = p.parse_args(argv)

    methods = default_methods() if args.methods == "all" else args.methods.split(",")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    print(f"sweep dynamic_range={args.dynamic_range} methods={methods} seeds={args.seeds} "
          f"N={args.n} D={args.dim} K={args.n_clusters}")

    rows, skipped = [], set()
    for dr in args.dynamic_range:
        base = make_clusters(n=args.n, d=args.dim, seed=args.data_seed,
                             n_clusters=args.n_clusters, dynamic_range=dr)
        X, X_truth = base["clean"], base["truth_coords"]
        print(f"\n=== dynamic_range={dr:g}  (intra_nn={base['intra_nn']:.3f}) ===")
        for method in methods:
            if method in skipped:
                continue
            m = get_method(method)
            seeds = range(args.seeds) if m.stochastic else [0]
            ctx = {"root": str(ROOT), "dataset": "clusters_sweep", "snr": float("inf"),
                   "tag": f"dr{dr:g}"}
            k = 0
            for seed in seeds:
                set_seeds(seed)
                try:
                    Y = m.embed(X, seed=seed, device=args.device, context=ctx)
                except SkipMethod as e:
                    print(f"  [skip] {method}: {e}"); skipped.add(method); break
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] {method} dr{dr:g} seed{seed}: {type(e).__name__}: {e}"); continue
                row = dict(dynamic_range=dr, method=method, seed=seed, stochastic=m.stochastic)
                row.update(compute_all(X, Y, X_truth, include_per_point=False))
                rows.append(row); k += 1
            if k:
                print(f"  {method:9s} runs={k}")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "sweep_per_run.csv", index=False)

    # aggregate: median + bootstrap 95% CI per (method, dynamic_range, metric)
    agg = []
    for (method, dr), g in per_run.groupby(["method", "dynamic_range"]):
        for metric in SWEEP_METRICS:
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if not vals.size:
                continue
            lo, hi = bootstrap_median_ci(vals)
            agg.append(dict(method=method, dynamic_range=dr, metric=metric,
                            median=float(np.median(vals)), ci_lo=lo, ci_hi=hi, n_runs=int(vals.size)))
    agg = pd.DataFrame(agg)
    agg.to_csv(out_dir / "sweep_aggregated.csv", index=False)

    present = [m for m in methods if (agg.method == m).any()]
    figures.dynamic_range_curve(agg, present, figdir)
    print(f"\nDONE. sweep rows={len(per_run)}  -> {out_dir}/sweep_*.csv ; figure -> {figdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
