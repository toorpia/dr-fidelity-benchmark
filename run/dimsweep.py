"""Curse-of-dimensionality sweep on the ``noise_dims`` dataset (3 tight clusters + noise dims).

For each total dimensionality D, generate the noise-dims dataset (3 tight clusters in 3 standardized
signal columns + D-3 standard-normal noise columns), embed every method over R seeds, and record
distance fidelity vs the 3-D truth plus 2-D label-separation scores (kNN accuracy / silhouette).
Quantified replication of the notebook experiment ``01_hi_dimensional_data_anlaysis.ipynb``.

    python run/dimsweep.py --dims 6 10 20 40 80 100 200 400 768 --methods all --seeds 3 --n 500

Unlike ``run/sweep.py``, a ``SkipMethod`` here skips only the current dim (no permanent blacklist):
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


def main(argv=None):
    p = argparse.ArgumentParser(description="dimensionality sweep on the noise-dims dataset")
    p.add_argument("--dims", nargs="+", type=int,
                   default=[6, 10, 20, 40, 80, 100, 200, 400, 768])
    p.add_argument("--methods", default="all")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-principal", type=int, default=3)
    p.add_argument("--cluster-std", type=float, default=0.005)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures" / "noise_dims"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    methods = default_methods() if args.methods == "all" else args.methods.split(",")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    knn_key = f"knn_acc_k{args.knn_k}"
    print(f"dimsweep dims={args.dims} methods={methods} seeds={args.seeds} "
          f"N={args.n} n_principal={args.n_principal} cluster_std={args.cluster_std:g}")

    rows = []
    grid = {m: {} for m in methods}
    labels_ref = None
    for dim in args.dims:
        base = make_noise_dims(n=args.n, d=dim, seed=args.data_seed,
                               n_principal=args.n_principal, cluster_std=args.cluster_std)
        X, X_truth, labels = base["clean"], base["truth_coords"], base["labels"]
        labels_ref = labels
        print(f"\n=== D={dim}  ({dim - args.n_principal} noise dims) ===", flush=True)
        for method in methods:
            m = get_method(method)
            seeds = range(args.seeds) if m.stochastic else [0]
            ctx = {"root": str(ROOT), "dataset": "noise_dims", "snr": float("inf"),
                   "tag": f"n{args.n}_dim{dim}"}
            k = 0
            for seed in seeds:
                set_seeds(seed)
                try:
                    Y = m.embed(X, seed=seed, device=args.device, context=ctx)
                except SkipMethod as e:
                    print(f"  [skip] {method} dim{dim}: {e}"); break
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] {method} dim{dim} seed{seed}: {type(e).__name__}: {e}"); continue
                emb_dir = out_dir / "embeddings" / "noise_dims" / f"dim{dim}" / method
                emb_dir.mkdir(parents=True, exist_ok=True)
                np.save(emb_dir / f"seed{seed}.npy", Y)
                if dim not in grid[method]:
                    grid[method][dim] = Y
                row = dict(dim=dim, n=args.n, method=method, seed=seed, stochastic=m.stochastic)
                row.update(compute_all(X, Y, X_truth, include_per_point=False, labels=labels))
                row[knn_key] = knn_label_accuracy(Y, labels, k=args.knn_k)
                row["silhouette_2d"] = silhouette_by_label(Y, labels)
                rows.append(row); k += 1
            if k:
                print(f"  {method:9s} runs={k}", flush=True)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "dimsweep_per_run.csv", index=False)

    # aggregate: median + bootstrap 95% CI per (method, dim, metric)
    sweep_metrics = ["shepard_p100__vs_truth", "shepard_p20__vs_truth", "full_shepard",
                     "shepard_p5__vs_ambient", knn_key, "silhouette_2d",
                     "cluster_over_compression", "recall_k15"]
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
    agg.to_csv(out_dir / "dimsweep_aggregated.csv", index=False)

    if not args.no_figures and len(per_run):
        present = [m for m in methods if grid.get(m)]
        dims_present = sorted({dm for m in present for dm in grid[m]})
        # landmark subset keeps the grid readable when many methods run (7 x 9 = 63 panels is
        # unreadable); fall back to everything for small custom sweeps
        landmark = [dm for dm in (6, 40, 100, 200, 768) if dm in dims_present]
        grid_dims = landmark if len(landmark) >= 3 else dims_present
        figures.dims_grid(grid, labels_ref, present, grid_dims, figdir)
        if len(agg):
            figures.dimension_curve(
                agg, present, figdir,
                panels=(("full_shepard",
                         "global Shepard ρ — vs ambient (the features as given)"),
                        (knn_key, f"2-D kNN label accuracy (k={args.knn_k})")))

    if len(per_run):
        pivot = per_run.pivot_table(index="method", columns="dim", values=knn_key, aggfunc="median")
        print(f"\n{knn_key} (chance = 1/{args.n_principal}), median over seeds:")
        print(pivot.round(3).to_string())
    print(f"\nDONE. rows={len(per_run)}  -> {out_dir}/dimsweep_*.csv ; figures -> {figdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
