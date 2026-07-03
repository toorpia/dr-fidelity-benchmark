"""Noise-robustness panel: clusters at a fixed high dynamic range under INTRA-cluster-relative noise.

Sweeps a per-cluster-relative SNR (noise calibrated to the within-cluster spread, not the global
scale — see synth.noise.add_noise_relative) so that noise is a meaningful level for the local
structure being measured. Checks whether the near-band ordering persists at realistic noise, not only
in the clean limit.

    python run/robustness.py --snr inf 8 2 --dynamic-range 20 --methods all --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from synth.noise import add_noise_relative  # noqa: E402

SWEEP_METRICS = ["shepard_p5__vs_ambient", "full_shepard"]


def parse_snr(tokens):
    return [float("inf") if str(t).lower() in ("inf", "clean") else float(t) for t in tokens]


def main(argv=None):
    p = argparse.ArgumentParser(description="intra-cluster-relative noise robustness on clusters")
    p.add_argument("--snr", nargs="+", default=["inf", "8", "2"])
    p.add_argument("--dynamic-range", type=float, default=20.0)
    p.add_argument("--methods", default="all")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures" / "clusters"))
    args = p.parse_args(argv)

    methods = default_methods() if args.methods == "all" else args.methods.split(",")
    snrs = parse_snr(args.snr)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)

    base = make_clusters(n=args.n, d=args.dim, seed=args.data_seed, dynamic_range=args.dynamic_range)
    clean, labels, X_truth = base["clean"], base["labels"], base["truth_coords"]
    rows, skipped = [], set()
    for snr in snrs:
        snr_lab = "inf" if not np.isfinite(snr) else f"{snr:g}"
        key = 10 ** 9 if not np.isfinite(snr) else int(round(snr * 1000))
        X = add_noise_relative(clean, labels, snr, np.random.default_rng(
            np.random.SeedSequence([args.data_seed, key])))
        print(f"\n=== relative SNR={snr_lab} (dynamic_range={args.dynamic_range:g}) ===")
        for method in methods:
            if method in skipped:
                continue
            m = get_method(method)
            seeds = range(args.seeds) if m.stochastic else [0]
            ctx = {"root": str(ROOT), "dataset": "clusters_robust", "snr": snr,
                   "tag": f"dr{args.dynamic_range:g}_rsnr{snr_lab}"}
            kk = 0
            for seed in seeds:
                np.random.seed(seed); torch.manual_seed(seed)
                try:
                    Y = m.embed(X, seed=seed, device="cpu", context=ctx)
                except SkipMethod as e:
                    print(f"  [skip] {method}: {e}"); skipped.add(method); break
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] {method} rsnr{snr_lab} seed{seed}: {e}"); continue
                row = dict(rel_snr=snr, method=method, seed=seed, stochastic=m.stochastic)
                row.update(compute_all(X, Y, X_truth, include_per_point=False))
                rows.append(row); kk += 1
            if kk:
                print(f"  {method:9s} runs={kk}")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "robustness_per_run.csv", index=False)
    agg = []
    for (method, snr), g in per_run.groupby(["method", "rel_snr"], dropna=False):
        for metric in SWEEP_METRICS:
            vals = g[metric].to_numpy(dtype=float); vals = vals[~np.isnan(vals)]
            if not vals.size:
                continue
            lo, hi = bootstrap_median_ci(vals)
            agg.append(dict(method=method, rel_snr=snr, metric=metric, median=float(np.median(vals)),
                            ci_lo=lo, ci_hi=hi, n_runs=int(vals.size)))
    pd.DataFrame(agg).to_csv(out_dir / "robustness_aggregated.csv", index=False)
    print(f"\nDONE robustness. rows={len(per_run)} -> {out_dir}/robustness_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
