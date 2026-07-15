"""Baseline-hyperparameter sensitivity check: does tuning move the tunable methods' headline scores?

The benchmark runs every open-source method at its library defaults (t-SNE perplexity=30, UMAP
n_neighbors=15, DREAMS reg_lambda=0.15) and toorPIA is placement-knob-free, so the standing
objection is "these methods would rank differently if tuned". This supplement sweeps the one
placement-critical knob of each method (t-SNE / UMAP neighborhood size over 5-100; DREAMS
local-global regularization strength over 0.05-0.5) on two datasets (density, clusters; SNR=1 --
the report's canonical noise level) and records the two RANKED metrics (full Shepard rho,
first-mode near-band rho) plus the neighbor methods' home-turf reference metric (recall@15).

Existing rows for (dataset, method, value, seed) combinations NOT re-run are kept verbatim
(same merge convention as run/benchmark.py), so a single-method sweep extends the committed
tables without perturbing them.

    python run/hyperparam_sensitivity.py --seeds 3              # all tunable methods
    python run/hyperparam_sensitivity.py --methods DREAMS       # extend with one method
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

from methods.dreams_method import embed_dreams  # noqa: E402
from methods.tsne import embed_tsne  # noqa: E402
from methods.umap_method import embed_umap  # noqa: E402
from metrics.compute import compute_all  # noqa: E402
from run.aggregate import bootstrap_median_ci  # noqa: E402
from run.benchmark import merge_with_existing, noise_rng  # noqa: E402
from synth import add_noise, make_dataset  # noqa: E402

# method -> (knob, embed fn, sweep values, library default)
KNOBS = {
    "t-SNE": ("perplexity", embed_tsne, [5, 15, 30, 50, 100], 30),
    "UMAP": ("n_neighbors", embed_umap, [5, 15, 30, 50, 100], 15),
    "DREAMS": ("reg_lambda", embed_dreams, [0.05, 0.1, 0.15, 0.3, 0.5], 0.15),
}
DETERMINISTIC = {"DREAMS"}  # one run per knob value (same treatment as in run/benchmark.py)
METRICS = ["full_shepard", "shepard_near__vs_ambient", "recall_k15"]


def main(argv=None):
    p = argparse.ArgumentParser(description="baseline hyperparameter sensitivity (t-SNE/UMAP/DREAMS)")
    p.add_argument("--methods", default="all", help="comma list or 'all'")
    p.add_argument("--datasets", default="density,clusters")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--snr", type=float, default=1.0)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures" / "hyperparam"))
    args = p.parse_args(argv)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    methods = list(KNOBS) if args.methods == "all" else args.methods.split(",")
    rows = []
    for ds in args.datasets.split(","):
        base = make_dataset(ds, n=args.n, d=args.dim, seed=args.data_seed)
        X = add_noise(base["clean"], args.snr, noise_rng(args.data_seed, args.snr))
        for method in methods:
            knob, fn, values, _default = KNOBS[method]
            for v in values:
                seeds = [0] if method in DETERMINISTIC else range(args.seeds)
                for seed in seeds:
                    np.random.seed(seed); torch.manual_seed(seed)
                    Y = fn(X, seed, device="cpu", **{knob: v})
                    row = dict(dataset=ds, method=method, knob=knob, value=v, seed=seed)
                    row.update({k: val for k, val in
                                compute_all(X, Y, base["truth_coords"], include_per_point=False,
                                            labels=base["labels"]).items() if k in METRICS})
                    rows.append(row)
                print(f"{ds} {method} {knob}={v} done", flush=True)
    per_run = merge_with_existing(pd.DataFrame(rows),
                                  out_dir / "hyperparam_sensitivity_per_run.csv",
                                  ["dataset", "method", "value", "seed"])
    # int knobs (perplexity 5..100) and float knobs (reg_lambda 0.05..0.5) share this column;
    # keep ints as ints (object dtype -- plain float64 would render 5 as "5.0") so kept rows
    # survive the rewrite byte-identically
    per_run["value"] = pd.Series([int(v) if float(v).is_integer() else float(v)
                                  for v in per_run["value"]],
                                 dtype=object, index=per_run.index)
    per_run.to_csv(out_dir / "hyperparam_sensitivity_per_run.csv", index=False)

    agg = []
    for (ds, method, v), g in per_run.groupby(["dataset", "method", "value"]):
        for metric in METRICS:
            vals = g[metric].to_numpy(dtype=float)
            lo, hi = bootstrap_median_ci(vals)
            agg.append(dict(dataset=ds, method=method, knob=g.knob.iloc[0], value=v,
                            metric=metric, median=float(np.median(vals)), ci_lo=lo, ci_hi=hi,
                            n_runs=len(vals)))
    agg = pd.DataFrame(agg)
    agg.to_csv(out_dir / "hyperparam_sensitivity_aggregated.csv", index=False)

    # figure: datasets x tunable methods, metric curves vs knob value, default marked
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    dss = args.datasets.split(",")
    fig_methods = [m for m in KNOBS if (agg.method == m).any()]
    fig, axes = plt.subplots(len(dss), len(fig_methods),
                             figsize=(5.75 * len(fig_methods), 4.6 * len(dss)), squeeze=False)
    colors = {"full_shepard": "#0b6", "shepard_near__vs_ambient": "#234a9e", "recall_k15": "#a33"}
    names = {"full_shepard": "full ρ (ranked)", "shepard_near__vs_ambient": "near ρ (ranked, first-mode band)",
             "recall_k15": "recall@15 (reference)"}
    for i, ds in enumerate(dss):
        for j, method in enumerate(fig_methods):
            knob, _fn, values, default = KNOBS[method]
            ax = axes[i][j]
            sub = agg[(agg.dataset == ds) & (agg.method == method)]
            for metric in METRICS:
                r = sub[sub.metric == metric].sort_values("value")
                ax.plot(r.value, r["median"], marker="o", label=names[metric],
                        color=colors[metric])
                ax.fill_between(r.value, r.ci_lo, r.ci_hi, alpha=0.15, color=colors[metric])
            ax.axvline(default, color="#888", lw=1.0, ls="--")
            ax.annotate("default", xy=(default, 0.02),
                        xycoords=("data", "axes fraction"), fontsize=8, color="#888",
                        ha="center")
            ax.set_xscale("log"); ax.set_xticks(values); ax.set_xticklabels(values)
            ax.set_xlabel(f"{method} {knob}")
            ax.set_ylabel("score"); ax.set_title(f"{ds} (SNR=1) — {method}", fontsize=10)
            ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Baseline hyperparameter sensitivity (t-SNE perplexity / UMAP n_neighbors, 5–100; "
                 "DREAMS reg_lambda, 0.05–0.5; defaults dashed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "sensitivity_snr1.png", dpi=130)
    print(f"\nDONE. rows={len(per_run)} -> {out_dir}/hyperparam_sensitivity_*.csv ; "
          f"figure -> {figdir}/sensitivity_snr1.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
