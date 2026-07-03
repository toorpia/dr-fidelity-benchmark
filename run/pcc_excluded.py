"""Supplement (Phase 1): PCC with outliers EXCLUDED from the reference-point set.

Mechanism probe for the constraint-density hypothesis on the `outliers` dataset — NOT part of the
main benchmark. The main results table (all seven methods, stock settings) is the headline; this
experiment asks *why* burial can happen under sparse constraints, using PCC only because its
reference-point sampling is the one place where the constrained pair set can be intervened on with
a one-line change (PyMDE is the other sparse-constraint case; the main table already covers it).

Intervention (minimal, everything else stock): pccdr's ``PCC.get_reference_points`` with
``sampling="random"`` draws ``np.random.choice(range(N), num_points)`` — WITH replacement. The API
does not accept an explicit reference set, so we subclass and restrict the same call to the bulk
indices: ``np.random.choice(bulk_indices, num_points)``. Same ``num_points = N``, same
with-replacement draw, same loss/optimizer/epochs/seeding as the stock ``PCC`` method wrapper.

Honest notes (also rendered into the REPORT supplement):

* **No causal isolation yet.** There is no INCLUDED control arm (outliers forced INTO the reference
  set), so if burial is observed here, this experiment alone cannot separate mechanism (a)
  "outlier-involving pairs are unconstrained" from mechanism (b) "the Pearson loss is dominated by
  bulk pairs and the outlier's contribution is diluted". Phase 2 (future work): an INCLUDED arm
  would separate them.
* **EXCLUDED is not an adversarial rarity.** Under the stock setting (``num_points = N``, with
  replacement) any given point is absent from the reference set with probability
  ``(1 - 1/N)^N ~= 37%``; the EXCLUDED arm is a deterministic reproduction of a state that stock
  PCC is in for each outlier on roughly every third seed — not a contrived corner case.
* Not swept here (future work): ``num_points`` sweeps, and per-seed logging of whether each outlier
  happened to be sampled as a reference in the stock runs.

    python run/pcc_excluded.py --seeds 3 --dim 768 --n 1000 --snr 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU determinism, as in the driver
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402
torch.set_num_threads(1)
import pandas as pd  # noqa: E402

from metrics.compute import compute_all  # noqa: E402
from run.aggregate import bootstrap_median_ci  # noqa: E402
from run.benchmark import noise_rng  # noqa: E402
from synth import add_noise  # noqa: E402
from synth.outliers import make_outliers  # noqa: E402

METHOD_NAME = "PCC-EXCLUDED"
AGG_METRICS = ["osp_median__vs_ambient", "osp_min__vs_ambient", "log2_osp_median__vs_ambient",
               "iso_rank_delta_mean__vs_ambient", "pair_cohesion_median__vs_ambient",
               "pair_angle_2d_max", "shepard_p5__vs_ambient", "full_shepard"]


def make_excluded_pcc(bulk_idx: np.ndarray):
    """Stock label-free PCC except reference points are drawn from ``bulk_idx`` only."""
    from pcc import PCC

    class PCCExcluded(PCC):
        def get_reference_points(self, data, Np):
            if self.sampling != "random":       # the intervention is defined for the stock sampler
                raise ValueError("PCCExcluded only supports sampling='random'")
            # upstream: np.random.choice(list(range(N)), Np) -- same draw, bulk indices only
            return np.random.choice(bulk_idx, Np)

    return PCCExcluded


def main(argv=None):
    p = argparse.ArgumentParser(description="PCC with outliers excluded from reference sampling")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--snr", default="1")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--num-epochs", type=int, default=500)
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args(argv)

    snr = float("inf") if str(args.snr).lower() in ("inf", "clean", "none") else float(args.snr)
    base = make_outliers(n=args.n, d=args.dim, seed=args.data_seed)
    X = add_noise(base["clean"], snr, noise_rng(args.data_seed, snr))
    X_truth = base["truth_coords"]
    oi = base["outlier_idx"]
    bulk_idx = np.setdiff1d(np.arange(args.n), oi)
    snr_lab = "inf" if not np.isfinite(snr) else f"{snr:g}"

    PCCExcluded = make_excluded_pcc(bulk_idx)
    emb_dir = Path(args.out) / "embeddings" / "outliers" / f"snr{snr_lab}" / METHOD_NAME
    emb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.seeds):
        np.random.seed(int(seed)); torch.manual_seed(int(seed))   # same seeding as the driver
        model = PCCExcluded(cluster=False, pearson=True, spearman=False, n_components=2,
                            num_points=args.n, num_epochs=args.num_epochs)
        Y = model.fit_transform(np.ascontiguousarray(X, dtype=np.float64), np.zeros(args.n))
        Y = np.asarray(Y, dtype=np.float64)
        assert not np.intersect1d(model.indices, oi).size, "an outlier leaked into the references"
        np.save(emb_dir / f"seed{seed}.npy", Y)
        row = dict(dataset="outliers", snr=snr, method=METHOD_NAME, seed=seed, stochastic=True,
                   n=args.n, d=args.dim)
        row.update(compute_all(X, Y, X_truth, include_per_point=False, labels=base["labels"],
                               outlier_idx=oi, outlier_dir=base.get("outlier_dir")))
        rows.append(row)
        print(f"seed{seed}: osp_median(vs ambient)={row['osp_median__vs_ambient']:.3g}")

    per_run = pd.DataFrame(rows)
    out_dir = Path(args.out)
    per_run.to_csv(out_dir / "pcc_excluded_per_run.csv", index=False)

    agg = []
    for metric in AGG_METRICS:
        vals = per_run[metric].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if not vals.size:
            continue
        lo, hi = bootstrap_median_ci(vals)
        agg.append(dict(dataset="outliers", snr=snr, method=METHOD_NAME, metric=metric,
                        median=float(np.median(vals)), ci_lo=lo, ci_hi=hi, n_runs=int(vals.size)))
    pd.DataFrame(agg).to_csv(out_dir / "pcc_excluded_aggregated.csv", index=False)
    print(f"\nDONE. rows={len(per_run)} -> {out_dir}/pcc_excluded_*.csv ; embeddings -> {emb_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
