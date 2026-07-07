"""Extend the committed main-benchmark tables with derived metric columns.

Currently: the structure-adaptive near-band family and the tightest-cluster scale family
(the latter replaces the legacy cluster_over_compression columns, which are dropped).

Embeds nothing and touches no existing column: for every archived embedding
``results/embeddings/{dataset}/snr{lab}/{method}/seed{seed}.npy`` it regenerates the exact
benchmark inputs (same data / noise rng) and computes ONLY the first-mode near-band columns
(``shepard_near`` / ``stress_near`` / ``stress_fidelity_near`` / ``near_band_pct`` /
``near_band_fallback``, each ``__vs_ambient`` and ``__vs_truth``), then merges them into
``results/metrics_per_run.csv`` by (dataset, snr, method, seed) and rebuilds the aggregate.
Used when the metric set gains columns so the committed tables extend without re-running any
method; fresh ``run/benchmark.py`` runs emit the same columns natively via ``compute_all``.

    python run/remetric.py --snr 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from metrics.distances import condensed_distances, first_mode_threshold  # noqa: E402
from metrics.stress import normalized_stress, stress_to_fidelity  # noqa: E402
from run import aggregate  # noqa: E402
from run.benchmark import noise_rng  # noqa: E402
from run.figures import snr_label  # noqa: E402
from synth import add_noise, list_datasets, make_dataset  # noqa: E402

NEAR_COLS = [f"{k}__{t}" for t in ("vs_ambient", "vs_truth")
             for k in ("shepard_near", "stress_near", "stress_fidelity_near",
                       "near_band_pct", "near_band_fallback")]
TIGHT_COLS = ["tight_cluster", "tight_ratio_truth", "tight_ratio_2d", "tight_over_compression"]
# superseded by the tightest-cluster metric; dropped from the tables on recompute
LEGACY_COLS = ["cluster_ratio_truth", "cluster_ratio_2d", "cluster_over_compression"]


def near_block(d_ref: np.ndarray, d_2d: np.ndarray, tag: str) -> dict:
    thr, fallback = first_mode_threshold(d_ref)
    m = d_ref <= thr
    rho, _ = spearmanr(d_ref[m], d_2d[m])
    st = normalized_stress(d_ref[m], d_2d[m])
    return {f"shepard_near__{tag}": float(rho), f"stress_near__{tag}": st,
            f"stress_fidelity_near__{tag}": stress_to_fidelity(st),
            f"near_band_pct__{tag}": float(100.0 * m.mean()),
            f"near_band_fallback__{tag}": float(fallback)}


def main(argv=None):
    p = argparse.ArgumentParser(description="merge first-mode near-band metrics into the tables")
    p.add_argument("--snr", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    per_run = pd.read_csv(out_dir / "metrics_per_run.csv", float_precision="round_trip")
    lab = snr_label(args.snr)

    from metrics.cluster_scale import tight_cluster_metrics
    rows = []
    for ds in list_datasets():
        base = make_dataset(ds, n=args.n, d=args.dim, seed=args.data_seed)
        X = add_noise(base["clean"], args.snr, noise_rng(args.data_seed, args.snr))
        d_amb = condensed_distances(X)
        d_truth = condensed_distances(base["truth_coords"])
        emb_root = out_dir / "embeddings" / ds / f"snr{lab}"
        for mdir in sorted(emb_root.iterdir()):
            for f in sorted(mdir.glob("seed*.npy")):
                Y = np.load(f)
                d_2d = condensed_distances(Y)
                row = dict(dataset=ds, snr=args.snr, method=mdir.name,
                           seed=int(f.stem.replace("seed", "")))
                row.update(near_block(d_amb, d_2d, "vs_ambient"))
                row.update(near_block(d_truth, d_2d, "vs_truth"))
                row.update(tight_cluster_metrics(base["truth_coords"], Y, base["labels"]))
                rows.append(row)
        print(f"{ds}: near band = p{rows[-1]['near_band_pct__vs_ambient']:.1f} "
              f"(fallback={bool(rows[-1]['near_band_fallback__vs_ambient'])})", flush=True)
    new = pd.DataFrame(rows)

    key = ["dataset", "snr", "method", "seed"]
    merged = per_run.drop(columns=[c for c in NEAR_COLS + TIGHT_COLS + LEGACY_COLS
                                   if c in per_run], errors="ignore")
    merged = merged.merge(new, on=key, how="left", validate="one_to_one")
    assert len(merged) == len(per_run), "row count changed"
    assert merged["shepard_near__vs_ambient"].notna().all(), "some rows got no near metrics"
    assert merged["tight_over_compression"].notna().all(), "some rows got no tight metrics"
    merged.to_csv(out_dir / "metrics_per_run.csv", index=False)

    agg = aggregate.aggregate_runs(merged, n_boot=args.bootstrap)
    agg.to_csv(out_dir / "metrics_aggregated.csv", index=False)
    print(f"DONE. rows={len(merged)} (+{len(NEAR_COLS)} cols), aggregated rows={len(agg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
