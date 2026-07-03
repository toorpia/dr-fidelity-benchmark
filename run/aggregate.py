"""Aggregation & statistics over the R seeds, plus run-to-run stability.

Per (dataset, snr, method, metric): median + bootstrap 95% CI + std over seeds. Deterministic
methods have a single run (CI degenerate). Rankings must respect CI overlap (the README states: when
two methods' CIs overlap on a metric, no strict winner is asserted).

Stability: per (dataset, snr, stochastic method), Procrustes-align the R embeddings and report
positional dispersion + mean pairwise disparity, alongside the std of a few headline fidelity metrics
-- showing where "coordinates wobble but structural fidelity is stable".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics.stability import mean_pairwise_disparity, position_dispersion

# columns that identify a run (everything else in a per-run row is a metric value)
ID_COLS = ["dataset", "snr", "method", "seed", "stochastic", "n", "d"]
HEADLINE_METRICS = ["full_shepard", "full_stress_fidelity", "shepard_p5__vs_ambient",
                    "shepard_p100__vs_truth", "recall_k15"]


def bootstrap_median_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the MEDIAN of ``values`` (NaNs dropped)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"), float("nan"))
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(v, size=(n_boot, v.size), replace=True), axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def aggregate_runs(per_run: pd.DataFrame, n_boot: int = 2000, seed: int = 12345) -> pd.DataFrame:
    """Long-format aggregated table: one row per (dataset, snr, method, metric)."""
    metric_cols = [c for c in per_run.columns if c not in ID_COLS]
    records = []
    group_keys = ["dataset", "snr", "method"]
    for (dataset, snr, method), g in per_run.groupby(group_keys, dropna=False):
        stochastic = bool(g["stochastic"].iloc[0])
        for metric in metric_cols:
            vals = g[metric].to_numpy(dtype=np.float64)
            finite = vals[~np.isnan(vals)]
            if finite.size == 0:
                continue
            lo, hi = bootstrap_median_ci(vals, n_boot=n_boot, seed=seed)
            records.append(dict(dataset=dataset, snr=snr, method=method, metric=metric,
                                stochastic=stochastic, n_runs=int(finite.size),
                                median=float(np.median(finite)), mean=float(np.mean(finite)),
                                std=float(np.std(finite, ddof=0)), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame.from_records(records)


def compute_stability(embeddings: dict, per_run: pd.DataFrame) -> pd.DataFrame:
    """Stability table from per-(dataset,snr,method) embedding lists.

    ``embeddings`` maps ``(dataset, snr, method) -> {seed: Y}``. Only stochastic methods with >= 2
    runs yield a row.
    """
    records = []
    for (dataset, snr, method), seed_map in embeddings.items():
        Ys = [seed_map[s] for s in sorted(seed_map)]
        if len(Ys) < 2:
            continue
        sub = per_run[(per_run.dataset == dataset) & (per_run.snr == snr) & (per_run.method == method)]
        rec = dict(dataset=dataset, snr=snr, method=method, n_runs=len(Ys),
                   position_dispersion=position_dispersion(Ys),
                   mean_pairwise_disparity=mean_pairwise_disparity(Ys))
        for m in HEADLINE_METRICS:
            if m in sub.columns:
                vals = sub[m].to_numpy(dtype=np.float64)
                vals = vals[~np.isnan(vals)]
                rec[f"{m}__std"] = float(np.std(vals, ddof=0)) if vals.size else float("nan")
        records.append(rec)
    return pd.DataFrame.from_records(records)
