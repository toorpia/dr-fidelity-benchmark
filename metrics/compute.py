"""Top-level metric aggregation producing one flat dict per (method, seed, dataset, snr) run.

Every metric is computed EXACTLY on all pairwise distances. Where the dataset supplies a ground-truth
generating geometry (``X_truth``), distance-band metrics are reported BOTH against the ambient
(noisy) high-D distances the method actually saw (``__vs_ambient``) and against the clean
ground-truth distances (``__vs_truth``). Neighbor-preservation metrics (recall@k / trustworthiness /
continuity) are defined on the input the method received and are reported vs-ambient only.
"""
from __future__ import annotations

import numpy as np

from .distances import DEFAULT_CUTOFFS, condensed_distances, square_distances
from .neighbors import DEFAULT_KS, recall_at_k, trustworthiness_continuity
from .shepard import band_shepard, per_point_band_shepard
from .stress import band_stress, stress_to_fidelity


def _band_block(d_hd: np.ndarray, d_2d: np.ndarray, tag: str, cutoffs) -> dict:
    out = {}
    for p, rho in band_shepard(d_hd, d_2d, cutoffs).items():
        out[f"shepard_p{p}__{tag}"] = rho
    for p, st in band_stress(d_hd, d_2d, cutoffs).items():
        out[f"stress_p{p}__{tag}"] = st
        out[f"stress_fidelity_p{p}__{tag}"] = stress_to_fidelity(st)
    return out


def compute_all(X_ambient: np.ndarray, Y: np.ndarray, X_truth: np.ndarray | None = None,
                cutoffs=DEFAULT_CUTOFFS, ks=DEFAULT_KS, include_per_point: bool = True,
                labels: np.ndarray | None = None,
                outlier_idx: np.ndarray | None = None,
                outlier_dir: np.ndarray | None = None,
                population: np.ndarray | None = None) -> dict:
    """Compute the full metric row for one embedding ``Y`` of ``X_ambient``.

    Returns a flat ``{metric_name: value}`` dict. Keys use ``shepard_p{p}``, ``stress_p{p}``,
    ``stress_fidelity_p{p}``, ``pp_shepard_p{p}`` (per-point variant), suffixed ``__vs_ambient`` or
    ``__vs_truth``; plus ``recall_k{k}``, ``trust_k{k}``, ``cont_k{k}`` and the full-data specials
    ``full_shepard``/``full_stress``/``full_stress_fidelity`` (= the p=100 numbers, vs-ambient).
    """
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    d_2d = condensed_distances(Y)

    # --- distance-band metrics vs ambient (the noisy high-D the method embedded) ---
    d_amb = condensed_distances(X_ambient)
    row = _band_block(d_amb, d_2d, "vs_ambient", cutoffs)
    if include_per_point:
        for p, rho in per_point_band_shepard(X_ambient, Y, cutoffs).items():
            row[f"pp_shepard_p{p}__vs_ambient"] = rho

    # classic global specials (p = 100)
    row["full_shepard"] = row["shepard_p100__vs_ambient"]
    row["full_stress"] = row["stress_p100__vs_ambient"]
    row["full_stress_fidelity"] = row["stress_fidelity_p100__vs_ambient"]

    # --- distance-band metrics vs ground-truth generating distances ---
    if X_truth is not None:
        d_truth = condensed_distances(X_truth)
        row.update(_band_block(d_truth, d_2d, "vs_truth", cutoffs))
        if include_per_point:
            d_truth_sq = square_distances(X_truth)
            for p, rho in per_point_band_shepard(X_truth, Y, cutoffs, d_hd_sq=d_truth_sq).items():
                row[f"pp_shepard_p{p}__vs_truth"] = rho

    # --- within-cluster scale preservation (value/density metric; needs cluster labels) ---
    if labels is not None and X_truth is not None:
        from .cluster_scale import cluster_scale_metrics
        labs = np.asarray(labels)
        if np.unique(labs[labs >= 0]).size >= 2:
            row.update(cluster_scale_metrics(X_truth, Y, labs))

    # --- outlier separation preservation (single-point metric; needs ground-truth outlier ids) ---
    # Reported vs-ambient ONLY, matching the repo's primary axis (how faithfully the 2-D map
    # renders the D-dim input the method actually saw); outlier_metrics itself stays generic.
    if outlier_idx is not None and len(outlier_idx):
        from .outlier import outlier_metrics, outlier_pair_metrics, outlier_shepard
        # standard Shepard rho over the anomaly-involving pairs (same statistic as the band rho,
        # pair subset selected by endpoint membership) -- the primary outlier reading
        row["outlier_shepard__vs_ambient"] = outlier_shepard(X_ambient, Y, outlier_idx)
        row.update(outlier_metrics(X_ambient, Y, outlier_idx, tag="vs_ambient"))
        if outlier_dir is not None:
            row.update(outlier_pair_metrics(X_ambient, Y, outlier_idx, outlier_dir,
                                            tag="vs_ambient"))

    # --- population-membership metrics (imbalanced two-population dataset) ---
    if population is not None and labels is not None and X_truth is not None:
        from .populations import population_metrics
        row.update(population_metrics(X_ambient, Y, X_truth, population, labels))

    # --- conventional neighbor-preservation metrics (vs ambient, documented-as-biased) ---
    for k, v in recall_at_k(X_ambient, Y, ks).items():
        row[f"recall_k{k}"] = v
    tc = trustworthiness_continuity(X_ambient, Y, ks)
    for k, v in tc["trustworthiness"].items():
        row[f"trust_k{k}"] = v
    for k, v in tc["continuity"].items():
        row[f"cont_k{k}"] = v

    return row
