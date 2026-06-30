"""Run-to-run embedding stability for stochastic methods.

Stochastic DR methods produce a different 2-D coordinate frame on every run (a rotation / reflection
/ translation / scale gauge that carries no structural meaning). To separate "the coordinates wobble"
from "the structure is unstable", we align the R embeddings with Procrustes (which removes exactly
that gauge) and then report:

* per-point position dispersion AFTER alignment (mean over points of the std of aligned coordinates
  across runs) -- how much the *coordinates* move; and
* the variance / std of each fidelity metric across runs -- how much the *structural fidelity* moves.

A method can show large coordinate dispersion yet tiny fidelity variance ("coordinates wobble but
structural fidelity is stable"). We use ``scipy.spatial.procrustes`` for full gauge removal
(translation + uniform scale + rotation + reflection).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import procrustes


def _standardize(Y: np.ndarray) -> np.ndarray:
    """Center and Frobenius-normalize, matching scipy.spatial.procrustes' internal convention."""
    Yc = Y - Y.mean(axis=0)
    norm = np.linalg.norm(Yc)
    if norm == 0:
        return Yc
    return Yc / norm


def align_to_reference(embeddings: list[np.ndarray], ref_index: int = 0) -> list[np.ndarray]:
    """Procrustes-align every embedding onto a common reference frame.

    Returns standardized, aligned copies (each centered + Frobenius-normalized, mapped onto the
    standardized reference). ``procrustes(ref, Y)`` returns ``(ref_std, Y_aligned, disparity)``.
    """
    ref = _standardize(embeddings[ref_index])
    aligned = []
    for Y in embeddings:
        _, Y_aligned, _ = procrustes(ref, Y)
        aligned.append(Y_aligned)
    return aligned


def position_dispersion(embeddings: list[np.ndarray], ref_index: int = 0) -> float:
    """Mean per-point positional std across runs, after Procrustes alignment.

    For each point, compute the std (over runs) of its aligned x and y, take the Euclidean magnitude,
    then average over points. 0 means the aligned coordinates are identical across runs.
    """
    if len(embeddings) < 2:
        return 0.0
    aligned = np.stack(align_to_reference(embeddings, ref_index), axis=0)  # (R, n, 2)
    per_point_std = aligned.std(axis=0)            # (n, 2)
    per_point_mag = np.sqrt((per_point_std ** 2).sum(axis=1))  # (n,)
    return float(per_point_mag.mean())


def mean_pairwise_disparity(embeddings: list[np.ndarray]) -> float:
    """Mean Procrustes disparity over all unordered run-pairs (0 = identical up to gauge)."""
    R = len(embeddings)
    if R < 2:
        return 0.0
    vals = []
    for i in range(R):
        for j in range(i + 1, R):
            _, _, d = procrustes(embeddings[i], embeddings[j])
            vals.append(d)
    return float(np.mean(vals))


def metric_variance(values: np.ndarray) -> dict:
    """Median / std / variance of a 1-D array of a fidelity metric across runs."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"median": float("nan"), "std": float("nan"), "var": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "std": float(v.std(ddof=0)),
            "var": float(v.var(ddof=0)), "n": int(v.size)}
