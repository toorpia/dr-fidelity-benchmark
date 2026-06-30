"""Pairwise-distance helpers and distance-band definitions.

All metrics in this package operate on EXACT pairwise distances (``scipy.spatial.distance.pdist``),
independent of any DR method's internal reference-point approximation.

Distance bands are the central methodological device of this benchmark. A *cumulative* band at
cutoff ``p`` percent is the set of point-pairs whose HIGH-D distance falls in the lowest ``p`` percent
of the global pairwise-distance distribution. ``p = 100`` therefore selects all pairs and recovers the
classic global Shepard/stress numbers. Sweeping ``p`` yields a near -> mid -> far fidelity profile.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

# Default band cutoffs (percent of the global high-D pairwise-distance distribution).
DEFAULT_CUTOFFS = (5, 10, 20, 30, 50, 75, 100)


def condensed_distances(X: np.ndarray) -> np.ndarray:
    """Return the condensed (1-D) vector of Euclidean pairwise distances for ``X`` (n x d)."""
    return pdist(np.ascontiguousarray(X, dtype=np.float64), metric="euclidean")


def band_thresholds(d_ref: np.ndarray, cutoffs=DEFAULT_CUTOFFS) -> dict:
    """Map each cutoff ``p`` to the p-th percentile of the reference distance vector ``d_ref``.

    Pairs with ``d_ref <= threshold[p]`` constitute the cumulative band for cutoff ``p``.
    """
    return {int(p): float(np.percentile(d_ref, p)) for p in cutoffs}


def band_mask(d_ref: np.ndarray, p: float) -> np.ndarray:
    """Boolean mask selecting the cumulative lowest-``p``-percent band on ``d_ref``."""
    if p >= 100:
        return np.ones_like(d_ref, dtype=bool)
    thr = np.percentile(d_ref, p)
    return d_ref <= thr


def square_distances(X: np.ndarray) -> np.ndarray:
    """Full (n x n) Euclidean distance matrix (diagonal = 0)."""
    return squareform(condensed_distances(X))
