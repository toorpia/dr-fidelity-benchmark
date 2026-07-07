"""Pairwise-distance helpers and distance-band definitions.

All metrics in this package operate on EXACT pairwise distances (``scipy.spatial.distance.pdist``),
independent of any DR method's internal reference-point approximation.

Distance bands are the central methodological device of this benchmark. A *cumulative* band at
cutoff ``p`` percent is the set of point-pairs whose HIGH-D distance falls in the lowest ``p`` percent
of the global pairwise-distance distribution. ``p = 100`` therefore selects all pairs and recovers the
classic global Shepard/stress numbers. Sweeping ``p`` yields a near -> mid -> far fidelity profile.

The HEADLINE near band is structure-adaptive rather than a fixed percentile: the pairwise-distance
profile of structured data is multimodal (first mode = within-structure pairs), and
``first_mode_threshold`` places the near/far boundary at the density valley where that first mode
decays into the tail. On this benchmark's datasets the boundary lands at p14-p20 and, on the
clustered datasets, coincides exactly with the true within-cluster pair fraction. The fixed
percentile bands remain in the CSVs as the reference profile.
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


def first_mode_threshold(d_ref: np.ndarray, bins: int = 256, smooth: int = 9,
                         order: int = 4, rel_height: float = 0.05,
                         dip: float = 0.95) -> tuple[float, bool]:
    """Radius where the FIRST mode of the pairwise-distance profile decays into the tail.

    The near band of structured data is not an arbitrary percentile: the pairwise-distance
    profile is multimodal, its first mode being the within-structure pairs. This estimator is
    deterministic and uses all pairs — histogram over ``bins`` equal-width bins, smoothed twice
    with a length-``smooth`` boxcar, then the first local minimum (``order``-neighborhood) after
    the first local maximum. The valley must sit below the median distance (a "first mode" that
    covers most pairs is not a near structure).

    Returns ``(threshold, fallback)``. When the smoothed profile has no second mode
    (effectively unimodal), falls back to the 5th-percentile radius with ``fallback=True``.
    """
    from scipy.signal import argrelmax

    counts, edges = np.histogram(d_ref, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = np.ones(smooth) / smooth
    ys = np.convolve(np.convolve(counts.astype(float), w, mode="same"), w, mode="same")
    # valley = argmin BETWEEN the first two modes (argrelmax is strict, so plateaus between the
    # modes cannot hide the valley the way a strict local-minimum search can). A candidate first
    # mode must be substantial (>= rel_height of the profile maximum -- tail-noise bumps are not
    # modes) and the valley must be a real dip (< dip of the first mode's height).
    maxima = [int(i) for i in argrelmax(ys, order=order)[0]
              if ys[i] >= rel_height * float(ys.max())]
    if len(maxima) >= 2:
        m0, m1 = maxima[0], maxima[1]
        v = m0 + int(np.argmin(ys[m0:m1 + 1]))
        if centers[v] < float(np.median(d_ref)) and ys[v] < dip * ys[m0]:
            return float(centers[v]), False
    return float(np.percentile(d_ref, 5)), True


def square_distances(X: np.ndarray) -> np.ndarray:
    """Full (n x n) Euclidean distance matrix (diagonal = 0)."""
    return squareform(condensed_distances(X))
