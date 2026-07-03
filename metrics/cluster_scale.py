"""Within-cluster scale preservation — a VALUE/density metric that the rank-based Shepard ρ misses.

The rank Shepard ρ is scale-invariant: if a method crushes each dense cluster to a near-point in 2-D
(while spreading the clusters apart), the within-cluster RANK order can stay weakly positive even
though the cluster's internal structure is visually destroyed. This metric instead compares the
*scale* of the within-cluster spread relative to the between-cluster spread, in 2-D vs the ground
truth:

    ratio(D) = median(within-cluster pairwise distance) / median(between-cluster pairwise distance)

``over_compression = ratio_truth / ratio_2D`` is how many times a method shrinks the within-cluster
scale relative to the between-cluster scale, beyond the truth. ≈1 means the relative scale is
preserved; ≫1 means dense clusters are crushed toward points (their internal structure becomes
unreadable). This directly captures the "clusters collapse to points" failure visible in the Shepard
scatter. Requires cluster labels (points with label < 0, e.g. transition bridges, are excluded).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def _within_between_masks(labels):
    labels = np.asarray(labels)
    n = len(labels)
    iu = np.triu_indices(n, 1)
    li, lj = labels[iu[0]], labels[iu[1]]
    valid = (li >= 0) & (lj >= 0)
    within = valid & (li == lj)
    between = valid & (li != lj)
    return within, between


def scale_ratio(D_condensed, within, between):
    """median(within) / median(between) for a condensed distance vector."""
    w, b = D_condensed[within], D_condensed[between]
    if w.size == 0 or b.size == 0:
        return float("nan")
    mb = float(np.median(b))
    if mb <= 0:
        return float("nan")
    return float(np.median(w)) / mb


def cluster_scale_metrics(X_truth, Y, labels) -> dict:
    """Return within-cluster scale-preservation metrics for embedding ``Y`` given cluster ``labels``.

    Keys: ``cluster_ratio_truth``, ``cluster_ratio_2d``, ``cluster_over_compression``
    (= ratio_truth / ratio_2d; ≈1 preserved, ≫1 clusters crushed toward points).
    """
    within, between = _within_between_masks(labels)
    rt = scale_ratio(pdist(np.ascontiguousarray(X_truth, dtype=np.float64)), within, between)
    r2 = scale_ratio(pdist(np.ascontiguousarray(Y, dtype=np.float64)), within, between)
    over = (rt / r2) if (r2 and r2 == r2 and r2 > 0) else float("nan")
    return {"cluster_ratio_truth": rt, "cluster_ratio_2d": r2, "cluster_over_compression": over}
