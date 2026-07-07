"""Tightest-cluster scale preservation — a VALUE/density metric that the rank-based Shepard ρ misses.

The rank Shepard ρ is scale-invariant: a method can crush a dense cluster to a near-point (or
inflate it) while the within-cluster RANK order stays weakly positive. This metric watches the
structure that a global-layout trade-off sacrifices first: the TIGHTEST cluster in the data.

Selection (once per dataset, in TRUTH space — method-independent, every method is scored on the
same cluster): among clusters (label >= 0) with at least ``min_pts`` points (median needs enough
pairs; tiny clusters are excluded), take the one with the smallest median within-cluster pairwise
distance. Ties break by cluster id (tied clusters are exchangeable by construction).

Scale reference: the median of ALL pairwise distances in the same space — the population's overall
spread. It is always defined and presupposes nothing about how the map arranges clusters (unlike a
between-cluster reference, which is only meaningful when the clusters are actually drawn apart).

    ratio(space)            = median(within tightest cluster) / median(all pairs)
    tight_over_compression  = ratio_truth / ratio_2D

≈1 = the tightest cluster keeps its scale relative to the overall spread; ≫1 = it is crushed
toward a point; ≪1 = it is inflated relative to the map's spread (local exaggeration). Crushing
is the real harm (information cannot be read back from a crushed cluster), so the report's
failure flag marks only the crush-side worst case of each table (when it exceeds 5×).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist

MIN_PTS = 20


def select_tight_cluster(d_truth_condensed, labels, min_pts: int = MIN_PTS):
    """Id and within-pair mask of the tightest sufficient cluster, chosen on truth distances."""
    labels = np.asarray(labels)
    n = len(labels)
    iu = np.triu_indices(n, 1)
    best, best_med, best_mask = None, np.inf, None
    for c in sorted(int(v) for v in set(labels[labels >= 0])):
        if int((labels == c).sum()) < min_pts:
            continue
        m = (labels[iu[0]] == c) & (labels[iu[1]] == c)
        med = float(np.median(d_truth_condensed[m]))
        if med < best_med:
            best, best_med, best_mask = c, med, m
    return best, best_mask


def tight_cluster_metrics(X_truth, Y, labels, min_pts: int = MIN_PTS) -> dict:
    """Tightest-cluster scale preservation for embedding ``Y``.

    Keys: ``tight_cluster`` (selected id), ``tight_ratio_truth``, ``tight_ratio_2d``,
    ``tight_over_compression`` (= ratio_truth / ratio_2d; ≈1 preserved, ≫1 crushed, ≪1 inflated).
    """
    d_tru = pdist(np.ascontiguousarray(X_truth, dtype=np.float64))
    cid, mask = select_tight_cluster(d_tru, labels, min_pts)
    if cid is None:
        return {"tight_cluster": float("nan"), "tight_ratio_truth": float("nan"),
                "tight_ratio_2d": float("nan"), "tight_over_compression": float("nan")}
    d_2d = pdist(np.ascontiguousarray(Y, dtype=np.float64))
    rt = float(np.median(d_tru[mask])) / float(np.median(d_tru))
    r2 = float(np.median(d_2d[mask])) / float(np.median(d_2d))
    over = (rt / r2) if r2 > 0 else float("nan")
    return {"tight_cluster": float(cid), "tight_ratio_truth": rt, "tight_ratio_2d": r2,
            "tight_over_compression": over}
