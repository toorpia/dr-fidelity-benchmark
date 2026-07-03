"""Outlier separation / pair metrics for the outliers dataset.

STATUS: the OSP family below is RETIRED FROM ALL REPORTING (maintainer decision): it is not an
established metric and its double normalization did not track the Shepard/embedding pictures.
It is still computed into the per-run CSVs for raw-data continuity; the reported quantities are
the standard band-Shepard rho / stress blocks plus the plain-geometry pair angle
(``pair_angle_2d_*``) and the addplot readouts.

Original motivation (kept for the record) -- does a single far-away point STAY separated in 2-D?

Why a dedicated metric: the band-Shepard rho is a statistic over MANY pairs, so the fate of one
point is invisible to it -- a single outlier contributes only O(N) of the O(N^2) pairs (O(1/N) of
any band), and whether it collapses onto the bulk cannot move a many-pair correlation. OSP scores
exactly that single-point fate. Like every metric in this package it is computed EXACTLY on all
pairwise distances, independent of any method's internal approximation.

The benchmark reports OSP **vs-ambient only** (``compute_all`` calls this with the ambient
features): the repo's primary axis is how faithfully the 2-D map renders the D-dim input the
method actually saw, and every PRIMARY metric follows that convention. Note that at finite SNR the
D-dimensional noise floor cannot be represented in 2-D, so vs-ambient OSP can far exceed 1 for
every method — the reading is limited to the burial threshold (OSP < 1) and between-method
comparison. This function itself is generic (pass any high-D reference and a ``tag``).

Separation margin of outlier ``o`` in a space (HD or 2D), scale-normalized within that space::

    s_space(o) = d_space(o, nearest non-outlier point) / median_bulk_nn(space)

where ``median_bulk_nn`` is the median over bulk points of the distance to their nearest OTHER bulk
point (outliers are excluded from the neighbor pool on both sides, so one outlier cannot mask
another and cannot perturb the bulk scale). Then::

    OSP(o) = s_2D(o) / s_HD(o)

OSP ~= 1: the separation margin is preserved; OSP << 1: the outlier is buried in the bulk;
OSP > 1: over-separated. ``log2(OSP)`` is also reported (symmetric axis around 0). Because both
``s`` values are ratios of distances measured within one space, OSP is invariant to uniform
scaling, rotation, reflection, and translation of the embedding -- no Procrustes gauge removal is
needed (asserted in tests).

Secondary, ISOLATION-RANK preservation -- the practitioner's view "is the most isolated point still
the most isolated?": rank ALL n points by their nearest-neighbor distance (rank 1 = most isolated;
ties broken deterministically by index) and compare each outlier's rank in HD vs 2-D.
``rank_2d - rank_hd > 0`` means the outlier became less isolated in the embedding.

Aggregation: per-outlier values are emitted (``*_o{j}``, j ordered by ascending dataset index) so
the seed-level aggregation (median + bootstrap 95% CI over R seeds) applies to each outlier as well
as to the per-run summaries (``osp_median``, ``osp_min``). These are single-point metrics, so no
pair-direction bootstrap is meaningful -- the CI is over seeds only, per the existing convention.
"""
from __future__ import annotations

import numpy as np

from .distances import square_distances


def _separation_and_rank(X: np.ndarray, outlier_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-outlier scale-normalized separation margin ``s`` and isolation rank, in one space.

    Returns ``(s, rank)``: ``s[j]`` = distance from outlier j to its nearest bulk point divided by
    the median bulk nearest-neighbor distance; ``rank[j]`` = outlier j's position (1 = most
    isolated) when all n points are ordered by descending nearest-neighbor distance.
    """
    D = square_distances(X)
    np.fill_diagonal(D, np.inf)
    n = D.shape[0]
    is_out = np.zeros(n, dtype=bool)
    is_out[outlier_idx] = True
    bulk = np.flatnonzero(~is_out)

    d_out = D[np.ix_(outlier_idx, bulk)].min(axis=1)          # nearest NON-outlier point
    bulk_nn = D[np.ix_(bulk, bulk)].min(axis=1)               # bulk NN among bulk only
    med = float(np.median(bulk_nn))
    s = d_out / med if med > 0 else np.full(len(outlier_idx), np.nan)

    nn_all = D.min(axis=1)                                    # isolation = NN distance among ALL
    order = np.argsort(-nn_all, kind="stable")                # ties -> lower index first (determ.)
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(1, n + 1)
    return s, rank[outlier_idx]


def outlier_shepard(X_hd: np.ndarray, Y: np.ndarray, outlier_idx) -> float:
    """Standard Shepard rho restricted to the ANOMALY-INVOLVING pairs.

    Spearman rank correlation between high-D and 2-D distances over exactly the pairs where at
    least one endpoint is a ground-truth outlier -- the same standard statistic as the global
    Shepard rho and the band-restricted rho, with the pair subset selected by endpoint membership
    instead of by distance percentile. It is the direct quantification of the outlier-related
    blocks in the Shepard density figure: a method that places the anomalies among other clusters,
    fuses same-kind pairs, or tears them apart scrambles the ordering of these pairs and scores
    low, regardless of how well the bulk-only pairs are ordered.
    """
    from scipy.stats import spearmanr

    X_hd = np.ascontiguousarray(X_hd, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    n = len(X_hd)
    is_out = np.zeros(n, dtype=bool)
    is_out[np.asarray(outlier_idx, dtype=np.int64)] = True
    iu = np.triu_indices(n, 1)
    mask = is_out[iu[0]] | is_out[iu[1]]
    D_hd, D_2d = square_distances(X_hd), square_distances(Y)
    rho = spearmanr(D_hd[iu][mask], D_2d[iu][mask]).statistic
    return float(rho)


def outlier_pair_metrics(X_hd: np.ndarray, Y: np.ndarray, outlier_idx, outlier_dir,
                         tag: str) -> dict:
    """Same-direction PAIR fidelity: near-duplicate anomalies of one direction should stay
    together and on one ray from the bulk in the embedding.

    The dataset places ``per_direction = 2`` outliers on each anomalous direction, offset by a
    small radial shift (their true separation ~ the bulk's own NN spacing). Two questions, per
    direction ``j`` (directions without exactly 2 members are skipped):

    * ``pair_cohesion_dir{j}`` -- scale-normalized ratio (computed but not reported) of the pair's internal
      distance: ``[d_2D(pair)/median_bulk_nn_2D] / [d_HD(pair)/median_bulk_nn_HD]``. ~1 = the
      near-duplicate pair stays exactly as close as in the input; ``>>1`` = the pair is torn
      apart (same anomaly rendered as two unrelated points); ``<<1`` = fused into one point.
    * ``pair_angle_2d_dir{j}`` -- angle (degrees) between the two members as seen from the BULK
      CENTROID in the 2-D map (ground truth: 0, same direction). ``pair_angle_hd_dir{j}__{tag}``
      is the same angle in the high-D reference (~0; noise makes it slightly positive) so the 2-D
      value has an honest baseline.

    Summary keys: ``pair_cohesion_median__{tag}``, ``pair_angle_2d_max``.
    """
    oi = np.sort(np.asarray(outlier_idx, dtype=np.int64))
    od = np.asarray(outlier_dir)
    X_hd = np.ascontiguousarray(X_hd, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    n = len(X_hd)
    is_out = np.zeros(n, dtype=bool); is_out[oi] = True
    bulk = ~is_out

    def bulk_nn_median(Z):
        D = square_distances(Z)
        np.fill_diagonal(D, np.inf)
        idx = np.flatnonzero(bulk)
        return float(np.median(D[np.ix_(idx, idx)].min(axis=1)))

    def angle_deg(Z, a, b):
        c = Z[bulk].mean(axis=0)
        va, vb = Z[a] - c, Z[b] - c
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return float("nan")
        return float(np.degrees(np.arccos(np.clip(va @ vb / (na * nb), -1.0, 1.0))))

    bn_hd, bn_2d = bulk_nn_median(X_hd), bulk_nn_median(Y)
    row, cohesions, angles2d = {}, [], []
    for j in sorted(set(int(v) for v in od)):
        members = oi[od == j]
        if len(members) != 2:
            continue
        a, b = int(members[0]), int(members[1])
        s_hd = np.linalg.norm(X_hd[a] - X_hd[b]) / bn_hd
        s_2d = np.linalg.norm(Y[a] - Y[b]) / bn_2d
        coh = float(s_2d / s_hd) if s_hd > 0 else float("nan")
        ang2d, anghd = angle_deg(Y, a, b), angle_deg(X_hd, a, b)
        row[f"pair_cohesion_dir{j}__{tag}"] = coh
        row[f"pair_angle_2d_dir{j}"] = ang2d
        row[f"pair_angle_hd_dir{j}__{tag}"] = anghd
        cohesions.append(coh); angles2d.append(ang2d)
    if cohesions:
        row[f"pair_cohesion_median__{tag}"] = float(np.nanmedian(cohesions))
        row["pair_angle_2d_max"] = float(np.nanmax(angles2d))
    return row


def outlier_metrics(X_hd: np.ndarray, Y: np.ndarray, outlier_idx, tag: str) -> dict:
    """OSP + isolation-rank preservation of embedding ``Y`` against high-D coords ``X_hd``.

    ``tag`` is ``"vs_truth"`` or ``"vs_ambient"`` (which high-D space ``X_hd`` is). Keys:
    ``osp_o{j}__{tag}``, ``log2_osp_o{j}__{tag}``, ``osp_median__{tag}``, ``osp_min__{tag}``,
    ``log2_osp_median__{tag}``; ``iso_rank_hd_o{j}__{tag}``, ``iso_rank_2d_o{j}`` (embedding-only,
    tag-independent), ``iso_rank_delta_o{j}__{tag}`` (= 2d - hd), ``iso_rank_delta_mean__{tag}``.
    """
    oi = np.sort(np.asarray(outlier_idx, dtype=np.int64))
    s_hd, rank_hd = _separation_and_rank(np.ascontiguousarray(X_hd, dtype=np.float64), oi)
    s_2d, rank_2d = _separation_and_rank(np.ascontiguousarray(Y, dtype=np.float64), oi)
    with np.errstate(divide="ignore", invalid="ignore"):
        osp = np.where(s_hd > 0, s_2d / s_hd, np.nan)
        log2_osp = np.where(osp > 0, np.log2(osp), -np.inf)
    delta = (rank_2d - rank_hd).astype(np.float64)

    row = {}
    for j in range(len(oi)):
        row[f"osp_o{j}__{tag}"] = float(osp[j])
        row[f"log2_osp_o{j}__{tag}"] = float(log2_osp[j])
        row[f"iso_rank_hd_o{j}__{tag}"] = float(rank_hd[j])
        row[f"iso_rank_2d_o{j}"] = float(rank_2d[j])
        row[f"iso_rank_delta_o{j}__{tag}"] = float(delta[j])
    row[f"osp_median__{tag}"] = float(np.nanmedian(osp))
    row[f"osp_min__{tag}"] = float(np.nanmin(osp)) if np.any(~np.isnan(osp)) else float("nan")
    med = row[f"osp_median__{tag}"]
    row[f"log2_osp_median__{tag}"] = float(np.log2(med)) if med > 0 else float("-inf")
    row[f"iso_rank_delta_mean__{tag}"] = float(np.mean(delta))
    return row
