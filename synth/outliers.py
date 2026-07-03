"""Dense bulk clusters plus a few injected OFF-SUBSPACE outliers at a controlled separation.

Models a contaminated feature set: a large "bulk" drawn from several dense sub-populations (e.g.
feature vectors of images captured under a handful of acquisition conditions), plus ``m`` points
whose generating condition is entirely different. The single knob ``outlier_factor`` controls HOW
far each outlier sits, as a multiple of the bulk's own extent -- it plays the same sweep-parameter
role that ``dynamic_range`` plays for the ``clusters`` dataset.

Construction (latent space -> random orthonormal projection -> SNR noise, like all generators):

* **bulk** (``n - m_outliers`` points): ``K`` dense Gaussian clusters on mutually orthogonal axes --
  the same arrangement as ``clusters``, at a moderate default dynamic range -- spanning the first
  ``latent_dim`` latent dimensions.
* **outliers** (``n_directions`` anomalous DIRECTIONS x ``per_direction`` points each, all sharing
  one ``outlier_factor``): direction ``j`` gets its OWN dedicated extra latent axis (dimension
  ``latent_dim + j``), **orthogonal to the entire subspace the bulk spans** ("off-subspace"). The
  members of direction ``j`` sit on that SAME axis at radii ``(outlier_factor + k*pair_offset) * Rg``
  for ``k = 0..per_direction-1`` -- i.e. same direction exactly, slightly offset radially. With the
  defaults (3 directions x 2, ``pair_offset = 0.1``) each pair's true separation is ``0.1 Rg``
  (~ the bulk's own median nearest-neighbor spacing), so the pair is "two near-duplicate anomalies
  of the same kind": a map that renders the anomaly structure faithfully must place pair members
  adjacent and in the same direction from the bulk, and different directions apart. **Rg is the
  bulk's radius of gyration** (root-mean-square distance of the bulk points from their centroid) --
  the compact scale of the central mass, so the default ``outlier_factor = 3`` reads "three bulk
  radii out". An earlier revision used the 99th percentile of the bulk pairwise distances (~ the
  diameter); that placed outliers so far (~5.5 Rg default, ~15 Rg at the sweep top) that the map's
  relative relation between the outliers and the central mass was lost.

Why off-subspace (and not a random direction WITHIN the bulk's latent subspace): a sample from a
different acquisition condition varies along feature directions the bulk does not span, so its
displacement should be orthogonal to the bulk's subspace -- an in-subspace far point would be the
easy case for variance-based linear projections (its direction carries large variance and lands in
the top principal components). Off-subspace placement makes the outlier direction carry only
``(outlier_factor * Rg)^2 / n`` variance, which a variance-truncating linear map may legally
drop -- exactly the distinction between constraining distances and constraining projected variance.
One dedicated axis per DIRECTION keeps the directions mutually separated (cross-direction distance
``~ sqrt(2) * outlier_factor * Rg``) and makes the placement deterministic given the seed; the
random orthonormal projection to ``D`` dimensions randomizes the ambient directions anyway. The
total latent dimension is ``latent_dim + n_directions``, so ``D >= latent_dim + n_directions``.

Design choice (documented in the README): all outliers share ONE factor per dataset, so within a
run the directions are i.i.d.-equivalent replicates of the same condition and the dose-response
over factors comes from the sweep (``run/sweep.py --sweep outlier_factor``), one dataset per factor.

Ground truth saved: ``outlier_idx`` (post-shuffle indices, ascending), ``outlier_dir`` (direction
id per outlier, aligned with ``outlier_idx``), ``outlier_factors`` (radius of each outlier in Rg
units), ``bulk_scale`` (= Rg). Outliers get label ``OUTLIER_LABEL = -1`` so the within-cluster
scale metric (which excludes labels < 0) is computed on the bulk only.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


OUTLIER_LABEL = -1


def make_outliers(n: int = 1000, d: int = 768, seed: int = 0, latent_dim: int = 8,
                  n_clusters: int = 5, dynamic_range: float = 10.0, cluster_sigma: float = 1.0,
                  n_directions: int = 3, per_direction: int = 2, pair_offset: float = 0.1,
                  outlier_factor: float = 3.0,
                  n_add_bulk: int = 0, add_per_direction: int = 0,
                  add_offset: float = 0.05) -> dict:
    """Generate the outliers dataset (bulk clusters + off-subspace outliers, paired per direction).

    ``outlier_factor`` = latent distance from the bulk centroid to the first outlier of each
    direction, in units of ``Rg`` = the bulk's radius of gyration; the k-th member of a direction
    sits at ``(outlier_factor + k*pair_offset) * Rg`` on the SAME axis (same direction exactly,
    slightly offset -- for checking whether same-direction anomalies stay together in the map).
    Returns the standard dict plus ``outlier_idx``, ``outlier_dir``, ``outlier_factors``,
    ``bulk_scale`` (= Rg).

    ADDPLOT data (out-of-sample test): with ``n_add_bulk`` / ``add_per_direction`` > 0 the dict
    gains an ``"addplot"`` sub-dict of EXTRA points in the SAME latent frame projected with the
    SAME orthonormal map: ``clean`` (rows: added bulk first, then one anomaly per direction at
    radius ``(outlier_factor + add_offset) * Rg`` on that direction's axis -- "a new anomaly of a
    known kind"), plus ``labels`` / ``dir`` / ``factors``. All extra randomness is drawn AFTER the
    fit data's draws, so the fit part is bit-identical to a call without addplot arguments
    (asserted in tests).
    """
    rng = np.random.default_rng(seed)
    K = n_clusters
    m = int(n_directions) * int(per_direction)
    if latent_dim < K:
        raise ValueError(f"latent_dim ({latent_dim}) must be >= n_clusters ({K}) for orthogonal axes")
    if not (0 < m < n // 10):
        raise ValueError(f"total outliers ({m}) must be small relative to n ({n})")
    n_bulk = n - m
    per = [n_bulk // K] * K
    for i in range(n_bulk - sum(per)):
        per[i] += 1

    # 1. bulk: isotropic Gaussian clusters on orthogonal axes (same recipe as synth/clusters.py)
    offsets, labels = [], []
    for i in range(K):
        offsets.append(rng.normal(0.0, cluster_sigma, size=(per[i], latent_dim)))
        labels.append(np.full(per[i], i, int))
    offsets = np.vstack(offsets)
    labels = np.concatenate(labels)

    nn_d = []
    for i in range(K):
        Xi = offsets[labels == i]
        nn = NearestNeighbors(n_neighbors=2).fit(Xi)
        dd, _ = nn.kneighbors(Xi)
        nn_d.append(dd[:, 1])
    intra_nn = float(np.median(np.concatenate(nn_d)))

    centroid_scale = dynamic_range * intra_nn / np.sqrt(2.0)
    centroids = np.zeros((K, latent_dim))
    for i in range(K):
        centroids[i, i] = centroid_scale
    bulk = offsets + centroids[labels]

    # 2. off-subspace outliers: embed the bulk in latent_dim + n_directions dims (zeros in the
    #    extra dims); direction j's k-th member sits on extra axis j at (factor + k*pair_offset)*Rg
    bulk_scale = float(np.sqrt(np.mean(np.sum((bulk - bulk.mean(axis=0)) ** 2, axis=1))))
    total_dim = latent_dim + int(n_directions)
    bulk_ext = np.hstack([bulk, np.zeros((n_bulk, int(n_directions)))])
    center = bulk_ext.mean(axis=0)
    outliers = np.tile(center, (m, 1))
    out_dir, out_factors = [], []
    for j in range(int(n_directions)):
        for k in range(int(per_direction)):
            i = j * int(per_direction) + k
            f = float(outlier_factor) + k * float(pair_offset)
            outliers[i, latent_dim + j] += f * bulk_scale
            out_dir.append(j); out_factors.append(f)

    latent = np.vstack([bulk_ext, outliers])
    labels = np.concatenate([labels, np.full(m, OUTLIER_LABEL, int)])
    dir_all = np.concatenate([np.full(n_bulk, -1, int), np.asarray(out_dir, int)])
    fac_all = np.concatenate([np.full(n_bulk, np.nan), np.asarray(out_factors, float)])

    perm = rng.permutation(n)
    latent, labels, dir_all, fac_all = latent[perm], labels[perm], dir_all[perm], fac_all[perm]
    outlier_idx = np.flatnonzero(labels == OUTLIER_LABEL)   # ascending -> deterministic o0..o{m-1}

    # projection: inlined equivalent of project_to_D (identical rng consumption and result), so the
    # SAME Q can also map the optional addplot points below without touching the fit data
    if d < total_dim:
        raise ValueError(f"ambient dim D={d} must be >= latent dim m={total_dim}")
    G = rng.standard_normal((d, total_dim))
    Q, _ = np.linalg.qr(G)
    clean = np.ascontiguousarray(latent @ Q.T, dtype=np.float64)

    addplot = None
    if int(n_add_bulk) > 0 or int(add_per_direction) > 0:
        # all draws below happen AFTER every fit-data draw -> fit output is bit-identical
        parts, a_labels, a_dir, a_fac = [], [], [], []
        if int(n_add_bulk) > 0:
            assign = rng.integers(0, K, size=int(n_add_bulk))
            ab = centroids[assign] + rng.normal(0.0, cluster_sigma, size=(int(n_add_bulk), latent_dim))
            parts.append(np.hstack([ab, np.zeros((int(n_add_bulk), int(n_directions)))]))
            a_labels += list(assign); a_dir += [-1] * int(n_add_bulk); a_fac += [np.nan] * int(n_add_bulk)
        for j in range(int(n_directions)):
            for k in range(int(add_per_direction)):
                f = float(outlier_factor) + (k + 1) * float(add_offset)
                p = center.copy()
                p[latent_dim + j] += f * bulk_scale
                parts.append(p[None, :])
                a_labels.append(OUTLIER_LABEL); a_dir.append(j); a_fac.append(f)
        add_latent = np.vstack(parts)
        addplot = {
            "clean": np.ascontiguousarray(add_latent @ Q.T, dtype=np.float64),
            "labels": np.asarray(a_labels, int),
            "dir": np.asarray(a_dir, int),
            "factors": np.asarray(a_fac, float),
        }

    label_names = {i: f"cluster {i}" for i in range(K)}
    label_names[OUTLIER_LABEL] = "outlier"
    return {
        "name": "outliers",
        "clean": clean,
        "truth_coords": clean,
        "labels": labels,
        "label_names": label_names,
        "color_value": labels.astype(float),
        "color_name": "cluster id (-1 = outlier)",
        "outlier_idx": outlier_idx,
        "outlier_dir": dir_all[outlier_idx],
        "outlier_factors": fac_all[outlier_idx],
        "bulk_scale": bulk_scale,
        "centroids_latent": centroids,
        "intra_nn": intra_nn,
        "params": dict(n=n, d=d, seed=seed, latent_dim=latent_dim, n_clusters=K,
                       dynamic_range=dynamic_range, cluster_sigma=cluster_sigma,
                       n_directions=int(n_directions), per_direction=int(per_direction),
                       pair_offset=float(pair_offset), outlier_factor=float(outlier_factor),
                       total_latent_dim=total_dim),
        **({"addplot": addplot} if addplot is not None else {}),
    }


def make_anchored_addplot(base: dict, deviation: float = 3.0, pair_offset: float = 0.1,
                          extra_dims: int = 8, seed: int = 0) -> dict:
    """CLUSTER-ANCHORED anomalies for the addplot (out-of-sample) test.

    Each anomaly shares a normal cluster's signal and deviates only along NEW dimensions: its
    clean vector is ``(cluster k's clean centroid) + r * Rg * u_k``, where ``u_k`` is a unit
    direction inside an ``extra_dims``-dimensional orthonormal block orthogonal to the entire
    subspace the clean bulk spans (one direction per cluster, drawn deterministically from
    ``seed``), ``Rg`` = the bulk's radius of gyration, and each cluster contributes a
    near-duplicate pair at radii ``deviation`` and ``deviation + pair_offset`` (in Rg units).

    So in the measured features an anomaly looks like a member of cluster ``k`` whose deviation
    lives in directions the normal data never varies along -- the map must show BOTH that the
    point is an anomaly (outside the normal region) and WHICH cluster it comes from (direction).

    Returns ``clean`` (2K x d, pairs grouped per cluster), ``anchor`` (source-cluster id per
    row), ``radii`` (in Rg units), and ``extra_basis`` (d x extra_dims).
    """
    labs = np.asarray(base["labels"])
    bulk = base["clean"][labs != OUTLIER_LABEL]
    labs_b = labs[labs != OUTLIER_LABEL]
    Rg = float(np.sqrt(np.mean(np.sum((bulk - bulk.mean(axis=0)) ** 2, axis=1))))
    centered = bulk - bulk.mean(axis=0)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    P_bulk = Vt[: int((s > s[0] * 1e-8).sum())]           # exact basis of the bulk's subspace
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 424242]))
    G = rng.normal(size=(bulk.shape[1], int(extra_dims)))
    G -= P_bulk.T @ (P_bulk @ G)
    U, _ = np.linalg.qr(G)                                # orthonormal, orthogonal to the bulk
    K = int(labs_b.max()) + 1
    dirs = rng.normal(size=(K, int(extra_dims)))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    cents = np.array([bulk[labs_b == k].mean(axis=0) for k in range(K)])
    rows, anchor, radii = [], [], []
    for k in range(K):
        u = U @ dirs[k]
        for r in (float(deviation), float(deviation) + float(pair_offset)):
            rows.append(cents[k] + r * Rg * u)
            anchor.append(k); radii.append(r)
    return {"clean": np.ascontiguousarray(np.array(rows), dtype=np.float64),
            "anchor": np.asarray(anchor, int), "radii": np.asarray(radii, float),
            "extra_basis": U}
