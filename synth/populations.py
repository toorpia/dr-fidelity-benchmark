"""Two populations with internal cluster structure, IMBALANCED in size.

Real-world motivation (generally occurring, not an edge case): measurement datasets routinely mix
a dominant population with a much smaller second population that occupies a DIFFERENT region of
the measurement space yet has internal structure of its own -- e.g. normal production runs vs a
rarely-used operating mode (start-up, alternative recipe), the main manufacturing line vs a small
pilot-lot series, data before vs after an instrument replacement or recalibration, seasonal or
day/night operating regimes, or a healthy majority vs a small patient group that itself splits
into subtypes. The practical question for a 2-D map is twofold: does the map keep the two
populations' relative placement, and does the SMALL population keep its internal structure -- or
is it collapsed to a blob (its subtypes lost) precisely because it has few points?

Construction (latent space -> random orthonormal projection -> SNR noise, like all generators):

* **Population A (majority)**: ``K`` dense Gaussian clusters on mutually orthogonal axes in its
  own ``latent_block`` dimensions -- the standard ``clusters`` recipe. Cluster-center scale ``c``
  is set from A's within-cluster spacing via ``dynamic_range`` (same convention as ``clusters``).
* **Population B (minority)**: the SAME geometry (same ``c``, same cluster arrangement, same
  within-cluster sigma) built in a DISJOINT block of latent dimensions, with fewer points --
  ``minority_frac`` of ``n``. Using A's scale for both isolates the size effect from geometry.
* **Group separation**: population B is offset by ``delta`` along one shared extra dimension, with
  ``delta = sqrt(2)*c*sqrt(group_range^2 - 1)`` so that EVERY cross-population cluster-center
  distance equals ``group_range`` x the (universal) within-population center distance
  ``sqrt(2)*c``. ``group_range`` is the group-level analogue of ``dynamic_range``: a strict
  two-level hierarchy with a single knob (``group_range = 1`` degenerates to 2K equidistant
  clusters; the experiment default is 2).

Ground truth saved: ``labels`` (cluster id ``0..2K-1``; B's clusters are ``K..2K-1``),
``population`` (0 = majority A, 1 = minority B), plus the scales. Total latent dimension is
``2*latent_block + 1``, so ``d`` must be at least that.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def make_populations(n: int = 1000, d: int = 768, seed: int = 0, latent_block: int = 8,
                     n_clusters: int = 5, dynamic_range: float = 10.0, cluster_sigma: float = 1.0,
                     group_range: float = 2.0, minority_frac: float = 0.05) -> dict:
    """Generate the imbalanced two-population dataset.

    ``minority_frac`` = fraction of the ``n`` points belonging to population B (0 < frac <= 0.5);
    ``group_range`` = cross-population center distance / within-population center distance.
    Returns the standard dict plus ``population`` (0/1 per point), ``center_scale`` (c),
    ``group_delta`` (the offset), and per-population point counts in ``params``.
    """
    rng = np.random.default_rng(seed)
    K = int(n_clusters)
    if latent_block < K:
        raise ValueError(f"latent_block ({latent_block}) must be >= n_clusters ({K})")
    if not (0.0 < minority_frac <= 0.5):
        raise ValueError(f"minority_frac must be in (0, 0.5], got {minority_frac}")
    if group_range < 1.0:
        raise ValueError(f"group_range must be >= 1, got {group_range}")
    n_b = int(round(minority_frac * n))
    n_a = n - n_b

    def cluster_block(n_pop):
        per = [n_pop // K] * K
        for i in range(n_pop - sum(per)):
            per[i] += 1
        offs, labs = [], []
        for i in range(K):
            offs.append(rng.normal(0.0, cluster_sigma, size=(per[i], latent_block)))
            labs.append(np.full(per[i], i, int))
        return np.vstack(offs), np.concatenate(labs)

    offs_a, lab_a = cluster_block(n_a)
    offs_b, lab_b = cluster_block(n_b)

    # one shared center scale, measured on the MAJORITY (isolates the size effect from geometry)
    nn_d = []
    for i in range(K):
        Xi = offs_a[lab_a == i]
        dd, _ = NearestNeighbors(n_neighbors=2).fit(Xi).kneighbors(Xi)
        nn_d.append(dd[:, 1])
    intra_nn = float(np.median(np.concatenate(nn_d)))
    c = float(dynamic_range) * intra_nn / np.sqrt(2.0)
    centers = np.zeros((K, latent_block))
    for i in range(K):
        centers[i, i] = c
    A = offs_a + centers[lab_a]
    B = offs_b + centers[lab_b]

    total_dim = 2 * latent_block + 1
    delta = np.sqrt(2.0) * c * np.sqrt(float(group_range) ** 2 - 1.0)
    LA = np.zeros((n_a, total_dim)); LA[:, :latent_block] = A
    LB = np.zeros((n_b, total_dim)); LB[:, latent_block:2 * latent_block] = B; LB[:, -1] = delta
    latent = np.vstack([LA, LB])
    labels = np.concatenate([lab_a, lab_b + K])
    population = np.concatenate([np.zeros(n_a, int), np.ones(n_b, int)])

    perm = rng.permutation(n)
    latent, labels, population = latent[perm], labels[perm], population[perm]

    if d < total_dim:
        raise ValueError(f"ambient dim D={d} must be >= latent dim {total_dim}")
    G = rng.standard_normal((d, total_dim))
    Q, _ = np.linalg.qr(G)
    clean = np.ascontiguousarray(latent @ Q.T, dtype=np.float64)

    label_names = {i: f"A cluster {i}" for i in range(K)}
    label_names.update({K + i: f"B cluster {i}" for i in range(K)})
    return {
        "name": "populations",
        "clean": clean,
        "truth_coords": clean,
        "labels": labels,
        "label_names": label_names,
        "color_value": labels.astype(float),
        "color_name": "cluster id (0-4 majority A, 5-9 minority B)",
        "population": population,
        "center_scale": c,
        "group_delta": float(delta),
        "intra_nn": intra_nn,
        "params": dict(n=n, d=d, seed=seed, latent_block=latent_block, n_clusters=K,
                       dynamic_range=dynamic_range, cluster_sigma=cluster_sigma,
                       group_range=float(group_range), minority_frac=float(minority_frac),
                       n_majority=n_a, n_minority=n_b, total_latent_dim=total_dim),
    }
