"""Continuous-transition dataset with known, GENUINELY HIGH-DIMENSIONAL global geometry.

Several "typical-state" clusters sit at KNOWN centroid positions and are connected by a continuous
transition region parameterised by ``t`` that runs through every centroid and CLOSES into a loop
(state 0 → 1 → … → K-1 → 0), so ``t`` is cyclic. The transition region is HETEROGENEOUS — its
perpendicular spread widens toward the middle of each bridge — to mirror a "Mixed / H2" distribution.

Crucially the centroids are placed on **K mutually orthogonal axes** (``scale · e_i``) rather than on
a 2-D circle: the known geometry then spans ~K dimensions (effective/participation dimension ≈ K), so
a linear method (PCA, 2 components) CANNOT reproduce the global structure trivially. This makes the
global-distance test discriminative. The clusters are made **dense** (small ``cluster_sigma``) so the
near-distance band is dominated by tight within-cluster structure — together with the loop's
between-cluster geometry this gives a clean near→far structure to score.

Ground truth is twofold: the known centroid geometry (global-distance scoring) AND the continuum
parameter ``t`` (transition-continuity scoring). Everything is built in latent space and mapped
isometrically into D dims, so the clean Euclidean distance is the ground truth.
"""
from __future__ import annotations

import numpy as np

from ._common import project_to_D

TRANSITION_LABEL = -1


def make_transition(n: int = 1000, d: int = 768, seed: int = 0, latent_dim: int = 9,
                    n_clusters: int = 7, cluster_frac: float = 0.6, centroid_scale: float = 5.0,
                    cluster_sigma: float = 0.138, path_sigma: float = 0.22, spread_amp: float = 3.0,
                    spread_dims: int = 3, closed_loop: bool = True) -> dict:
    """Generate the continuous-transition dataset (high-dim global geometry, closed loop, dense states).

    Returns a dict with ``clean`` (n x d), ``truth_coords`` (== clean), ``labels`` (cluster id, or
    ``-1`` for transition points), ``label_names``, ``color_value`` (continuum ``t`` in [0,1)),
    ``color_name``, ``centroids_latent`` (K x latent_dim), ``centroids_D`` (K x d), ``name``,
    ``params``.
    """
    rng = np.random.default_rng(seed)
    K = n_clusters
    if latent_dim < K:
        raise ValueError(f"latent_dim ({latent_dim}) must be >= n_clusters ({K}) for orthogonal axes")
    # centroids on K mutually orthogonal axes -> equidistant, span K dims (effective dim ~ K)
    centroids = np.zeros((K, latent_dim))
    for i in range(K):
        centroids[i, i] = centroid_scale

    # edges: closed loop (cycle, including K-1 -> 0) or open chain
    edges = [(i, (i + 1) % K) for i in range(K)] if closed_loop else [(i, i + 1) for i in range(K - 1)]
    n_edges = len(edges)

    n_cluster_total = int(round(cluster_frac * n))
    n_trans_total = n - n_cluster_total
    per_cluster = [n_cluster_total // K] * K
    for i in range(n_cluster_total - sum(per_cluster)):
        per_cluster[i] += 1
    per_edge = [n_trans_total // n_edges] * n_edges
    for i in range(n_trans_total - sum(per_edge)):
        per_edge[i] += 1

    pts, labels, tvals = [], [], []

    # dense Gaussian blobs at the centroids; cyclic node parameter t = i/K
    for i in range(K):
        blob = centroids[i] + rng.normal(0.0, cluster_sigma, size=(per_cluster[i], latent_dim))
        pts.append(blob)
        labels.append(np.full(per_cluster[i], i, int))
        tvals.append(np.full(per_cluster[i], i / K))

    # transition bridges between consecutive centroids, with heterogeneous (mid-fat) spread
    sd = min(spread_dims, latent_dim)
    for e_idx, (a, b) in enumerate(edges):
        ca, cb = centroids[a], centroids[b]
        u = rng.uniform(0.0, 1.0, size=per_edge[e_idx])
        base = ca[None, :] + u[:, None] * (cb - ca)[None, :]
        width = path_sigma * (1.0 + spread_amp * np.sin(np.pi * u))   # widest at u=0.5 (both sides)
        spread = np.zeros((per_edge[e_idx], latent_dim))
        spread[:, :sd] = rng.standard_normal((per_edge[e_idx], sd))
        spread = spread * width[:, None]
        pts.append(base + spread)
        labels.append(np.full(per_edge[e_idx], TRANSITION_LABEL, int))
        tvals.append((e_idx + u) / n_edges)   # global cyclic t in [0, 1)

    latent = np.vstack(pts)
    labels = np.concatenate(labels)
    tvals = np.concatenate(tvals)

    perm = rng.permutation(len(latent))
    latent, labels, tvals = latent[perm], labels[perm], tvals[perm]

    clean = project_to_D(latent, d, rng)
    centroids_D = project_to_D(centroids, d, np.random.default_rng(seed + 1))

    label_names = {i: f"state {i}" for i in range(K)}
    label_names[TRANSITION_LABEL] = "transition"

    return {
        "name": "transition",
        "clean": clean,
        "truth_coords": clean,
        "labels": labels,
        "label_names": label_names,
        "color_value": tvals,
        "color_name": "continuum t (cyclic)",
        "centroids_latent": centroids,
        "centroids_D": centroids_D,
        "params": dict(n=n, d=d, seed=seed, latent_dim=latent_dim, n_clusters=K,
                       cluster_frac=cluster_frac, centroid_scale=centroid_scale,
                       cluster_sigma=cluster_sigma, path_sigma=path_sigma, spread_amp=spread_amp,
                       spread_dims=spread_dims, closed_loop=closed_loop),
    }
