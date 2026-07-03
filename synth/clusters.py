"""K dense clusters in a high-dimensional rigid arrangement, with a tunable DYNAMIC RANGE.

``K`` small dense Gaussian clusters are placed on ``K`` mutually orthogonal axes (so the global
geometry spans ``K-1`` dimensions -- genuinely high-D, not a low-effective-dim trap), and a single
knob ``dynamic_range`` controls the ratio of the inter-cluster distance to the intra-cluster spacing.

This probes both scales at once: the near band (``p5``) measures within-cluster fine structure and the
full band (``full``) measures the global cluster layout, so a method is scored on whether it keeps the
within-cluster arrangement while also placing the clusters correctly. Models the common real-world
case of distinct dense sub-populations separated by large gaps in a high-D feature space.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ._common import local_log_density, project_to_D


def make_clusters(n: int = 1000, d: int = 768, seed: int = 0, latent_dim: int = 8,
                  n_clusters: int = 7, dynamic_range: float = 20.0,
                  cluster_sigma: float = 1.0) -> dict:
    """Generate the high-dim clusters dataset.

    ``dynamic_range`` = inter-centroid distance / median intra-cluster nearest-neighbor distance.
    Returns the standard dict: ``clean`` (n x d), ``truth_coords`` (== clean), ``labels`` (cluster id),
    ``label_names``, ``color_value`` (cluster id), ``color_name``, ``centroids_latent``, ``intra_nn``,
    ``name``, ``params``.
    """
    rng = np.random.default_rng(seed)
    K = n_clusters
    if latent_dim < K:
        raise ValueError(f"latent_dim ({latent_dim}) must be >= n_clusters ({K}) for orthogonal axes")
    per = [n // K] * K
    for i in range(n - sum(per)):
        per[i] += 1

    # 1. isotropic Gaussian offsets per cluster (centered at origin), and cluster ids
    offsets, labels = [], []
    for i in range(K):
        offsets.append(rng.normal(0.0, cluster_sigma, size=(per[i], latent_dim)))
        labels.append(np.full(per[i], i, int))
    offsets = np.vstack(offsets)
    labels = np.concatenate(labels)

    # 2. measure intra-cluster nearest-neighbor spacing (the "intra spread")
    nn_d = []
    for i in range(K):
        Xi = offsets[labels == i]
        nn = NearestNeighbors(n_neighbors=2).fit(Xi)
        dd, _ = nn.kneighbors(Xi)
        nn_d.append(dd[:, 1])
    intra_nn = float(np.median(np.concatenate(nn_d)))

    # 3. centroids on orthogonal axes, scaled so inter-centroid distance = dynamic_range * intra_nn
    #    (distance between scale*e_i and scale*e_j is scale*sqrt(2))
    centroid_scale = dynamic_range * intra_nn / np.sqrt(2.0)
    centroids = np.zeros((K, latent_dim))
    for i in range(K):
        centroids[i, i] = centroid_scale
    latent = offsets + centroids[labels]

    perm = rng.permutation(n)
    latent, labels = latent[perm], labels[perm]

    clean = project_to_D(latent, d, rng)
    return {
        "name": "clusters",
        "clean": clean,
        "truth_coords": clean,
        "labels": labels,
        "label_names": {i: f"cluster {i}" for i in range(K)},
        "color_value": labels.astype(float),
        "color_name": "cluster id",
        "centroids_latent": centroids,
        "intra_nn": intra_nn,
        "params": dict(n=n, d=d, seed=seed, latent_dim=latent_dim, n_clusters=K,
                       dynamic_range=dynamic_range, cluster_sigma=cluster_sigma),
    }
