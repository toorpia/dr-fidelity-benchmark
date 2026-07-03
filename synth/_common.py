"""Shared helpers for the synthetic generators.

Structure is built in a low-dimensional LATENT space (where the geometry is easy to reason about and
the ground-truth distances are defined), then mapped into the ambient dimension ``D`` by an
ORTHONORMAL projection. An orthonormal map is an isometry, so Euclidean distances in latent space and
in the clean D-dimensional space are identical -- the ground-truth distance is therefore unambiguous,
and it equals the clean ambient Euclidean distance (before noise is added in all D dimensions).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def project_to_D(latent: np.ndarray, D: int, rng: np.random.Generator) -> np.ndarray:
    """Map ``latent`` (n x m) into ``D`` dims via a random orthonormal projection (distance-preserving).

    Requires ``D >= m``. Columns of the (D x m) projection are orthonormal, so
    ``||clean_i - clean_j|| == ||latent_i - latent_j||``.
    """
    n, m = latent.shape
    if D < m:
        raise ValueError(f"ambient dim D={D} must be >= latent dim m={m}")
    G = rng.standard_normal((D, m))
    Q, _ = np.linalg.qr(G)            # Q: (D x m), orthonormal columns
    clean = latent @ Q.T              # (n x D); isometric image of latent
    return np.ascontiguousarray(clean, dtype=np.float64)


def local_log_density(latent: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-point log local-density proxy: ``-log(distance to k-th nearest neighbour)``.

    Higher = denser. Computed in latent space (the clean generating geometry).
    """
    k = min(k, len(latent) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    dist, _ = nn.kneighbors(latent)
    kth = dist[:, -1]
    kth = np.maximum(kth, 1e-12)
    return -np.log(kth)
