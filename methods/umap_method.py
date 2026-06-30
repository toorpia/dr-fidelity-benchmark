"""UMAP -- neighbor-preserving, stochastic (umap-learn).

Passing ``random_state`` makes UMAP reproducible (it internally drops to single-threaded). Different
seeds give different embeddings, as required for the stability analysis. ``n_neighbors`` is clamped
below ``n`` for small-N smoke tests.
"""
from __future__ import annotations

import warnings

from .base import register


@register("UMAP", stochastic=True, n_neighbors=15, min_dist=0.1)
def embed_umap(X, seed, device="cpu", context=None, n_neighbors=15, min_dist=0.1):
    import umap
    n = len(X)
    nn = int(min(n_neighbors, max(2, n - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=min_dist,
                            random_state=int(seed))
        return reducer.fit_transform(X)
