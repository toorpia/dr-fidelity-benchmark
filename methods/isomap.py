"""Isomap -- geodesic / global, deterministic baseline (sklearn). Document ``n_neighbors``."""
from __future__ import annotations

from sklearn.manifold import Isomap

from .base import register


@register("Isomap", stochastic=False, n_neighbors=15)
def embed_isomap(X, seed, device="cpu", context=None, n_neighbors=15):
    n = len(X)
    nn = min(n_neighbors, max(2, n - 1))
    return Isomap(n_neighbors=nn, n_components=2).fit_transform(X)
