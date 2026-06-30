"""PCA -- linear, deterministic baseline (sklearn)."""
from __future__ import annotations

from sklearn.decomposition import PCA

from .base import register


@register("PCA", stochastic=False)
def embed_pca(X, seed, device="cpu", context=None):
    return PCA(n_components=2, random_state=seed).fit_transform(X)
