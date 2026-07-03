"""Conventional neighbor-preservation metrics (recall@k, trustworthiness, continuity).

These are included for completeness and comparability with the DR literature, but are KNOWN to be
biased toward neighbor-preserving methods (UMAP / t-SNE) and structurally unfavorable to
distance-preserving methods: they use variable-radius k-NN sets and a hard inclusion threshold that
over-penalizes near-ties in dense regions. The fair local alternative is the near-band Shepard rho /
stress in ``shepard.py`` / ``stress.py``. See README for the full bias discussion.
"""
from __future__ import annotations

import numpy as np
from sklearn.manifold import trustworthiness as _sk_trustworthiness
from sklearn.neighbors import NearestNeighbors

DEFAULT_KS = (5, 15, 30)


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` nearest neighbours of each row of ``X`` (excluding self)."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]


def recall_at_k(X: np.ndarray, Y: np.ndarray, ks=DEFAULT_KS) -> dict:
    """Mean overlap (intersection / k) of high-D and 2-D k-NN sets. Returns ``{k: recall}``."""
    n = X.shape[0]
    out = {}
    for k in ks:
        if k >= n:
            out[int(k)] = float("nan")
            continue
        hd = _knn_indices(X, k)
        ld = _knn_indices(Y, k)
        hd_sets = [set(hd[i]) for i in range(n)]
        overlap = np.mean([len(hd_sets[i] & set(ld[i])) for i in range(n)]) / k
        out[int(k)] = float(overlap)
    return out


def trustworthiness_continuity(X: np.ndarray, Y: np.ndarray, ks=DEFAULT_KS) -> dict:
    """sklearn trustworthiness and its dual, continuity, at each k.

    Continuity is trustworthiness with the spaces swapped (``trustworthiness(Y, X)``).
    Returns ``{"trustworthiness": {k: v}, "continuity": {k: v}}``.
    """
    n = X.shape[0]
    tw, cont = {}, {}
    for k in ks:
        if k >= n // 2:  # sklearn requires k < n/2
            tw[int(k)] = float("nan")
            cont[int(k)] = float("nan")
            continue
        tw[int(k)] = float(_sk_trustworthiness(X, Y, n_neighbors=k))
        cont[int(k)] = float(_sk_trustworthiness(Y, X, n_neighbors=k))
    return {"trustworthiness": tw, "continuity": cont}
