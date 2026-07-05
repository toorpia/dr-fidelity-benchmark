"""Label-separation metrics on the 2-D embedding: are the labeled clusters still distinct?

``cluster_scale_metrics`` measures within/between SCALE (over-compression); these measure MIXING --
whether points of different labels remain separable in the 2-D map, which is the visual question of
the curse-of-dimensionality (noise-dims) experiment. Chance level for leave-one-out kNN accuracy
with K balanced classes is 1/K.

Not folded into ``metrics.compute.compute_all`` (yet): only the noise-dims runner consumes these, so
the existing benchmark output schema stays untouched.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def knn_label_accuracy(Y: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Leave-one-out kNN majority-vote accuracy of ``labels`` in the embedding ``Y``."""
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    n = Y.shape[0]
    k = min(int(k), n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    _, idx = nn.kneighbors(Y)
    neigh = labels[idx[:, 1:]]                       # drop self (column 0)
    pred = np.array([np.bincount(row).argmax() for row in neigh])
    return float(np.mean(pred == labels))


def silhouette_by_label(Y: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score of ``labels`` in the embedding ``Y`` (range [-1, 1]; higher = separated)."""
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return float("nan")
    return float(silhouette_score(np.ascontiguousarray(Y, dtype=np.float64), labels))
