"""Curse-of-dimensionality dataset: 3 tight clusters in 3 principal dims + pure-noise dimensions.

Replicates the classroom experiment in ``01_hi_dimensional_data_anlaysis.ipynb``: ``make_blobs``
places ``n_principal`` tight Gaussian clusters (std 0.005) at the unit-vector corners of an
``n_principal``-dim signal subspace; the remaining ``d - n_principal`` columns are standard-normal
noise; finally every column is z-scored with ``StandardScaler``. As ``d`` grows the unit-variance
noise columns dominate every pairwise distance and the cluster structure drowns -- the "curse of
dimensionality" under test. The signal/noise distance ratio decays like ~3/sqrt(2(d-3)); that decay
IS the phenomenon, so nothing compensates for it.

Documented deviation from the notebook: the notebook drew the noise columns from the GLOBAL numpy
RNG (cell-execution-order dependent), so re-running a different subset of dims would silently change
the data. Here noise comes from ``default_rng(SeedSequence([seed, d]))`` -- each (seed, d) config is
independently reproducible in any sweep order, which the toorPIA on-disk cache requires (a cached
embedding must always correspond to byte-identical input).

NOT registered in ``synth.registry``: ``truth_coords`` (the standardized signal columns) is
deliberately NOT isometric to ``clean`` (all d columns) -- the noise dims are the phenomenon --
which violates the registry's ``pdist(truth_coords) == pdist(clean)`` contract. Runners import this
generator directly (same pattern as ``run/sweep.py`` importing ``make_clusters``).
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def make_noise_dims(n: int = 500, d: int = 100, seed: int = 42, n_principal: int = 3,
                    cluster_std: float = 0.005, standardize: bool = True) -> dict:
    """Generate the noise-dims dataset.

    Returns the standard dict: ``clean`` (n x d, standardized), ``truth_coords`` (the standardized
    signal columns == ``clean[:, :n_principal]`` -- the true cluster geometry embedded in what the
    methods actually see), ``labels`` (cluster id), ``label_names``, ``color_value``, ``color_name``,
    ``name``, ``params``.
    """
    if d < n_principal:
        raise ValueError(f"d ({d}) must be >= n_principal ({n_principal})")
    centers = np.eye(n_principal).tolist()
    features, labels = make_blobs(n_samples=n, centers=centers, n_features=n_principal,
                                  random_state=seed, cluster_std=cluster_std)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(d)]))
    noise = rng.standard_normal((n, d - n_principal))
    features = np.hstack([features, noise])
    if standardize:
        features = StandardScaler().fit_transform(features)
    return {
        "name": "noise_dims",
        "clean": features,
        "truth_coords": features[:, :n_principal],
        "labels": labels.astype(int),
        "label_names": {i: f"cluster {i}" for i in range(n_principal)},
        "color_value": labels.astype(float),
        "color_name": "cluster id",
        "params": dict(n=n, d=d, seed=seed, n_principal=n_principal, cluster_std=cluster_std,
                       standardize=standardize),
    }
