"""Multi-series high-dimensional random-walk dataset: time-series (continuous change) probe.

``n_series`` independent random walks in ``ndim`` dimensions (uniform steps in
``[-step, +step]`` per dimension, starting near the origin), concatenated. The regime is the
shape of real multivariate monitoring data: OPEN trajectories (filamentary structure, no clusters
at all), with two scale-separated properties a faithful 2-D map must render at once:

1. LOCAL -- the walk's saw-tooth step structure (successive displacements are independent).
2. GLOBAL -- the series separate radially over time: in high dimensions two random directions
   are almost surely near-orthogonal, so different series diverge from the shared origin.

Each series draws from ``default_rng(SeedSequence([seed, s]))`` so every (seed, series) is
independently reproducible in any order (required by the toorPIA on-disk cache contract).

NOT registered in ``synth.registry``: like ``noise_dims`` this is a probe. Its ground truth IS the
ambient geometry (the walk is generated directly in D dimensions), so ``truth_coords == clean``
and the vs-truth metrics coincide with vs-ambient.
"""
from __future__ import annotations

import numpy as np


def make_random_walks(ndim: int = 50, npoints: int = 500, n_series: int = 6,
                      step: float = 0.001, seed: int = 0) -> dict:
    """Generate the concatenated multi-series random walk.

    Returns the standard dict: ``clean`` ((n_series * npoints) x ndim), ``truth_coords``
    (== ``clean``), ``series`` (series id per row), ``t`` (time index within the series),
    ``color_value`` (global row index -- the time-gradient coloring of the gallery), ``name``,
    ``params``.
    """
    blocks = []
    for s in range(n_series):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(s)]))
        steps = rng.uniform(-step, step, size=(npoints, ndim))
        walk = np.cumsum(steps, axis=0)          # row 0 = first small displacement from origin
        blocks.append(walk)
    X = np.ascontiguousarray(np.vstack(blocks), dtype=np.float64)
    series = np.repeat(np.arange(n_series), npoints)
    t = np.tile(np.arange(npoints), n_series)
    return {
        "name": "random_walk",
        "clean": X,
        "truth_coords": X,
        "series": series,
        "t": t,
        "color_value": np.arange(len(X), dtype=float),
        "color_name": "time (concatenated series)",
        "params": dict(ndim=ndim, npoints=npoints, n_series=n_series, step=step, seed=seed),
    }
