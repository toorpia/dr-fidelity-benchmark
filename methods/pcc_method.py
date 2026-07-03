"""PCC -- Preserving Clusters and Correlations, used LABEL-FREE (pccdr; Gildenblat & Pahnke 2025).

Configuration (verified against installed pccdr): ``PCC(cluster=False, pearson=True, spearman=False,
n_components=2, num_points=N)``.

* ``cluster=False`` disables the cluster-supervision (MiCS / CrossEntropy) term entirely; the only
  objective is the **Pearson** correlation between high-D and 2-D distances to sampled reference
  points -- label-free.
* **Pearson, NOT Spearman, on purpose** (non-circularity): the primary scoring metric is the
  Spearman-rank band-Shepard rho. Optimizing Pearson (value) while scoring on Spearman (rank) is the
  honest, non-circular outcome -- we are not teaching to the test. See README.
* ``fit_transform(X, y)`` requires ``y``; with ``cluster=False`` it is unused -> pass ``np.zeros(N)``.
* ``num_points`` reference sampling is WITH REPLACEMENT (``np.random.choice``); for N < num_points it
  does not enumerate all points. We set ``num_points = N`` and document the with-replacement caveat;
  this is reported, not claimed "exact".

Stochastic in BOTH numpy (reference sampling) and torch (init): the driver seeds both; we also seed
locally for standalone use.
"""
from __future__ import annotations

import numpy as np

from .base import register


@register("PCC", stochastic=True, num_epochs=500)
def embed_pcc(X, seed, device="cpu", context=None, num_epochs=500):
    import torch
    from pcc import PCC

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    n = len(X)
    model = PCC(cluster=False, pearson=True, spearman=False, n_components=2,
                num_points=n, num_epochs=num_epochs)
    Y = model.fit_transform(np.ascontiguousarray(X, dtype=np.float64), np.zeros(n))
    return np.asarray(Y, dtype=np.float64)
