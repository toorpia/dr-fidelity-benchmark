"""Distance-band Shepard correlation (the primary, fair local-fidelity metric).

Primary (global / absolute-band): for each cumulative band at cutoff ``p`` (pairs whose HIGH-D
distance is in the lowest ``p`` percent of ALL pairwise distances), the Spearman rank correlation
between high-D and 2-D pairwise distances. Bands are defined on the GLOBAL high-D distance
distribution, so every method is scored on the same, absolute set of near/mid/far pairs.

Secondary (per-point / variable-radius): for each point, take that point's own lowest-``p``-percent
pairs (a per-point variable radius, like a variable k-NN set), pool them, and correlate. This mirrors
the variable-radius selection that biases recall@k, and is provided only for contrast. The global
(absolute) variant is PRIMARY.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from .distances import DEFAULT_CUTOFFS, band_mask, square_distances


def band_shepard(d_hd: np.ndarray, d_2d: np.ndarray, cutoffs=DEFAULT_CUTOFFS) -> dict:
    """PRIMARY. Spearman rho within each cumulative global band.

    Parameters
    ----------
    d_hd, d_2d : condensed pairwise-distance vectors (same ordering, e.g. both from ``pdist``).
    Returns ``{p: rho}``. ``p = 100`` is the classic full-data Shepard rho.
    """
    out = {}
    for p in cutoffs:
        m = band_mask(d_hd, p)
        if m.sum() < 3:
            out[int(p)] = float("nan")
            continue
        rho, _ = spearmanr(d_hd[m], d_2d[m])
        out[int(p)] = float(rho)
    return out


def per_point_band_shepard(X: np.ndarray, Y: np.ndarray, cutoffs=DEFAULT_CUTOFFS,
                           d_hd_sq: np.ndarray | None = None) -> dict:
    """SECONDARY (variable-radius view). Pool each point's lowest-``p``-percent pairs, then correlate.

    ``d_hd_sq`` may be a precomputed (n x n) high-D distance matrix (e.g. ground-truth distances);
    if ``None`` it is computed from ``X`` as Euclidean.
    """
    Dhd = (square_distances(X) if d_hd_sq is None else d_hd_sq).copy()
    D2d = square_distances(Y)
    n = Dhd.shape[0]
    np.fill_diagonal(Dhd, np.inf)        # exclude self-pairs from every "lowest-p%" set
    rows = np.arange(n)[:, None]
    out = {}
    for p in cutoffs:
        # per-point variable radius: each point keeps its own lowest-p% pairs
        k = min(n - 1, max(1, int(round((p / 100.0) * (n - 1)))))
        idx = np.argpartition(Dhd, k - 1, axis=1)[:, :k]   # (n, k) nearest-by-high-D per point
        hd_sel = Dhd[rows, idx].ravel()
        td_sel = D2d[rows, idx].ravel()
        if hd_sel.size < 3:
            out[int(p)] = float("nan")
            continue
        rho, _ = spearmanr(hd_sel, td_sel)
        out[int(p)] = float(rho)
    return out
