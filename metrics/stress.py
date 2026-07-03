"""Distance-band stress (value-based, complements the rank-based band-Shepard rho).

Normalized stress measures how well 2-D distances reproduce high-D distance VALUES after the single
best global scale factor (Kruskal-style residual). It catches methods that preserve the ORDER of
distances (good Shepard rho) yet distort their magnitudes -- e.g. scale-invariant or strongly
nonlinear objectives.

For a band's high-D distances ``a`` and 2-D distances ``b``::

    alpha  = <a, b> / <b, b>            # optimal isotropic scale of b onto a
    stress = sqrt( sum((a - alpha*b)^2) / sum(a^2) )

``stress`` is in [0, 1+]; lower is better. ``fidelity = max(0, 1 - stress)`` (higher is better) is
also reported so it can sit alongside Shepard rho in summary tables. ``p = 100`` recovers the classic
full-data stress. This optimal-scale formulation matches the prior campari benchmark
(he-dr-benchmark-campari/code/reproduce_fidelity.py).
"""
from __future__ import annotations

import numpy as np

from .distances import DEFAULT_CUTOFFS, band_mask


def normalized_stress(a: np.ndarray, b: np.ndarray) -> float:
    """Optimal-scale normalized (Kruskal) stress between high-D distances ``a`` and 2-D ``b``."""
    bb = float(np.dot(b, b))
    if bb <= 0:
        return float("nan")
    alpha = float(np.dot(a, b)) / bb
    denom = float(np.dot(a, a))
    if denom <= 0:
        return float("nan")
    resid = a - alpha * b
    return float(np.sqrt(float(np.dot(resid, resid)) / denom))


def band_stress(d_hd: np.ndarray, d_2d: np.ndarray, cutoffs=DEFAULT_CUTOFFS) -> dict:
    """Normalized stress within each cumulative global band. Returns ``{p: stress}`` (lower=better)."""
    out = {}
    for p in cutoffs:
        m = band_mask(d_hd, p)
        if m.sum() < 2:
            out[int(p)] = float("nan")
            continue
        out[int(p)] = normalized_stress(d_hd[m], d_2d[m])
    return out


def stress_to_fidelity(stress: float) -> float:
    """Convert a stress value to a (higher-is-better) fidelity score ``max(0, 1 - stress)``."""
    if stress != stress:  # NaN
        return float("nan")
    return float(max(0.0, 1.0 - stress))
