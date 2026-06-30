"""SNR-controlled additive noise for the synthetic datasets.

The generators return CLEAN coordinates whose Euclidean distances define the ground truth. The driver
then adds isotropic Gaussian noise at a configurable signal-to-noise ratio (SNR) to form the ambient
features ``X`` the methods actually embed. Reporting metrics over an SNR sweep (rather than a single
noise level) is mandatory in this benchmark.

SNR is defined on power: ``SNR = signal_power / noise_power``, where signal power is the mean per-
coordinate variance of the clean data. Thus ``noise_std = sqrt(signal_var / SNR)`` per coordinate.
``SNR = inf`` (or non-finite) returns the clean data unchanged.
"""
from __future__ import annotations

import numpy as np


def signal_variance(clean: np.ndarray) -> float:
    """Mean per-coordinate variance of the clean signal (its 'power')."""
    return float(np.var(clean, axis=0).mean())


def add_noise(clean: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    """Return ``clean + isotropic Gaussian`` at the requested SNR (power ratio).

    ``snr`` non-finite or <= 0-handling: ``inf`` -> clean unchanged.
    """
    if snr is None or not np.isfinite(snr):
        return np.ascontiguousarray(clean, dtype=np.float64)
    sig_var = signal_variance(clean)
    noise_std = float(np.sqrt(sig_var / float(snr)))
    noise = rng.normal(scale=noise_std, size=clean.shape)
    return np.ascontiguousarray(clean + noise, dtype=np.float64)


def intra_cluster_variance(clean: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-coordinate variance computed WITHIN clusters (the intra-cluster signal scale).

    Unlike :func:`signal_variance` (which is dominated by between-cluster separation at high dynamic
    range), this measures only the spread inside each cluster.
    """
    total, count = 0.0, 0
    for lab in np.unique(labels):
        Xc = clean[labels == lab]
        if len(Xc) > 1:
            total += float(np.var(Xc, axis=0).mean()) * len(Xc)
            count += len(Xc)
    return total / max(count, 1)


def add_noise_relative(clean: np.ndarray, labels: np.ndarray, snr: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Additive noise whose power is set relative to the INTRA-cluster scale, not the global scale.

    At high dynamic range the global variance is dominated by between-cluster separation, so the
    global-SNR :func:`add_noise` swamps dense clusters at any modest SNR. This calibrates the noise to
    the within-cluster spread, so a given SNR is a meaningful noise level for the local structure being
    measured. ``snr=inf`` -> clean unchanged.
    """
    if snr is None or not np.isfinite(snr):
        return np.ascontiguousarray(clean, dtype=np.float64)
    intra_var = intra_cluster_variance(clean, labels)
    noise_std = float(np.sqrt(intra_var / float(snr)))
    noise = rng.normal(scale=noise_std, size=clean.shape)
    return np.ascontiguousarray(clean + noise, dtype=np.float64)
