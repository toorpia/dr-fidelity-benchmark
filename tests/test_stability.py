import numpy as np

from metrics.stability import (align_to_reference, mean_pairwise_disparity,
                               metric_variance, position_dispersion)


def _gauge_transform(Y, rng):
    """Apply a random rotation/reflection + scale + translation (a pure gauge change)."""
    th = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    if rng.random() < 0.5:
        R = R @ np.array([[1, 0], [0, -1]])      # reflection
    return Y @ R * rng.uniform(0.5, 2.0) + rng.uniform(-3, 3, size=2)


def test_pure_gauge_has_zero_dispersion():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(50, 2))
    embs = [_gauge_transform(base, rng) for _ in range(5)]
    assert position_dispersion(embs) < 1e-6
    assert mean_pairwise_disparity(embs) < 1e-6


def test_alignment_reduces_inter_run_distance():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(50, 2))
    embs = [_gauge_transform(base, rng) for _ in range(4)]

    def spread(arrs):
        A = np.stack(arrs, 0)
        return float(A.std(0).mean())

    before = spread([e - e.mean(0) for e in embs])   # centered but unaligned
    after = spread(align_to_reference(embs))
    assert after < before


def test_dispersion_grows_with_noise():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(50, 2))
    small = [base + rng.normal(0, 0.01, base.shape) for _ in range(5)]
    big = [base + rng.normal(0, 0.2, base.shape) for _ in range(5)]
    assert position_dispersion(small) < position_dispersion(big)


def test_metric_variance_handles_nan():
    v = metric_variance(np.array([0.5, 0.6, np.nan, 0.7]))
    assert v["n"] == 3
    assert 0.5 <= v["median"] <= 0.7
