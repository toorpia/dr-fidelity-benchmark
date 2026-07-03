import numpy as np

from metrics import (band_shepard, band_stress, compute_all, condensed_distances,
                     normalized_stress, recall_at_k)
from metrics.distances import band_mask


def test_identity_embedding_is_perfect():
    rng = np.random.default_rng(0)
    X2 = rng.normal(size=(120, 2))             # 2-D input embedded as itself
    row = compute_all(X2, X2.copy(), X_truth=None)
    assert abs(row["full_shepard"] - 1.0) < 1e-9
    assert row["full_stress"] < 1e-9
    assert abs(row["shepard_p5__vs_ambient"] - 1.0) < 1e-9
    assert abs(row["recall_k5"] - 1.0) < 1e-9


def test_band_mask_is_cumulative_and_p100_is_all():
    d = np.arange(100.0)
    assert band_mask(d, 100).all()
    m5 = band_mask(d, 5)
    assert m5.sum() <= band_mask(d, 50).sum() <= band_mask(d, 100).sum()
    # only the smallest distances are selected
    assert d[m5].max() <= np.percentile(d, 5)


def test_normalized_stress_scale_invariance():
    rng = np.random.default_rng(1)
    a = rng.uniform(1, 5, size=200)
    assert normalized_stress(a, a) < 1e-12          # identical
    assert normalized_stress(a, 7.3 * a) < 1e-9     # pure rescale -> optimal-scale stress ~ 0


def test_band_shepard_monotone_for_isometry():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 5))
    d_hd = condensed_distances(X)
    rho = band_shepard(d_hd, d_hd)          # X vs itself
    assert all(abs(v - 1.0) < 1e-9 for v in rho.values())
    st = band_stress(d_hd, d_hd)
    assert all(v < 1e-9 for v in st.values())


def test_recall_at_k_keys():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 8))
    Y = rng.normal(size=(100, 2))
    r = recall_at_k(X, Y, ks=(5, 15))
    assert set(r) == {5, 15}
    assert all(0.0 <= v <= 1.0 for v in r.values())


def test_truth_block_present_when_truth_given():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 10))
    Y = rng.normal(size=(60, 2))
    row = compute_all(X, Y, X_truth=X)
    assert any(k.endswith("__vs_truth") for k in row)
    assert "shepard_p100__vs_truth" in row
