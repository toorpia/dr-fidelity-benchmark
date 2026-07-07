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


def test_first_mode_threshold_bimodal_and_fallback():
    from metrics.distances import first_mode_threshold
    rng = np.random.default_rng(0)
    # bimodal profile: near mode at ~1, far mode at ~10 -> valley between them, no fallback
    d = np.concatenate([rng.normal(1.0, 0.15, 20000), rng.normal(10.0, 1.0, 80000)])
    d = d[d > 0]
    thr, fb = first_mode_threshold(d)
    assert not fb
    assert 1.5 < thr < 8.0
    frac = np.mean(d <= thr)
    assert 0.15 < frac < 0.30          # captures the near mode, not the far bulk
    # unimodal profile -> p5 fallback, flagged
    u = rng.normal(10.0, 1.0, 100000)
    thr_u, fb_u = first_mode_threshold(u)
    assert fb_u
    assert abs(np.mean(u <= thr_u) - 0.05) < 0.01


def test_tight_cluster_metrics_identity_crush_and_min_pts():
    from metrics.cluster_scale import tight_cluster_metrics
    rng = np.random.default_rng(1)
    # three clusters: one tight (60 pts, sigma .01), one loose (60, sigma 1), one tiny-tight (5 pts)
    tight = rng.normal(0, 0.01, (60, 4))
    loose = rng.normal(8, 1.0, (60, 4))
    tiny = rng.normal(-8, 0.001, (5, 4))
    X = np.vstack([tight, loose, tiny])
    labels = np.array([0] * 60 + [1] * 60 + [2] * 5)
    row = tight_cluster_metrics(X, X.copy(), labels)
    assert row["tight_cluster"] == 0          # tiny cluster excluded by min_pts, tight one wins
    assert abs(row["tight_over_compression"] - 1.0) < 1e-9
    Y = X.copy()
    Y[:60] = Y[:60].mean(axis=0) + (Y[:60] - Y[:60].mean(axis=0)) / 10.0   # crush cluster 0 by 10x
    row2 = tight_cluster_metrics(X, Y, labels)
    assert 5.0 < row2["tight_over_compression"] < 15.0
