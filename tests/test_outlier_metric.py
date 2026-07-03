import numpy as np

from metrics import compute_all, outlier_metrics, outlier_pair_metrics, outlier_shepard  # noqa: F401


def _toy_2d(seed=0, m=2):
    """2-D latent toy: Gaussian bulk + m clear outliers. HD *is* 2-D, so the identity embedding is
    the trivial perfect case (OSP must be exactly 1)."""
    rng = np.random.default_rng(seed)
    bulk = rng.normal(0.0, 1.0, size=(120, 2))
    ang = rng.uniform(0, 2 * np.pi, size=m)
    out = 15.0 * np.column_stack([np.cos(ang), np.sin(ang)])
    X = np.vstack([bulk, out])
    oi = np.arange(len(bulk), len(bulk) + m)
    return X, oi


def test_osp_identity_embedding_is_one():
    X, oi = _toy_2d()
    row = outlier_metrics(X, X.copy(), oi, tag="vs_ambient")
    for j in range(len(oi)):
        assert abs(row[f"osp_o{j}__vs_ambient"] - 1.0) < 1e-12
        assert row[f"iso_rank_delta_o{j}__vs_ambient"] == 0.0
    assert abs(row["osp_median__vs_ambient"] - 1.0) < 1e-12
    assert abs(row["log2_osp_median__vs_ambient"]) < 1e-12


def test_osp_collapsed_outlier_is_small():
    X, oi = _toy_2d()
    Y = X.copy()
    Y[oi] = X[:oi[0]].mean(axis=0)      # bury every outlier at the bulk centroid
    row = outlier_metrics(X, Y, oi, tag="vs_ambient")
    assert row["osp_median__vs_ambient"] < 0.2
    assert row["osp_min__vs_ambient"] < 0.2
    # the buried outliers lose their top isolation ranks
    assert row["iso_rank_delta_mean__vs_ambient"] > 0


def test_osp_invariant_to_similarity_transform():
    """Uniform scale + rotation + reflection + translation of the embedding leave OSP unchanged
    (it is a ratio within each space) -- no Procrustes gauge removal needed."""
    rng = np.random.default_rng(5)
    X, oi = _toy_2d(seed=1)
    Y = rng.normal(size=X.shape)                     # arbitrary (imperfect) embedding
    base = outlier_metrics(X, Y, oi, tag="vs_ambient")
    th = 1.234
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    flip = np.array([[1.0, 0.0], [0.0, -1.0]])
    Y2 = 37.5 * (Y @ rot @ flip) + np.array([100.0, -3.0])
    row = outlier_metrics(X, Y2, oi, tag="vs_ambient")
    for k, v in base.items():
        assert np.isclose(row[k], v, rtol=1e-9, atol=1e-9), k


def _toy_pairs_2d(seed=0):
    """2-D toy with 2 anomalous directions x 2 near-duplicate members each."""
    rng = np.random.default_rng(seed)
    bulk = rng.normal(0.0, 1.0, size=(120, 2))
    out = np.array([[15.0, 0.0], [16.0, 0.0], [0.0, 15.0], [0.0, 16.0]])
    X = np.vstack([bulk, out])
    oi = np.arange(len(bulk), len(bulk) + 4)
    od = np.array([0, 0, 1, 1])
    return X, oi, od


def test_pair_metrics_identity_is_perfect():
    from metrics import outlier_pair_metrics
    X, oi, od = _toy_pairs_2d()
    row = outlier_pair_metrics(X, X.copy(), oi, od, tag="vs_ambient")
    for j in (0, 1):
        assert abs(row[f"pair_cohesion_dir{j}__vs_ambient"] - 1.0) < 1e-12
        assert row[f"pair_angle_2d_dir{j}"] < 0.1
    assert abs(row["pair_cohesion_median__vs_ambient"] - 1.0) < 1e-12


def test_pair_metrics_detect_split_pair():
    from metrics import outlier_pair_metrics
    X, oi, od = _toy_pairs_2d()
    Y = X.copy()
    Y[oi[1]] = [0.0, -16.0]          # tear direction-0's pair apart (opposite side of the bulk)
    row = outlier_pair_metrics(X, Y, oi, od, tag="vs_ambient")
    assert row["pair_cohesion_dir0__vs_ambient"] > 5
    assert row["pair_angle_2d_dir0"] > 60
    assert abs(row["pair_cohesion_dir1__vs_ambient"] - 1.0) < 1e-12


def test_outlier_shepard_identity_and_subset():
    from itertools import combinations
    from scipy.stats import spearmanr
    from metrics import outlier_shepard
    X, oi = _toy_2d(seed=4, m=3)
    assert abs(outlier_shepard(X, X.copy(), oi) - 1.0) < 1e-12       # identity embedding -> 1
    rng = np.random.default_rng(6)
    Y = rng.normal(size=X.shape)
    # equals a hand-built Spearman over exactly the anomaly-involving pairs
    pairs = [(i, j) for i, j in combinations(range(len(X)), 2) if i in set(oi) or j in set(oi)]
    a = [np.linalg.norm(X[i] - X[j]) for i, j in pairs]
    b = [np.linalg.norm(Y[i] - Y[j]) for i, j in pairs]
    assert np.isclose(outlier_shepard(X, Y, oi), spearmanr(a, b).statistic, rtol=1e-12)
    # burying the outliers in the bulk scrambles those pairs -> clearly below identity
    Yb = X.copy(); Yb[oi] = X[:oi[0]].mean(axis=0)
    assert outlier_shepard(X, Yb, oi) < 0.5


def test_compute_all_emits_outlier_block():
    rng = np.random.default_rng(2)
    X, oi = _toy_2d(seed=2)
    Y = rng.normal(size=X.shape)
    row = compute_all(X, Y, X_truth=X, include_per_point=False, outlier_idx=oi)
    assert "osp_median__vs_ambient" in row
    assert "osp_o0__vs_ambient" in row and "iso_rank_2d_o0" in row
    # pair block appears when direction ids are supplied
    Xp, oip, odp = _toy_pairs_2d(seed=3)
    rowp = compute_all(Xp, rng.normal(size=Xp.shape), X_truth=Xp, include_per_point=False,
                       outlier_idx=oip, outlier_dir=odp)
    assert "pair_cohesion_median__vs_ambient" in rowp and "pair_angle_2d_max" in rowp
    # OSP is reported vs-ambient ONLY (the repo's primary axis); no vs-truth OSP keys are emitted
    assert not any(("osp" in k or "iso_rank" in k) and k.endswith("__vs_truth") for k in row)
    # without outlier_idx the block is absent (no impact on existing datasets)
    row2 = compute_all(X, Y, X_truth=X, include_per_point=False)
    assert not any(k.startswith("osp") for k in row2)
