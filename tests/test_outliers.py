import numpy as np
from scipy.spatial.distance import pdist

from synth import make_dataset
from synth.outliers import OUTLIER_LABEL


def test_outliers_generation_is_deterministic():
    a = make_dataset("outliers", n=300, d=50, seed=7)
    b = make_dataset("outliers", n=300, d=50, seed=7)
    assert np.array_equal(a["clean"], b["clean"])
    assert np.array_equal(a["labels"], b["labels"])
    assert np.array_equal(a["outlier_idx"], b["outlier_idx"])
    assert np.array_equal(a["outlier_dir"], b["outlier_dir"])


def test_outliers_radii_match_factor_and_pair_offset():
    for factor in (1.5, 3.0, 8.0):
        ds = make_dataset("outliers", n=400, d=64, seed=0, outlier_factor=factor)
        labs, oi, od = ds["labels"], ds["outlier_idx"], ds["outlier_dir"]
        assert np.array_equal(oi, np.flatnonzero(labs == OUTLIER_LABEL))
        assert len(oi) == 6 and sorted(set(od)) == [0, 1, 2]
        # linear isometry -> each outlier's distance to the bulk centroid equals its recorded
        # factor (factor or factor + pair_offset) times Rg, exactly
        bulk_center = ds["clean"][labs != OUTLIER_LABEL].mean(axis=0)
        d_o = np.linalg.norm(ds["clean"][oi] - bulk_center, axis=1)
        assert np.allclose(d_o, ds["outlier_factors"] * ds["bulk_scale"], rtol=1e-9)
        # each direction has one member at factor and one at factor + 0.1 (default pair_offset)
        for j in range(3):
            fs = np.sort(ds["outlier_factors"][od == j])
            assert np.allclose(fs, [factor, factor + 0.1])
        # same-direction pair separation = pair_offset * Rg; cross-direction ~ sqrt(f1^2+f2^2)*Rg
        for j in range(3):
            a, b = oi[od == j]
            assert np.isclose(np.linalg.norm(ds["clean"][a] - ds["clean"][b]),
                              0.1 * ds["bulk_scale"], rtol=1e-9)


def test_outliers_are_off_subspace_and_pairs_collinear():
    """Every outlier's displacement from the bulk centroid is orthogonal to the whole subspace the
    bulk spans, and same-direction pair members are exactly collinear as seen from the centroid.
    The projection has orthonormal columns (preserves inner products), so both hold EXACTLY in the
    clean ambient space."""
    ds = make_dataset("outliers", n=400, d=64, seed=2, outlier_factor=3.0)
    labs, oi, od = ds["labels"], ds["outlier_idx"], ds["outlier_dir"]
    bulk = ds["clean"][labs != OUTLIER_LABEL]
    center = bulk.mean(axis=0)
    B = bulk - center
    for o in oi:
        v = ds["clean"][o] - center
        cos = (B @ v) / (np.linalg.norm(B, axis=1) * np.linalg.norm(v) + 1e-300)
        assert np.max(np.abs(cos)) < 1e-9
    for j in range(3):
        a, b = oi[od == j]
        va, vb = ds["clean"][a] - center, ds["clean"][b] - center
        cos = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))
        assert cos > 1 - 1e-12                        # same direction exactly
    # distinct directions are orthogonal
    a, b = oi[od == 0][0], oi[od == 1][0]
    va, vb = ds["clean"][a] - center, ds["clean"][b] - center
    assert abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)) < 1e-9


def test_outliers_bulk_scale_is_radius_of_gyration():
    ds = make_dataset("outliers", n=400, d=64, seed=3)
    bulk = ds["clean"][ds["labels"] != OUTLIER_LABEL]
    # isometric projection -> the bulk's radius of gyration is preserved exactly in clean space
    rg = np.sqrt(np.mean(np.sum((bulk - bulk.mean(axis=0)) ** 2, axis=1)))
    assert np.isclose(ds["bulk_scale"], rg, rtol=1e-9)


def test_addplot_points_do_not_change_fit_data():
    plain = make_dataset("outliers", n=300, d=50, seed=7)
    with_add = make_dataset("outliers", n=300, d=50, seed=7, n_add_bulk=20, add_per_direction=1)
    assert np.array_equal(plain["clean"], with_add["clean"])          # fit part bit-identical
    assert np.array_equal(plain["labels"], with_add["labels"])
    ap = with_add["addplot"]
    assert ap["clean"].shape == (23, 50)
    # each added anomaly: radius (factor + 0.05) * Rg, exactly co-directional with its fit pair
    center = with_add["clean"][with_add["labels"] != OUTLIER_LABEL].mean(axis=0)
    for j in range(3):
        row = ap["clean"][ap["dir"] == j][0]
        r = np.linalg.norm(row - center)
        assert np.isclose(r, 3.05 * with_add["bulk_scale"], rtol=1e-9)
        a = with_add["outlier_idx"][with_add["outlier_dir"] == j][0]
        v1, v2 = row - center, with_add["clean"][a] - center
        assert v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)) > 1 - 1e-12


def test_outliers_shapes_and_truth():
    ds = make_dataset("outliers", n=250, d=40, seed=1)
    assert ds["clean"].shape == (250, 40)
    assert np.allclose(pdist(ds["truth_coords"]), pdist(ds["clean"]))
    assert set(np.unique(ds["labels"])) == set(range(5)) | {OUTLIER_LABEL}
    assert ds["color_value"].shape == (250,)


def test_anchored_addplot_geometry():
    """Anchored anomalies: anchor = the cluster's clean centroid, deviation entirely orthogonal
    to the bulk's spanned subspace, radii (dev, dev+offset) x Rg, deterministic in the seed."""
    from synth.outliers import make_anchored_addplot
    base = make_dataset("outliers", n=400, d=64, seed=3)
    a = make_anchored_addplot(base, deviation=3.0, pair_offset=0.1, seed=3)
    b = make_anchored_addplot(base, deviation=3.0, pair_offset=0.1, seed=3)
    assert np.array_equal(a["clean"], b["clean"])
    labs = base["labels"]; bulk = base["clean"][labs != -1]; labs_b = labs[labs != -1]
    K = int(labs_b.max()) + 1
    assert a["clean"].shape == (2 * K, 64)
    assert list(a["anchor"]) == [k for k in range(K) for _ in range(2)]
    Rg = float(np.sqrt(np.mean(np.sum((bulk - bulk.mean(0)) ** 2, axis=1))))
    centered = bulk - bulk.mean(0)
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)
    P = Vt[: int((s > s[0] * 1e-8).sum())]
    for i in range(2 * K):
        k = a["anchor"][i]
        cent = bulk[labs_b == k].mean(0)
        dev = a["clean"][i] - cent
        # deviation lies entirely outside the bulk subspace and has the designed radius
        assert np.linalg.norm(P @ dev) < 1e-8 * np.linalg.norm(dev)
        assert abs(np.linalg.norm(dev) - a["radii"][i] * Rg) < 1e-6 * Rg
