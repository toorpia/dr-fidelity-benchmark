import numpy as np
from scipy.spatial.distance import pdist

from metrics import population_metrics
from synth import make_dataset


def test_populations_deterministic_and_counts():
    a = make_dataset("populations", n=1000, d=64, seed=7)
    b = make_dataset("populations", n=1000, d=64, seed=7)
    assert np.array_equal(a["clean"], b["clean"])
    assert np.array_equal(a["labels"], b["labels"])
    assert np.array_equal(a["population"], b["population"])
    assert (a["population"] == 1).sum() == 50 and (a["population"] == 0).sum() == 950
    # labels 0-4 are the majority's clusters, 5-9 the minority's; both evenly split
    assert set(np.unique(a["labels"])) == set(range(10))
    for k in range(5):
        assert (a["labels"] == k).sum() == 190
        assert (a["labels"] == 5 + k).sum() == 10
    assert np.all((a["labels"] >= 5) == (a["population"] == 1))
    assert np.allclose(pdist(a["truth_coords"]), pdist(a["clean"]))


def test_populations_center_geometry_is_exact():
    """With near-zero within-cluster scatter the sample centroids are the designed centers, so the
    two-level distance contract can be checked exactly: every within-population center distance is
    sqrt(2)*c and every cross-population center distance is group_range times that."""
    ds = make_dataset("populations", n=1000, d=64, seed=0, cluster_sigma=1e-9, group_range=2.0)
    c = ds["center_scale"]
    cents = np.array([ds["clean"][ds["labels"] == i].mean(axis=0) for i in range(10)])
    within = [np.linalg.norm(cents[i] - cents[j]) for a, b in ((0, 5), (5, 10))
              for i in range(a, b) for j in range(i + 1, b)]
    cross = [np.linalg.norm(cents[i] - cents[j]) for i in range(5) for j in range(5, 10)]
    assert np.allclose(within, np.sqrt(2.0) * c, rtol=1e-6)
    assert np.allclose(cross, 2.0 * np.sqrt(2.0) * c, rtol=1e-6)


def test_population_metrics_identity_and_shuffled_minority():
    rng = np.random.default_rng(3)
    ds = make_dataset("populations", n=400, d=40, seed=3)
    X, pop, labs = ds["clean"], ds["population"], ds["labels"]
    row = population_metrics(X, X.copy(), X, pop, labs)
    for k in ("within_majority_shepard__vs_ambient", "within_minority_shepard__vs_ambient",
              "cross_population_shepard__vs_ambient", "minority_shepard__vs_ambient"):
        assert abs(row[k] - 1.0) < 1e-9
    assert abs(row["population_over_compression"] - 1.0) < 1e-9
    # shuffling the minority's positions among themselves destroys the minority-internal ordering
    # but leaves the majority-internal pairs untouched
    Y = X.copy()
    mi = np.flatnonzero(pop == 1)
    Y[mi] = Y[mi[rng.permutation(len(mi))]]
    row2 = population_metrics(X, Y, X, pop, labs)
    assert abs(row2["within_majority_shepard__vs_ambient"] - 1.0) < 1e-9
    assert row2["within_minority_shepard__vs_ambient"] < 0.3
