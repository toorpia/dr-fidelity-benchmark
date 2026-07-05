"""Tests for the noise-dims generator and the label-separation metrics."""
import numpy as np
import pytest

from metrics.label_separation import knn_label_accuracy, silhouette_by_label
from synth.noise_dims import make_noise_dims


def test_shapes_and_contract():
    d = make_noise_dims(n=120, d=20)
    assert d["clean"].shape == (120, 20)
    assert d["truth_coords"].shape == (120, 3)
    assert np.array_equal(d["truth_coords"], d["clean"][:, :3])
    assert d["labels"].shape == (120,)
    assert set(np.unique(d["labels"])) <= {0, 1, 2}
    assert d["color_value"].shape == (120,)
    assert d["name"] == "noise_dims"


def test_deterministic_per_seed_and_dim():
    a = make_noise_dims(n=100, d=30, seed=42)
    b = make_noise_dims(n=100, d=30, seed=42)
    assert np.array_equal(a["clean"], b["clean"])
    c = make_noise_dims(n=100, d=30, seed=1)
    assert not np.array_equal(a["clean"], c["clean"])


def test_signal_columns_invariant_across_d():
    # StandardScaler is column-wise, so the standardized signal columns (and labels) must be
    # identical for every total dimensionality -- grid columns show the same clusters.
    a = make_noise_dims(n=100, d=10, seed=42)
    b = make_noise_dims(n=100, d=200, seed=42)
    assert np.allclose(a["truth_coords"], b["truth_coords"])
    assert np.array_equal(a["labels"], b["labels"])


def test_standardized_columns():
    d = make_noise_dims(n=300, d=50)
    assert np.allclose(d["clean"].mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(d["clean"].std(axis=0), 1.0, atol=1e-6)


def test_d_equals_principal_has_zero_noise_dims():
    d = make_noise_dims(n=60, d=3)
    assert d["clean"].shape == (60, 3)


def test_d_below_principal_raises():
    with pytest.raises(ValueError):
        make_noise_dims(n=60, d=2)


def test_knn_accuracy_and_silhouette_on_separated_clusters():
    rng = np.random.default_rng(0)
    Y = np.vstack([rng.normal(0, 0.05, (40, 2)), rng.normal(10, 0.05, (40, 2))])
    labels = np.array([0] * 40 + [1] * 40)
    assert knn_label_accuracy(Y, labels, k=10) == 1.0
    assert silhouette_by_label(Y, labels) > 0.9


def test_knn_accuracy_near_chance_on_shuffled_labels():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(300, 2))
    labels = rng.integers(0, 3, 300)
    assert knn_label_accuracy(Y, labels, k=10) < 0.55


def test_silhouette_single_label_is_nan():
    Y = np.random.default_rng(0).normal(size=(50, 2))
    assert np.isnan(silhouette_by_label(Y, np.zeros(50, dtype=int)))
