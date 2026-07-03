import numpy as np
from scipy.spatial.distance import pdist

from synth import add_noise, make_dataset, signal_variance
from synth._common import project_to_D
from synth.transition import TRANSITION_LABEL


def test_orthonormal_projection_is_isometric():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(60, 8))
    clean = project_to_D(latent, 100, rng)
    assert clean.shape == (60, 100)
    assert np.allclose(pdist(latent), pdist(clean), atol=1e-9)


def test_density_shapes_and_labels():
    ds = make_dataset("density", n=200, d=64, seed=0)
    assert ds["clean"].shape == (200, 64)
    assert ds["truth_coords"].shape == (200, 64)
    assert set(np.unique(ds["labels"])) <= {0, 1, 2}
    assert ds["color_value"].shape == (200,)
    # truth distance == clean Euclidean (truth_coords is the clean image)
    assert np.allclose(pdist(ds["truth_coords"]), pdist(ds["clean"]))


def test_transition_has_states_and_continuum():
    ds = make_dataset("transition", n=300, d=64, seed=1, n_clusters=4)
    labs = set(np.unique(ds["labels"]))
    assert TRANSITION_LABEL in labs            # bridge points exist
    assert {0, 1, 2, 3} <= labs                # all states present
    t = ds["color_value"]
    assert t.min() >= 0.0 and t.max() <= 1.0   # continuum parameter in [0, 1]
    assert ds["centroids_latent"].shape[0] == 4


def test_noise_snr_inf_is_identity_and_finite_snr_scales():
    ds = make_dataset("density", n=150, d=32, seed=2)
    rng = np.random.default_rng(3)
    assert np.allclose(add_noise(ds["clean"], float("inf"), rng), ds["clean"])
    X = add_noise(ds["clean"], 4.0, np.random.default_rng(3))
    # noise power should be ~ signal_var / snr
    noise_var = np.var(X - ds["clean"])
    assert np.isclose(noise_var, signal_variance(ds["clean"]) / 4.0, rtol=0.2)


def test_dataset_generation_is_deterministic():
    a = make_dataset("transition", n=120, d=40, seed=7)
    b = make_dataset("transition", n=120, d=40, seed=7)
    assert np.array_equal(a["clean"], b["clean"])
    assert np.array_equal(a["color_value"], b["color_value"])
