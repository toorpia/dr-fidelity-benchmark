import numpy as np
from sklearn.decomposition import PCA

from synth import add_noise_relative, intra_cluster_variance, make_dataset


def test_clusters_dynamic_range_knob():
    ds = make_dataset("clusters", n=700, d=64, seed=0, n_clusters=7, dynamic_range=20.0)
    cen = ds["centroids_latent"]
    inter = np.linalg.norm(cen[0] - cen[1])
    # inter-centroid / intra-cluster spacing == requested dynamic range
    assert np.isclose(inter / ds["intra_nn"], 20.0, rtol=1e-6)
    # a different dynamic range scales the separation proportionally
    ds2 = make_dataset("clusters", n=700, d=64, seed=0, n_clusters=7, dynamic_range=5.0)
    inter2 = np.linalg.norm(ds2["centroids_latent"][0] - ds2["centroids_latent"][1])
    assert np.isclose(inter2 / ds2["intra_nn"], 5.0, rtol=1e-6)


def test_clusters_high_effective_dim():
    ds = make_dataset("clusters", n=700, d=128, seed=0, n_clusters=7, dynamic_range=20.0)
    X = ds["clean"] - ds["clean"].mean(0)
    ev = PCA().fit(X).explained_variance_
    pr = (ev.sum() ** 2) / np.square(ev).sum()
    assert pr > 4.0          # orthogonal K=7 arrangement -> effective dim ~6 (no low-dim trap)
    assert set(np.unique(ds["labels"])) == set(range(7))


def test_add_noise_relative_uses_intra_scale():
    ds = make_dataset("clusters", n=700, d=64, seed=1, dynamic_range=20.0)
    intra = intra_cluster_variance(ds["clean"], ds["labels"])
    X = add_noise_relative(ds["clean"], ds["labels"], 4.0, np.random.default_rng(0))
    assert np.isclose(np.var(X - ds["clean"]), intra / 4.0, rtol=0.2)
    # snr=inf -> unchanged
    same = add_noise_relative(ds["clean"], ds["labels"], float("inf"), np.random.default_rng(0))
    assert np.allclose(same, ds["clean"])
