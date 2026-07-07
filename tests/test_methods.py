import os
import tempfile

import numpy as np
import pytest

from methods import get_method
from methods.base import SkipMethod
from methods.external import save_external

FAST_METHODS = ["PCA", "Isomap", "PyMDE", "PCC", "t-SNE"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(80, 12))


@pytest.mark.parametrize("name", FAST_METHODS)
def test_method_returns_n_by_2_finite(name, data):
    m = get_method(name)
    Y = m.embed(data, seed=0, device="cpu", context=None)
    assert Y.shape == (data.shape[0], 2)
    assert np.isfinite(Y).all()


def test_pca_is_deterministic_across_seeds(data):
    pca = get_method("PCA")
    y0 = pca.embed(data, seed=0, device="cpu")
    y1 = pca.embed(data, seed=999, device="cpu")
    assert np.allclose(y0, y1)


def test_stochastic_method_varies_with_seed(data):
    pcc = get_method("PCC")
    np.random.seed(0)
    y0 = pcc.embed(data, seed=0, device="cpu")
    np.random.seed(1)
    y1 = pcc.embed(data, seed=1, device="cpu")
    assert not np.allclose(y0, y1)


def test_toorpia_skips_without_cache_or_key(data):
    os.environ.pop("TOORPIA_API_KEY", None)
    m = get_method("toorPIA")
    ctx = {"root": tempfile.mkdtemp(), "dataset": "density", "snr": float("inf"),
           "tag": "n80_d12_snrinf"}
    with pytest.raises(SkipMethod):
        m.embed(data, seed=0, device="cpu", context=ctx)


def test_toorpia_loads_cache_without_api(data):
    root = tempfile.mkdtemp()
    fake = np.random.default_rng(5).normal(size=(data.shape[0], 2))
    save_external(fake, root, "density", "toorpia", "embedding", "n80_d12_snrinf", 0)
    os.environ["TOORPIA_API_KEY"] = "dummy-should-not-be-used"
    ctx = {"root": root, "dataset": "density", "snr": float("inf"), "tag": "n80_d12_snrinf"}
    Y = get_method("toorPIA").embed(data, seed=0, device="cpu", context=ctx)
    assert np.allclose(Y, fake)
    os.environ.pop("TOORPIA_API_KEY", None)
