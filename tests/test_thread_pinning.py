"""Order-independence of torch-based methods: a prior UMAP run must not change their results.

UMAP's first execution (numba threading-layer init) silently resets the process's torch thread
count (1 -> n_cores). Torch-based methods (PyMDE, PCC) then run multi-threaded, changing float
reduction order and hence their optimization trajectory: results would depend on which methods ran
BEFORE them in the same process. The wrappers re-pin ``torch.set_num_threads(1)`` on every call;
these tests are the regression net for that.
"""
import numpy as np
import torch

from methods import get_method


def _tiny_X(n=80, d=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def test_pymde_repins_torch_threads():
    X = _tiny_X()
    torch.set_num_threads(8)                      # simulate the post-UMAP state
    np.random.seed(0); torch.manual_seed(0)
    get_method("PyMDE").embed(X, seed=0)
    assert torch.get_num_threads() == 1


def test_pcc_repins_torch_threads():
    X = _tiny_X()
    torch.set_num_threads(8)
    np.random.seed(0); torch.manual_seed(0)
    get_method("PCC").embed(X, seed=0)
    assert torch.get_num_threads() == 1


def test_pymde_result_independent_of_prior_thread_state():
    X = _tiny_X()
    m = get_method("PyMDE")
    torch.set_num_threads(1)
    np.random.seed(0); torch.manual_seed(0)
    Y1 = m.embed(X, seed=0)
    torch.set_num_threads(8)                      # simulate the post-UMAP state
    np.random.seed(0); torch.manual_seed(0)
    Y2 = m.embed(X, seed=0)
    assert np.array_equal(Y1, Y2)
