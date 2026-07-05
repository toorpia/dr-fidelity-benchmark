"""PyMDE -- Minimum-Distortion Embedding, distance-based, stochastic (pymde).

Objective: ``pymde.preserve_distances`` with the default ``Absolute`` loss and a ``Standardized``
constraint (zero-mean, identity-covariance up to scale), which preserves distance VALUES while
fixing the trivial scale/translation gauge. Verified signature:
``preserve_distances(data, embedding_dim=2, loss=Absolute, constraint=None, max_distances=5e7,
device='cpu')`` -> ``MDE``; ``MDE.embed()`` returns the (n x 2) tensor. Default ``max_distances`` is
5e7, comfortably above n*(n-1)/2 for the benchmark's N, so all pairwise distances are used.

Seeded via ``torch.manual_seed`` (the driver also seeds numpy and forces CPU).
"""
from __future__ import annotations

from .base import register


@register("PyMDE", stochastic=True, max_iter=300)
def embed_pymde(X, seed, device="cpu", context=None, max_iter=300):
    import pymde
    import torch

    # re-pin every call: a prior UMAP run (numba threading-layer init) silently resets the
    # process's torch thread count (1 -> n_cores), changing float reduction order and hence the
    # optimization trajectory -- PyMDE results must not depend on which methods ran before it
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    data = torch.as_tensor(X, dtype=torch.float32)
    mde = pymde.preserve_distances(data, embedding_dim=2, constraint=pymde.Standardized(),
                                   device=device, verbose=False)
    Y = mde.embed(max_iter=max_iter, verbose=False)
    return Y.detach().cpu().numpy()
