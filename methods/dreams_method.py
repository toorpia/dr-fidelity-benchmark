"""DREAMS -- t-SNE regularized toward the PCA embedding (Kury, Kobak & Damrich, TMLR 2026).

DREAMS (arXiv:2508.13747) adds to t-SNE's KL objective an MSE pull toward a precomputed reference
embedding (default: PCA-2), weighted by ``reg_lambda``, to preserve local and global structure at
once. Implementation is the authors' openTSNE fork::

    pip install "git+https://github.com/berenslab/DREAMS.git@tp"

which installs AS ``openTSNE`` (version 1.0.2 + regularization; replaces vanilla openTSNE). Nothing
else in this repo imports openTSNE, so the swap affects only this method.

Configuration follows the DREAMS README defaults exactly: the 2-component PCA embedding, rescaled to
std(first column) = 1e-4 (openTSNE's initialization-scale convention), is used as BOTH the
initialization and the regularization embedding; ``reg_lambda=0.15`` (the published default);
perplexity 30 (openTSNE default, same value as this benchmark's t-SNE); Barnes--Hut gradients pinned
(``negative_gradient_method="bh"`` -- the "auto" choice for N < 10k, pinned so a config change can
never silently switch the estimator); single-threaded.

Registered DETERMINISTIC: with a fixed PCA initialization and single-threaded BH gradient descent
the optimizer has no remaining randomness -- three independent seeds produce byte-identical
embeddings (verified on clusters, SNR=1, D=768, N=1000). Same treatment as PCA/Isomap/toorPIA:
one run, no CI.

Upstream bug, patched at import time below: the ``tp`` branch HEAD (eeea6a6) leaks an ``"X"`` entry
into ``gradient_descent_params`` (tsne.py, ``prepare_initial``), which ``gradient_descent.__call__``
rejects with a TypeError -- every plain ``TSNE(...).fit(X)`` crashes. The value is dead code (never
consumed anywhere in the fork), so the wrapper drops the keyword before delegating. Keeping the
patch here, rather than in a modified fork, means the installed package is exactly the published
source.
"""
from __future__ import annotations

import numpy as np

from .base import SkipMethod, register


def _patched_gradient_descent():
    """Import the DREAMS fork, apply the dead-``X``-kwarg patch once, and return the TSNE class."""
    try:
        import openTSNE
        import openTSNE.tsne as _tsne_mod
    except ImportError as e:
        raise SkipMethod(f"openTSNE (DREAMS fork) not installed: {e}")
    import inspect

    if "regularization" not in inspect.signature(openTSNE.TSNE.__init__).parameters:
        raise SkipMethod(
            "installed openTSNE lacks DREAMS regularization -- install the fork: "
            'pip install "git+https://github.com/berenslab/DREAMS.git@tp"'
        )
    gd = _tsne_mod.gradient_descent
    if not getattr(gd, "_dreams_x_patch", False):
        orig = gd.__call__

        def _call_dropping_dead_X(self, *args, X=None, **kwargs):
            return orig(self, *args, **kwargs)

        gd.__call__ = _call_dropping_dead_X
        gd._dreams_x_patch = True
    return openTSNE.TSNE


def fit_dreams(X, seed, reg_lambda=0.15, perplexity=30):
    """Fit DREAMS and return the openTSNE ``TSNEEmbedding`` object (kept for ``transform``)."""
    from sklearn.decomposition import PCA

    TSNE = _patched_gradient_descent()
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = len(X)
    perp = float(min(perplexity, max(5, (n - 1) // 3)))  # small-N clamp, as in tsne.py
    pca2 = PCA(n_components=2, svd_solver="full").fit_transform(X)
    init = pca2 / pca2[:, 0].std() * 1e-4  # openTSNE init-scale convention (DREAMS README)
    t = TSNE(initialization=init, regularization=True, reg_lambda=float(reg_lambda),
             reg_embedding=init, perplexity=perp, negative_gradient_method="bh",
             random_state=int(seed), n_jobs=1, verbose=False)
    return t.fit(X)


@register("DREAMS", stochastic=False, reg_lambda=0.15, perplexity=30)
def embed_dreams(X, seed, device="cpu", context=None, reg_lambda=0.15, perplexity=30):
    return np.asarray(fit_dreams(X, seed, reg_lambda=reg_lambda, perplexity=perplexity),
                      dtype=np.float64)
