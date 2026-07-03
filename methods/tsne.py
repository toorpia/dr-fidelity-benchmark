"""t-SNE -- neighbor-preserving, stochastic (sklearn).

Uses ``init='random'`` (seeded) so that independent seeds genuinely vary the embedding, which makes
the multi-seed run-to-run stability analysis meaningful. ``init='pca'`` would give a near-fixed
embedding and is documented as the lower-variance alternative in the README. Perplexity is clamped
below ``n`` for small-N smoke tests.
"""
from __future__ import annotations

from sklearn.manifold import TSNE

from .base import register


@register("t-SNE", stochastic=True, perplexity=30)
def embed_tsne(X, seed, device="cpu", context=None, perplexity=30):
    n = len(X)
    perp = float(min(perplexity, max(5, (n - 1) // 3)))
    tsne = TSNE(n_components=2, init="random", perplexity=perp, random_state=int(seed),
                n_jobs=1, method="barnes_hut")
    return tsne.fit_transform(X)
