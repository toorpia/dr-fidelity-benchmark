"""toorPIA -- closed-source DR exposed via a remote API client (cache-first wrapper).

We never inspect toorPIA's internals; we only characterize its input->output behavior. The wrapper is
strictly cache-first, which makes the published benchmark reproducible offline WITHOUT a key:

1. If the cached/injected embedding exists at the on-disk contract path -> load it, NO API call.
2. Else if ``TOORPIA_API_KEY`` is set -> call ``toorPIA().basemap_embedding(...)``, write the cache.
3. Else -> raise ``SkipMethod`` so the benchmark runs end-to-end without toorPIA.

Verified installed API (toorpia 1.2.0): ``toorPIA(api_key=None)`` (reads ``TOORPIA_API_KEY`` /
``TOORPIA_API_URL`` from the environment); ``basemap_embedding(data, l2_normalization=...)`` accepts
an (n, d) ndarray directly and returns ``{"xyData": (n, 2) array, "mapNo": ..., "shareUrl": ...}``.

The embedding endpoint applies NO per-column normalization and NO centering; the only preprocessing
option is a per-row L2 normalization, which we pass as ``l2_normalization=False`` so toorPIA embeds
the very same raw feature vectors the other methods receive and the plain-Euclidean metrics measure
(preprocessing aligned with the evaluation basis). The endpoint exposes no random seed and is
deterministic up to server jitter (measured run-to-run mean |diff| ~1e-3 of the map scale), so the
method is registered ``stochastic=False`` and cached as ``seed0`` only.

History: earlier benchmark revisions used ``fit_transform(vector_normalization=False)``, whose
engine unconditionally normalizes every input column by 2 sigma before computing high-D distances --
so toorPIA alone saw a standardized geometry while the other methods and the ambient reference saw
the raw one. Those embeddings remain cached under ``toorpia/fit/`` for provenance; the current
``toorpia/embedding/`` tree is what the benchmark consumes.
"""
from __future__ import annotations

import os

import numpy as np

from .base import SkipMethod, register
from .external import config_tag, external_path, load_external, save_external

MODE = "embedding"


def _coords_from_result(res) -> np.ndarray:
    """Normalize the toorPIA return value to an (n, 2) float array."""
    if res is None:
        raise SkipMethod("toorPIA API returned no data (authentication / transport failure)")
    if isinstance(res, dict):
        for key in ("xyData", "xy", "coords"):
            if key in res:
                return np.asarray(res[key], dtype=np.float64)
        raise ValueError(f"toorPIA returned dict without coordinate key; keys={list(res)}")
    arr = np.asarray(res, dtype=np.float64)
    if arr.ndim != 2:
        raise SkipMethod(f"toorPIA API returned non-2D result of shape {arr.shape} "
                         "(likely an authentication / transport failure)")
    return arr


@register("toorPIA", stochastic=False, l2_normalization=False)
def embed_toorpia(X, seed, device="cpu", context=None, l2_normalization=False):
    if context is None or "root" not in context or "dataset" not in context:
        raise SkipMethod("toorPIA needs run context (root, dataset, tag); none provided")
    root, dataset = context["root"], context["dataset"]
    tag = context.get("tag") or config_tag(X.shape[0], X.shape[1], context.get("snr"))

    # 1. cache / injection
    cached = load_external(root, dataset, "toorpia", MODE, tag, seed)
    if cached is not None:
        if cached.shape != (X.shape[0], 2):
            raise ValueError(f"cached toorPIA embedding shape {cached.shape} != {(X.shape[0], 2)}")
        return cached

    # 2. API (only if a key is available)
    if not os.environ.get("TOORPIA_API_KEY"):
        raise SkipMethod(
            f"no cached toorPIA embedding at "
            f"{external_path(root, dataset, 'toorpia', MODE, tag, seed)} and TOORPIA_API_KEY is unset")

    import time

    from toorpia import toorPIA

    data = np.ascontiguousarray(X, dtype=np.float64)
    # Retry transient API failures (rate-limit / transport blips) with backoff before giving up, so a
    # single hiccup does not drop toorPIA for the rest of a long sweep.
    last_err = None
    for attempt in range(4):
        try:
            client = toorPIA()
            res = client.basemap_embedding(data, l2_normalization=l2_normalization)
            Y = _coords_from_result(res)
            if Y.shape != (X.shape[0], 2):
                raise SkipMethod(f"toorPIA returned shape {Y.shape}, expected {(X.shape[0], 2)}")
            save_external(Y, root, dataset, "toorpia", MODE, tag, seed)
            return Y
        except SkipMethod as e:
            last_err = e
            if attempt < 3:
                time.sleep(3.0 * (attempt + 1))  # 3s, 6s, 9s backoff
    raise SkipMethod(f"toorPIA failed after 4 attempts: {last_err}")
