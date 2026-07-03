"""toorPIA -- closed-source DR exposed via a remote API client (cache-first wrapper).

We never inspect toorPIA's internals; we only characterize its input->output behavior. The wrapper is
strictly cache-first, which makes the published benchmark reproducible offline WITHOUT a key:

1. If the cached/injected embedding exists at the on-disk contract path -> load it, NO API call.
2. Else if ``TOORPIA_API_KEY`` is set -> call ``toorPIA().fit_transform(...)``, write the cache.
3. Else -> raise ``SkipMethod`` so the benchmark runs end-to-end without toorPIA.

Verified installed API (toorpia 1.1.1): ``toorPIA(api_key=None)`` (reads ``TOORPIA_API_KEY`` /
``TOORPIA_API_URL`` from the environment); ``fit_transform(data, label=None, random_seed=42, ...)``
returns an (n, 2) array. NOTE vs the original spec: this version DOES expose ``random_seed`` -- we pass
``random_seed=seed`` per run (documented deviation).
"""
from __future__ import annotations

import os

import numpy as np

from .base import SkipMethod, register
from .external import config_tag, external_path, load_external, save_external

MODE = "fit"


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


@register("toorPIA", stochastic=True, vector_normalization=False)
def embed_toorpia(X, seed, device="cpu", context=None, vector_normalization=False):
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

    import pandas as pd
    from toorpia import toorPIA

    df = pd.DataFrame(np.ascontiguousarray(X, dtype=np.float64),
                      columns=[f"d{i}" for i in range(X.shape[1])])
    # Retry transient API failures (rate-limit / transport blips) with backoff before giving up, so a
    # single hiccup does not drop toorPIA for the rest of a long sweep.
    last_err = None
    for attempt in range(4):
        try:
            client = toorPIA()
            # vector_normalization=False disables toorPIA's internal unit-norm (norm-1) rescaling so it
            # embeds the same raw feature vectors the other methods receive and the metrics measure.
            res = client.fit_transform(df, label=None, random_seed=int(seed),
                                       vector_normalization=vector_normalization)
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
