"""On-disk contract for externally-injected / cached embeddings.

Cache/injection path::

    {root}/external_embeddings/{dataset}/{method}/{mode}/{tag}/seed{seed}.npy

``mode`` distinguishes the workflow (``fit`` vs ``addplot`` for toorPIA); ``tag`` encodes the data
config (``n{N}_d{D}_snr{snr}``) so different sweep points never collide -- this extends the simpler
``{dataset}/{method}/seed{seed}.npy`` contract in the original spec to support the noise sweep. A precomputed
file dropped at this path is loaded verbatim (the injection path); methods that compute embeddings via
an API also WRITE here (the cache path), so public users reproduce everything offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def config_tag(n: int, d: int, snr) -> str:
    snr_str = "inf" if (snr is None or not np.isfinite(snr)) else f"{float(snr):g}"
    return f"n{n}_d{d}_snr{snr_str}"


def external_path(root, dataset: str, method: str, mode: str, tag: str, seed: int) -> Path:
    return (Path(root) / "external_embeddings" / dataset / method / mode / tag / f"seed{seed}.npy")


def load_external(root, dataset: str, method: str, mode: str, tag: str, seed: int) -> Optional[np.ndarray]:
    p = external_path(root, dataset, method, mode, tag, seed)
    if p.exists():
        return np.asarray(np.load(p), dtype=np.float64)
    return None


def save_external(Y: np.ndarray, root, dataset: str, method: str, mode: str, tag: str, seed: int) -> Path:
    p = external_path(root, dataset, method, mode, tag, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, np.asarray(Y, dtype=np.float64))
    return p
