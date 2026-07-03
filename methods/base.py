"""Uniform DR-method interface + registry.

Every method is wrapped behind ``embed(X, seed, device, context) -> Y (n x 2)``. Methods self-register
via the ``@register`` decorator; importing ``methods`` (see ``__init__``) populates the registry.

``context`` is an optional dict carrying run metadata that a few methods need (notably toorPIA, which
keys its on-disk cache by dataset + config). Most methods ignore it.

Determinism: the DRIVER seeds BOTH numpy (``np.random.seed`` / ``default_rng``) and torch
(``torch.manual_seed``) and sets ``torch.set_num_threads(1)`` before each call, and forces CPU via
``CUDA_VISIBLE_DEVICES`` -- because some methods draw randomness from numpy (PCC reference-point
sampling) and some from torch (PCC / PyMDE init). Seeding only one is insufficient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


class SkipMethod(Exception):
    """Raised by a wrapper when a method cannot run but should be skipped gracefully.

    Example: toorPIA with no cached embedding and no ``TOORPIA_API_KEY``.
    """


@dataclass
class Method:
    name: str
    stochastic: bool
    fn: Callable
    params: dict = field(default_factory=dict)

    def embed(self, X: np.ndarray, seed: int, device: str = "cpu",
              context: Optional[dict] = None) -> np.ndarray:
        Y = self.fn(X, seed, device=device, context=context, **self.params)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.shape != (X.shape[0], 2):
            raise ValueError(f"{self.name} returned shape {Y.shape}, expected {(X.shape[0], 2)}")
        return Y


REGISTRY: dict[str, Method] = {}


def register(name: str, stochastic: bool, **params):
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = Method(name=name, stochastic=stochastic, fn=fn, params=dict(params))
        return fn
    return deco


def get_method(name: str) -> Method:
    if name not in REGISTRY:
        raise KeyError(f"unknown method '{name}'. available: {list_methods()}")
    return REGISTRY[name]


def list_methods() -> list[str]:
    return list(REGISTRY)
