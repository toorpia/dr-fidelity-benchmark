"""Uniform DR-method wrappers. Importing this package self-registers every method.

Default method ordering (used when ``--methods all``) puts deterministic baselines first, then
distance-preserving methods, then neighbor-preserving methods, then toorPIA.
"""
from __future__ import annotations

# import side effects populate REGISTRY
from . import pca, isomap, pymde_method, pcc_method, tsne, umap_method, toorpia_method  # noqa: F401
from .base import REGISTRY, Method, SkipMethod, get_method, list_methods, register

DEFAULT_ORDER = ["PCA", "Isomap", "PyMDE", "PCC", "t-SNE", "UMAP", "toorPIA"]


def default_methods() -> list[str]:
    """All registered methods in the canonical display order."""
    return [m for m in DEFAULT_ORDER if m in REGISTRY] + [m for m in REGISTRY if m not in DEFAULT_ORDER]


__all__ = ["REGISTRY", "Method", "SkipMethod", "get_method", "list_methods", "register",
           "DEFAULT_ORDER", "default_methods"]
