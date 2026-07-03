"""Dataset registry: name -> generator callable.

Each generator returns the dict documented in ``density.py`` / ``transition.py``: at minimum
``clean`` (n x d clean coords), ``truth_coords`` (ground-truth coords whose Euclidean distance is the
truth), ``labels``, ``label_names``, ``color_value``, ``color_name``, ``name``, ``params``. The
driver adds SNR-controlled noise to ``clean`` to form the ambient features.
"""
from __future__ import annotations

from .clusters import make_clusters
from .density import make_density
from .outliers import make_outliers
from .populations import make_populations
from .transition import make_transition

GENERATORS = {
    "density": make_density,
    "transition": make_transition,
    "clusters": make_clusters,
    "outliers": make_outliers,
    "populations": make_populations,
}


def list_datasets():
    return list(GENERATORS)


def make_dataset(name: str, **kwargs) -> dict:
    if name not in GENERATORS:
        raise KeyError(f"unknown dataset '{name}'. available: {list_datasets()}")
    return GENERATORS[name](**kwargs)
