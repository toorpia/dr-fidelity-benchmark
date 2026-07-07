"""Membership-restricted fidelity metrics for the imbalanced two-population dataset.

Same standard statistics as everywhere else in this package, with the pair subset selected by the
endpoints' population membership (exactly as the outlier metrics subset by "involves an outlier"
and the distance bands subset by distance percentile):

* ``within_majority_shepard`` -- Spearman rho over the pairs whose BOTH endpoints are in the
  majority population (does the big population's internal structure survive?).
* ``within_minority_shepard`` -- same for the minority population (the practical question: does
  the SMALL population keep its internal structure, or is it collapsed because it has few
  points?).
* ``cross_population_shepard`` -- rho over the pairs with one endpoint in each population (is the
  relative placement of the two populations kept?).
* ``minority_shepard`` -- rho over ALL pairs involving at least one minority endpoint
  ([minority]-[minority] plus [minority]-[majority]) -- the direct analog of the outliers
  dataset's ``outlier_shepard`` (all pairs minus the majority-only pairs). One number that drops
  if EITHER the minority's internal structure or its placement is scrambled.

All computed exactly on all pairwise distances, vs-ambient (the repo's primary axis).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def _pair_masks(population: np.ndarray):
    pop = np.asarray(population)
    n = len(pop)
    iu = np.triu_indices(n, 1)
    both_a = (pop[iu[0]] == 0) & (pop[iu[1]] == 0)
    both_b = (pop[iu[0]] == 1) & (pop[iu[1]] == 1)
    cross = pop[iu[0]] != pop[iu[1]]
    return both_a, both_b, cross


def population_metrics(X_ambient: np.ndarray, Y: np.ndarray,
                       population: np.ndarray) -> dict:
    """All population-membership metrics for one embedding. Keys as documented in the module."""
    both_a, both_b, cross = _pair_masks(population)
    d_amb = pdist(np.ascontiguousarray(X_ambient, dtype=np.float64))
    d_2d = pdist(np.ascontiguousarray(Y, dtype=np.float64))
    minority_involving = both_b | cross
    row = {
        "within_majority_shepard__vs_ambient": float(spearmanr(d_amb[both_a], d_2d[both_a]).statistic),
        "within_minority_shepard__vs_ambient": float(spearmanr(d_amb[both_b], d_2d[both_b]).statistic),
        "cross_population_shepard__vs_ambient": float(spearmanr(d_amb[cross], d_2d[cross]).statistic),
        "minority_shepard__vs_ambient": float(
            spearmanr(d_amb[minority_involving], d_2d[minority_involving]).statistic),
    }
    return row
