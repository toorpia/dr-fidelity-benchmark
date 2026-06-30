"""Exact distance-fidelity metrics for the DR benchmark.

Primary contribution: distance-band-restricted Shepard rho (rank) + stress (value), reported as a
near->far curve over global percentile cutoffs. Conventional neighbor metrics (recall@k,
trustworthiness, continuity) are included for comparability but documented as biased.
"""
from .compute import compute_all
from .distances import DEFAULT_CUTOFFS, condensed_distances, square_distances
from .neighbors import DEFAULT_KS, recall_at_k, trustworthiness_continuity
from .shepard import band_shepard, per_point_band_shepard
from .stress import band_stress, normalized_stress, stress_to_fidelity
from .stability import (align_to_reference, mean_pairwise_disparity, metric_variance,
                        position_dispersion)

__all__ = [
    "compute_all", "DEFAULT_CUTOFFS", "condensed_distances", "square_distances",
    "DEFAULT_KS", "recall_at_k", "trustworthiness_continuity",
    "band_shepard", "per_point_band_shepard",
    "band_stress", "normalized_stress", "stress_to_fidelity",
    "align_to_reference", "mean_pairwise_disparity", "metric_variance", "position_dispersion",
]
