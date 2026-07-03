"""Synthetic datasets with known ground-truth geometry/distance for the DR benchmark."""
from .noise import add_noise, add_noise_relative, intra_cluster_variance, signal_variance
from .registry import GENERATORS, list_datasets, make_dataset

__all__ = ["add_noise", "add_noise_relative", "intra_cluster_variance", "signal_variance",
           "GENERATORS", "list_datasets", "make_dataset"]
