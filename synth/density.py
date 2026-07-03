"""Non-uniform density dataset.

A mixture of three regions with deliberately different densities, built in a latent space and mapped
isometrically into D dims:

* **uniform**  -- points uniform in a hypercube (medium density, fills the centre);
* **core**     -- a tight Gaussian blob at the origin (very high density, tiny volume);
* **shell**    -- points on a thin spherical shell at large radius (low density, large volume).

Ground-truth distance = Euclidean in the (clean) generating space. Region labels and a continuous
local-density value are saved for colouring.

Purpose: quantify density distortion (do t-SNE / UMAP inflate the dense core?), demonstrate the
recall@k bias, and show distance-preserving methods' advantage on the near distance band.
"""
from __future__ import annotations

import numpy as np

from ._common import local_log_density, project_to_D

REGION_NAMES = {0: "uniform", 1: "core", 2: "shell"}


def make_density(n: int = 1000, d: int = 768, seed: int = 0, latent_dim: int = 10,
                 core_frac: float = 0.45, uniform_frac: float = 0.40, shell_frac: float = 0.15,
                 core_sigma: float = 0.15, uniform_halfwidth: float = 3.0,
                 shell_radius: float = 6.0, shell_jitter: float = 0.10) -> dict:
    """Generate the non-uniform-density dataset.

    Returns a dict with ``clean`` (n x d), ``truth_coords`` (== clean; truth distance = Euclidean of
    clean), ``labels`` (region id), ``label_names``, ``color_value`` (log local density),
    ``color_name``, ``name`` and ``params``.
    """
    rng = np.random.default_rng(seed)
    fr = np.array([uniform_frac, core_frac, shell_frac], dtype=float)
    fr = fr / fr.sum()
    n_uniform, n_core = int(round(fr[0] * n)), int(round(fr[1] * n))
    n_shell = n - n_uniform - n_core

    # uniform region: medium density filling a hypercube centred at the origin
    uni = rng.uniform(-uniform_halfwidth, uniform_halfwidth, size=(n_uniform, latent_dim))
    # tight Gaussian core: very high density at the origin
    core = rng.normal(0.0, core_sigma, size=(n_core, latent_dim))
    # sparse shell: low density on a thin spherical shell at large radius
    dirs = rng.standard_normal((n_shell, latent_dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    radii = shell_radius + rng.normal(0.0, shell_jitter, size=(n_shell, 1))
    shell = dirs * radii

    latent = np.vstack([uni, core, shell])
    labels = np.concatenate([np.zeros(n_uniform, int), np.ones(n_core, int), np.full(n_shell, 2, int)])

    # shuffle so region order does not coincide with index order
    perm = rng.permutation(n)
    latent, labels = latent[perm], labels[perm]

    color_value = local_log_density(latent, k=10)
    clean = project_to_D(latent, d, rng)

    return {
        "name": "density",
        "clean": clean,
        "truth_coords": clean,            # isometric image of latent -> distances are the truth
        "labels": labels,
        "label_names": REGION_NAMES,
        "color_value": color_value,
        "color_name": "log local density",
        "params": dict(n=n, d=d, seed=seed, latent_dim=latent_dim, core_frac=core_frac,
                       uniform_frac=uniform_frac, shell_frac=shell_frac, core_sigma=core_sigma,
                       uniform_halfwidth=uniform_halfwidth, shell_radius=shell_radius,
                       shell_jitter=shell_jitter),
    }
