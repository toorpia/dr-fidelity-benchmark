# DR Fidelity Benchmark — distance-preservation focus

A reproducible benchmark characterizing how faithfully dimensionality-reduction (DR) methods preserve
**distance structure** when mapping high-dimensional data to 2D, on synthetic datasets whose true
geometry is known.

> **New here?** Read *What this measures* → *Results at a glance* → *Key terms* below (≈2 min), then
> open **`REPORT.html`** for the results with figures and a step-by-step reading guide. The rest of
> this file is the detailed methodology.

## What this measures (plain-language)

A **dimensionality-reduction (DR)** method takes high-dimensional data (here, 768-dimensional vectors)
and draws each point as a dot on a 2-D plot. Information is inevitably lost. This benchmark measures
**how faithfully the 2-D distances between points match the original high-dimensional distances** —
separately for *near* pairs (fine, within-group structure) and *far* pairs (the overall layout).
Because the test data is synthetic with a **known true geometry**, every method is scored against the
ground truth instead of by eye.

We compare **seven** methods — PCA, Isomap, t-SNE, UMAP, PyMDE, PCC, and the closed-source
**toorPIA** — on **three** synthetic datasets (density, clusters, transition).

**Bottom line:** no single number tells the whole story, because these methods optimize different
things. **toorPIA preserves both near and global distance structure best overall** — it tops the
composite (near + global) ranking on all three datasets and has the **highest global Shepard ρ on two
of the three** (clusters and transition; PCC edges it out only on the density set). **PCC** posts a
high global correlation but **crushes dense clusters almost to points**; **t-SNE / UMAP** lead the
conventional neighbor metrics (and the near band) but distort global, large-scale distances.

## Results at a glance

- **No method wins everything** — the ranking flips depending on whether you care about *near*
  (within-cluster) or *global* (overall layout) structure.
- **toorPIA** — best all-rounder on distance fidelity: **tops the composite (near + global) ranking on all three datasets**,
  has the **highest global Shepard ρ on two of the three** (clusters and transition; 2nd on density),
  preserves the within-cluster scale (over-compression ≈0.4–0.6×), and shows ≈zero run-to-run wobble.
- **PCC** — a **high** global Shepard ρ (**highest only on density**; 3rd on clusters, 2nd on
  transition) **but it crushes dense clusters** (within-cluster over-compression ≈9–50×). The high
  global ρ *hides* this collapse — it shows up only in the over-compression metric and the Shepard
  density plots.
- **t-SNE / UMAP** — lead the conventional recall@k / trustworthiness / continuity *and* the near band
  (p5), but distort global, large-scale distances (low *full* Shepard ρ). Note recall@k is itself a
  **biased** local metric (see *Metrics*).
- **PyMDE** — good scale preservation (best on the density set) and mid-pack global ρ, but weak on the
  near band (p5).
- **Why a new local metric?** recall@k is structurally unfair to distance-preserving methods, so the
  primary near-neighbor score here is the **fixed-radius near-band Shepard ρ (p=5)**. See `REPORT.html`
  for the full tables and the two figure-backed explanations (why full ρ hides local structure, and
  why recall@k is biased).

## Key terms (quick glossary)

- **Dimensionality reduction (DR) / embedding** — mapping each high-D point to a 2-D coordinate; the
  2-D output is the *embedding*.
- **Fidelity** — how well the 2-D distances reproduce the high-D distances.
- **Ground truth — vs-truth / vs-ambient** — each dataset is a known clean geometry plus added noise.
  *vs-truth* scores against the clean generating distances; *vs-ambient* against the noisy distances
  the method actually saw.
- **Shepard ρ** — Spearman rank correlation between high-D and 2-D pairwise distances (1 = perfect
  distance ordering).
- **Distance band (p)** — the pairs whose high-D distance is in the lowest *p* % of all pairs.
  **p=5** = near-neighbor band; **p=100 (full)** = all pairs = the classic global number.
- **Stress** — value-based distance error (lower = better); complements the rank-based ρ by catching
  distorted distance *values*.
- **Over-compression ×** — how much a method shrinks the within-cluster scale vs the truth.
  ≈1× = preserved; ≫1× = clusters crushed toward points.
- **recall@k / trustworthiness / continuity** — neighbor-overlap metrics (variable radius, hard
  cutoff); kept only as a biased reference, not a fair near-distance metric.
- **SNR** — signal-to-noise ratio of the added noise; the canonical report uses SNR=1 (realistic).
- **Procrustes stability** — run-to-run wobble of a stochastic embedding after removing the
  rotation / scale / flip gauge; small = reproducible.

## How to read this repo

- **Just want the results** → open **`REPORT.html`** (build it with `python run/make_report.py` if it
  is stale); it has the ranking tables, figures, and a reading guide.
- **Want to reproduce** → follow *Quick start* below.
- **Want the methodology in depth** → *Synthetic datasets* and *Metrics* below.
- **Optional analyses** → `run/sweep.py` (dynamic-range curve), `run/robustness.py` (noise),
  `run/refigure.py` (rebuild figures without re-running methods).

The central methodological contribution is **how local fidelity is measured**. The standard local
metric (recall@k / kNN agreement) is biased toward neighbor-preserving methods (UMAP / t-SNE) and
structurally unfair to distance-preserving methods, because it uses variable-radius k-NN sets and a
hard inclusion threshold that over-penalizes near-ties in dense regions. We replace/augment it with
**distance-band-restricted Shepard ρ + stress**, reported as a near→far curve, which is fair to
distance-preserving methods and non-circular with respect to neighbor-preservation objectives.

## Motivation: citable characterization of a closed-source method

This benchmark is designed to be a reproducible, externally-citable characterization so that a
closed-source method (**toorPIA**) can be referenced in academic papers despite its internal
algorithm being non-public. We never inspect toorPIA's internals — we only characterize its
**input → output** behavior on data whose true structure is known, exactly as we do for the
open-source methods. toorPIA's output coordinates (not its algorithm) are committed to the repo so
the published results are fully reproducible offline.

## Quick start

```bash
pip install -r requirements.txt
# Smoke test (~1 min): two runs produce identical metrics
python run/benchmark.py --n 300 --dim 50 --seeds 3 --snr inf
# Full benchmark (all three datasets, all methods, R=20, D=768, N=1000; SNR=1)
python run/benchmark.py --dataset all --methods all --seeds 20 --dim 768 --n 1000 --snr 1
pytest tests/ -q
```

Outputs: per-run + aggregated metric tables and stability table in `results/`; figures in `figures/`;
saved embeddings in `results/embeddings/`; recorded library versions in `ENVIRONMENT.md`.

A consolidated **`REPORT.html`** dashboard is generated from the result tables with
`python run/make_report.py` (add `--embed` for a portable single file with base64-embedded figures;
open from the repo root). It focuses the discussion on **p=5 (near) and full (global)** Shepard ρ and
includes: per-config ranking tables with bootstrap CIs and **best (green) / worst (red)** cell
highlighting; a **composite ranking** (1st→5 … 5th→1 points awarded on the full-ρ order AND the p5-ρ
order, summed); and a **near-vs-global scatter** (x = Shepard ρ@p5, y = Shepard ρ full) showing which
methods reproduce both near and global structure (top-right is best).

## Repository layout

| path | contents |
|---|---|
| `synth/` | synthetic generators (true geometry / true distances known) + SNR noise |
| `methods/` | uniform `embed(X, seed, device, context) -> Y(N×2)` wrappers + registry |
| `metrics/` | exact fidelity metrics on all pairwise distances |
| `run/` | driver (`benchmark.py`), aggregation, figures, `make_report.py` (REPORT.html); optional `sweep.py` / `robustness.py` / `refigure.py` |
| `figures/`, `results/` | generated plots and metric tables |
| `external_embeddings/` | committed toorPIA coordinate caches + the external-injection path |
| `tests/` | small-N sanity tests |

## Synthetic datasets

All three datasets model structures that occur in real high-dimensional feature spaces. Each is built
in a low dimensional **latent** space (where the geometry is unambiguous) and mapped into the ambient
dimension `D` by a random **orthonormal** projection. An orthonormal map is an isometry, so the
ground-truth distance equals the clean ambient Euclidean distance; **SNR-controlled isotropic
Gaussian noise** is then added in all `D` dimensions to form the features `X` the methods embed.
Because truth and ambient differ only by that noise, metrics are reported BOTH **vs-truth** (clean
generating distances) and **vs-ambient** (noisy distances the method actually saw). N, D, noise (SNR),
and seed are CLI-configurable. The code supports an SNR sweep (`--snr inf 4 1`), but the canonical
report uses **SNR=1**: with D=768 the high-dimensional structure is most discriminative under noise
(at SNR=∞ even simple linear methods look good), so a single informative noise level is reported.

1. **Non-uniform density** (`synth/density.py`) — a uniform region + a tight Gaussian core + a sparse
   spherical shell, with deliberately different densities. Ground-truth distance = Euclidean in the
   generating space. Region labels and a continuous local-density value are saved for coloring.
   *Purpose:* quantify density distortion (do t-SNE / UMAP inflate the dense core?), demonstrate the
   recall@k bias, and show the distance-preservers' advantage on the near band.
2. **Distinct dense clusters at a tunable dynamic range** (`synth/clusters.py`) — `K=7` small dense
   Gaussian clusters placed on `K` mutually orthogonal axes (so the global geometry spans `K−1`
   dimensions — genuinely high-D, not a low-effective-dim trap), with a single knob `dynamic_range`
   = inter-cluster distance ÷ intra-cluster spacing. *Purpose:* test whether a method preserves the
   fine **within-cluster** structure (near band, p5) while also placing the clusters correctly
   (global); `run/sweep.py` traces both bands as `dynamic_range` varies. Models distinct dense
   sub-populations separated by large gaps in a high-D feature space.
3. **Continuous transition with known, genuinely high-dimensional global geometry**
   (`synth/transition.py`) — `K=7` "typical-state" clusters at KNOWN centroid positions placed on **K
   mutually orthogonal axes** (`scale·eᵢ`), connected by a continuous transition region parameterized
   by `t` that runs through every centroid and **closes into a loop** (state 0→1→…→K-1→0, so `t` is
   cyclic). The transition is HETEROGENEOUS — its perpendicular spread widens toward the middle of
   each bridge (spreading to both sides) — to mirror a "Mixed / H2" distribution, and the clusters are
   **dense**, with their inter-point spacing matched to the density dataset's tight core (median
   nearest-neighbor distance ≈ 0.29), so the near band is dominated by tight within-cluster structure. Ground truth is twofold: the known centroid geometry (global-distance scoring) and the
   continuum parameter `t` (transition-continuity scoring; saved for coloring). *Purpose:* test
   reproduction of structure from **near (dense clusters) to far (the loop geometry)** distances, and
   whether a method keeps the transition continuous or fragments it.

   **Why orthogonal axes (not a circle).** Centroids on a 2-D circle make the global geometry
   *intrinsically 2-dimensional*, so a linear method (PCA, 2 components) reproduces it trivially —
   an artifact, not a real high-dimensional test. Placing `K=7` centroids on orthogonal axes makes
   the centroid configuration span `K−1 = 6` affine dimensions (measured participation/effective
   dimension ≈ 6, with PCA's top-2 components capturing only ~38% of the variance), so the global
   structure genuinely requires more than two linear dimensions and PCA can no longer win by
   construction. (By contrast the density dataset already has effective dimension ≈ 10.)

## DR methods

All wrapped behind `embed(X, seed, device, context) -> Y (N×2)` (`methods/base.py`). Stochastic
methods run **R independent seeds** (default R=20); deterministic methods run once. Exact settings
(verified against the installed packages):

| method | library | hyperparameters | stochastic |
|---|---|---|---|
| **PCA** | scikit-learn | `n_components=2` | no |
| **Isomap** | scikit-learn | `n_neighbors=15`, `n_components=2` | no |
| **t-SNE** | scikit-learn | `init='random'`, `perplexity=30`, `random_state=seed`, `n_jobs=1` | yes |
| **UMAP** | umap-learn | `n_neighbors=15`, `min_dist=0.1`, `random_state=seed` | yes |
| **PyMDE** | pymde | `preserve_distances`, `Absolute` loss, `Standardized` constraint, CPU | yes |
| **PCC** | pccdr | `cluster=False, pearson=True, spearman=False, n_components=2, num_points=N` | yes |
| **toorPIA** | toorpia (remote API) | `fit_transform(..., random_seed=seed, vector_normalization=False)` | yes |

Notes:
- **t-SNE uses `init='random'`** (seeded) so independent seeds genuinely vary the embedding, making
  the run-to-run stability analysis meaningful. `init='pca'` would give a near-fixed embedding and is
  the lower-variance alternative.
- **toorPIA is called with `vector_normalization=False`**: toorPIA's internal `vector_normalization`
  rescales each input vector to unit (norm-1) length before embedding. We disable it so toorPIA
  embeds the SAME raw feature vectors that the other methods receive and that the fidelity metrics
  are computed on — otherwise toorPIA would be characterized on a different (unit-normalized)
  representation, breaking comparability.
- **toorPIA is placement knob-free** — unlike t-SNE (`perplexity`) or UMAP (`n_neighbors`), no
  user-tunable hyperparameter changes its embedding layout, so the same input yields a unique placement.
- **PCC is used label-free** (`cluster=False`): the cluster-supervision (MiCS / CrossEntropy) term is
  disabled entirely; the only objective is the **Pearson** correlation between high-D and 2-D
  distances to sampled reference points. `fit_transform(X, y)` requires `y`; with `cluster=False` it
  is unused, so we pass `np.zeros(N)`.
- **PyMDE** uses the `preserve_distances` objective (distance VALUES), with the `Standardized`
  constraint fixing the trivial scale/translation gauge; default `max_distances` (5e7) exceeds
  `N·(N−1)/2`, so all pairwise distances are used.

### toorPIA — external-injection / caching contract

toorPIA is closed-source and runs on a remote server. Its wrapper (`methods/toorpia_method.py`) is
strictly **cache-first**:

1. If the embedding exists at the on-disk path → load it, **no API call**.
2. Else if `TOORPIA_API_KEY` is set → call the API, then **write the cache**.
3. Else → skip toorPIA gracefully (the benchmark runs end-to-end without it).

On-disk path (`methods/external.py`):
```
external_embeddings/{dataset}/toorpia/{mode}/{tag}/seed{seed}.npy     mode=fit   tag=n{N}_d{D}_snr{snr}
```
The `{tag}` extends the original spec's simpler `{dataset}/{method}/seed{seed}.npy` so different SNR-sweep
points never collide. Committing these `.npy` files (plain coordinates, not the algorithm) lets public
users reproduce every figure/metric **without a toorPIA key**. The same path is the generic
external-injection point: drop a precomputed `Y` there for any method and it is used verbatim.

## Metrics

All metrics are computed **exactly on all pairwise distances** (`scipy.spatial.distance.pdist`),
independent of any method's internal reference-point approximation. High-D distance is the dataset's
defined distance (Euclidean here); 2-D distance is Euclidean.

### Global / multi-scale (the core contribution)

**Why bands.** Shepard ρ over *all* pairs (`p=100`) reflects **global-structure** reproduction — but in
high dimensions distances *concentrate*, so most pairs sit in a narrow far band and the full ρ is
dominated by far pairs; the accuracy of **near** distances is buried. We therefore restrict ρ to
cumulative distance **bands** defined on the GLOBAL high-D pairwise-distance distribution: the band at
cutoff `p` is the set of pairs whose high-D distance is in the lowest `p`% of all pairs. Sweeping
`p ∈ {5,10,20,30,50,75,100}%` gives a near→mid→far profile; **`p=5` (the globally-nearest 5% of pairs)
isolates near-neighbor descriptive power**, while `p=100` recovers the classic global number.

Crucially the band uses **one fixed absolute radius for the whole dataset** — every point is judged on
the same distance threshold. This is what makes it a *fair* near-neighbor metric, in contrast to the
variable-radius k-NN sets used by recall@k (see the bias note below).

- **Distance-band Shepard ρ** (`metrics/shepard.py`, PRIMARY) — Spearman rank correlation between
  high-D and 2-D distances **within each band**. A per-point variant (each point's lowest-`p`% pairs,
  a variable radius) is provided as a SECONDARY view to expose the variable-radius bias that recall@k
  suffers from; the global/absolute-band variant is primary.
- **Distance-band stress** (`metrics/stress.py`) — value-based normalized stress within the same
  bands: `stress = sqrt( Σ(a − α·b)² / Σa² )` with optimal scale `α = ⟨a,b⟩/⟨b,b⟩` (a = high-D, b =
  2-D distances). This catches methods that preserve distance ORDER but distort distance VALUES
  (e.g. scale-invariant objectives). `fidelity = max(0, 1 − stress)` is also reported.
- **Full Shepard ρ** and **full 1−stress** — the `p=100` specials, for the classic global numbers.

Each band metric is reported BOTH `__vs_ambient` and `__vs_truth`.

### Local fidelity, standard (for completeness, with documented bias)

- **recall@k** (kNN-set overlap / k) at `k ∈ {5,15,30}`, **trustworthiness**, **continuity**
  (`metrics/neighbors.py`; sklearn `trustworthiness`, continuity = its dual with spaces swapped).

> **Known bias — recall@k is not a fair neighborhood evaluation.** recall@k / trustworthiness /
> continuity take each point's own k nearest neighbors — a **per-point variable radius** — with a hard
> inclusion threshold. They over-penalize near-ties in dense regions and are structurally **favorable
> to k-NN-based methods (t-SNE / UMAP)**; they measure agreement with a k-NN neighborhood, not faithful
> reproduction of near distances. We keep them for comparability but treat them as a **biased
> reference**, not a correct near-neighbor metric. The fair near-neighbor metric is the **fixed-radius
> near-band Shepard ρ (p=5)** above. (Empirically, distance-preservers dominate the full / near band
> Shepard ρ and stress, while t-SNE / UMAP dominate recall@k — the band curve interpolates between.)
>
> A **per-point band-Shepard** variant (each point's lowest-`p`% pairs) is provided only for contrast:
> it re-introduces the variable radius, so the fixed-radius global band remains primary. On
> non-uniform-density data the global near band is weighted toward dense regions (where pairs are
> dense); the trade-off is point-uniformity vs fixed-radius fairness.

### Avoiding circularity

We never optimize a method on the same quantity used to score it.
- **PCC is run with the Pearson (value) loss while the primary global metric is Spearman (rank)
  Shepard ρ.** Optimizing Spearman would be "teaching to the test"; optimizing Pearson yet scoring
  well on the rank-based Shepard ρ is the honest, non-circular outcome. We keep **Pearson, not
  Spearman, on purpose**.
- No included method's training objective coincides with a scoring metric: t-SNE/UMAP optimize
  neighbor embeddings (KL / cross-entropy), PyMDE optimizes an `Absolute` distance loss (value, not
  the rank metric), PCA optimizes variance, Isomap optimizes geodesic MDS. PyMDE's value loss is
  closest in spirit to the (secondary) stress metric; this is flagged here explicitly.

## Aggregation, statistics, and stability

- Per (method, dataset, SNR, metric, band): **median + bootstrap 95% CI + std** over the R seeds
  (`run/aggregate.py`). When two methods' CIs overlap on a metric, **no strict winner is asserted**.
- **Run-to-run stability** (`metrics/stability.py`): the R embeddings are aligned with Procrustes
  (`scipy.spatial.procrustes`, removing the translation/rotation/reflection/scale gauge); we report
  per-point positional dispersion AND the std of the fidelity metrics across runs. This reveals where
  "coordinates wobble but structural fidelity is stable".

## Reproducibility / determinism

- Every stochastic step is seeded. The driver seeds **both** numpy (`np.random.seed`) **and** torch
  (`torch.manual_seed`) before each embed — necessary because PCC draws randomness from numpy
  (reference-point sampling, `np.random.choice`) AND torch (init), while PyMDE draws from torch.
- The driver sets `torch.set_num_threads(1)` and forces CPU (`CUDA_VISIBLE_DEVICES=""`) by default to
  avoid floating-point non-determinism from thread/GPU reduction order. Device is configurable.
- Data generation and noise realizations are seeded deterministically per (dataset, SNR). **Re-running
  with the same arguments reproduces identical numbers** (verified in `tests/` and by diffing two
  runs of `results/metrics_per_run.csv`).
- Library versions are recorded into `ENVIRONMENT.md` at runtime; pinned in `requirements.txt`.

## Deviations from the original specification (honest notes)

Verified against the installed packages; differences are reported rather than silently adapted:

1. **toorPIA exposes a seed.** The installed `toorpia==1.1.1` `fit_transform` accepts `random_seed`
   (default 42). We pass `random_seed=seed` per run (cleaner than treating runs as uncontrolled
   draws). The spec assumed no seed was available.
2. **PCC `fit_transform(X, y)` requires `y`** (positional). With `cluster=False` it is unused; we pass
   `np.zeros(N)`. The installed defaults are already `pearson=True, spearman=False`; the class default
   `cluster=True` is always overridden to `cluster=False`.
3. **PCC reference sampling is with replacement** (`np.random.choice`, `num_points`); for `N <
   num_points` it does not enumerate all points. We set `num_points = N` and report this — it is not
   claimed to be "exact".

## How to add a new method

Create `methods/yourmethod.py`:
```python
from .base import register

@register("YourMethod", stochastic=True, your_param=1.0)
def embed_yourmethod(X, seed, device="cpu", context=None, your_param=1.0):
    ...                      # return an (N, 2) array
```
then add it to the import line in `methods/__init__.py`. The driver, metrics, aggregation, and
figures pick it up automatically. To inject precomputed coordinates instead, drop them at the
external-embedding path (see the toorPIA contract above).

## How to cite

- **PCC**: J. Gildenblat and J. Pahnke, *Principal Component Correlation* (PCC), arXiv:2503.07609
  (2025); https://github.com/jacobgil/pcc
- **t-SNE**: L. van der Maaten and G. Hinton, *Visualizing Data using t-SNE*, JMLR 9 (2008).
- **UMAP**: L. McInnes, J. Healy, J. Melville, *UMAP*, arXiv:1802.03426 (2018).
- **PyMDE**: A. Agrawal, A. Ali, S. Boyd, *Minimum-Distortion Embedding*, FnT in ML (2021).
- **Isomap**: J. B. Tenenbaum, V. de Silva, J. C. Langford, *Science* 290 (2000).
- **Trustworthiness/Continuity**: Venna & Kaski (2001); **Shepard diagram**: Shepard (1962);
  **stress**: Kruskal (1964).

## Scope

This benchmark characterizes **distance/structure preservation on synthetic, known-structure data**.
It is **not** a claim about any method's superiority on real downstream tasks. Results quantify
fidelity profiles and their dispersion; rankings respect CI overlap.
