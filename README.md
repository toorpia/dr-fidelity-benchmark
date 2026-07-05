# DR Fidelity Benchmark — distance-preservation focus

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21189374.svg)](https://doi.org/10.5281/zenodo.21189374)

A reproducible benchmark characterizing how faithfully dimensionality-reduction (DR) methods preserve
**distance structure** when mapping high-dimensional data to 2D, on synthetic datasets whose true
geometry is known.

> **New here?** Read *What this measures* → *Results at a glance* → *Key terms* below (≈2 min), then
> open the **published report** at <https://toorpia.github.io/dr-fidelity-benchmark/> for the
> results with figures and a step-by-step reading guide (the same content as **`REPORT.html`** in
> this repo; **`REPORT.md`** is the same content rendered as Markdown, readable directly on
> GitHub). The rest of this file is the detailed methodology.

## What this measures (plain-language)

A **dimensionality-reduction (DR)** method takes high-dimensional data (here, 768-dimensional vectors)
and draws each point as a dot on a 2-D plot. Information is inevitably lost. This benchmark measures
**how faithfully the 2-D distances between points match the original high-dimensional distances** —
separately for *near* pairs (fine, within-group structure) and *far* pairs (the overall layout).
Because the test data is synthetic with a **known true geometry**, every method is scored against the
ground truth instead of by eye.

We compare **seven** methods — PCA, Isomap, t-SNE, UMAP, PyMDE, PCC, and the closed-source
**toorPIA** — on **five** synthetic datasets (density, clusters, transition, outliers, imbalanced
populations). All five share a single, deliberately noise-friendly noise model; see *Noise
regimes* below for a direct stress test of the opposite regime.

**Bottom line:** no single number tells the whole story, because these methods optimize different
things. **toorPIA preserves both near and global distance structure best overall** — it tops the
composite (near + global) ranking on all three datasets and has the **highest global Shepard ρ on two
of the three** (clusters and transition; PCC edges it out only on the density set). **PCC** posts a
high global correlation but **crushes dense clusters almost to points**; **t-SNE / UMAP** lead the
conventional neighbor metrics (and the near band) but distort global, large-scale distances. On the
fourth dataset (paired same-kind anomalies, off-subspace), the anomaly-pair Shepard ρ — the
standard ρ restricted to the outlier-involving pairs, scored into the ranking — puts **toorPIA
clearly first (0.254, near-identical across seeds)** with t-SNE / UMAP / PyMDE at the bottom
(their anomalies sit at or inside bulk clusters, or same-kind pairs are fused/torn); in the
out-of-sample (addplot) test — monitoring on an anomaly-free basemap — **toorPIA is the only
method that plots a never-seen anomaly outside the normal region at all, and its direction points
back to the anomaly's source cluster**; t-SNE / PyMDE / PCC cannot perform the add-data operation
at all. On the fifth dataset (imbalanced two populations with internal structure), only
**toorPIA keeps a 5-% minority population both recognizable as a separate group and internally
trustworthy**; PCA and PCC lose the minority's internal structure — *silently*, their global ρ
stays high — and t-SNE / UMAP never render the two-population separation at all.

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
- **Outliers dataset (R=3, SNR=1)** — read the **Shepard density figure** (high-D vs 2-D pairwise
  distance) and the **star gallery** first; the quantitative anchor is the same standard statistic
  restricted to the pairs that matter here: the **anomaly-pair Shepard ρ** (`outlier ρ` — Spearman
  over exactly the pairs involving a ground-truth outlier), which feeds the composite ranking as a
  third 5..1 column so that **no local reading can outrank the baseline**: a method that plots its
  anomalies among other clusters, fuses same-kind pairs, or tears them apart scrambles precisely
  these pairs. Result: **toorPIA 0.254 [0.253, 0.254]** — clearly first and near-identical across
  seeds — with PCA 0.156, PCC 0.154 [0.114, 0.164], Isomap 0.153, then **t-SNE 0.083
  [0.047, 0.136]** and **UMAP 0.062 [−0.005, 0.111]** (their anomalies sit at or inside
  bulk clusters — visible in the gallery and as off-trend blocks in the Shepard density panel) and
  **PyMDE 0.033** last (same-kind pair torn to opposite ends, different kinds fused). toorPIA also
  holds the highest global ρ (0.755) and the only clean monotone Shepard band; PCC's ρ 0.707
  coexists with visibly wrong anomaly blocks — few pairs, so the all-pair statistic barely moves,
  which is exactly why the restricted ρ is scored. (Descriptive, not scored: same-kind pair angles
  — toorPIA ≤ 10°, PCC ≈ 86°, PCA/Isomap/PyMDE ≥ 150°; t-SNE's ≈ 0° is a fusion artifact, one of
  the reasons a local reading must not stand alone.) The variance-truncation prediction for PCA,
  made before the run, held. See the outcome note below.
- **addplot / out-of-sample (the monitoring operation itself)** — the basemap is fitted on
  normal data only, and **cluster-anchored anomalies** arrive afterwards, one at a time: each
  shares a normal cluster's profile in the measured features and deviates 3 Rg along new
  dimensions the normal data never varies in (a near-duplicate pair per cluster) — the realistic
  shape of a fault: a known operating state plus a never-seen effect. Two questions in order:
  **detection** (does the anomaly land visibly outside the normal region?) and **attribution**
  (does its direction on the map point back to its source cluster?). The high-D features resolve
  attribution 10/10, so a faithful map can too. Result (R=3, SNR=1): **only toorPIA answers both
  — every anomaly lands far outside the normal region (5.2–6.4× the bulk radius, identical
  across seeds) AND its direction points at the source cluster (attribution 10/10, angle to own
  cluster 0.6°, pair angle 0.6°)**. For PCA / Isomap / UMAP the anomalies land **inside or at
  the normal clusters** (median radius ratio 0.96–1.34, minima down to 0.26) — the anomaly is
  drawn as just another normal point of its source cluster, so the "attribution" (0.8–1.0)
  carries no alarm: detection silently fails. Added normal controls land correctly for every
  operable method (ratio 0.87–1.01).
  **t-SNE, PyMDE, and PCC cannot perform the operation at all** (no out-of-sample transform —
  adding data means re-fitting, which re-arranges the map).
- **Populations dataset — minority-structure preservation under data imbalance (R=3, SNR=1,
  canonical setting 95% vs 5%)** — the practical task is **extracting an unknown minority
  population from the map**, which needs two readings positive **at once**: the minority drawn as
  a recognizable separate group (cross-population ρ) AND a trustworthy internal structure
  (minority-internal ρ); failing either disqualifies the task. At 5%: **toorPIA is the only
  method with both readings clearly positive** (minority-internal 0.275 / cross-population 0.497,
  plus the best majority-internal 0.745 and the best full ρ 0.809, zero run-to-run wobble), at
  the cost of the strongest population-level over-compression (3.5×: the group gap dominates and
  the internal layouts are drawn relatively small — read that column together with
  cross-population ρ). **PCA** places the minority correctly (cross ρ 0.683, the best) but its
  internal structure is gone (minority-internal −0.025 — a correctly placed featureless blob).
  **t-SNE / UMAP** are the mirror image: the best minority internals (0.658 / 0.451) but
  cross-population ρ ≈ 0 — the minority's clusters are scattered among the majority's, so the
  two-population structure is not rendered at all. **PCC** fails both readings (minority-internal
  0.023, cross-population 0.198) — and the failures are **silent**: PCC still posts the 2nd-best
  full ρ (**0.761**), so nothing in the global metric warns that the minority was destroyed (PCA
  likewise: full 0.608 vs minority-internal −0.025). This is consistent with PCC's
  reference-point sampling: pairs internal to the minority carry a share of its loss that shrinks
  with the *square* of the minority fraction (0.25 % of terms at 5%), so minority points end up
  placed almost entirely by their relations to majority reference points.
- **Minority-fraction sweep (50% → 5%)** — how the canonical 5% failures develop. **PCC** is
  actually the best cross-population renderer when balanced (cross ρ 0.821 at 50/50,
  minority-internal 0.539) and collapses monotonically as the minority share shrinks (cross
  0.821 → 0.423 → 0.262 → 0.198; minority-internal 0.539 → 0.492 → 0.246 → 0.023). **PCA**'s
  minority internals are already dead at 25% (0.029). **t-SNE / UMAP** keep the minority
  internals at every fraction but never render the separation (cross ≈ 0 throughout). **toorPIA**
  keeps both readings clearly positive at every fraction (minority-internal 0.698 → 0.275;
  cross-population 0.316 → 0.497 — the separation reading actually strengthens as the imbalance
  grows), with the trade-off growing in the same direction (population over-compression
  2.2× → 3.5× as the fraction shrinks).
- **Dimension sweep (noise-dims supplement)** — with 3 signal columns plus D−3 pure-noise columns
  (effective SNR = 3/(D−3), zero signal redundancy — the opposite regime of the five datasets),
  every method loses the three true clusters between D=40 and D=200 (kNN label accuracy falls to
  chance ≈1/3); **toorPIA alone keeps them visible through D=768** (accuracy 0.98 at effective
  SNR ≈0.004) and first degrades at a toorPIA-only D=2000 probe (kNN accuracy 0.56–0.88 depending
  on the noise realization — the onset of its breaking regime, still above chance in every draw).
  PyMDE from D≈40 draws crisp but fully label-mixed clumps — plausible-looking false
  structure. See *Noise regimes* below and `figures/noise_dims/`.
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
- **Outlier ρ (anomaly-pair Shepard ρ)** — the standard Shepard ρ restricted to the pairs where at
  least one endpoint is a ground-truth outlier (subset by endpoint membership, exactly as the bands
  subset by distance percentile). Quantifies the outlier-related blocks of the Shepard density
  figure; feeds the outliers dataset's third ranking column.
- **Majority-internal / minority-internal / cross-population ρ** — the same membership-restricted
  Shepard ρ on the populations dataset: both endpoints in the majority, both in the minority, or
  one in each. Minority-internal ρ answers "is the small population's internal structure
  trustworthy?"; cross-population ρ answers "is the minority drawn as a recognizable separate
  group?" — extracting an unknown minority needs both.
- **SNR** — signal-to-noise ratio of the added noise; the canonical report uses SNR=1 (realistic).
- **Procrustes stability** — run-to-run wobble of a stochastic embedding after removing the
  rotation / scale / flip gauge; small = reproducible.

## How to read this repo

- **Just want the results** → open **`REPORT.html`** (build it with `python run/make_report.py` if it
  is stale); it has the ranking tables, figures, and a reading guide.
- **Want to reproduce** → follow *Quick start* below.
- **Want the methodology in depth** → *Synthetic datasets* and *Metrics* below.
- **Optional analyses** → `run/sweep.py` (dynamic-range curve and outlier-factor curve),
  `run/robustness.py` (noise), `run/dimsweep.py` (noise-dims dimension sweep — see *Noise
  regimes*), `run/refigure.py` (rebuild figures without re-running methods).

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

## Conflict of interest & trust measures

**Disclosure: this repository is maintained by the vendor of toorPIA.** The benchmark is therefore
designed so that its claims do not rest on the maintainers' judgment:

- **Metrics are computed independently of every method** — exactly, on all pairwise distances
  (`scipy.spatial.distance.pdist`), never through any method's internal reference-point or
  neighbor-graph approximation (toorPIA's included; we only see its output coordinates).
- **Hypotheses are documented before the results** — the constraint-density hypothesis for the
  outliers dataset (below) is committed to git before the corresponding result tables, and the
  results are reported however they come out, including when they contradict the hypothesis or are
  unfavorable to toorPIA.
- **Every number is third-party recomputable offline** — toorPIA's output coordinates (the only
  ones a third party could not regenerate, since the method is a remote API) are committed under
  `external_embeddings/`; every open-source method's embedding is byte-reproducible from the
  deterministic driver (same CLI args → identical numbers, enforced by tests), and all per-run
  metric tables are committed.
- **No strict winner is asserted when CIs overlap**, and the biased-but-standard reference metrics
  (recall@k family) are always reported alongside the primary ones.

## Quick start

```bash
pip install -r requirements.txt
# Smoke test (~1 min): two runs produce identical metrics
python run/benchmark.py --n 300 --dim 50 --seeds 3 --snr inf --out /tmp/dr_smoke --figdir /tmp/dr_smoke/figs
# Full benchmark (all five datasets, all methods, R=3, D=768, N=1000; SNR=1)
python run/benchmark.py --dataset all --methods all --seeds 3 --dim 768 --n 1000 --snr 1
# Or a single dataset — results MERGE into results/ (only the re-run dataset+SNR rows are replaced)
python run/benchmark.py --dataset outliers --methods all --seeds 3 --dim 768 --n 1000 --snr 1
# Sweeps: dynamic range (clusters) and outlier factor (outliers)
python run/sweep.py --dynamic-range 2 5 10 20 50 --methods all --seeds 3
python run/sweep.py --sweep outlier_factor --outlier-factor 1.5 2 3 5 8 --methods all --seeds 3
python run/sweep.py --sweep minority_frac --minority-frac 0.5 0.25 0.1 0.05 --methods all --seeds 3
# Noise-dims dimension sweep (curse-of-dimensionality supplement; see "Noise regimes")
python run/dimsweep.py --dims 6 10 20 40 80 100 200 400 768 --methods all --seeds 3 --n 1000
python run/dimsweep.py --dims 2000 --methods toorPIA --seeds 3 --n 1000   # extreme-D extension
pytest tests/ -q
```

Outputs: per-run + aggregated metric tables and stability table in `results/`; figures in `figures/`;
saved embeddings in `results/embeddings/`; recorded library versions in `ENVIRONMENT.md`.
`run/benchmark.py` merges into the existing `results/` tables (replacing only the (dataset, SNR)
configs being re-run), so partial runs never clobber the committed results of other datasets — use
`--out` for scratch runs you do not want merged.

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
| `synth/` | synthetic generators (true geometry / true distances known) + SNR noise; `noise_dims.py` (unregistered noise-dims probe) |
| `methods/` | uniform `embed(X, seed, device, context) -> Y(N×2)` wrappers + registry |
| `metrics/` | exact fidelity metrics on all pairwise distances; `label_separation.py` (noise-dims readouts) |
| `run/` | driver (`benchmark.py`), aggregation, figures, `make_report.py` (REPORT.html); optional `sweep.py` / `robustness.py` / `dimsweep.py` / `refigure.py` |
| `figures/`, `results/` | generated plots and metric tables |
| `external_embeddings/` | committed toorPIA coordinate caches + the external-injection path |
| `tests/` | small-N sanity tests |

## Synthetic datasets

All five datasets model structures that occur in real high-dimensional feature spaces. Each is built
in a low dimensional **latent** space (where the geometry is unambiguous) and mapped into the ambient
dimension `D` by a random **orthonormal** projection. An orthonormal map is an isometry, so the
ground-truth distance equals the clean ambient Euclidean distance; **SNR-controlled isotropic
Gaussian noise** is then added in all `D` dimensions to form the features `X` the methods embed.
Because truth and ambient differ only by that noise, metrics are reported BOTH **vs-truth** (clean
generating distances) and **vs-ambient** (noisy distances the method actually saw). N, D, noise (SNR),
and seed are CLI-configurable. The code supports an SNR sweep (`--snr inf 4 1`), but the canonical
report uses **SNR=1**: with D=768 the high-dimensional structure is most discriminative under noise
(at SNR=∞ even simple linear methods look good), so a single informative noise level is reported.
This design is deliberately **redundancy-rich**: the orthonormal projection spreads each latent
coordinate over all D columns, so isotropic noise self-averages and no curse of dimensionality
operates. See *Noise regimes* below for the direct test of the opposite, redundancy-free regime.

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

4. **Bulk clusters + injected off-subspace outliers at a controlled separation**
   (`synth/outliers.py`) — a bulk of `K=5` dense Gaussian clusters (the `clusters` recipe at a
   moderate dynamic range) spanning the first `latent_dim` latent dimensions, plus **3 anomalous
   directions × 2 near-duplicate outliers each**: direction `j` has its **own dedicated extra
   latent axis — orthogonal to the entire subspace the bulk spans** ("off-subspace"), and its two
   members sit on that same axis at radii `outlier_factor × Rg` and `(outlier_factor + 0.1) × Rg`
   (same direction exactly, offset by `0.1 Rg` ≈ the bulk's own median NN spacing — two
   near-duplicate anomalies of the same kind, for checking whether the map keeps them adjacent and
   co-directional). `Rg` = the bulk's **radius of gyration** (RMS distance of the bulk points from
   their centroid — the compact scale of the central mass, so the default factor 3 reads "three
   bulk radii out"; an earlier revision used the p99 pairwise distance ≈ the diameter, which placed
   outliers so far — ~5.5 Rg default, ~15 Rg at the sweep top — that the map's relative relation
   between the outliers and the central mass was lost). *Why off-subspace:* a sample from a different
   acquisition condition varies along feature directions the bulk does not span; an in-subspace
   far point would also be the easy case for variance-based linear projections (its direction
   carries large variance and lands in the top principal components), whereas an off-subspace
   single point carries only `(outlier_factor·Rg)²/N` variance, which a variance-truncating
   map may legally drop — separating "constrains distances" from "constrains projected variance".
   *Motivation (anonymized real case):* in a set of feature vectors from material-microscopy
   images, a single image acquired under a different imaging condition should appear as one clearly
   isolated point in the 2-D map — whether it actually does depends on the DR method. *Purpose:*
   test whether a **single far-away point keeps its separation margin** in 2-D — exactly the single-point property that many-pair statistics such as the band-Shepard ρ
   cannot resolve. *Knob:*
   `outlier_factor` is the single sweep parameter (peer of `dynamic_range`), default 3, swept over
   {1.5, 2, 3, 5, 8} by `run/sweep.py --sweep outlier_factor`. *Design decision:* all m outliers
   share ONE factor per dataset (so they are i.i.d. replicates within a run and the dose-response
   comes from the sweep); the alternative — mixing several factors in one dataset — was rejected
   because it would make the per-outlier values non-exchangeable across seeds. Ground truth saved:
   outlier indices, per-outlier factor, and labels (outliers are excluded from the within-cluster
   scale metric).

5. **Imbalanced two populations, each with internal cluster structure** (`synth/populations.py`;
   the analysis theme: **minority-structure preservation under data imbalance**) —
   a **majority** population (the standard 5-orthogonal-axes cluster recipe) and a much smaller
   **minority** population with the *same* 5-cluster geometry (same center scale, measured on the
   majority so that only the point count differs), built in a disjoint block of latent dimensions
   and offset along one shared extra dimension so that **every cross-population cluster-center
   distance is exactly `group_range` × the universal within-population center distance** — a
   strict two-level hierarchy with two knobs: `group_range` (default 2) and `minority_frac`
   (default 0.05, i.e. 95% vs 5% — the canonical setting of this dataset; swept over {0.5, 0.25, 0.1, 0.05} by `run/sweep.py --sweep minority_frac`, at
   the report's SNR=1). *Motivation — a ubiquitous real situation, not an edge case:* measurement
   datasets routinely mix a dominant population with a much smaller second population that
   occupies a different region of the measurement space yet has internal structure of its own —
   normal production runs vs a rarely-used operating mode (start-up, an alternative recipe), the
   main manufacturing line vs a small pilot-lot series, data taken before vs after an instrument
   replacement or recalibration, seasonal or day/night operating regimes, or a large healthy
   cohort vs a small patient group that itself splits into subtypes. *Purpose:* in a real project
   the composition of the data is **unknown in advance** — nobody tells the analyst that a minority
   population exists, let alone where it sits. Minority shares of a few percent are common, and the
   minority is very often the actual object of the analysis: anomaly analysis, finding the positive
   cases in medical data, identifying transient operating states of a manufacturing process.
   Extracting an unknown minority from a 2-D map therefore requires two things **at once**: the
   **minority must be drawn as a recognizable separate group** (cross-population ρ) and its
   **internal structure must be trustworthy** (minority-internal ρ). A method that fails either one
   cannot be used for this task — and since the task is ubiquitous, such a method's applicability
   to real data analysis is severely limited. The imbalance is exactly what makes the second
   question hard, and the dataset connects continuously to the outliers dataset (a small minority
   *without* internal structure is the anomaly case).

## Noise regimes: what the isotropic design assumes — and a direct stress test

### Why the five datasets are curse-of-dimensionality-free by design

The five datasets share one deliberately noise-friendly design. The random orthonormal projection
gives every latent factor dense, balanced loadings across **all** D=768 ambient columns, so the
ambient data is in effect **768 noisy re-measurements of the same ~10 latent quantities**. In every
pairwise distance the signal contributions add coherently while the isotropic noise contributions
cancel (the signal-noise cross terms average to ≈0, and the pure-noise term is a near-constant
offset for every pair, fluctuating only ~1/√D). The driver also anchors the noise **power** to the
signal power (`SNR`), independent of D. Two consequences: the ambient dimension is **nominal** (the
between/within distance contrast of `clusters` at SNR=1 is ≈1.46 at D=8 and still ≈1.46 at D=768),
and **no curse of dimensionality operates — by construction**. This is the noise-*friendly*
extreme: real feature tables also contain uninformative columns and correlated noise components
that do **not** self-average.

### Supplement: the noise-dims dimension sweep (`run/dimsweep.py`)

The **noise-dims sweep** (runner `run/dimsweep.py`; results files `dimsweep_*`) probes the
opposite, redundancy-free extreme directly. *Design* (`synth/noise_dims.py`, ported from a teaching
notebook): three tight Gaussian clusters live in **3 signal columns** (`make_blobs`, std 0.005,
centers at the unit vectors) and every additional column is pure unit-variance noise; all columns
are then z-scored. *Mechanism:* each added dimension adds unit noise power at fixed signal power,
so the effective SNR is **3/(D−3)** — from ≈1 at D=6 to ≈0.004 at D=768; total dimensionality is
the noise knob. *Contract honesty:* unlike the five registry datasets this generator is
deliberately **not isometric** (the ground truth is the 3 signal columns, not the ambient
distances) and is deliberately **not in the synth registry** — it is a probe, not a sixth dataset.
*Readouts:* leave-one-out **kNN label accuracy** (k=10, chance 1/3) and 2-D silhouette
(`metrics/label_separation.py`) answer the operational question "are the three true clusters still
visible in the map?"; distance fidelity is reported per the benchmark's primary axis (**vs-ambient**
band Shepard ρ) — note the ambient distances themselves become noise-dominated as D grows, so every
method's full ρ declines *by construction* and must not be read as method failure. *Sweep spec:*
D = 6…768 for all seven methods (the same ambient dimension as the main benchmark, now with zero
redundancy) plus a **toorPIA-only extension at D=2000**; n=1000 (matching both the main benchmark
and the source notebook), R=3 seeds.

**Result (D = 6–768 all methods + D=2000 toorPIA-only, R = 3, n = 1000).** Every method holds the
three clusters to D≈20–40 (kNN accuracy ≥0.96, except PyMDE which collapses first at D=40: 0.56).
The collapse then proceeds in order: PCC and Isomap fade from D=80 (0.72 / 0.76 → 0.39 / 0.42 at
D=200), PCA / t-SNE / UMAP hold 0.85–0.92 at D=80 but drop to 0.46–0.51 by D=200, and from
D=200–400 every method except toorPIA sits near chance (0.33–0.51 vs chance 0.33). **toorPIA holds
accuracy 1.00 through D=200, 0.99 at D=400, and 0.98 at D=768** (effective SNR ≈0.004); pushed to
**D=2000** (effective SNR ≈0.0015, 1997 noise columns) it enters its breaking regime — and there
the outcome becomes **noise-realization dependent**: across six independent noise realizations
(data seeds 0–3, the committed sweep's SeedSequence realization, and the source notebook's global
RNG stream — all through the same API at `random_seed=42`) the kNN accuracy spans **0.56–0.88**
(median ≈0.79); the committed sweep's realization happens to be the lowest of the six. Method-seed
variance stays negligible (±0.01) — the variance is across noise *draws*, the signature of a
critical regime, and every realization stays well above chance. On the vs-ambient axis the ordering inverts: PCC
tracks the noise-dominated ambient distances best (full ρ ≈0.54 at D=768 vs toorPIA ≈0.40) while
its map shows no clusters — the two readouts together separate "faithful to the features as given"
from "true structure still visible", which is exactly the regime dependence this supplement
documents. The PyMDE panels are a useful caution: from D≈40 it draws crisp, well-separated clumps
whose label composition is fully mixed — plausible-looking structure that is entirely false.
Figures: `figures/noise_dims/` (`dimension_curve.png`, `dims_grid.png`); tables:
`results/dimsweep_{per_run,aggregated}.csv`; rendered in `REPORT.html#noise-dims`. Reproduce:
`python run/dimsweep.py --dims 6 10 20 40 80 100 200 400 768 --methods all --seeds 3 --n 1000`,
then `python run/dimsweep.py --dims 2000 --methods toorPIA --seeds 3 --n 1000` (results merge by
(dim, method, seed); the committed toorPIA caches make both replays offline).

The idealized datasets and the noise-dims sweep bracket reality from the two extremes (maximal
signal redundancy vs none); realistic middle-ground variants (sparse loadings, correlated noise)
are under construction and will be documented here once finalized.

## DR methods

All wrapped behind `embed(X, seed, device, context) -> Y (N×2)` (`methods/base.py`). Stochastic
methods run **R independent seeds** (default R=3); deterministic methods run once. Exact settings
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

## Constraint density: which pair relations does each method's loss include?

A neutral, technical axis that organizes much of what this benchmark measures: DR methods differ in
**which pairwise relations enter their loss (or construction) at all**. A pair relation that is not
in the loss is not optimized for — its preservation in 2-D is a side effect, not a guarantee. The
choice is a **design trade-off, not a defect**: constraining fewer pairs is what buys scalability
(O(N·k) or O(N·m) instead of O(N²) work per step), and every sparse/local design below is a
reasonable engineering answer to "all pairs is too expensive". The question this axis raises is
only: *what, concretely, is traded away for that speed?*

| method | pair relations constrained by the loss / construction | traded for |
|---|---|---|
| **PCA** | all pairs — but of **projected** distances: variance is a full-pair statistic, yet the linear map is chosen by variance, so a direction carrying little variance (e.g. one off-subspace point) can be dropped entirely | linear maps only |
| **Isomap** | all pairs (classical MDS on the full graph-geodesic distance matrix) | O(N²·log N)+ cost |
| **toorPIA** | all pairs (vendor statement — internals not public; see the COI section) | O(N²)-class cost |
| **PyMDE** (as configured) | by design a **sparse subsample** of pairs (`max_distances`); at this benchmark's N=1000 the default cap exceeds N(N−1)/2, so **effectively all pairs here** | scalability at large N |
| **PCC** (label-free) | **distances to a sampled reference set** (`num_points=N`, sampled **with replacement**) — a column subset of the full pair matrix | O(N·m) per epoch instead of O(N²) |
| **t-SNE** | **local neighborhoods** (perplexity-scaled affinities; far-pair attractions vanish) | scalability + cluster legibility |
| **UMAP** | **local neighborhoods** (k-NN graph, k=15, with negative sampling) | scalability + cluster legibility |

Two consequences of this axis are already measured by the benchmark. The **within-cluster scale
collapse** (over-compression metric): when the loss is dominated by, or restricted to, a subset of
relations, the remaining structure can be squeezed without penalty. And — the reason for the fourth
dataset — **outlier separation**:

> **Hypothesis (documented before the outliers results; see the COI section).** A pair relation
> not included in the loss has no preservation guarantee. A single outlier participates in only
> O(1/N) of all pairs, so under sparse constraints two mechanisms can shrink its separation margin
> in 2-D: **(a) omission** — the outlier's pairs may not be in the constrained subset at all (for
> PCC's reference sampling with `num_points = N` drawn *with replacement*, any given point is
> absent from the reference set with probability (1 − 1/N)^N ≈ 37% — a routine event, not a rare
> corner case); **(b) dilution** — even when included, the outlier's few pair terms are outvoted by
> the O(N²) bulk terms in the loss. Methods that constrain **all** pairs cannot omit any single
> point's relations, so the margin should be more stable. Local-neighborhood methods constrain the
> outlier only through its (distant) nearest neighbors, so the *magnitude* of its separation is
> not represented in the loss at all. This is the same mechanism family as the known
> "few-point-scale structure buried by the bulk loss" effect on the clusters dataset — applied to
> m points instead of a within-cluster scale. If the results contradict this (e.g. a
> sparse-constraint method preserves the margin well), they are reported as-is.

Because the 3 anomalous directions are i.i.d.-equivalent replicates and **all seven methods** run
on the dataset, the main results table is itself the test of "does sparse constraint density
generally imply margin shrinkage" — PyMDE (sparse by design, effectively full-pair at this N) and
PCC (reference-point sparse) provide the two informative sparse cases.

> **Outcome (added after the results; the hypothesis above and the off-subspace PCA prediction are
> unchanged from their pre-results commits).** At the realistic separation (3 Rg, SNR=1) the
> question splits into *presence* (is the anomaly separated at all?) and *structure* (are its
> relations rendered right?), and the constraint types part ways on both — both readable directly
> from the Shepard density figure and the star gallery. **Presence:** every
> pairwise-distance-constraining method keeps the anomalies separated; the galleries show burial
> only for the variance and neighborhood constraints, deepening with the true factor (PCA — the
> pre-registered variance-truncation prediction; UMAP; t-SNE weakly at SNR=1). **Structure
> (same-kind pairs; pair angle vs truth 0°):** the neighborhood methods keep near-duplicates
> together — UMAP nearly perfectly (≤ 2°), t-SNE by fusing them into one point — but place them on
> bulk-cluster edges; the sparse reference-point design tears same-kind pairs apart (PCC ≈ 80–140°,
> seed-arbitrary), as do PCA/Isomap/PyMDE (≥ 150°); toorPIA keeps pairs co-directional (≤ 10°,
> ≤ 1° clean) at a stretched radial scale. The **burial arm was NOT borne out for the
> reference-point design** (an exclusion probe confirmed this is structural — a reference-point
> loss constrains every point's *row* of reference distances, so complete omission cannot occur;
> null result, see the note below): PCC's constraint-density trade-off surfaces as *relational*
> distortion — pair tearing, and its Shepard diagram's off-trend blocks — not as burial.
> Net reading: **no single constraint type gets both halves right; the axis that decides the
> monitoring use case is the pair/direction structure plus the out-of-sample operation (see the
> addplot supplement), not the separation margin alone.**

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

### How the outliers dataset is scored (the anomaly-pair Shepard ρ)

The primary quantity is the **same standard statistic as everywhere else, restricted to the pairs
that matter here**: the **anomaly-pair Shepard ρ** (`outlier_shepard`, "outlier ρ") — Spearman
rank correlation between high-D and 2-D distances over exactly the pairs where at least one
endpoint is a ground-truth outlier. It is the direct quantification of the outlier-related blocks
of the **Shepard density figure** (pair-internal ≈ bulk-NN scale, outlier-to-bulk, and
outlier-to-outlier across kinds): a method that plots its anomalies among other clusters, fuses
same-kind pairs, or tears them apart scrambles precisely these pairs and scores low, no matter how
well the pairs among the normal points are ordered. This restriction exists because those violations involve
*few* pairs, so the all-pair ρ moves only slightly — the same standard machinery as the distance
bands, with the subset selected by endpoint membership instead of by distance percentile. **On the
outliers dataset the composite ranking scores a third column on it** (1st→5 … 5th→1, alongside
full and p5), so a method cannot rank well there while failing the anomaly baseline; the
`outlier ρ vs full ρ` scatter (the analogue of the near-vs-global scatter) shows both at once.

Descriptive only (not scored): the same-kind **pair angles** from `outlier_pair_metrics` (truth
0°) — useful for describing *how* a method fails (torn vs fused), but never a stand-alone
evaluation, because a method can score a perfect angle by fusing the pair while failing the
baseline (t-SNE does exactly this). Aggregation follows the repo convention (median + bootstrap
95% CI over the R seeds).

**Retired metric (for the record):** an earlier bespoke ratio metric, OSP ("outlier separation
preservation", a double normalization of nearest-non-outlier margins), was removed from all
reporting: it is not an established metric and its normalization did not track the
Shepard/embedding pictures. Its raw columns remain in the per-run CSVs and its code in
`metrics/outlier.py` for the record; no conclusion in this repository rests on it.

### The imbalanced-populations dataset's diagnostics (membership-restricted ρ)

The analysis theme is **minority-structure preservation under data imbalance**. The ranking table
keeps the benchmark's standard two-sided evaluation — global (full ρ) and local (near-band p5) —
with no extra scored columns. The diagnostics below use the same standard machinery, with the pair
subset selected by the endpoints' **population membership** (exactly as the outliers dataset
subsets by "involves an outlier"): **majority-internal ρ** (both endpoints in the majority),
**minority-internal ρ** (both in the minority — the hard question: does the small population keep
its internal structure?), and **cross-population ρ** (one endpoint in each — is the minority
placed correctly relative to the majority?). All vs-ambient, per the repo convention. They are
reported in the results CSVs. The sweep curve reads exactly like the outliers dataset's: global
Shepard ρ next to the pooled **minority-pair ρ** (`minority_shepard` — the same standard ρ over
all pairs involving at least one minority endpoint, i.e. all pairs minus the majority-only pairs;
the direct analog of the anomaly-pair ρ, and one number that drops if either the minority's
internal structure or its placement is scrambled). The **population over-compression ×** diagnostic
applies the repo's over-compression formula one level up (median within-population between-cluster
distance ÷ median cross-population distance, truth over 2-D): ≈1 = the internal layouts keep their
scale relative to the group separation, ≫1 = they are squeezed toward knots while the group gap
dominates. Caveat stated once: read it together with cross-population ρ — a method that never
renders the group separation at all (cross ρ ≈ 0) can post an unsuspicious value here without
deserving it.

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
- **The pair angle and the addplot readouts coincide with no method's training objective either**:
  the angle between two designated points seen from the bulk centroid, and the nearest-fit-anomaly
  kind assignment, are geometric readouts of the finished map — none of the seven methods optimizes
  these quantities.

### Note: reference-set exclusion probe (null result; data retained)

`run/pcc_excluded.py` re-ran PCC with the outliers excluded from the reference set (everything
else stock; intervention listed under *Deviations*). **Null result** — indistinguishable from the
stock run; the structural reason is in the outcome note above. Not rendered in the REPORT; script
and CSVs (`results/pcc_excluded_*.csv`) remain as the record.

### Supplement: addplot / out-of-sample test (`run/addplot_test.py`)

The direct test of the anomaly-monitoring criterion. The basemap is fitted on **normal data
only** (the outliers dataset's bulk — no anomaly ever seen at fit time) and new points arrive
afterwards, one at a time. The added set holds **cluster-anchored anomalies**
(`synth.outliers.make_anchored_addplot`): each shares a normal cluster's clean profile in the
measured features and deviates `3 Rg` along new dimensions orthogonal to everything the normal
data varies in — one near-duplicate pair per cluster (radii 3.0 / 3.1 Rg, 5 clusters × 2 = 10
anomalies) — the realistic shape of a fault: a known operating state plus an effect the
historical data never showed. 50 fresh normal points ride along as controls. Two questions, in
order: **detection** — does a never-seen anomaly land visibly outside the normal region at all
(`anomaly_radius_ratio`: 2-D distance from the map centroid over the bulk's median radius)? —
and **attribution** — is the anomaly's **direction** from the centroid closest to its own source
cluster's direction (`attribution_accuracy`, `angle_to_own`), and do the two near-duplicates of
one cluster stay co-directional (`pair_angle`)? The direction of an addplot point is
information: it should say *which* normal condition the anomaly departed from. The ambient
high-D features resolve attribution 10/10 (the anchor signal survives SNR=1 noise), so a
faithful map can too. Each method maps the added points with its own out-of-sample operation:
PCA / Isomap `transform` (deterministic), UMAP seeded `transform`, toorPIA server-side `addplot`
on the fitted map (live fit + addplot per seed, committed as a self-consistent
`basemap_fit`/`basemap_add` cache pair — the benchmark's fit cache is not reusable because the
server keeps the fitted-map state and is not bit-deterministic across sessions). **t-SNE
(sklearn), PyMDE, and PCC expose no out-of-sample operation** — reported as "not operable" rows:
for monitoring, adding data means re-fitting, and a re-fit re-arranges the map. Results in
`results/addplot_*.csv` and the REPORT's *addplot* section.

**Result (R=3, SNR=1).** **toorPIA is the only method that answers both questions**: all 10
anchored anomalies land far outside the normal region (radius ratio median 5.87, min 5.23 —
identical across seeds) **and** each one's direction points at its source cluster (attribution
10/10, angle to own cluster 0.6° median / 3.0° max, pair angle 0.6°) — the map says "this is an
anomaly, and it departed from cluster k" in one reading. **PCA, Isomap, and UMAP draw the
anomalies inside or at the normal clusters** (median radius ratio 0.97 / 1.34 / 0.96–1.04,
minima down to 0.26): the orthogonal deviation is dropped or interpolated away, so the anomaly
looks like one more normal point of its source cluster — the nominally high attribution
(8–10/10) carries no alarm, and detection fails silently. Added normal controls land correctly
for every operable method (bulk-control ratio 0.87–1.01). t-SNE / PyMDE / PCC: not operable (no
out-of-sample transform).

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
- **Order-independence of torch-based methods**: UMAP's first execution (numba threading-layer
  init) silently resets the process's torch thread count, which would change the float reduction
  order — and hence the optimization trajectory — of any PyMDE/PCC run that follows it in the same
  process. The PyMDE and PCC wrappers therefore **re-pin `torch.set_num_threads(1)` on every
  call**, so their results do not depend on which methods ran before them
  (`tests/test_thread_pinning.py` is the regression net).
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
4. **PCC-EXCLUDED supplement uses a subclass override** (`run/pcc_excluded.py`): pccdr's API does
   not accept an explicit reference set, so the supplement subclasses `PCC` and overrides
   `get_reference_points` to draw the same with-replacement `np.random.choice` over bulk indices
   only. No other pccdr code path is touched; the stock benchmark runs use the unmodified package.

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
fidelity profiles and their dispersion; rankings respect CI overlap. This applies equally to the
outliers dataset and its pair/addplot readouts: they characterize whether a synthetic anomaly
structure survives the 2-D map, not any method's usefulness for real-world outlier *detection*
(which is a downstream task with its own tooling). Note also that all five-dataset results are
obtained under the redundancy-rich isotropic noise model — the noise-*friendly* extreme (see
*Noise regimes*); the noise-dims sweep shows the rankings need not transfer to
sparse/irrelevant-feature regimes, and none of the rankings should be extrapolated there.
