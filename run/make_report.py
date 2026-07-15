"""Generate REPORT.html — a consolidated results dashboard from the benchmark outputs.

Reads ``results/metrics_aggregated.csv`` and ``results/stability.csv`` and assembles ranking tables
(median + bootstrap 95% CI, best-in-column highlighted), the figures (Shepard density heatmaps,
multi-scale Shepard/stress curves, embedding panels, summary heatmaps), the stability tables, and the
methodology / bias / non-circularity notes into a single HTML file.

    python run/make_report.py                 # figures referenced by relative path (git-friendly)
    python run/make_report.py --embed         # base64-embed figures -> portable standalone file

By default figures are referenced relative to the repo root, so open REPORT.html from the repo.
"""
from __future__ import annotations

import argparse
import base64
import html
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["density", "clusters", "transition", "outliers", "populations"]
# display titles ("populations" stays the internal key for results/caches/figures paths)
DATASET_TITLES = {"populations":
                  "imbalanced populations — minority-structure preservation under data imbalance"}
DATASET_NAV = {"populations": "imbalanced populations"}
SNRS = [1.0]   # SNR=1 (noisy = realistic).
METHOD_ORDER = ["PCA", "Isomap", "PyMDE", "PCC", "t-SNE", "UMAP", "DREAMS", "toorPIA"]

# Column groups. The band-Shepard family (fixed-radius, fair) is the PRIMARY block; the recall@k
# family (variable-radius k-NN, favorable to k-NN methods) is explicitly a REFERENCE block.
# Each entry: (metric key, header, direction) with direction True = higher is better,
# False = lower is better, "one" = closest to 1 is best, "zero" = closest to 0 is best.
TABLE_GROUPS = [
    ("rank", "Distance-fidelity ranking (scores the ρ columns only; 1st→5 … 5th→1 pts)", [
        ("__pts_full", "full pts", True),
        ("__pts_near", "near pts", True),
        ("__pts_total", "Σ", True),
    ]),
    ("primary", "Band-Shepard ρ — fixed-radius, fair (PRIMARY)", [
        ("full_shepard", "full · global", True),
        ("shepard_near__vs_ambient", "near · first-mode band", True),
    ]),
    ("crush", "Tightest-cluster scale  (× vs overall spread; ≫1 = crushed to a point, ≪1 = inflated)", [
        ("tight_over_compression", "tight-cluster scale ×", "one"),
    ]),
    ("ref", "Reference — k-NN-favorable, biased (not a fair neighborhood test)", [
        ("recall_k15", "recall@15", True),
        ("trust_k15", "trust@15", True),
        ("cont_k15", "continuity@15", True),
    ]),
]
# metrics scored for the composite ranking: full Shepard ρ and the first-mode near-band ρ.
# On the outliers dataset a third key is scored the same way (1st→5 … 5th→1): the standard Shepard
# ρ restricted to the ANOMALY-INVOLVING pairs — the direct quantification of the outlier-related
# blocks of the Shepard density figure. A method that plots the anomalies among other clusters,
# fuses same-kind pairs, or tears them apart scrambles exactly these pairs and scores low there no
# matter how well it orders the pairs among the normal points — so local readings can never outrank the baseline.
RANK_KEYS = [("shepard_near__vs_ambient", "__pts_near"), ("full_shepard", "__pts_full")]
OUTLIER_RANK_KEY = ("outlier_shepard__vs_ambient", "__pts_out")

# Extra column group shown only for the outliers dataset: the outlier-pair Shepard ρ that feeds
# the third ranking column above.
OUTLIER_GROUP = ("osp", "Outlier pairs — standard Shepard ρ over anomaly-involving pairs", [
    ("outlier_shepard__vs_ambient", "outlier ρ", True),
])


def _badness(v, mode):
    """Distance from the ideal value for 'one' / 'zero' direction modes (smaller = better)."""
    if v is None:
        return float("inf")
    if mode == "one":
        # symmetric in ratio terms: 0.5x and 2x are equally far from the ideal 1
        return abs(np.log2(v)) if v > 0 else float("inf")
    return abs(v)                                   # mode == "zero"

DATASET_BLURB = {
    "clusters": "Distinct dense sub-populations in a high-dimensional feature space (7 well-separated "
                "dense clusters). Tests whether a method keeps the fine within-cluster structure while "
                "also placing the clusters correctly.",
    "density": "Non-uniform density (uniform + tight core + sparse shell). Tests density distortion "
               "and the recall@k bias; distance-preservers should win the global/near Shepard bands.",
    "transition": "Continuous transition: dense clusters CONNECTED by bridge regions. The bridges (plus "
                  "SNR=1 noise) dilute the tightest-cluster over-compression here — PCC's clusters are "
                  "squeezed but not to points (≈2× under the canonical noise). Included to show "
                  "the effect is weaker when clusters are connected; toorPIA still preserves scale and "
                  "leads the global ρ.",
    "outliers": "Bulk of dense clusters plus 3 anomalous DIRECTIONS × 2 near-duplicate outliers "
                "each, at outlier_factor × Rg (bulk radius of gyration) along dedicated directions "
                "orthogonal to the bulk's subspace (modeled on a real contamination case: images "
                "acquired under a different condition hiding in a feature set). Read it from the "
                "two figures first: the Shepard density panel (high-D vs 2-D distance — the "
                "outlier-related pair blocks are directly visible, and violations such as same-kind "
                "pairs thrown to huge 2-D distances or far pairs collapsed to 0 show up as "
                "off-trend blocks) and the star gallery (outliers colored by direction; a/b = the "
                "near-duplicate pair). The quantitative anchor is the same standard statistic as "
                "everywhere else: the Shepard ρ restricted to the ANOMALY-INVOLVING pairs (the "
                "'outlier ρ' column) — it scores exactly the blocks the density panel shows and "
                "feeds the ranking score as a third 5..1 column, so a method that plots its "
                "anomalies among other clusters or fuses/tears same-kind pairs scores low there no "
                "matter how well it orders the pairs among the normal points.",
    "populations": "Theme: minority-structure preservation under data imbalance. An IMBALANCED "
                   "pair of populations, each with internal cluster structure: a "
                   "majority (5 dense clusters) and a much smaller minority (the same 5-cluster "
                   "geometry, fewer points — default 5%, i.e. 95% vs 5%), placed in disjoint dimension regions "
                   "and separated so that every cross-population center distance is exactly "
                   "group_range (=2) × the within-population center distance. Models a ubiquitous "
                   "real situation: normal production data mixed with a rarely-used operating "
                   "mode, a main line vs a small pilot-lot series, data before/after an instrument "
                   "change, or a large healthy cohort vs a small patient group with subtypes. In a "
                   "real project this composition is UNKNOWN in advance, and the minority is very "
                   "often the actual object of the analysis (anomaly analysis, positive cases in "
                   "medical data, transient operating states of a process) — so extracting it from "
                   "the map needs two readings positive AT ONCE: is the minority drawn as a "
                   "recognizable separate group (cross-population ρ), and does it KEEP a "
                   "trustworthy internal structure despite having few points (minority-internal "
                   "ρ)? A method that fails either one cannot be used for this ubiquitous task. "
                   "The gallery (circles = majority, triangles = minority) shows both at a "
                   "glance; the sweep curve reads exactly like the outliers dataset's: global "
                   "Shepard ρ next to the SAME standard ρ restricted to the minority-involving "
                   "pairs ([minority]-[minority] plus [minority]-[majority] — one number that "
                   "drops if either reading fails). Beware the failure is SILENT: the global ρ "
                   "(left panel, and the table below) can stay high while the minority-pair ρ "
                   "(right panel) collapses. The table keeps the benchmark's standard two-sided "
                   "evaluation (global + local Shepard bands); the per-method "
                   "membership-restricted ρ values are in the results CSVs.",
}


def snr_label(snr):
    return "inf" if not np.isfinite(snr) else f"{snr:g}"


def load():
    agg = pd.read_csv(ROOT / "results" / "metrics_aggregated.csv")
    agg["snr"] = agg["snr"].astype(float)
    stab = pd.read_csv(ROOT / "results" / "stability.csv")
    stab["snr"] = stab["snr"].astype(float)
    return agg, stab


def cell(agg, dataset, snr, method, metric):
    r = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == method)
            & (agg.metric == metric)]
    if not len(r):
        return None
    return float(r["median"].iloc[0]), float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0]), \
        int(r["n_runs"].iloc[0])


def fmt(c):
    if c is None:
        return "—"
    med, lo, hi, n = c
    if n <= 1 or (abs(hi - lo) < 5e-4):
        return f"{med:.3f}"
    return f"{med:.3f}<span class='ci'> [{lo:.3f}, {hi:.3f}]</span>"


def ranking_points(agg, dataset, snr, methods):
    """Composite ranking points: for full Shepard ρ AND the first-mode near-band ρ — plus, on the
    outliers dataset, the anomaly-pair Shepard ρ — the 1st..5th method gets 5,4,3,2,1 points
    (others 0); the total is their sum. Returns ``{method: {__pts_*: int}}``."""
    rank_keys = list(RANK_KEYS) + ([OUTLIER_RANK_KEY] if dataset == "outliers" else [])
    pts = {m: {pk: 0 for _, pk in rank_keys} for m in methods}
    for metric_key, pts_key in rank_keys:
        ranked = sorted(((m, cell(agg, dataset, snr, m, metric_key)) for m in methods),
                        key=lambda t: -(t[1][0] if t[1] else -1e9))
        for i, (m, c) in enumerate(ranked):
            if c is not None:
                pts[m][pts_key] = max(0, 5 - i)
    for m in methods:
        pts[m]["__pts_total"] = sum(pts[m][pk] for _, pk in rank_keys)
    return pts


def ranking_table(agg, dataset, snr):
    groups = TABLE_GROUPS + ([OUTLIER_GROUP] if dataset == "outliers" else [])
    if dataset == "outliers":
        # the rank group gains the outlier-pair-ρ points column (same 5..1 scheme)
        kind, label, items = groups[0]
        items = items[:-1] + [("__pts_out", "outlier pts", True)] + items[-1:]
        groups = [(kind, label, items)] + groups[1:]
    methods = [m for m in METHOD_ORDER
               if ((agg.dataset == dataset) & (agg.snr == snr) & (agg.method == m)).any()]
    pts = ranking_points(agg, dataset, snr, methods)

    def value(key, m):
        """(numeric, display_html) for any column key, including synthetic ranking-point keys."""
        if key.startswith("__pts"):
            v = pts[m][key]
            return float(v), str(v)
        c = cell(agg, dataset, snr, m, key)
        return (None if c is None else c[0]), fmt(c)

    keys = [k for _, _, items in groups for (k, _, _) in items]
    numeric = {m: {k: value(k, m)[0] for k in keys} for m in methods}
    disp = {m: {k: value(k, m)[1] for k in keys} for m in methods}

    # best / worst per column — by direction (higher / lower / closest-to-ideal); only when varying
    hib = {k: h for _, _, items in groups for (k, _, h) in items}
    best, worst = {}, {}
    for k in keys:
        vs = [numeric[m][k] for m in methods if numeric[m][k] is not None]
        if vs and max(vs) != min(vs):
            mode = hib.get(k, True)
            if mode in ("one", "zero"):
                best[k] = min(vs, key=lambda v: _badness(v, mode))
                worst[k] = max(vs, key=lambda v: _badness(v, mode))
            elif mode:
                best[k], worst[k] = max(vs), min(vs)
            else:
                best[k], worst[k] = min(vs), max(vs)
    # the tightest-cluster column shows only the closest-to-1 highlight (light green) and the
    # >5x crush flag; everything else in it stays uncolored
    worst.pop("tight_over_compression", None)

    # order rows by composite total (desc), tie-break by full Shepard ρ (desc)
    def order_key(m):
        c = cell(agg, dataset, snr, m, "full_shepard")
        return (-pts[m]["__pts_total"], -(c[0] if c else -1e9))
    methods = sorted(methods, key=order_key)

    grp_th = ["<th class='method' rowspan='2'>method</th>"]
    sub_th = []
    for kind, label, items in groups:
        grp_th.append(f"<th class='grp grp-{kind}' colspan='{len(items)}'>{html.escape(label)}</th>")
        for _, h, _ in items:
            sub_th.append(f"<th class='sub-{kind}'>{html.escape(h)}</th>")
    head = f"<tr>{''.join(grp_th)}</tr><tr>{''.join(sub_th)}</tr>"

    rows = []
    for m in methods:
        tds = []
        for kind, _, items in groups:
            for k, _, _ in items:
                n = numeric[m][k]
                cls = []
                # absolute-threshold reds (regardless of rank): a negative near-band ρ, a
                # negative anomaly-pair ρ, or THE WORST tightest-cluster crush of the table when
                # it exceeds 5x -- crushing destroys information (it cannot be read back), so only
                # an unambiguous crush-side worst case earns the failure flag; inflation and
                # moderate crush stay uncolored (only closest-to-1 is highlighted).
                col_max = max((numeric[mm]["tight_over_compression"]
                               for mm in methods
                               if numeric[mm].get("tight_over_compression") is not None),
                              default=None) if k == "tight_over_compression" else None
                force_red = n is not None and (
                    (k == "shepard_near__vs_ambient" and n < 0)
                    or (k == "tight_over_compression" and col_max is not None
                        and n == col_max and n > 5)
                    or (k == "outlier_shepard__vs_ambient" and n < 0))
                if force_red:
                    cls.append("crit")          # darker red than rank-worst — an outright failure
                elif n is not None and k in best:
                    if n == best[k]:
                        cls.append("best")
                    elif k in worst and n == worst[k]:
                        cls.append("worst")
                if kind == "ref":
                    cls.append("refcol")
                if k == "__pts_total":
                    cls.append("total")
                attr = f" class='{' '.join(cls)}'" if cls else ""
                tds.append(f"<td{attr}>{disp[m][k]}</td>")
        rows.append(f"<tr><th class='method'>{html.escape(m)}</th>{''.join(tds)}</tr>")
    return (f"<table class='rank'><thead>{head}</thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def stability_table(stab, dataset):
    sub = stab[stab.dataset == dataset].copy()
    if not len(sub):
        return ""
    sub = sub.sort_values(["method", "snr"])
    cols = ["method", "snr", "n_runs", "position_dispersion",
            "full_shepard__std", "full_stress_fidelity__std"]
    cols = [c for c in cols if c in sub.columns]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows = []
    for _, r in sub.iterrows():
        tds = []
        for c in cols:
            v = r[c]
            if c == "snr":
                v = snr_label(float(v))
            elif isinstance(v, float):
                v = f"{v:.4f}"
            tds.append(f"<td>{html.escape(str(v))}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")
    return (f"<table class='rank'><thead><tr>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def img(dataset, fname, embed):
    p = ROOT / "figures" / dataset / fname
    if not p.exists():
        return f"<div class='missing'>missing: {html.escape(fname)}</div>"
    if embed:
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        src = f"data:image/png;base64,{b64}"
    else:
        src = f"figures/{dataset}/{fname}"
    return f"<img loading='lazy' src='{src}' alt='{html.escape(fname)}'>"


def figure_one(dataset, fname, caption, embed):
    """A single captioned, full-width figure (used by the reading guide)."""
    return (f"<figure class='guidefig'>{img(dataset, fname, embed)}"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def figure_block(dataset, snr, embed):
    lab = snr_label(snr)
    figs = [
        (f"distance_distribution_snr{lab}.png",
         "High-D pairwise-distance profile — why the near band (the profile's first mode) is a slice the full ρ buries"),
        (f"score_scatter_snr{lab}.png", "Near (first-mode band) vs global (full) Shepard ρ — top-right captures both"),
        (f"shepard_scatter_snr{lab}.png", "Shepard density (jet, log counts) — high-D vs 2-D distance"),
        (f"summary_heatmap_snr{lab}.png", "Summary heatmap (methods × metrics, median)"),
        (f"embeddings_snr{lab}.png", "2-D embeddings colored by the dataset variable"),
    ]
    if dataset == "transition":
        figs[4] = (f"embeddings_snr{lab}.png",
                   "2-D embeddings colored by the cyclic continuum t. The dataset's defining "
                   "geometry has two features at once: seven DENSE typical-state clusters, and "
                   "their CLOSED ring connectivity (state 0 → 1 → … → 6 → 0 through bridges). "
                   "Read the gallery against both: toorPIA is the only method that reproduces "
                   "both simultaneously — seven distinct dense lobes arranged in the correct "
                   "cyclic order with the bridge points between adjacent states. The others "
                   "each recover at most one feature: Isomap draws the cleanest closed ring but "
                   "smears the dense clusters along it; PCA keeps the cyclic order in a ragged "
                   "ring with blurred, overlapping clusters; t-SNE and UMAP recover dense "
                   "clusters but tear the connecting bridges, fragmenting the loop; DREAMS "
                   "recovers dense clusters AND (via its PCA scaffold) the correct cyclic "
                   "arrangement, but it too tears bridges, so the ring stays open; PCC draws "
                   "the seven states as radial spokes fused at a central hub, so every state "
                   "becomes adjacent to every other and the cyclic adjacency is lost. The "
                   "connectivity reading is quantified by the bridge bottleneck-gap diagnostic "
                   "(results/bridge_gaps.csv, python run/bridge_gaps.py): the smallest radius "
                   "at which adjacent states connect through their own bridge points, as a "
                   "fraction of the inter-state distance. Among the four methods that draw "
                   "seven dense, separated clusters, toorPIA keeps every bridge connected "
                   "(largest void 14% of the inter-cluster distance), while t-SNE tears 4 of "
                   "the 7 bridges (up to 49%), UMAP 4 (up to 85%), and DREAMS 3 (up to 58%); "
                   "the low gap values of PCA/Isomap/PyMDE/PCC are connectivity-by-blurring "
                   "(overlapping or hub-fused clusters), which the tables and the gallery "
                   "expose. This qualitative reading sits alongside the quantitative table "
                   "below: toorPIA holds the best global ρ here (0.729) while its first-mode "
                   "near ρ trails, which is what drops it to joint 3rd in the composite with "
                   "DREAMS")
    if dataset == "populations":
        figs = [
            (f"population_gallery_snr{lab}.png",
             "Embedding gallery at the canonical 95% vs 5% — circles = majority population, "
             "triangles = minority population, colored by cluster (0-4 majority, 5-9 minority). "
             "PCA draws the minority as a correctly placed featureless blob, "
             "PCC strings it out on an arc around the majority "
             "(placement rank-uninformative, cross ρ 0.20), t-SNE fuses its five clusters into "
             "one tiny clump set apart, UMAP scatters them as islands along the map edge, "
             "DREAMS draws it as one coherent clump flung to an arbitrary distance (internal "
             "structure intact, minB ρ 0.65; placement rank-uninformative, cross ρ 0.07 — the "
             "composite Σ ties toorPIA's while the reading the dataset tests still fails), "
             "toorPIA keeps it a recognizable group with internal structure"),
            ("populations_sweep_curve.png",
             "Sweep vs minority fraction (SNR=1, CI bands): global Shepard ρ and the "
             "minority-pair-restricted ρ"),
        ] + figs
        figs.insert(0, figs.pop(4))     # lead with the Shepard density panel
        # the gallery (majority/minority markers) supersedes the plain embeddings panel; the
        # minority-pair-ρ scatter replaces the plain near-vs-global scatter (same substitution
        # as on the outliers dataset)
        figs = [f for f in figs if not f[0].startswith(("embeddings_", "score_scatter_"))]
        figs.insert(2, (f"population_score_scatter_snr{lab}.png",
                        "Minority-pair ρ vs global (full) ρ — top-right renders both the "
                        "minority population and the global layout correctly"))
    if dataset == "outliers":
        figs = [
            (f"outlier_gallery_snr{lab}.png",
             "Embedding gallery, ground-truth outliers starred (one color per anomalous direction; "
             "a/b = the near-duplicate pair). toorPIA is the only method that renders the "
             "anomalies faithfully INCLUDING the relations among them: each same-kind a/b pair "
             "lands adjacent and co-directional (pair angle ≤ 10°), the three kinds point to "
             "three separate directions well away from the bulk, and the bulk's 5-cluster layout "
             "is kept — the picture behind its outlier ρ 0.615 (1st) and global ρ 0.759 (1st). "
             "PCA and Isomap drop the anomalies onto the bulk clusters; PyMDE tears same-kind "
             "pairs to opposite ends of the map and fuses different kinds; PCC keeps the bulk "
             "clean but throws same-kind pair members to different corners (angles ≈ 86°); t-SNE "
             "fuses each a/b pair into a single point among the bulk clusters; UMAP attaches the "
             "anomalies to bulk-cluster edges; DREAMS draws them between the bulk clusters, "
             "inside the normal region (outlier ρ 0.055)"),
            (f"outlier_score_scatter_snr{lab}.png",
             "Outlier-pair ρ vs global (full) ρ — top-right renders both the anomalies and the "
             "global layout correctly"),
            ("sweep_outliers_curve.png",
             "Sweep vs outlier_factor (clean, CI bands): global Shepard ρ and the "
             "anomaly-pair-restricted ρ"),
        ] + figs
        # lead with the most intuitive picture: high-D vs 2-D distance density
        figs.insert(0, figs.pop(5))
        # the outlier-pair ρ vs full ρ scatter is the anomaly-placement reading here; the plain
        # near-vs-global scatter adds nothing on this dataset
        figs = [f for f in figs if not f[0].startswith("score_scatter_")]
    # the 2-D map gallery IS these methods' primary output, so it is returned separately and
    # placed BEFORE the comparison table; the diagnostic figures follow the table
    map_prefix = {"populations": "population_gallery_", "outliers": "outlier_gallery_"}.get(
        dataset, "embeddings_")
    map_figs = [f for f in figs if f[0].startswith(map_prefix)]
    figs = [f for f in figs if not f[0].startswith(map_prefix)]

    def render(fs):
        return "".join(
            f"<figure>{img(dataset, f, embed)}<figcaption>{html.escape(cap)}</figcaption></figure>"
            for f, cap in fs)

    map_html = f"<div class='figgrid mapfig'>{render(map_figs)}</div>" if map_figs else ""
    return map_html, f"<div class='figgrid'>{render(figs)}</div>"


def random_walk_section(embed: bool) -> str:
    """The time-series dataset: multi-series high-D random walks (continuous change, no
    clusters). Runs on exactly the main-benchmark footing (toorPIA through the same
    basemap_embedding endpoint), so it is a main section. Rendered only when the probe's
    output exists."""
    p = ROOT / "results" / "random_walk_aggregated.csv"
    if not p.exists():
        return ""
    agg = pd.read_csv(p)
    COLS = [("full_shepard", "full · all pairs"),
            ("cross_series_shepard", "cross-series · star arrangement (GLOBAL)"),
            ("within_series_shepard", "within-series · trajectory shape"),
            ("within_series_near_shepard", "within-series near |Δt|≤20 · step structure (LOCAL)")]

    def cellv(m, k):
        r = agg[(agg.method == m) & (agg.metric == k)]
        if not len(r):
            return None, "—"
        c = (float(r["median"].iloc[0]), float(r["ci_lo"].iloc[0]),
             float(r["ci_hi"].iloc[0]), int(r["n_runs"].iloc[0]))
        return c[0], fmt(c)

    methods = [m for m in METHOD_ORDER if (agg.method == m).any()]
    numeric = {m: {k: cellv(m, k)[0] for k, _ in COLS} for m in methods}
    best = {k: max(v[k] for v in numeric.values() if v[k] is not None) for k, _ in COLS}
    worst = {k: min(v[k] for v in numeric.values() if v[k] is not None) for k, _ in COLS}
    methods = sorted(methods, key=lambda m: -(numeric[m]["full_shepard"] or -1e9))

    sub = "".join(f"<th class='sub-primary'>{html.escape(h)}</th>" for _, h in COLS)
    head = (f"<tr><th class='method' rowspan='2'>method</th>"
            f"<th class='grp grp-primary' colspan='{len(COLS)}'>Trajectory Shepard ρ — the "
            f"standard statistic, pairs split by (series, time) membership</th></tr>"
            f"<tr>{sub}</tr>")
    body = []
    for m in methods:
        tds = []
        for k, _ in COLS:
            n, disp = cellv(m, k)
            cls = (" class='best'" if n is not None and n == best[k]
                   else " class='worst'" if n is not None and n == worst[k] else "")
            tds.append(f"<td{cls}>{disp}</td>")
        body.append(f"<tr><th class='method'>{html.escape(m)}</th>{''.join(tds)}</tr>")
    table = (f"<table class='rank'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>")

    return (
        "<h2 id='random-walk'>time series — multi-series high-D random walks (continuous "
        "change, no clusters)</h2>"
        "<p class='blurb'>Theme: <b>trajectory rendering — the shape of multivariate "
        "time-series data.</b> Six independent random walks in D=50 (each time step adds an "
        "independent uniform displacement in every dimension; all series start at the origin) "
        "are concatenated into one dataset of 6 × 500 = 3000 points and embedded together. "
        "This is the regime that real multivariate monitoring data lives in — process sensors, "
        "equipment logs, longitudinal measurements: <b>OPEN, filamentary trajectories indexed "
        "by (series, time), with no clusters at all</b> — the structural opposite of the five "
        "datasets above. All eight methods run on exactly the main-benchmark footing (toorPIA "
        "through the same <code>basemap_embedding</code> endpoint).</p>"
        "<p class='blurb'><b>What the data really looks like — read this before the maps.</b> "
        "A random walk in D=50 does not look like the doodle that 2-D intuition suggests. "
        "Because every next step is drawn with D fresh degrees of freedom, three geometric "
        "facts hold with near-certainty — every panel of the figure below is <i>measured on "
        "the actual dataset</i>, not drawn schematically. <b>(1) Radial escape:</b> the "
        "distance from the origin grows as √t (‖x_t‖ ≈ step·√(tD/3)) — in high "
        "dimensions the walk never wanders back; each series pushes steadily OUTWARD (panel 2; "
        "compare the same recipe in D=2, panel 1, which meanders, recrosses itself, and "
        "overlaps its neighbors). <b>(2) Zigzag at every step (local):</b> successive steps "
        "are independent, and in high dimensions two independent directions are almost surely "
        "near-orthogonal — the angle concentrates at 90° with spread ~1/√D ≈ a few degrees "
        "(panel 3, blue) — so the trajectory is JAGGED at every single step and never smooths "
        "into a curve. <b>(3) Mutually orthogonal series (global):</b> the same "
        "near-orthogonality holds between the position vectors of different series (panel 3, "
        "red), so the six walks radiate along mutually perpendicular directions and never "
        "approach one another — every cross-series distance obeys the Pythagorean relation "
        "d ≈ √(r_i² + r_j²) (panel 4). The true shape is therefore a "
        "<b>star of six jagged spokes radiating from the shared origin at mutual right "
        "angles</b>. Six mutually orthogonal directions cannot all be drawn at right angles "
        "on paper — but a faithful 2-D map must still render what CAN be rendered: one shared "
        "origin with the series fanning out separately (never crossing), monotone outward "
        "time order along each spoke, and the saw-tooth jitter on top of every spoke.</p>"
        + figure_one("random_walk", "geometry_explainer.png",
                     "The geometric ground truth, measured on the actual dataset. (1) The same "
                     "walk recipe in D=2 — the intuition trap: walks meander, recross, and "
                     "overlap. (2) In D=50 the distance from the origin hugs the √t law "
                     "(colored series; grey = the D=2 walks wandering around it): monotone "
                     "radial escape. (3) Angles between successive steps (blue) and between "
                     "positions of different series (red) both concentrate at 90° in D=50, "
                     "while in D=2 they are spread over everything (grey): the walk is jagged "
                     "at every step, and the series are mutually orthogonal. (4) Cross-series "
                     "distances match the Pythagorean prediction √(r_i²+r_j²): the series "
                     "never approach each other.", embed)
        + "<div class='figgrid mapfig'><figure>"
        + img("random_walk", "trajectory_gallery.png", embed)
        + "<figcaption>2-D maps, points colored by time (dark = early, yellow = late; squares "
        "= series end points). Read against the three facts above. toorPIA renders the full "
        "truth — six jagged spokes radiating separately from one shared origin with clean "
        "outward time order (the picture behind its leading within-series 0.970 / "
        "cross-series 0.897 / full 0.924 ρ). PCA and PyMDE keep the radial fan but tangle it "
        "near the origin (PyMDE also scatters fragments of one series). PCC draws a clean "
        "six-spoke star but STRAIGHTENS the spokes — the saw-tooth jitter is visibly smoothed "
        "away (its step-structure ρ 0.470 is the lowest of all eight) — and throws a detached "
        "patch of one series across the map. Isomap collapses each walk to a nearly straight "
        "ray. DREAMS recovers the six-way radial arrangement (cross-series 0.778, second "
        "only to toorPIA) with clean outward time order (0.952) — but the shared origin is "
        "rendered only partly: every series' t=0 point does land at the common center, yet "
        "two of the six spokes detach from it within the first plotted steps (their ribbons "
        "jump to ≈18% and ≈6% of their spoke length before the next point is drawn, leaving "
        "a visible gap at the base), and all six spokes are smooth, "
        "wavy ribbons: the saw-tooth jitter amplitude is erased, exactly the value-level "
        "information a rank statistic cannot see. t-SNE and UMAP cut the walks into smooth, "
        "disconnected ribbons scattered over "
        "the canvas: the time order along each fragment is beautifully preserved — t-SNE and "
        "DREAMS share the step-structure lead (0.951 / 0.952; UMAP 0.885), a rank statistic "
        "that smoothing does not "
        "hurt — but for t-SNE and UMAP the shared origin and the radial arrangement are gone "
        "(cross-series ρ 0.366 / 0.477, the two lowest), and the jitter amplitude, which the "
        "rank statistic cannot see, is erased from the picture.</figcaption></figure></div>"
        + table +
        "<p class='note'><b>Honest notes.</b> (1) <b>Same footing as the five datasets "
        "above:</b> toorPIA runs through the identical deterministic "
        "<code>basemap_embedding(l2_normalization=False)</code> endpoint (coordinates "
        "committed under <code>external_embeddings/random_walk/</code> — offline replay, no "
        "API key needed); stochastic methods use R=3 seeds and brackets are bootstrap 95% "
        "CIs. (2) <b>Parameters:</b> D=50, 6 series × 500 steps, and NO added noise: the "
        "walk's own randomness IS the data, and the walk is generated directly in the ambient "
        "space, so the ground truth is the ambient geometry itself (vs-truth ≡ vs-ambient) — "
        "there is no separate latent space or SNR knob to harmonize with the main grid. D=50 "
        "already places the geometry deep in the high-dimensional regime (the angle spread "
        "around 90° is ~1/√D ≈ 8°); raising D only tightens the concentration. (3) "
        "<b>Columns:</b> the standard Spearman ρ with the pair set selected by (series, time) "
        "membership — the same construction as the outlier ρ and the minority-pair ρ: "
        "<i>within-series</i> = both endpoints in the same series; <i>within-series near</i> "
        "= same series and |Δt| ≤ 20 steps (the step-level saw-tooth; the time-lag window "
        "replaces the distance-percentile near band because here \"local\" is defined by "
        "time, not by a mode of the distance profile); <i>cross-series</i> = endpoints in "
        "different series (the star arrangement); <i>full</i> = all pairs. (4) No composite "
        "ranking points on this table — the split columns are the reading (rows are ordered "
        "by the full ρ; best per column green, worst light red as everywhere). (5) "
        "<b>Reproduce:</b> <code>python run/timeseries_probe.py</code> (defaults: "
        "<code>--ndim 50 --npoints 500 --n-series 6 --lag 20 --seeds 3</code>).</p>")


def noise_dims_section(embed: bool) -> str:
    """Supplement: the noise-dims dimension sweep — a direct stress test of noise tolerance in
    the redundancy-free regime (the opposite extreme of the five datasets' isotropic design).
    Rendered only when the sweep's output exists."""
    p = ROOT / "results" / "dimsweep_aggregated.csv"
    if not p.exists():
        return ""
    agg = pd.read_csv(p).replace({"method": {"toorPIA": "toorPIA (basemap_csvform)"}})
    emb_p = ROOT / "results" / "dimsweep_embedding_aggregated.csv"
    if emb_p.exists():
        emb = pd.read_csv(emb_p).replace(
            {"method": {"toorPIA-embedding": "toorPIA (basemap_embedding)"}})
        agg = pd.concat([agg, emb], ignore_index=True)
    row_order = ([m for m in METHOD_ORDER if m != "toorPIA"]
                 + ["toorPIA (basemap_embedding)", "toorPIA (basemap_csvform)"])
    dims_avail = sorted(int(d) for d in agg["dim"].unique())
    landmark = [d for d in (6, 40, 80, 200, 250, 300, 768, 1500, 2000)
                if d in dims_avail] or dims_avail[:5]
    knn_keys = sorted({str(m) for m in agg["metric"].unique() if str(m).startswith("knn_acc_k")})
    if not knn_keys:
        return ""
    groups = [(knn_keys[0], "2-D kNN label accuracy (chance = 1/3)")]

    def cell(m, dim, metric):
        r = agg[(agg.method == m) & (agg.dim == dim) & (agg.metric == metric)]
        if not len(r):
            return "<td>—</td>"
        med, lo, hi = (float(r["median"].iloc[0]), float(r["ci_lo"].iloc[0]),
                       float(r["ci_hi"].iloc[0]))
        ci = ("" if abs(hi - lo) < 5e-4
              else f"<span class='ci'> [{lo:.3f}, {hi:.3f}]</span>")
        return f"<td>{med:.3f}{ci}</td>"

    grp = "".join(f"<th class='grp grp-primary' colspan='{len(landmark)}'>{html.escape(t)}</th>"
                  for _, t in groups)
    sub = "".join("".join(f"<th class='sub-primary'>D={d}</th>" for d in landmark) for _ in groups)
    body = []
    for m in row_order:
        if not len(agg[agg.method == m]):
            continue
        tds = "".join(cell(m, d, k) for k, _ in groups for d in landmark)
        body.append(f"<tr><th class='method'>{html.escape(m)}</th>{tds}</tr>")
    table = (f"<table class='rank'><thead>"
             f"<tr><th class='method' rowspan='2'>method</th>{grp}</tr>"
             f"<tr>{sub}</tr></thead><tbody>{''.join(body)}</tbody></table>")

    return (
        "<h2 id='noise-dims'>Supplement — noise-dims dimension sweep (when dimensionality itself "
        "is the noise)</h2>"
        "<p class='blurb'><b>Why this supplement exists.</b> It makes a two-stage point about "
        "toorPIA under noise dimensions. <b>Stage 1:</b> on the same footing as the main "
        "benchmark — the raw-geometry <code>basemap_embedding</code> endpoint — toorPIA already "
        "tolerates noise dimensions better than every generic method. <b>Stage 2:</b> toorPIA "
        "additionally ships <code>basemap_csvform</code>, its <b>default endpoint for CSV/tabular "
        "data analysis</b> (a distinct pipeline with per-item preprocessing), and that endpoint "
        "shows an extraordinary tolerance to noise dimensions — it is essentially indifferent to "
        "them out to hundreds of pure-noise columns. This second point is what a practitioner "
        "needs for real tabular data, where irrelevant columns are numerous and their number is "
        "unknown in advance — and it is the reason this probe is a supplement rather than a sixth "
        "dataset row. <b>Two noise regimes.</b> The five datasets above share one deliberately "
        "noise-friendly design: the random orthonormal projection spreads every latent factor "
        "across all D=768 ambient columns (D-fold redundancy), so the driver's isotropic noise "
        "self-averages in every pairwise distance and the ambient dimension is nominal — no curse "
        "of dimensionality operates, <i>by construction</i>. Real CSV data is usually the "
        "opposite: the signal lives in a few columns and every further column mostly adds noise. "
        "This probe is that regime in its pure form: 3 tight clusters live in 3 signal columns "
        "and every additional column is pure unit-variance noise (per-column standardized), so "
        "each added dimension adds noise power at fixed signal power — the effective SNR is "
        "3/(D−3) and falls toward 0 as D grows. The readout is the <b>kNN label accuracy</b> — "
        "the direct operational answer to \"are the three true clusters still visible in the 2-D "
        "map?\" — shown both absolute and as the chance-corrected <b>skill ratio against kNN run "
        "directly on the raw D-dimensional features</b> (the ambient baseline: what a "
        "practitioner gets with no dimensionality reduction at all). <b>Stage-1 reading "
        "(same endpoint as the main benchmark):</b> the seven generic methods collapse between "
        "D≈40 and 400 (DREAMS is the most resistant of them — kNN accuracy 0.59 at D=200, "
        "skill ratio ≈ 0.6 — but it too is at chance by D≈400, 0.38); "
        "<code>basemap_embedding</code> instead tracks the ambient baseline itself "
        "(skill ratio ≈ 1: the 2-D map delivers essentially everything the raw feature space's "
        "neighborhoods still contain) and at D=200 stands at <b>0.75 vs ≤ 0.61</b> for the best "
        "generic method; its own breaking regime starts later (D≈250–400, "
        "realization-sensitive: 0.57 / 0.33 / 0.44 at D=250/300/400). <b>Stage-2 reading:</b> "
        "<code>basemap_csvform</code> is unaffected where every raw-geometry method — toorPIA's "
        "own embedding endpoint included — has already collapsed: <b>1.00 at D=200–300, 0.99 at "
        "D=400, 0.98 at D=768</b> (effective SNR ≈ 0.004), 0.92 at D=1500, first breaking at "
        "D=2000. Its skill ratio grows to ≈ 2.8 at D=768 and ≈ 3.9 at D=1500 — the 2-D map reads "
        "roughly three to four times more above-chance class signal than kNN on the raw features "
        "themselves. The probe is deliberately NOT a sixth registry dataset: its ground truth "
        "(the 3 signal columns) is intentionally not isometric to the ambient features, and "
        "distance-band Shepard ρ is deliberately not shown here: a global ρ is only meaningful "
        "paired with its near band (the point of this benchmark's methodology), and neither band "
        "reads cleanly against this probe's noise-dominated ambient distances — the full metric "
        "set remains in <code>results/dimsweep_per_run.csv</code>. Read the results as "
        "<b>regime dependence</b> — rankings from the redundancy-rich datasets above need not "
        "transfer to sparse/irrelevant-feature regimes, and vice versa.</p>"
        + table
        + figure_one("noise_dims", "dimension_curve.png",
                     "Left: 2-D kNN label accuracy vs total dimensionality D (log10 axis; chance "
                     "= 1/3 dashed; dotted grey = the ambient baseline, kNN run directly on the "
                     "raw D-dim features); median + bootstrap 95% CI ribbon per method. Right: "
                     "the same data as the chance-corrected skill ratio against that ambient "
                     "baseline — ratio 1 = as class-readable as the raw feature space itself. "
                     "toorPIA's two endpoints are the solid lines (generic methods dashed): "
                     "basemap_embedding (grey) "
                     "tracks the ambient baseline until its breaking regime at D≈250–400; "
                     "basemap_csvform (black) is unaffected out to D=768 and reaches skill "
                     "ratio ≈ 3.9 at D=1500. The D=1500 and D=2000 points pool 5 noise "
                     "realizations × 3 method seeds each, so their ribbons show realization "
                     "spread.", embed)
        + figure_one("noise_dims", "dims_grid.png",
                     "2-D embeddings at landmark dimensions, 3 true clusters colored — watch "
                     "where each method's clusters dissolve as noise dimensions are added. The "
                     "last two rows are toorPIA's two endpoints: basemap_embedding dissolves in "
                     "the D≈250–400 breaking regime like a raw-geometry method, while "
                     "basemap_csvform keeps the three clusters cleanly separated through "
                     "D=768.", embed)
        + "<p class='note'><b>Honest notes.</b> (1) <b>Endpoints:</b> the "
        "<code>basemap_csvform</code> rows drive the default CSV-analysis pipeline (it exposes "
        "<code>random_seed</code>, hence R=3 seeds and CI brackets; the sweep drives the same "
        "engine through its DataFrame form — spot-verified against a direct "
        "<code>basemap_csvform</code> call at D=768: pairwise-distance ρ 0.998, kNN accuracy "
        "0.975 vs 0.976; committed coordinates under "
        "<code>external_embeddings/noise_dims/toorpia/fit/</code>). The "
        "<code>basemap_embedding</code> rows drive the deterministic raw-geometry endpoint the "
        "main benchmark uses (single run per D; coordinates under "
        "<code>external_embeddings/noise_dims/toorpia/embedding/</code>; tables in "
        "<code>results/dimsweep_embedding_*.csv</code>). Each D is one noise realization, so "
        "values inside a breaking regime fluctuate between adjacent D (the embedding endpoint's "
        "0.57 / 0.33 / 0.44 at D=250/300/400 is that fluctuation, not a recovery). "
        "(1b) <b>Ambient baseline / skill ratio:</b> the baseline is leave-one-out kNN run "
        "directly on the raw standardized D-dim features — an operational reference (what no "
        "dimensionality reduction at all would give), not an information ceiling. The ratio is "
        "chance-corrected, (acc − 1/3)/(acc<sub>ambient</sub> − 1/3), because both accuracies "
        "approach chance (not 0) as D grows — the raw uncorrected ratio of a fully collapsed "
        "method would drift toward 1 instead of 0. Above D≈1000 the denominator itself is small "
        "(ambient 0.48 at D=1500, 0.44 at D=2000), so the ratio is increasingly "
        "noise-sensitive there. "
        "(2) n=1000, matching both the main benchmark and "
        "the source notebook. (3) toorPIA is additionally probed at <b>D=2000</b> (effective SNR "
        "≈ 0.0015); the other methods are already at chance well before D=768, so the extension "
        "is run for toorPIA only (the empty cells read as 'not run', not as failures). At this "
        "extreme the outcome is <b>noise-realization dependent</b> (the signature of a critical "
        "regime), so the D=1500 and D=2000 cells each aggregate <b>5 independent noise "
        "realizations × 3 method seeds</b> (data seeds 42, 0–3): at D=1500 the realizations agree "
        "(0.90–0.93) while at D=2000 they span 0.58–0.88; D ≤ 768 cells use one realization — "
        "realization variance is negligible below the breaking regime. "
        "(4) Bracketed ranges are bootstrap 95% CIs over seeds; deterministic methods show a "
        "point value. (5) kNN label accuracy is leave-one-out in the 2-D embedding (k=10; 3 "
        "balanced clusters, chance = 1/3). (6) Reproduce: "
        "<code>python run/dimsweep.py --dims 6 10 20 40 80 100 200 250 300 400 768 --methods all "
        "--seeds 3 --n 1000</code>, then "
        "<code>python run/dimsweep.py --dims 1500 2000 --methods toorPIA --seeds 3 --n 1000 "
        "--data-seed 42 0 1 2 3</code> (results merge by (dim, data_seed, method, seed)); the "
        "embedding arm: <code>python run/dimsweep.py --dims 6 10 20 40 80 100 200 250 300 400 "
        "768 --methods toorPIA --toorpia-endpoint embedding</code>; figures: "
        "<code>python run/dimsweep.py --figures-only</code>. The committed toorPIA caches make "
        "every replay offline.</p>")


def addplot_section(embed: bool) -> str:
    """Supplement: addplot / out-of-sample test on an anomaly-free basemap. Rendered only when
    the experiment's output exists. Two operational questions: does a never-seen anomaly land
    OUTSIDE the normal region (detection), and does its DIRECTION point back to its source
    cluster (attribution)?"""
    p = ROOT / "results" / "addplot_aggregated.csv"
    if not p.exists():
        return ""
    agg = pd.read_csv(p)
    per = pd.read_csv(ROOT / "results" / "addplot_per_run.csv")
    COLS = [("anomaly_radius_ratio_med", "anomaly distance ÷ bulk radius (med)"),
            ("anomaly_radius_ratio_min", "min"),
            ("attribution_accuracy", "source-cluster attribution acc."),
            ("angle_to_own_med", "angle to own cluster ° (med)"),
            ("pair_angle_med", "same-pair angle ° (med)"),
            ("add_bulk_radius_ratio", "bulk-control ratio (≈1 ideal)")]

    head = "".join(f"<th>{html.escape(h)}</th>" for _, h in COLS)
    body = []
    for m in METHOD_ORDER:
        sub = agg[agg.method == m]
        if len(sub):
            tds = []
            for k, _ in COLS:
                r = sub[sub.metric == k]
                if not len(r):
                    tds.append("<td>—</td>"); continue
                med, lo, hi = (float(r["median"].iloc[0]), float(r["ci_lo"].iloc[0]),
                               float(r["ci_hi"].iloc[0]))
                ci = ("" if abs(hi - lo) < 5e-4
                      else f"<span class='ci'> [{lo:.3f}, {hi:.3f}]</span>")
                tds.append(f"<td>{med:.3f}{ci}</td>")
            body.append(f"<tr><th class='method'>{html.escape(m)}</th>{''.join(tds)}</tr>")
        else:
            note = per[per.method == m]["note"]
            note = html.escape(str(note.iloc[0])) if len(note) else "unsupported"
            body.append(f"<tr><th class='method'>{html.escape(m)}</th>"
                        f"<td colspan='{len(COLS)}' style='text-align:center;color:#933'>"
                        f"not operable: {note}</td></tr>")
    table = (f"<table class='rank'><thead><tr><th class='method'>method</th>{head}</tr></thead>"
             f"<tbody>{''.join(body)}</tbody></table>")

    return (
        "<h2 id='addplot'>Supplement — addplot / out-of-sample test (operational criterion)</h2>"
        "<p class='blurb'><b>The monitoring scenario:</b> the basemap is fitted on NORMAL data "
        "only (no anomaly ever seen at fit time) and new points arrive afterwards, one at a "
        "time. The added set holds <b>cluster-anchored anomalies</b> — each shares a normal "
        "cluster's profile in the measured features and deviates 3 Rg along new dimensions "
        "orthogonal to everything the normal data varies in (a near-duplicate pair per cluster, "
        "5 clusters × 2), the realistic shape of a fault: a known operating state plus an effect "
        "the historical data never showed — plus 50 fresh normal points as controls. Each method "
        "maps them with its own out-of-sample operation (PCA/Isomap: <code>transform</code>; "
        "UMAP: seeded <code>transform</code>; DREAMS: openTSNE's partial-optimization "
        "<code>transform</code> onto the fixed DREAMS basemap — its regularization term acts only "
        "at fit time; toorPIA: server-side "
        "<code>addplot_embedding</code> on the fitted basemap). "
        "<b>Two questions, in order: detection</b> — does the anomaly land visibly outside the "
        "normal region at all (distance from the map centroid over the bulk's median radius)? — "
        "and <b>attribution</b> — is its direction from the centroid closest to its own source "
        "cluster's direction? The direction of an addplot point is information: it should say "
        "WHICH normal condition the anomaly departed from. The ambient high-D features resolve "
        "attribution 10/10 (the anchor signal survives SNR=1 noise), so a faithful map can too. "
        "<b>t-SNE (sklearn), PyMDE, and PCC expose no out-of-sample operation at all</b> — for "
        "monitoring that is itself the finding: adding data means re-fitting, and a re-fit "
        "re-arranges the map.</p>"
        + table +
        "<figure class='guidefig'>" + img("outliers", "basemap_addplot_gallery.png", embed) +
        "<figcaption>Anomaly-free basemap (normal clusters, faint colors) + added points "
        "(▲ = cluster-anchored anomalies colored by SOURCE cluster, · = added normal controls). "
        "Faithful = each ▲ outside the normal region AND in its own cluster's direction, pair "
        "members adjacent; dots inside the bulk.</figcaption></figure>"
        "<p class='note'><b>Honest notes.</b> (1) toorPIA's <code>addplot_embedding</code> targets "
        "the fitted basemap's server-side state, so this test performs a live "
        "<code>basemap_embedding(l2_normalization=False)</code> + one <code>addplot_embedding</code> "
        "call per added point (the monitoring semantics: points arrive one at a time) and commits "
        "the two coordinate sets as a self-consistent cache pair (<code>embedding_basemap</code> / "
        "<code>embedding_addplot</code>); the addplot inherits the basemap's preprocessing "
        "server-side, so basemap and added points are guaranteed identical treatment. "
        "(2) PCA/Isomap/DREAMS transforms and toorPIA are deterministic; "
        "UMAP's is seeded. DREAMS's added points are likewise mapped one per call. "
        "(3) A re-fit-based alternative for the methods without an out-of-sample "
        "operation (append the new data, re-fit, Procrustes-align, measure displacement) is future "
        "work — it measures a different, weaker property (map stability under re-fit), not the "
        "monitoring operation itself.</p>")


def build(embed: bool) -> str:
    agg, stab = load()
    css = """
    :root{--fg:#1a1a1a;--mut:#666;--line:#ddd;--accent:#0b6;--bg:#fff;--best:#d8f5e3}
    *{box-sizing:border-box} body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      color:var(--fg);margin:0;background:#fafafa;line-height:1.5}
    .wrap{max-width:1180px;margin:0 auto;padding:24px;background:var(--bg)}
    h1{font-size:26px;margin:0 0 4px} h2{margin-top:40px;border-bottom:2px solid var(--line);padding-bottom:6px}
    h3{margin-top:28px} .sub{color:var(--mut);font-size:14px}
    nav{position:sticky;top:0;background:#fffd;backdrop-filter:blur(4px);border-bottom:1px solid var(--line);
      padding:10px 0;margin-bottom:8px;z-index:9;font-size:14px}
    nav a{margin-right:14px;color:var(--accent);text-decoration:none} nav a:hover{text-decoration:underline}
    table.rank{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 4px}
    table.rank th,table.rank td{border:1px solid var(--line);padding:5px 8px;text-align:right}
    table.rank thead th{background:#f2f2f2}
    th.grp{font-size:12px;padding:6px 8px;text-align:center}
    th.grp-primary{background:#e3f6ec;color:#064;border-bottom:2px solid #0b6}
    th.grp-ref{background:#fdeee2;color:#a33;border-bottom:2px solid #e08a4a}
    th.sub-primary{background:#eef9f2} th.sub-ref{background:#fdf3ea;color:#a33}
    td.refcol{color:#7a7a7a;background:#fcfbfa} td.refcol.best{color:#064}
    td.worst{background:#fcd9d6} td.refcol.worst{background:#fcd9d6;color:#933} td.total{font-weight:700}
    td.crit{background:#cf3b30;color:#fff;font-weight:700} td.crit .ci{color:#ffd9d4}
    th.grp-rank{background:#e8eefb;color:#234a9e;border-bottom:2px solid #6a8cff} th.sub-rank{background:#eef2fc}
    th.grp-crush{background:#fdeecb;color:#8a5a00;border-bottom:2px solid #e0a000} th.sub-crush{background:#fff7e6}
    th.grp-osp{background:#f3e8fd;color:#5a2d8a;border-bottom:2px solid #9467bd} th.sub-osp{background:#f8f2fd}
    table.rank th.method{text-align:left;background:#f7f7f7} td.best{background:var(--best);font-weight:600}
    .ci{color:var(--mut);font-size:11px}
    .legend{font-size:13px;margin:8px 0 2px;color:#444}
    .legend .pill{display:inline-block;border-radius:10px;padding:1px 9px;margin-right:8px;font-size:12px}
    .pill-primary{background:#e3f6ec;color:#064} .pill-ref{background:#fdeee2;color:#a33}
    .figgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:14px 0}
    .figgrid.mapfig{grid-template-columns:1fr}  /* the map gallery spans the full width (2x) */
    figure{margin:0;border:1px solid var(--line);border-radius:8px;padding:8px;background:#fff}
    figure img{width:100%;display:block;border-radius:4px} figcaption{font-size:12px;color:var(--mut);margin-top:6px}
    .blurb{background:#f4f8ff;border-left:4px solid #6aa6ff;padding:10px 14px;border-radius:4px;font-size:14px}
    .note{background:#fff8f0;border-left:4px solid #f0a; padding:10px 14px;border-radius:4px;font-size:14px}
    .snr-tag{display:inline-block;background:#eee;border-radius:12px;padding:1px 10px;font-size:12px;color:#333}
    code{background:#f0f0f0;padding:1px 5px;border-radius:3px;font-size:90%}
    footer{margin-top:48px;color:var(--mut);font-size:13px;border-top:1px solid var(--line);padding-top:12px}
    .missing{color:#b00;font-size:12px;padding:30px;text-align:center;border:1px dashed #b00}
    dl.gloss{font-size:13px;margin:6px 0 2px} dl.gloss dt{font-weight:700;color:#234a9e;margin-top:7px}
    dl.gloss dd{margin:0 0 2px 16px;color:#333}
    figure.guidefig{margin:14px 0;max-width:1000px}
    """
    nav = ("<nav><b>REPORT</b> &nbsp; <a href='#guide'>reading guide</a> &nbsp; "
           + " ".join(f"<a href='#{d}'>{DATASET_NAV.get(d, d)}</a>" for d in DATASETS)
           + " <a href='#random-walk'>time series</a>"
           + " <a href='#stability'>stability</a> <a href='#notes'>notes</a>"
           + " <a href='#addplot'>addplot</a> <a href='#noise-dims'>noise-dims</a></nav>")

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>DR Fidelity Benchmark — Report</title>",
             f"<style>{css}</style></head><body><div class='wrap'>", nav]

    # header
    parts.append(
        "<h1>DR Fidelity Benchmark — Results Report</h1>"
        "<div class='sub'>Distance-preservation focus · 5 datasets (D=768 · N=1000 · "
        "<b>SNR=1</b>, realistic additive noise) + a time-series dataset (multi-series random "
        "walks: D=50 · N=3000 · clean) · seeds: R=3 (stochastic methods; PCA / Isomap / "
        "DREAMS / toorPIA are deterministic; transition — committed tables) · CPU/1-thread · reproducible · "
        "<a href='https://github.com/toorpia/dr-fidelity-benchmark'>code &amp; data on GitHub</a>.</div>"
        "<p class='blurb'><b>Thesis.</b> <b>Shepard ρ (full, p=100)</b> reflects <b>global-structure</b> "
        "reproduction. But in high dimensions distances concentrate, so most pairs sit at large "
        "distances and the full ρ is dominated by far pairs — the accuracy of <b>near</b> distances is "
        "buried. We therefore restrict ρ to the <b>near band</b>, defined structure-adaptively: the "
        "pairwise-distance profile of structured data is multimodal — its <b>first mode</b> is the "
        "within-structure pairs — and the near band is all pairs up to the density valley where "
        "that first mode decays into the tail (pX = the lowest X% of all pairwise distances; "
        "the boundary lands at p14–p20 on these datasets; on the "
        "clustered datasets the boundary coincides exactly with the true within-cluster pair "
        "fraction). It isolates <b>near-neighbor descriptive power</b>. Crucially the band uses a "
        "<b>fixed absolute radius for every point</b>, so it is fair. <b>recall@k</b> instead takes "
        "each point's own k nearest (a <b>variable radius</b>): it over-penalizes near-ties in dense "
        "regions and is <b>favorable to k-NN-based methods (t-SNE/UMAP)</b> — it is not a fair "
        "neighborhood evaluation, and is shown only as a reference. Best value per column is "
        "highlighted; bracketed ranges are bootstrap 95% CIs over seeds (deterministic methods show a "
        "point value).</p>"
        "<p class='legend'><span class='pill pill-primary'>PRIMARY · band-Shepard (fixed-radius, fair)</span>"
        "<span class='pill pill-ref'>REFERENCE · recall@k etc. (variable-radius k-NN, biased)</span>"
        "&nbsp; first-mode near band → p=100 global. The two groups are colored accordingly in every table below.</p>"
        "<p class='blurb'><b>Headline finding.</b> <b>PCC crushes dense clusters to points.</b> It "
        "over-compresses the tightest cluster's scale by ≈93× (density) and ≈9× (clusters) relative to the "
        "truth (≈1×), destroying each cluster's internal structure — even though its global ρ (full) "
        "stays <b>high</b> (it is the highest only on density; <b>toorPIA</b> leads the global ρ on "
        "clusters and transition). toorPIA preserves the within-cluster scale (≈0.4–0.5×) across all "
        "three datasets. This scale collapse is invisible to the rank-based Shepard ρ (which is "
        "scale-invariant) but is obvious in the Shepard density plots below and is quantified by the "
        "tightest-cluster scale column. <b>DREAMS</b> (t-SNE regularized toward the PCA layout) "
        "<b>takes the first-mode near-band ρ on all five datasets</b> — ahead of t-SNE on every "
        "one — but it leads the global ρ nowhere, and the "
        "membership-restricted questions expose the same silent failures as the neighbor-graph "
        "family: its outlier ρ is 0.055 (anomalies drawn amid the bulk) and its cross-population ρ "
        "is ≈0.07 (the minority placed with no distance information). <b>toorPIA tops the composite "
        "(near + global) ranking on density and clusters; on transition it leads the global ρ but "
        "drops to joint 3rd on the composite with DREAMS</b> (PCC 1st, t-SNE 2nd — toorPIA's "
        "first-mode near ρ trails there; the transition gallery shows the qualitative geometry this "
        "trade-off hides: toorPIA alone renders the seven dense states AND their closed-ring "
        "connectivity at once — the bridge bottleneck-gap diagnostic below makes that quantitative: "
        "toorPIA's largest bridge void is 14% of the inter-cluster distance, while t-SNE / UMAP / "
        "DREAMS tear 4 / 4 / 3 of the seven bridges). The fourth dataset "
        "(<a href='#outliers'>outliers</a>) asks a "
        "different, single-point question — whether one far-away point stays separated — scored by "
        "the standard Shepard/stress blocks plus a plain-geometry pair-angle column. There, "
        "<b>toorPIA keeps same-kind anomaly pairs co-directional</b> (angle ≤ 10°) with three kinds "
        "at three separate places, and in the <a href='#addplot'>addplot</a> out-of-sample test "
        "(monitoring on an anomaly-free basemap) it is <b>the only method that plots a never-seen "
        "anomaly outside the normal region at all — and its direction points back to the source "
        "cluster</b>; UMAP preserves pairs best in raw terms but attaches them to bulk-cluster "
        "edges; t-SNE fuses each pair into one point; <b>PCC tears same-kind pairs apart</b> "
        "(cohesion ≈650, angles ≈86°); PCA/Isomap send pair members to opposite sides; DREAMS is "
        "operable out-of-sample but lands its anomalies at the bulk's edge (radius ratio ≈1.0 — "
        "perfect direction, no alarm); and t-SNE / PyMDE / PCC cannot perform the add-data "
        "operation at all. On the "
        "<a href='#random-walk'>time-series dataset</a> (multi-series high-D random walks — "
        "the true shape is a star of jagged, mutually orthogonal spokes) <b>toorPIA leads "
        "every trajectory readout</b> (full ρ 0.924, within-series 0.970, cross-series "
        "0.897); DREAMS is second across the board (0.846 / 0.953 / 0.778) but draws the "
        "spokes as smooth ribbons, while the neighbor-graph methods keep only the step-level "
        "time order (t-SNE 0.951, DREAMS 0.952 there) and lose the star arrangement entirely "
        "(cross-series ρ 0.366 / 0.477).</p>")

    # ---- reading guide: what we measure + the two figure-backed methodology explanations ----
    lab1 = snr_label(SNRS[0])
    parts.append(
        "<h2 id='guide'>Reading guide — what we measure, and why not recall@k</h2>"
        "<p class='blurb'><b>New here?</b> This benchmark asks one question: when a "
        "dimensionality-reduction (DR) method squashes high-dimensional data into a 2-D picture, how "
        "faithfully does it keep the original <b>distances</b> between points? We run eight methods on "
        "synthetic data whose true geometry is known, so each method can be scored against the truth. A "
        "good method keeps <b>near</b> distances (fine, within-cluster structure) <i>and</i> <b>far</b> "
        "distances (the global layout). Below we explain how we measure that — and why the metric the DR "
        "literature usually reaches for is misleading — then show the results dataset by dataset.</p>"

        "<h3>Why the global (full, p=100) Shepard ρ hides near-neighbor structure</h3>"
        "<p>The classic global score is the <b>Shepard ρ</b>: the rank correlation between every pair's "
        "high-D distance and its 2-D distance. The catch is <b>distance concentration</b> — in high "
        "dimensions almost every pair of points sits at a similar mid-to-far distance, and only a thin "
        "sliver of pairs are genuinely close. In the plot below (the <code>clusters</code> dataset) the "
        "near band (green) is the profile's first mode — the within-cluster pairs, ~14% here — while the "
        "bulk of the ≈500,000 pairs piles up far away. Because the full ρ ranks <i>all</i> pairs "
        "together, near-distance errors are out-voted ~6:1 and "
        "averaged away. A method can crush every cluster to a blob yet still post a near-perfect "
        "full ρ. To actually see near-neighbor fidelity we restrict ρ to the <b>first-mode near "
        "band</b>: all pairs up to the density valley where the first mode of the distance profile "
        "decays into the tail, judged on one fixed distance radius.</p>"
        + figure_one("clusters", f"distance_distribution_snr{lab1}.png",
                     "clusters: most pairs are far (between-cluster); the first-mode near band (within-cluster "
                     "fine structure, here p14) is the green tail the full ρ averages away.", embed) +

        "<h3>Why recall@k / trustworthiness / continuity are a biased reference</h3>"
        "<p>The DR literature's usual <i>local</i> metric is <b>recall@k</b> (and its cousins "
        "trustworthiness and continuity): for each point, how many of its k high-D nearest neighbors are "
        "still neighbors in 2-D. Two structural choices make it an unfair test of distance fidelity. "
        "<b>(1) Variable radius</b> — with a fixed k, a point in a dense region encloses its k neighbors "
        "in a tiny radius while a point in a sparse region needs a much larger one (panel 1), so every "
        "point is judged on a different distance scale; on real data that radius genuinely varies "
        "point-to-point (panel 3). <b>(2) Hard 0/1 threshold</b> — the k-th and (k+1)-th neighbors can be "
        "almost equidistant, yet one counts fully and the other not at all (panel 2): a tiny coordinate "
        "wobble flips membership and the score jumps, even though the actual distances barely moved. Both "
        "choices make recall@k structurally favorable to neighbor-graph methods (t-SNE / UMAP) and a poor "
        "measure of faithful near-distance reproduction. The fixed-radius first-mode near-band Shepard ρ uses "
        "the <i>same</i> radius for every point and scores by actual distance, so it has neither problem. "
        "We keep recall@k only as a labelled <b>reference</b> column.</p>"
        + figure_one("density", f"recall_bias_snr{lab1}.png",
                     "density: recall@k judges each point on its own radius (panels 1 & 3) with a hard "
                     "in/out cutoff (panel 2); the fixed-radius band (green line) treats all points the "
                     "same.", embed) +

        "<h3>Key terms</h3>"
        "<dl class='gloss'>"
        "<dt>Dimensionality reduction (DR) / embedding</dt><dd>Mapping each high-D point to a 2-D "
        "coordinate for visualization; the 2-D output is the <i>embedding</i>.</dd>"
        "<dt>Fidelity</dt><dd>How well the 2-D distances reproduce the high-D distances.</dd>"
        "<dt>Ground truth — vs-truth / vs-ambient</dt><dd>Each dataset is built from a known clean "
        "geometry, then noise is added. <b>vs-truth</b> scores against the clean generating distances; "
        "<b>vs-ambient</b> scores against the noisy distances the method actually saw.</dd>"
        "<dt>Shepard ρ</dt><dd>Spearman rank correlation between high-D and 2-D pairwise distances "
        "(1 = perfect distance ordering).</dd>"
        "<dt>Distance band (p)</dt><dd>The pairs whose high-D distance is in the lowest <i>p</i>% of all "
        "pairs. The <b>near band</b> = the profile's first mode (up to its density valley; p14–p20 here); <b>p=100 (full)</b> = all pairs = the global number.</dd>"
        "<dt>Stress</dt><dd>Value-based distance error (lower = better); complements the rank-based "
        "Shepard ρ by catching distorted distance <i>values</i> (e.g. clusters crushed to points).</dd>"
        "<dt>Over-compression ×</dt><dd>How much a method shrinks the within-cluster scale vs the truth. "
        "≈1× = preserved; ≫1× = clusters crushed toward points.</dd>"
        "<dt>recall@k / trustworthiness / continuity</dt><dd>Neighbor-overlap metrics (variable radius, "
        "hard cutoff) — shown as a biased reference, not a fair near-distance metric (see above).</dd>"
        "<dt>Outlier ρ (anomaly-pair Shepard ρ)</dt><dd>The standard Shepard ρ restricted to the "
        "pairs where at least one endpoint is a ground-truth outlier — the same statistic as the "
        "global and band ρ, with the pair subset selected by endpoint membership instead of by "
        "distance percentile. It quantifies exactly the outlier-related blocks of the Shepard "
        "density figure and feeds the outliers dataset's third ranking column.</dd>"
        "<dt>SNR</dt><dd>Signal-to-noise ratio of the added noise; this report uses SNR=1 (realistic).</dd>"
        "<dt>Procrustes stability</dt><dd>Run-to-run wobble of a stochastic method's embedding after "
        "removing the rotation/scale/flip gauge; small = reproducible.</dd>"
        "</dl>"

        "<h3>How to read the ranking tables</h3>"
        "<p class='note'>Each per-dataset table has four column groups, left to right: <b>Ranking "
        "score</b> (composite points — for full ρ and for the first-mode near ρ the 1st→5th method scores 5→1 points; Σ "
        "is their sum and rows are sorted by Σ); immediately beside it the "
        "<span class='pill pill-primary'>PRIMARY</span> <b>band-Shepard ρ</b> block (fixed-radius, fair — "
        "full·global and near·first-mode) <i>that the ranking score is computed from</i>; then <b>Tightest-cluster "
        "scale</b> (the tight-cluster scale ×); and the <span class='pill pill-ref'>REFERENCE</span> "
        "<b>recall@k</b> block (variable-radius k-NN, biased — greyed out). In every column the "
        "<span style='background:#d8f5e3;padding:0 4px'><b>best</b></span> value is green and the "
        "<span style='background:#fcd9d6;padding:0 4px'><b>worst</b></span> is light red. A "
        "<span style='background:#cf3b30;color:#fff;padding:0 4px'><b>darker red</b></span> (regardless "
        "of rank) flags an outright failure: a <b>negative</b> near-band ρ, the table's <b>worst "
        "tightest-cluster crush when it exceeds 5×</b> (crushing destroys information — it cannot "
        "be read back from the map; inflation is legible, if exaggerated), or a <b>negative "
        "anomaly-pair ρ</b>. In the tightest-cluster column only the value closest to 1 is "
        "highlighted (light green); everything below the flag threshold stays uncolored. "
        "Note the composite ranking scores <b>distance "
        "fidelity only</b> (the PRIMARY ρ columns, plus the anomaly-pair ρ on the outliers "
        "dataset); the k-NN reference block is deliberately unscored — see the recall@k bias "
        "note. Bracketed ranges "
        "are bootstrap 95% CIs over seeds (deterministic methods show a single value). The "
        "<b>outliers</b> dataset adds a purple <b>outlier ρ</b> column — the standard Shepard ρ "
        "restricted to the anomaly-involving pairs — and a third ranking column scored on it "
        "(1st→5 … 5th→1), so the composite Σ there is full + near + outlier.</p>")

    # per-dataset
    for d in DATASETS:
        parts.append(f"<h2 id='{d}'>{DATASET_TITLES.get(d, d)}</h2>")
        parts.append(f"<p class='blurb'>{html.escape(DATASET_BLURB[d])}</p>")
        for snr in SNRS:
            parts.append(f"<h3>SNR = <span class='snr-tag'>{snr_label(snr)}</span></h3>")
            map_fig, rest_figs = figure_block(d, snr, embed)
            parts.append(map_fig)          # the map is the primary output -- shown before the table
            parts.append(ranking_table(agg, d, snr))
            parts.append(rest_figs)

    # the time-series dataset (multi-series random walks) -- same footing, own metric split
    parts.append(random_walk_section(embed))

    # stability
    parts.append("<h2 id='stability'>Run-to-run stability (Procrustes)</h2>")
    parts.append("<p class='sub'>After removing the rotation/reflection/translation/scale gauge: "
                 "per-point positional dispersion and the std of headline fidelity metrics across seeds. "
                 "Small dispersion with ~0 metric-std means coordinates may wobble but structural "
                 "fidelity is stable.</p>")
    for d in DATASETS:
        parts.append(f"<h3>{DATASET_NAV.get(d, d)}</h3>")
        parts.append(stability_table(stab, d))

    # notes
    parts.append(
        "<h2 id='notes'>Methodology notes</h2>"
        "<p class='note'><b>Why bands (and how the near band is defined).</b> Shepard ρ over "
        "<i>all</i> pairs measures global-structure reproduction, but high-dimensional distance "
        "concentration packs most pairs into a narrow far band, so the full ρ is dominated by far "
        "pairs and near-distance accuracy is buried. The fixed percentile cutoffs (the band at p "
        "holds the lowest p% of all pairs; p = 5…100 in the CSVs and the profile figures) expose "
        "the near→far profile; the HEADLINE near band is "
        "<b>structure-adaptive</b>: the pairwise-distance profile of structured data is multimodal "
        "— its first mode is the within-structure pairs — and the band is all pairs up to the "
        "density valley where that first mode decays into the tail. Estimator "
        "(<code>metrics.distances.first_mode_threshold</code>; deterministic, all pairs, constants "
        "disclosed): histogram over 256 equal-width bins, smoothed twice with a length-9 boxcar; "
        "the boundary is the profile minimum between the first two local maxima (order 4; a mode "
        "must reach ≥5% of the profile maximum, and the valley must dip below 95% of the first "
        "mode — tail-noise bumps are not modes), required to lie below the median distance; if no "
        "second mode exists the band falls back "
        "to the 5th-percentile radius (flagged in the CSVs; never triggered on these datasets). At SNR=1 the "
        "boundary lands at p20.5 (density), p14.2 (clusters), p14.9 (transition), p19.7 "
        "(outliers), p18.0 (populations) — on the three clustered datasets this coincides exactly "
        "with the true within-cluster pair fraction (14.2% / 19.7% / 18.0%), i.e. the data-driven "
        "band recovers the ground-truth notion of \"near\" without using labels. The detected "
        "percentile and fallback flag are recorded per row (<code>near_band_pct</code>, "
        "<code>near_band_fallback</code>).</p>"
        "<p class='note'><b>Fixed-radius (fair) vs variable-radius (biased).</b> The band uses one "
        "absolute distance threshold for the whole dataset — every point is judged on the same radius. "
        "<b>recall@k / trustworthiness / continuity</b> instead take each point's own k nearest "
        "(a per-point variable radius) with a hard inclusion threshold; they over-penalize near-ties in "
        "dense regions and are structurally favorable to k-NN-based methods (t-SNE/UMAP). They are "
        "<b>not a correct neighborhood evaluation</b> and are reported as a biased reference only. "
        "(A per-point band-Shepard variant exists for contrast, but it re-introduces the variable "
        "radius, so the fixed-radius global band is primary.)</p>"
        "<p class='note'><b>Density-weighting caveat.</b> On non-uniform-density data the global "
        "first-mode band is weighted toward dense regions, so the near ρ emphasizes near-structure where "
        "pairs are dense. A per-point-uniform view is the (variable-radius) per-point variant — the "
        "trade-off is point-uniformity vs fixed-radius fairness.</p>"
        "<p class='note'><b>Non-circularity.</b> PCC is run with the Pearson (value) loss while the "
        "primary metric is Spearman (rank) Shepard ρ — optimizing Pearson yet scoring on the rank "
        "metric is the honest, non-circular outcome. DREAMS optimizes the t-SNE KL plus a "
        "coordinate-level MSE pull toward the PCA layout — neither term is a quantity this "
        "benchmark scores.</p>"
        "<p class='note'><b>PCC crushes dense clusters (scale collapse).</b> PCC squeezes each dense "
        "cluster to a near-point in 2-D (tightest-cluster over-compression ≫1) while keeping high global "
        "ρ. Mechanistically this follows from its published objective (arXiv:2503.07609), which "
        "optimizes only distances to a sampled set of reference points, leaving non-reference points "
        "free to overlap; methods that constrain all pairwise distances keep the local scale. Note the "
        "rank-based Shepard ρ is scale-invariant and largely misses this — the tightest-cluster scale metric "
        "and the Shepard density plots are what reveal it.</p>"
        "<p class='note'><b>Outliers dataset — how it is scored.</b> Anchored in the standard, "
        "established readings only: the Shepard density figure (high-D vs 2-D pairwise distance — "
        "the outlier-related pair blocks and their violations are directly visible), the standard "
        "band-Shepard ρ / stress blocks over exact pairwise distances, and one plain-geometry "
        "column (the same-kind pair angle, truth 0°) plus the addplot kind-assignment accuracy "
        "below — both direct readouts of the embedding pictures, with no bespoke normalization. "
        "An earlier bespoke ratio metric (OSP) computed for this dataset was <b>retired from all "
        "reporting</b>: it is not an established metric and its double normalization did not track "
        "the Shepard/embedding pictures; its raw columns remain in the per-run CSVs for the "
        "record.</p>"
        "<p class='note'><b>toorPIA.</b> Called via the embedding endpoint "
        "<code>basemap_embedding(X, l2_normalization=False)</code> (toorpia 1.2.0): no per-column "
        "normalization, no centering, and per-row L2 normalization disabled, so it embeds the very "
        "same raw vectors the other methods see and the plain-Euclidean metrics measure "
        "(preprocessing aligned with the evaluation basis). The endpoint exposes no random seed and "
        "is deterministic up to server jitter (measured run-to-run mean |Δ| ≈ 1e-3 of the map "
        "scale), so toorPIA is treated as a deterministic method: point values without CI brackets, "
        "and no run-to-run stability row. Coordinates are cached/committed for offline reproduction "
        "(no API key needed).</p>"
        "<p class='note'><b>DREAMS.</b> t-SNE regularized toward the PCA embedding (Kury, Kobak &amp; "
        "Damrich, TMLR 2026, arXiv:2508.13747), run at the authors' published defaults: the "
        "2-component PCA embedding as both initialization and regularization target, "
        "reg_lambda=0.15, perplexity 30, Barnes–Hut, single-threaded (see "
        "<code>methods/dreams_method.py</code>). Installed from the authors' openTSNE fork "
        "(<code>pip install \"git+https://github.com/berenslab/DREAMS.git@tp\"</code>, replaces "
        "vanilla openTSNE); the fork's current head has a bug that crashes every plain "
        "<code>fit()</code> (a dead <code>\"X\"</code> kwarg leaks into the optimizer), patched "
        "at import time in the wrapper so the installed package stays exactly the published "
        "source. With the fixed PCA initialization and single-threaded gradient descent the "
        "optimizer has no remaining randomness — three seeds produce byte-identical embeddings — "
        "so DREAMS is treated as deterministic: point values, no CI brackets, no stability "
        "row.</p>"
        "<p class='note'><b>Baseline hyperparameters.</b> The open-source methods run at library "
        "defaults (t-SNE perplexity=30, UMAP n_neighbors=15, DREAMS reg_lambda=0.15; toorPIA has "
        "no placement knob), so "
        "the standing objection is \"these methods would rank differently if tuned\". "
        "<code>run/hyperparam_sensitivity.py</code> sweeps each method's placement-critical knob "
        "(t-SNE/UMAP neighborhood size over 5–100; DREAMS local–global regularization strength "
        "over 0.05–0.5) on density and clusters (SNR=1, R=3): raising t-SNE's perplexity moves BOTH "
        "ranked columns materially. On density, perplexity=100 lifts its full ρ from 0.30 to 0.60 "
        "(bottom tier → mid-pack, still below PCA 0.70 / toorPIA 0.82 / PCC 0.82) and its "
        "first-mode near ρ from 0.26 to 0.41 — at perplexity ≥ 50 t-SNE overtakes toorPIA's "
        "density near ρ (0.31); on clusters its near ρ rises 0.39 → 0.54 (overtaking DREAMS's "
        "0.47) while its full ρ stays flat at 0.37. Re-scored with those tuned values the "
        "composite still goes to toorPIA on clusters (8 vs t-SNE 5), but on density tuned t-SNE "
        "draws LEVEL with toorPIA (7 vs 7) — and the same settings erode t-SNE's own reference "
        "metric (recall@15: "
        "0.254 → 0.230 on density, 0.270 → 0.208 on clusters) — the global-vs-local trade-off is "
        "intrinsic to the method, not an artifact of the default. UMAP's gains are small "
        "everywhere (full ρ 0.16 → 0.26 on density, still last tier; near and recall "
        "flat-to-declining beyond n_neighbors ≈ 15–30). DREAMS's reg_lambda is a genuine "
        "local–global dial on density — its full ρ climbs 0.39 → 0.62 as λ rises 0.05 → 0.5 "
        "(still below toorPIA 0.82) while its near ρ stays 0.36–0.41 and keeps the near-band "
        "lead at every λ; on clusters both columns are flat (full 0.41–0.43, near 0.46–0.47). "
        "No λ changes any composite leader. Defaults therefore decide no leadership outright — "
        "but tuned t-SNE's density standing (near-band lead, composite tie) is a genuine "
        "sensitivity, disclosed here. Full tables: "
        "<code>results/hyperparam_sensitivity_*.csv</code>.</p>"
        "<figure class='guidefig'>" + img("hyperparam", "sensitivity_snr1.png", embed) +
        "<figcaption>Baseline-hyperparameter sensitivity (SNR=1, median + bootstrap 95% CI over "
        "3 seeds; library default dashed). Green/blue = the two ranked ρ columns; red = the "
        "k-NN-favorable reference metric the neighbor methods lead. Raising t-SNE's perplexity "
        "buys global and near-band ρ at the cost of its own recall@15 (drawing level with "
        "toorPIA's density composite at perplexity 100); DREAMS's λ trades its density full ρ "
        "up without losing its near-band lead; no setting changes a composite leader "
        "outright.</figcaption></figure>"
        "<p class='note'><b>Scope.</b> This characterizes distance/structure preservation on "
        "synthetic known-structure data; it is not a claim about downstream-task superiority. When "
        "CIs overlap, no strict winner is asserted.</p>")

    parts.append(addplot_section(embed))
    parts.append(noise_dims_section(embed))

    parts.append(
        "<footer>Generated by <code>run/make_report.py</code> from "
        "<code>results/metrics_aggregated.csv</code> + <code>results/stability.csv</code> "
        "(+ <code>results/dimsweep_aggregated.csv</code> for the noise-dims supplement; reproduce "
        "it with <code>python run/dimsweep.py --dims 6 10 20 40 80 100 200 400 768 --methods all "
        "--seeds 3 --n 1000</code> plus <code>--dims 1500 2000 --methods toorPIA "
        "--data-seed 42 0 1 2 3</code>). "
        "This page shows <b>SNR=1</b> (realistic additive noise). Reproduce the full SNR sweep with "
        "<code>python run/benchmark.py --dataset all --methods all --seeds 3 --dim 768 --n 1000 "
        "--snr inf 4 1</code> (or just the reported level with <code>--snr 1</code>), then rebuild this "
        "page with <code>python run/make_report.py</code>. Code, data, and the full methodology "
        "(<code>README.md</code>) live in the "
        "<a href='https://github.com/toorpia/dr-fidelity-benchmark'>GitHub repository</a>.</footer>")
    parts.append("</div></body></html>")
    return "".join(parts)


class _HtmlToMd(HTMLParser):
    """Convert the report's generated HTML to GitHub-flavored Markdown.

    Single source of truth: the Markdown is derived from the exact HTML string, so both outputs
    always carry the same content. Only the tags this report emits are handled. Formatting that
    GFM cannot express is mapped: 2-row table headers (colspan groups) become an italic
    column-groups line above the table; the best/worst/critical cell highlighting becomes
    **bold** / *italic* / ⚠."""

    BLOCK_NOTE = ("*(Markdown rendering of `REPORT.html` — same content, generated together. "
                  "In tables, **bold** = best in column, *italic* = worst, ⚠ = outright "
                  "failure flag.)*\n\n"
                  "*A paper version (arXiv submission in preparation) lives under "
                  "[`paper/`](paper/); it is frozen at release tag **v1.2.0**, while this "
                  "report tracks the current committed results.*")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.buf = [], []
        self.skip_depth = 0          # inside <style>/<nav>/<title>
        self.table = None            # {'rows': [...], 'in_head': bool} while inside a table
        self.cell = None
        self.list_item = None        # 'dt' | 'dd' while inside a glossary item

    # ---- helpers ----
    def _flush(self, prefix=""):
        text = "".join(self.buf).strip()
        self.buf = []
        if text:
            if prefix == "> ":
                text = "> " + text
            self.out.append(text + "\n")

    # ---- tag events ----
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag in ("style", "nav", "title"):
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if self.table is not None:
            if tag == "tr":
                self.table["rows"].append({"head": self.table["in_head"], "cells": []})
            elif tag in ("th", "td"):
                self.cell = {"tag": tag, "cls": a.get("class", ""),
                             "colspan": int(a.get("colspan", 1)),
                             "rowspan": int(a.get("rowspan", 1)), "text": []}
                self.table["rows"][-1]["cells"].append(self.cell)
            elif tag == "thead":
                self.table["in_head"] = True
            return
        if tag == "table":
            self.table = {"rows": [], "in_head": False}
        elif tag in ("h1", "h2", "h3"):
            self._flush(); self.buf = []
            self.buf.append({"h1": "# ", "h2": "## ", "h3": "### "}[tag])
        elif tag == "p":
            self._flush()
            self._p_class = cls
        elif tag in ("b", "strong"):
            self.buf.append("**")
        elif tag == "i":
            self.buf.append("*")
        elif tag == "code":
            self.buf.append("`")
        elif tag == "span":
            self._spans = getattr(self, "_spans", [])
            self._spans.append(cls)
        elif tag == "a":
            self.buf.append("[")
            self._href = a.get("href", "")
        elif tag == "img":
            src = a.get("src", "")
            alt = a.get("alt", "")
            self.buf.append(f"![{alt}]({src})")
        elif tag == "figcaption":
            self._flush()
            self.buf.append("*")
        elif tag == "dt":
            self._flush()
            self.buf.append("- **")
        elif tag == "dd":
            self.buf.append(" — ")
        elif tag == "footer":
            self._flush()
            self.out.append("\n---\n\n")
        elif tag == "br":
            self.buf.append("  \n")

    def handle_endtag(self, tag):
        if tag in ("style", "nav", "title"):
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if self.table is not None:
            if tag in ("th", "td"):
                self.cell = None
            elif tag == "thead":
                self.table["in_head"] = False
            elif tag == "table":
                self._emit_table()
                self.table = None
            return
        if tag in ("h1", "h2", "h3"):
            self._flush()
            self.out.append("\n")
            if "".join(x if isinstance(x, str) else "" for x in self.out).startswith("# ") \
                    and len([l for l in self.out if l.startswith("#")]) == 1:
                self.out.append(self.BLOCK_NOTE + "\n\n")
        elif tag == "p":
            self._flush("> " if getattr(self, "_p_class", "") in ("blurb", "note") else "")
            self.out.append("\n")
        elif tag in ("b", "strong"):
            self.buf.append("**")
        elif tag == "i":
            self.buf.append("*")
        elif tag == "code":
            self.buf.append("`")
        elif tag == "span":
            spans = getattr(self, "_spans", [])
            if spans and "pill" in spans.pop():
                self.buf.append(" · ")     # pills sit flush in HTML; keep them separated in MD
        elif tag == "a":
            self.buf.append(f"]({self._href})")
        elif tag == "figcaption":
            self.buf.append("*")
            self._flush()
            self.out.append("\n")
        elif tag == "figure":
            self._flush()
            self.out.append("\n")
        elif tag == "dt":
            self.buf.append("**")
        elif tag == "dd":
            self._flush()
        elif tag in ("dl", "div"):
            self._flush()
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return
        if self.cell is not None:
            self.cell["text"].append(data)
        elif self.buf is not None:
            self.buf.append(data)

    # ---- table rendering ----
    def _emit_table(self):
        rows = self.table["rows"]
        head_rows = [r for r in rows if r["head"]]
        body_rows = [r for r in rows if not r["head"]]

        def cell_text(c):
            t = " ".join("".join(c["text"]).split())
            cls = c["cls"]
            if "crit" in cls.split():
                t = f"⚠ **{t}**"
            elif "best" in cls.split():
                t = f"**{t}**"
            elif "worst" in cls.split():
                t = f"*{t}*"
            return t.replace("|", "\\|")

        if len(head_rows) == 2:      # grouped header: emit the group line, then the sub header
            groups = [c for c in head_rows[0]["cells"] if c["rowspan"] == 1]
            gline = " · ".join(
                f"{' '.join(''.join(c['text']).split())} [{c['colspan']} col"
                f"{'s' if c['colspan'] > 1 else ''}]" for c in groups)
            self.out.append(f"*Column groups: {gline}*\n\n")
            header = ["method"] + [cell_text(c) for c in head_rows[1]["cells"]]
        else:
            header = [cell_text(c) for c in head_rows[0]["cells"]] if head_rows else []

        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join("---" for _ in header) + "|"]
        for r in body_rows:
            cells = []
            for c in r["cells"]:
                cells.append(cell_text(c))
                cells.extend([""] * (c["colspan"] - 1))   # GFM has no colspan: pad
            cells += [""] * (len(header) - len(cells))
            lines.append("| " + " | ".join(cells) + " |")
        self.out.append("\n".join(lines) + "\n\n")

    def markdown(self):
        self._flush()
        return "".join(self.out).replace("\n\n\n", "\n\n")


def html_to_md(html_text: str) -> str:
    conv = _HtmlToMd()
    conv.feed(html_text)
    return conv.markdown()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true",
                    help="base64-embed figures into a single portable HTML (larger file)")
    ap.add_argument("--out", default=str(ROOT / "REPORT.html"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    html_text = build(args.embed)
    out.write_text(html_text, encoding="utf-8")
    md_out = out.with_suffix(".md")
    md_out.write_text(html_to_md(html_text), encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"wrote {out}  ({size:.0f} KB, embed={args.embed})")
    print(f"wrote {md_out}  ({md_out.stat().st_size / 1024:.0f} KB, same content as Markdown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
