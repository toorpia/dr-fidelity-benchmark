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
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["density", "clusters", "transition"]
SNRS = [1.0]   # SNR=1 (noisy = realistic).
METHOD_ORDER = ["PCA", "Isomap", "PyMDE", "PCC", "t-SNE", "UMAP", "toorPIA"]

# Column groups. The band-Shepard family (fixed-radius, fair) is the PRIMARY block; the recall@k
# family (variable-radius k-NN, favorable to k-NN methods) is explicitly a REFERENCE block.
# Each entry: (metric key, header, higher-is-better).
TABLE_GROUPS = [
    ("rank", "Ranking score (1st→5 … 5th→1 pts)", [
        ("__pts_full", "full pts", True),
        ("__pts_p5", "p5 pts", True),
        ("__pts_total", "Σ", True),
    ]),
    ("primary", "Band-Shepard ρ — fixed-radius, fair (PRIMARY)", [
        ("full_shepard", "full · global", True),
        ("shepard_p5__vs_ambient", "p5 · near-neighbor", True),
    ]),
    ("crush", "Within-cluster scale  (×over-compression; ≫1 = clusters crushed to points)", [
        ("cluster_over_compression", "over-compression ×", False),
    ]),
    ("ref", "Reference — k-NN-favorable, biased (not a fair neighborhood test)", [
        ("recall_k15", "recall@15", True),
        ("trust_k15", "trust@15", True),
        ("cont_k15", "continuity@15", True),
    ]),
]
# metrics scored for the composite ranking: full Shepard ρ and near-band p5 Shepard ρ
RANK_KEYS = [("shepard_p5__vs_ambient", "__pts_p5"), ("full_shepard", "__pts_full")]

DATASET_BLURB = {
    "clusters": "Distinct dense sub-populations in a high-dimensional feature space (7 well-separated "
                "dense clusters). Tests whether a method keeps the fine within-cluster structure while "
                "also placing the clusters correctly.",
    "density": "Non-uniform density (uniform + tight core + sparse shell). Tests density distortion "
               "and the recall@k bias; distance-preservers should win the global/near Shepard bands.",
    "transition": "Continuous transition: dense clusters CONNECTED by bridge regions. The bridges (plus "
                  "SNR=1 noise) dilute the within-cluster over-compression here — PCC's clusters are "
                  "squeezed but not to points (≈79× when clean, milder under noise). Included to show "
                  "the effect is weaker when clusters are connected; toorPIA still preserves scale and "
                  "leads the global ρ.",
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
    """Composite ranking points: for full Shepard ρ AND near-band p5 Shepard ρ, the 1st..5th method
    gets 5,4,3,2,1 points (others 0); the total is their sum. Returns ``{method: {__pts_*: int}}``."""
    pts = {m: {"__pts_full": 0, "__pts_p5": 0} for m in methods}
    for metric_key, pts_key in RANK_KEYS:
        ranked = sorted(((m, cell(agg, dataset, snr, m, metric_key)) for m in methods),
                        key=lambda t: -(t[1][0] if t[1] else -1e9))
        for i, (m, c) in enumerate(ranked):
            if c is not None:
                pts[m][pts_key] = max(0, 5 - i)
    for m in methods:
        pts[m]["__pts_total"] = pts[m]["__pts_full"] + pts[m]["__pts_p5"]
    return pts


def ranking_table(agg, dataset, snr):
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

    keys = [k for _, _, items in TABLE_GROUPS for (k, _, _) in items]
    numeric = {m: {k: value(k, m)[0] for k in keys} for m in methods}
    disp = {m: {k: value(k, m)[1] for k in keys} for m in methods}

    # best / worst per column — direction depends on higher-is-better; only when the column varies
    hib = {k: h for _, _, items in TABLE_GROUPS for (k, _, h) in items}
    best, worst = {}, {}
    for k in keys:
        vs = [numeric[m][k] for m in methods if numeric[m][k] is not None]
        if vs and max(vs) != min(vs):
            if hib.get(k, True):
                best[k], worst[k] = max(vs), min(vs)
            else:
                best[k], worst[k] = min(vs), max(vs)

    # order rows by composite total (desc), tie-break by full Shepard ρ (desc)
    def order_key(m):
        c = cell(agg, dataset, snr, m, "full_shepard")
        return (-pts[m]["__pts_total"], -(c[0] if c else -1e9))
    methods = sorted(methods, key=order_key)

    grp_th = ["<th class='method' rowspan='2'>method</th>"]
    sub_th = []
    for kind, label, items in TABLE_GROUPS:
        grp_th.append(f"<th class='grp grp-{kind}' colspan='{len(items)}'>{html.escape(label)}</th>")
        for _, h, _ in items:
            sub_th.append(f"<th class='sub-{kind}'>{html.escape(h)}</th>")
    head = f"<tr>{''.join(grp_th)}</tr><tr>{''.join(sub_th)}</tr>"

    rows = []
    for m in methods:
        tds = []
        for kind, _, items in TABLE_GROUPS:
            for k, _, _ in items:
                n = numeric[m][k]
                cls = []
                # absolute-threshold reds (regardless of rank): a negative near-band p5 ρ, or an
                # over-compression above 2× — both flag an outright failure, not just last place.
                force_red = n is not None and (
                    (k == "shepard_p5__vs_ambient" and n < 0)
                    or (k == "cluster_over_compression" and n > 2))
                if force_red:
                    cls.append("crit")          # darker red than rank-worst — an outright failure
                elif n is not None and k in best:
                    if n == best[k]:
                        cls.append("best")
                    elif n == worst[k]:
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
         "High-D pairwise-distance profile — why the near band (p≤5) is a thin slice the full ρ buries"),
        (f"score_scatter_snr{lab}.png", "Near (p=5) vs global (full) Shepard ρ — top-right captures both"),
        (f"shepard_scatter_snr{lab}.png", "Shepard density (jet, log counts) — high-D vs 2-D distance"),
        (f"summary_heatmap_snr{lab}.png", "Summary heatmap (methods × metrics, median)"),
        (f"embeddings_snr{lab}.png", "2-D embeddings colored by the dataset variable"),
    ]
    items = "".join(
        f"<figure>{img(dataset, f, embed)}<figcaption>{html.escape(cap)}</figcaption></figure>"
        for f, cap in figs)
    return f"<div class='figgrid'>{items}</div>"


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
    table.rank th.method{text-align:left;background:#f7f7f7} td.best{background:var(--best);font-weight:600}
    .ci{color:var(--mut);font-size:11px}
    .legend{font-size:13px;margin:8px 0 2px;color:#444}
    .legend .pill{display:inline-block;border-radius:10px;padding:1px 9px;margin-right:8px;font-size:12px}
    .pill-primary{background:#e3f6ec;color:#064} .pill-ref{background:#fdeee2;color:#a33}
    .figgrid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:14px 0}
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
           + " ".join(f"<a href='#{d}'>{d}</a>" for d in DATASETS)
           + " <a href='#stability'>stability</a> <a href='#notes'>notes</a></nav>")

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>DR Fidelity Benchmark — Report</title>",
             f"<style>{css}</style></head><body><div class='wrap'>", nav]

    # header
    parts.append(
        "<h1>DR Fidelity Benchmark — Results Report</h1>"
        "<div class='sub'>Distance-preservation focus · 3 datasets · R=20 seeds · D=768 · N=1000 · "
        "<b>SNR=1</b> (realistic additive noise) · CPU/1-thread · reproducible.</div>"
        "<p class='blurb'><b>Thesis.</b> <b>Shepard ρ (full, p=100)</b> reflects <b>global-structure</b> "
        "reproduction. But in high dimensions distances concentrate, so most pairs sit at large "
        "distances and the full ρ is dominated by far pairs — the accuracy of <b>near</b> distances is "
        "buried. We therefore restrict ρ to cumulative distance bands; <b>p=5 = the globally-nearest "
        "5% of pairs</b> isolates <b>near-neighbor descriptive power</b>. Crucially the band uses a "
        "<b>fixed absolute radius for every point</b>, so it is fair. <b>recall@k</b> instead takes "
        "each point's own k nearest (a <b>variable radius</b>): it over-penalizes near-ties in dense "
        "regions and is <b>favorable to k-NN-based methods (t-SNE/UMAP)</b> — it is not a fair "
        "neighborhood evaluation, and is shown only as a reference. Best value per column is "
        "highlighted; bracketed ranges are bootstrap 95% CIs over seeds (deterministic methods show a "
        "point value).</p>"
        "<p class='legend'><span class='pill pill-primary'>PRIMARY · band-Shepard (fixed-radius, fair)</span>"
        "<span class='pill pill-ref'>REFERENCE · recall@k etc. (variable-radius k-NN, biased)</span>"
        "&nbsp; p=5 near → p=100 global. The two groups are colored accordingly in every table below.</p>"
        "<p class='blurb'><b>Headline finding.</b> <b>PCC crushes dense clusters to points.</b> It "
        "over-compresses the within-cluster scale by ≈50× (density) and ≈9× (clusters) relative to the "
        "truth (≈1×), destroying each cluster's internal structure — even though its global ρ (full) "
        "stays <b>high</b> (it is the highest only on density; <b>toorPIA</b> leads the global ρ on "
        "clusters and transition). toorPIA preserves the within-cluster scale (≈0.4–0.6×) across all "
        "three datasets. This scale collapse is invisible to the rank-based Shepard ρ (which is "
        "scale-invariant) but is obvious in the Shepard density plots below and is quantified by the "
        "over-compression column. <b>toorPIA tops the composite (near + global) ranking on all three "
        "datasets.</b></p>")

    # ---- reading guide: what we measure + the two figure-backed methodology explanations ----
    lab1 = snr_label(SNRS[0])
    parts.append(
        "<h2 id='guide'>Reading guide — what we measure, and why not recall@k</h2>"
        "<p class='blurb'><b>New here?</b> This benchmark asks one question: when a "
        "dimensionality-reduction (DR) method squashes high-dimensional data into a 2-D picture, how "
        "faithfully does it keep the original <b>distances</b> between points? We run seven methods on "
        "synthetic data whose true geometry is known, so each method can be scored against the truth. A "
        "good method keeps <b>near</b> distances (fine, within-cluster structure) <i>and</i> <b>far</b> "
        "distances (the global layout). Below we explain how we measure that — and why the metric the DR "
        "literature usually reaches for is misleading — then show the results dataset by dataset.</p>"

        "<h3>Why the global (full, p=100) Shepard ρ hides near-neighbor structure</h3>"
        "<p>The classic global score is the <b>Shepard ρ</b>: the rank correlation between every pair's "
        "high-D distance and its 2-D distance. The catch is <b>distance concentration</b> — in high "
        "dimensions almost every pair of points sits at a similar mid-to-far distance, and only a thin "
        "sliver of pairs are genuinely close. In the plot below (the <code>clusters</code> dataset) the "
        "near band (p≤5, green) is a small left-hand tail, while the bulk of the ≈500,000 pairs piles up "
        "far away. Because the full ρ ranks <i>all</i> pairs together and only ~5% are near, "
        "near-distance errors are out-voted about 19:1 and averaged away. A method can crush every "
        "cluster to a blob yet still post a near-perfect full ρ. To actually see near-neighbor fidelity "
        "we restrict ρ to the <b>near band (p=5)</b>: the nearest 5% of pairs, judged on one fixed "
        "distance radius.</p>"
        + figure_one("clusters", f"distance_distribution_snr{lab1}.png",
                     "clusters: most pairs are far (between-cluster); the near 5% (within-cluster fine "
                     "structure) is the green tail the full ρ averages away.", embed) +

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
        "measure of faithful near-distance reproduction. The fixed-radius near-band Shepard ρ (p=5) uses "
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
        "pairs. <b>p=5</b> = near-neighbor band; <b>p=100 (full)</b> = all pairs = the global number.</dd>"
        "<dt>Stress</dt><dd>Value-based distance error (lower = better); complements the rank-based "
        "Shepard ρ by catching distorted distance <i>values</i> (e.g. clusters crushed to points).</dd>"
        "<dt>Over-compression ×</dt><dd>How much a method shrinks the within-cluster scale vs the truth. "
        "≈1× = preserved; ≫1× = clusters crushed toward points.</dd>"
        "<dt>recall@k / trustworthiness / continuity</dt><dd>Neighbor-overlap metrics (variable radius, "
        "hard cutoff) — shown as a biased reference, not a fair near-distance metric (see above).</dd>"
        "<dt>SNR</dt><dd>Signal-to-noise ratio of the added noise; this report uses SNR=1 (realistic).</dd>"
        "<dt>Procrustes stability</dt><dd>Run-to-run wobble of a stochastic method's embedding after "
        "removing the rotation/scale/flip gauge; small = reproducible.</dd>"
        "</dl>"

        "<h3>How to read the ranking tables</h3>"
        "<p class='note'>Each per-dataset table has four column groups, left to right: <b>Ranking "
        "score</b> (composite points — for full ρ and for p5 ρ the 1st→5th method scores 5→1 points; Σ "
        "is their sum and rows are sorted by Σ); immediately beside it the "
        "<span class='pill pill-primary'>PRIMARY</span> <b>band-Shepard ρ</b> block (fixed-radius, fair — "
        "full·global and p5·near) <i>that the ranking score is computed from</i>; then <b>Within-cluster "
        "scale</b> (the over-compression ×); and the <span class='pill pill-ref'>REFERENCE</span> "
        "<b>recall@k</b> block (variable-radius k-NN, biased — greyed out). In every column the "
        "<span style='background:#d8f5e3;padding:0 4px'><b>best</b></span> value is green and the "
        "<span style='background:#fcd9d6;padding:0 4px'><b>worst</b></span> is light red. A "
        "<span style='background:#cf3b30;color:#fff;padding:0 4px'><b>darker red</b></span> (regardless "
        "of rank) flags an outright failure: a <b>negative</b> near-band p5 ρ, or an "
        "<b>over-compression &gt; 2×</b>. Bracketed ranges are bootstrap 95% CIs over seeds "
        "(deterministic methods show a single value).</p>")

    # per-dataset
    for d in DATASETS:
        parts.append(f"<h2 id='{d}'>{d}</h2>")
        parts.append(f"<p class='blurb'>{html.escape(DATASET_BLURB[d])}</p>")
        for snr in SNRS:
            parts.append(f"<h3>SNR = <span class='snr-tag'>{snr_label(snr)}</span></h3>")
            parts.append(ranking_table(agg, d, snr))
            parts.append(figure_block(d, snr, embed))

    # stability
    parts.append("<h2 id='stability'>Run-to-run stability (Procrustes)</h2>")
    parts.append("<p class='sub'>After removing the rotation/reflection/translation/scale gauge: "
                 "per-point positional dispersion and the std of headline fidelity metrics across seeds. "
                 "Small dispersion with ~0 metric-std means coordinates may wobble but structural "
                 "fidelity is stable.</p>")
    for d in DATASETS:
        parts.append(f"<h3>{d}</h3>")
        parts.append(stability_table(stab, d))

    # notes
    parts.append(
        "<h2 id='notes'>Methodology notes</h2>"
        "<p class='note'><b>Why bands (and why p=5 is the near-neighbor metric).</b> Shepard ρ over "
        "<i>all</i> pairs measures global-structure reproduction, but high-dimensional distance "
        "concentration packs most pairs into a narrow far band, so the full ρ is dominated by far "
        "pairs and near-distance accuracy is buried. Cumulative cutoffs expose the near→far profile; "
        "<b>p=5</b> (globally-nearest 5% of pairs) is the near-neighbor descriptive-power measure that "
        "the full ρ hides.</p>"
        "<p class='note'><b>Fixed-radius (fair) vs variable-radius (biased).</b> The band uses one "
        "absolute distance threshold for the whole dataset — every point is judged on the same radius. "
        "<b>recall@k / trustworthiness / continuity</b> instead take each point's own k nearest "
        "(a per-point variable radius) with a hard inclusion threshold; they over-penalize near-ties in "
        "dense regions and are structurally favorable to k-NN-based methods (t-SNE/UMAP). They are "
        "<b>not a correct neighborhood evaluation</b> and are reported as a biased reference only. "
        "(A per-point band-Shepard variant exists for contrast, but it re-introduces the variable "
        "radius, so the fixed-radius global band is primary.)</p>"
        "<p class='note'><b>Density-weighting caveat.</b> On non-uniform-density data the globally "
        "nearest 5% of pairs is weighted toward dense regions, so p=5 emphasizes near-structure where "
        "pairs are dense. A per-point-uniform view is the (variable-radius) per-point variant — the "
        "trade-off is point-uniformity vs fixed-radius fairness.</p>"
        "<p class='note'><b>Non-circularity.</b> PCC is run with the Pearson (value) loss while the "
        "primary metric is Spearman (rank) Shepard ρ — optimizing Pearson yet scoring on the rank "
        "metric is the honest, non-circular outcome.</p>"
        "<p class='note'><b>PCC crushes dense clusters (scale collapse).</b> PCC squeezes each dense "
        "cluster to a near-point in 2-D (within-cluster over-compression ≫1) while keeping high global "
        "ρ. Mechanistically this follows from its published objective (arXiv:2503.07609), which "
        "optimizes only distances to a sampled set of reference points, leaving non-reference points "
        "free to overlap; methods that constrain all pairwise distances keep the local scale. Note the "
        "rank-based Shepard ρ is scale-invariant and largely misses this — the over-compression metric "
        "and the Shepard density plots are what reveal it.</p>"
        "<p class='note'><b>toorPIA.</b> Called <code>fit_transform(..., vector_normalization=False, "
        "random_seed=seed)</code> so it embeds the same vectors the other methods see; coordinates are "
        "cached/committed for offline reproduction (no API key needed). The installed toorpia 1.1.1 "
        "does expose a seed (vs the original spec).</p>"
        "<p class='note'><b>Scope.</b> This characterizes distance/structure preservation on "
        "synthetic known-structure data; it is not a claim about downstream-task superiority. When "
        "CIs overlap, no strict winner is asserted.</p>")

    parts.append(
        "<footer>Generated by <code>run/make_report.py</code> from "
        "<code>results/metrics_aggregated.csv</code> + <code>results/stability.csv</code>. "
        "This page shows <b>SNR=1</b> (realistic additive noise). Reproduce the full SNR sweep with "
        "<code>python run/benchmark.py --dataset all --methods all --seeds 20 --dim 768 --n 1000 "
        "--snr inf 4 1</code> (or just the reported level with <code>--snr 1</code>), then rebuild this "
        "page with <code>python run/make_report.py</code>. See <code>README.md</code> for full "
        "methodology.</footer>")
    parts.append("</div></body></html>")
    return "".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true",
                    help="base64-embed figures into a single portable HTML (larger file)")
    ap.add_argument("--out", default=str(ROOT / "REPORT.html"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.write_text(build(args.embed), encoding="utf-8")
    size = out.stat().st_size / 1024
    print(f"wrote {out}  ({size:.0f} KB, embed={args.embed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
