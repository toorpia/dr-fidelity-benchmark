"""Figure generation: multi-scale fidelity curves, Shepard scatters, embedding panels, heatmaps.

All figures are written under ``figures/{dataset}/``. Each function is defensive: a plotting failure
is reported but never aborts the benchmark run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics.distances import DEFAULT_CUTOFFS, band_thresholds, condensed_distances

_METHOD_COLORS = {
    "PCA": "#1f77b4", "Isomap": "#17becf", "PyMDE": "#2ca02c", "PCC": "#9467bd",
    "t-SNE": "#ff7f0e", "UMAP": "#d62728", "DREAMS": "#e377c2", "toorPIA": "#000000",
    # the noise-dims supplement shows toorPIA through both of its endpoints
    "toorPIA (basemap_csvform)": "#000000", "toorPIA (basemap_embedding)": "#7f7f7f",
}


def snr_label(snr) -> str:
    return "inf" if (snr is None or not np.isfinite(snr)) else f"{float(snr):g}"


def _color(method):
    return _METHOD_COLORS.get(method, None)


def _band_curve(agg, dataset, snr, method, prefix, suffix, cutoffs):
    xs, med, lo, hi = [], [], [], []
    for p in cutoffs:
        name = f"{prefix}_p{p}__{suffix}"
        r = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == method)
                & (agg.metric == name)]
        if len(r):
            xs.append(p); med.append(r["median"].iloc[0])
            lo.append(r["ci_lo"].iloc[0]); hi.append(r["ci_hi"].iloc[0])
    return np.array(xs), np.array(med), np.array(lo), np.array(hi)


def multiscale_curves(agg, dataset, snr, methods, out_dir, cutoffs=DEFAULT_CUTOFFS):
    """One figure per (prefix, suffix): band-cutoff on x, metric on y, one CI-ribboned line/method."""
    panels = [("shepard", "vs_ambient", "Shepard ρ (rank)  — vs ambient", (0, 1)),
              ("shepard", "vs_truth", "Shepard ρ (rank)  — vs truth", (0, 1)),
              ("stress", "vs_ambient", "normalized stress  — vs ambient (lower=better)", None),
              ("stress", "vs_truth", "normalized stress  — vs truth (lower=better)", None)]
    written = []
    for prefix, suffix, title, ylim in panels:
        fig, ax = plt.subplots(figsize=(7, 5))
        any_line = False
        for method in methods:
            xs, med, lo, hi = _band_curve(agg, dataset, snr, method, prefix, suffix, cutoffs)
            if xs.size == 0 or np.all(np.isnan(med)):
                continue
            any_line = True
            ax.plot(xs, med, marker="o", label=method, color=_color(method))
            ax.fill_between(xs, lo, hi, alpha=0.15, color=_color(method))
        if not any_line:
            plt.close(fig); continue
        ax.set_xlabel("distance-band cutoff  p  (lowest p% of high-D pairs)")
        ax.set_ylabel(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f"{dataset}  (SNR={snr_label(snr)})  —  {prefix} {suffix}")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        out = Path(out_dir) / f"curve_{prefix}_{suffix}_snr{snr_label(snr)}.png"
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        written.append(out)
    return written


def shepard_scatter(X_ambient, embeddings, dataset, snr, out_dir, cutoffs=DEFAULT_CUTOFFS,
                    gridsize=90, max_pairs=2_000_000, seed=0):
    """Shepard diagram per method as a POINT-DENSITY heatmap (jet), with band boundaries marked.

    Each panel is a 2-D density of (high-D distance, 2-D distance) pairs, rendered with hexagonal
    binning and LOG-scaled counts (jet colormap), so the true concentration of pairs is visible
    instead of being hidden behind overplotted translucent points. Each panel is independently scaled
    with its own colorbar (relative density within that panel). Dashed white lines mark the band
    cutoffs on the high-D axis.
    """
    d_hd = condensed_distances(X_ambient)
    rng = np.random.default_rng(seed)
    if d_hd.size > max_pairs:                       # cap only for very large N; keeps density faithful
        sel = rng.choice(d_hd.size, size=max_pairs, replace=False)
        d_hd_p = d_hd[sel]
    else:
        sel = None
        d_hd_p = d_hd
    from metrics.distances import first_mode_threshold
    thr = [first_mode_threshold(d_hd)[0]]   # mark the first-mode near-band boundary
    methods = list(embeddings)
    ncol = min(4, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.1 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, method in enumerate(methods):
        ax = axes.flat[i]; ax.axis("on")
        d2 = condensed_distances(embeddings[method])
        d2_p = d2[sel] if sel is not None else d2
        hb = ax.hexbin(d_hd_p, d2_p, gridsize=gridsize, cmap="jet", bins="log", mincnt=1)
        for t in thr:
            ax.axvline(t, color="white", lw=0.7, ls="--", alpha=0.85)
        cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("log10(pair count)", fontsize=6)
        ax.set_title(method, fontsize=9)
        ax.set_xlabel("high-D dist", fontsize=8); ax.set_ylabel("2-D dist", fontsize=8)
    fig.suptitle(f"Shepard density (jet) — {dataset} (SNR={snr_label(snr)}); dashed = first-mode near-band boundary",
                 fontsize=11)
    out = Path(out_dir) / f"shepard_scatter_snr{snr_label(snr)}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def _square_equal_axes(ax, Y, pad=1.06):
    """Equal x/y scale in a square panel: 2-D distances/distortions read true in every direction.

    Limits = the data's bounding square (common half-range on both axes), so panels differ only by
    the embedding's own geometry, never by anisotropic autoscaling.
    """
    cx = (float(Y[:, 0].min()) + float(Y[:, 0].max())) / 2
    cy = (float(Y[:, 1].min()) + float(Y[:, 1].max())) / 2
    half = max(float(Y[:, 0].max()) - float(Y[:, 0].min()),
               float(Y[:, 1].max()) - float(Y[:, 1].min())) / 2 * pad
    half = half or 1.0
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")


def embedding_panels(embeddings, color_value, color_name, dataset, snr, out_dir):
    """2-D embedding per method, colored by the dataset's meaningful variable.

    Panels are SQUARE with equal x/y scales (see :func:`_square_equal_axes`), so in-map distortion
    is visually faithful.
    """
    methods = list(embeddings)
    # cyclic parameters (closed-loop t) -> cyclic map; categorical (cluster id) -> discrete map
    name = (color_name or "").lower()
    cmap = "twilight" if "cyclic" in name else ("tab10" if "cluster id" in name else "viridis")
    ncol = min(4, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    sc = None
    for i, method in enumerate(methods):
        ax = axes.flat[i]; ax.axis("on")
        Y = embeddings[method]
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=color_value, s=6, cmap=cmap)
        _square_equal_axes(ax, Y)
        ax.set_title(method, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label=color_name)
    fig.suptitle(f"2-D embeddings — {dataset} (SNR={snr_label(snr)}), colored by {color_name}",
                 fontsize=11)
    out = Path(out_dir) / f"embeddings_snr{snr_label(snr)}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


_DIR_COLORS = ["#d62728", "#e377c2", "#ff7f0e", "#9467bd", "#8c564b"]   # star color per direction


def outlier_gallery(embeddings, labels, outlier_idx, dataset, snr, out_dir, outlier_dir=None):
    """2-D embedding per method with the ground-truth outliers highlighted.

    Bulk points are colored by cluster id (muted, small); the ground-truth outliers are drawn as
    stars with a black edge, COLORED BY ANOMALOUS DIRECTION (same-direction near-duplicate pairs
    share a color and are labeled o{dir}a / o{dir}b), so both burial and pair-splitting are visible
    at a glance. Panels are SQUARE with equal x/y scales (see :func:`_square_equal_axes`), so
    in-map distortion is visually faithful.
    """
    methods = list(embeddings)
    oi = np.sort(np.asarray(outlier_idx))
    od = (np.asarray(outlier_dir) if outlier_dir is not None else np.arange(len(oi)))
    is_out = np.zeros(len(labels), dtype=bool)
    is_out[oi] = True
    ncol = min(4, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, method in enumerate(methods):
        ax = axes.flat[i]; ax.axis("on")
        Y = embeddings[method]
        ax.scatter(Y[~is_out, 0], Y[~is_out, 1], c=np.asarray(labels)[~is_out], s=6,
                   cmap="tab10", alpha=0.55, linewidths=0)
        seen = {}
        for k, o in enumerate(oi):
            d = int(od[k])
            seen[d] = seen.get(d, -1) + 1
            ax.scatter([Y[o, 0]], [Y[o, 1]], marker="*", s=170,
                       c=_DIR_COLORS[d % len(_DIR_COLORS)], edgecolor="k",
                       linewidth=0.7, zorder=5)
            tag = f"o{d}{chr(ord('a') + seen[d])}" if outlier_dir is not None else f"o{k}"
            ax.annotate(tag, (Y[o, 0], Y[o, 1]), textcoords="offset points", xytext=(5, 4),
                        fontsize=8, color="#a00", fontweight="bold")
        _square_equal_axes(ax, Y)
        ax.set_title(method, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"2-D embeddings — {dataset} (SNR={snr_label(snr)}); "
                 "★ = ground-truth outliers (one color per anomalous direction; a/b = the "
                 "near-duplicate pair)", fontsize=11)
    out = Path(out_dir) / f"outlier_gallery_snr{snr_label(snr)}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def dims_grid(grid, labels, methods, dims, out_dir, fname="dims_grid.png"):
    """Notebook-style grid: rows = methods, cols = total dimensionality, colored by cluster label.

    ``grid`` is ``{method: {dim: (n, 2) embedding}}``; missing cells stay blank. Panels are forced
    square (equal half-range on both axes, centered) exactly like the source notebook, so cluster
    collapse reads as shape -- not as an axis-scaling artifact.
    """
    # CVD-safe cluster colors (Okabe-Ito family; validated: worst adjacent protan dE 37, all
    # >= 3:1 contrast) -- the notebook's raw r/g/b has protan dE 10 between red and green, and
    # cluster MIXING is exactly what this figure must show
    _cluster_colors = np.array(["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"])
    point_colors = _cluster_colors[np.asarray(labels, dtype=int) % 6]
    nrow, ncol = len(methods), len(dims)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, method in enumerate(methods):
        for j, dim in enumerate(dims):
            Y = grid.get(method, {}).get(dim)
            if Y is None:
                continue
            ax = axes[i][j]; ax.axis("on")
            ax.scatter(Y[:, 0], Y[:, 1], c=point_colors, s=8, alpha=0.7)
            x_center, y_center = np.mean(ax.get_xlim()), np.mean(ax.get_ylim())
            half = max(np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim())) / 2.0
            ax.set_xlim(x_center - half, x_center + half)
            ax.set_ylim(y_center - half, y_center + half)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{method} - {dim}D", fontsize=9)
    fig.suptitle("noise-dims sweep: 3 tight clusters in 3 signal dims + (D-3) noise dims",
                 fontsize=12)
    out = Path(out_dir) / fname
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def dimension_curve(agg_dims, methods, out_dir,
                    panels=(("knn_acc_k10", "2-D kNN label accuracy (k=10)"),)):
    """Cluster survival vs total dimensionality, one CI-ribboned line per method.

    ``agg_dims`` columns: method, dim, metric, median, ci_lo, ci_hi. kNN-accuracy panels get a dashed
    chance line at 1/3 (three balanced clusters). Distance-band Shepard ρ is deliberately not
    plotted here: a global ρ is only meaningful paired with its near band, and neither reads
    cleanly against the noise-dominated ambient of this probe (metrics remain in the CSVs).
    """
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 5), sharex=True,
                             squeeze=False)
    for ax, (metric, title) in zip(axes.flat, panels):
        any_line = False
        for method in methods:
            r = agg_dims[(agg_dims.method == method) & (agg_dims.metric == metric)]
            r = r.sort_values("dim")
            if not len(r):
                continue
            any_line = True
            x = r["dim"].to_numpy()
            ax.plot(x, r["median"], marker="o", label=method, color=_color(method))
            ax.fill_between(x, r["ci_lo"], r["ci_hi"], alpha=0.15, color=_color(method))
        # proper log10 axis: decade majors (10, 100, 1000) + minor ticks; the measured dims are
        # marked by the data points themselves
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("total dimensionality D, log10 axis  (3 signal + D-3 noise dims)")
        ax.set_ylabel(title, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        if metric.startswith("knn_acc"):
            ax.axhline(1 / 3, color="#888", lw=1.0, ls="--")
            ax.annotate("chance = 1/3", xy=(0.02, 1 / 3), xycoords=("axes fraction", "data"),
                        fontsize=8, color="#888", va="bottom")
        ax.grid(alpha=0.3, which="both")
        if any_line:
            ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Curse of dimensionality (noise-dims): cluster survival vs total dimensionality",
                 fontsize=12)
    out = Path(out_dir) / "dimension_curve.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def trajectory_gallery(embeddings, color_value, series, npoints, out_dir,
                       fname="trajectory_gallery.png"):
    """Multi-series random-walk maps, colored by concatenated time.

    One square panel per method; points colored by the global time gradient (viridis), so each
    series occupies a distinct color range; each series' end point is marked. Read against the
    two defining properties: the saw-tooth LOCAL jitter along each trajectory, and the GLOBAL
    radial separation of the series.
    """
    methods = list(embeddings)
    ncol = min(4, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    n_series = int(np.max(series)) + 1
    sc = None
    for i, method in enumerate(methods):
        ax = axes.flat[i]; ax.axis("on")
        Y = embeddings[method]
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=color_value, s=2, cmap="viridis", linewidths=0)
        for k in range(n_series):                       # mark each series' end point
            e = (k + 1) * npoints - 1
            ax.scatter([Y[e, 0]], [Y[e, 1]], s=42, marker="s", facecolor="none",
                       edgecolor="k", linewidth=1.0, zorder=3)
        _square_equal_axes(ax, Y)
        ax.set_title(method, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6,
                     label="time (concatenated series; squares = series end points)")
    fig.suptitle(f"Multi-series high-D random walks — {n_series} series, "
                 f"{npoints} steps each, colored by time", fontsize=11)
    out = Path(out_dir) / fname
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def random_walk_geometry(ndim, npoints, n_series, step, seed, out_dir,
                         fname="geometry_explainer.png"):
    """Empirical explainer of the TRUE geometry of multi-series high-D random walks.

    Every panel is MEASURED from the actual generator output (no schematic drawings):
    (1) the same walk recipe in D=2 — the intuition trap: walks meander, recross, overlap;
    (2) distance from the origin vs time in D=ndim — monotone ~sqrt(t) radial escape
        (D=2 shown in grey for contrast: it wanders around the theory curve);
    (3) angles: successive steps within a series, and positions of different series, both
        concentrate at 90 deg in D=ndim (broad in D=2) — zigzag locally, orthogonal globally;
    (4) cross-series distances match the Pythagorean prediction sqrt(r_i^2 + r_j^2) exactly —
        the series radiate along mutually orthogonal directions and never approach each other.
    Together: the true shape is a star of jagged spokes, mutually orthogonal, radiating from
    the shared origin.
    """
    from synth.random_walk import make_random_walks

    hi = make_random_walks(ndim=ndim, npoints=npoints, n_series=n_series, step=step, seed=seed)
    lo = make_random_walks(ndim=2, npoints=npoints, n_series=n_series, step=step, seed=seed)
    cmap = plt.get_cmap("tab10")

    def walk(base, s):
        return base["clean"][s * npoints:(s + 1) * npoints]

    def angles_deg(V, W):
        c = (np.einsum("ij,ij->i", V, W)
             / (np.linalg.norm(V, axis=1) * np.linalg.norm(W, axis=1)))
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.4))

    # (1) the same recipe in D=2: meander, recross, overlap
    ax = axes[0, 0]
    for s in range(n_series):
        W = walk(lo, s)
        ax.plot(W[:, 0], W[:, 1], lw=0.7, color=cmap(s % 10))
    ax.scatter([0], [0], marker="*", s=140, color="k", zorder=3, label="origin")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(1) the SAME recipe in D=2 — the intuition trap:\n"
                 "walks meander, recross, and overlap each other", fontsize=10)

    # (2) radial escape: ||x_t|| grows as sqrt(t) in high D
    ax = axes[0, 1]
    t = np.arange(1, npoints + 1)
    for s in range(n_series):
        r2 = np.linalg.norm(walk(lo, s), axis=1) / (step * np.sqrt(2.0 / 3.0))
        ax.plot(t, r2, lw=0.7, color="0.78", zorder=1)
    for s in range(n_series):
        r = np.linalg.norm(walk(hi, s), axis=1) / (step * np.sqrt(ndim / 3.0))
        ax.plot(t, r, lw=0.9, color=cmap(s % 10), zorder=2)
    ax.plot(t, np.sqrt(t), "k--", lw=1.5, zorder=3,
            label=r"theory  $\sqrt{t}$   ($\|x_t\| \approx s\sqrt{tD/3}$)")
    ax.set_xlabel("time step  t")
    ax.set_ylabel(r"distance from origin  /  $s\sqrt{D/3}$")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"(2) D={ndim}: monotone radial escape — each series (colors)\n"
                 "hugs the √t law; the D=2 walks (grey) wander around it", fontsize=10)

    # (3) direction statistics: everything is ~orthogonal in high D
    ax = axes[1, 0]
    def step_angles(base):
        out = []
        for s in range(n_series):
            st = np.diff(walk(base, s), axis=0)
            out.append(angles_deg(st[:-1], st[1:]))
        return np.concatenate(out)
    cross_pos = np.concatenate(
        [angles_deg(walk(hi, i), walk(hi, j))
         for i in range(n_series) for j in range(i + 1, n_series)])
    bins = np.linspace(0, 180, 91)
    ax.hist(step_angles(lo), bins=bins, density=True, histtype="step",
            color="0.55", lw=1.2, label="successive steps, D=2 (broad)")
    ax.hist(step_angles(hi), bins=bins, density=True, histtype="stepfilled",
            color="#1f77b4", alpha=0.55, label=f"successive steps, D={ndim} (zigzag)")
    ax.hist(cross_pos, bins=bins, density=True, histtype="stepfilled",
            color="#d62728", alpha=0.45,
            label=f"positions of two DIFFERENT series, D={ndim}")
    ax.axvline(90, color="k", ls=":", lw=1)
    ax.set_xlabel("angle between the two directions  (degrees)")
    ax.set_ylabel("density")
    ax.set_xlim(0, 180); ax.legend(fontsize=8)
    ax.set_title(f"(3) D={ndim}: independent directions concentrate at 90°\n"
                 "(spread ~1/√D) — jagged at every step, series mutually orthogonal",
                 fontsize=10)

    # (4) Pythagoras: cross-series distance = sqrt(r_i^2 + r_j^2)
    ax = axes[1, 1]
    X, series = hi["clean"], hi["series"]
    rng = np.random.default_rng(12345)          # fixed: the figure is reproducible
    a = rng.integers(0, len(X), 8000); b = rng.integers(0, len(X), 8000)
    m = series[a] != series[b]
    a, b = a[m], b[m]
    d = np.linalg.norm(X[a] - X[b], axis=1)
    pred = np.sqrt(np.linalg.norm(X[a], axis=1) ** 2 + np.linalg.norm(X[b], axis=1) ** 2)
    ax.scatter(pred, d, s=3, alpha=0.2, color="#1f77b4", linewidths=0, rasterized=True)
    lim = (0.0, float(max(pred.max(), d.max()) * 1.05))
    ax.plot(lim, lim, "k--", lw=1.4, label="exact orthogonality:  $d=\\sqrt{r_i^2+r_j^2}$")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel(r"Pythagorean prediction  $\sqrt{r_i^2+r_j^2}$")
    ax.set_ylabel("actual cross-series distance")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"(4) D={ndim}: cross-series distances are Pythagorean —\n"
                 "the series NEVER approach each other (no overlap possible)", fontsize=10)

    fig.suptitle("The true geometry of multi-series high-D random walks: jagged spokes, "
                 "mutually orthogonal,\nradiating from the shared origin "
                 f"(measured on the actual dataset — {n_series} series × {npoints} steps, "
                 f"D={ndim})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out_dir) / fname
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def noise_dims_endpoint_curve(agg_dims, methods, ambient, out_dir, knn_key="knn_acc_k10",
                              k=10, chance=1.0 / 3.0):
    """Noise-dims cluster survival, two readings side by side.

    Left  — absolute kNN label accuracy vs D (same semantics as :func:`dimension_curve`).
    Right — the chance-corrected skill RATIO against the ambient raw-feature kNN baseline:
            (acc_2D − chance) / (acc_ambient − chance). Ratio 1 = the 2-D map is exactly as
            class-readable as kNN run directly on the raw D-dim features; >1 = the map reads
            MORE class structure than the raw feature space itself; 0 = chance. The correction
            matters: both accuracies approach chance (not 0) as D grows, so the RAW ratio of a
            fully collapsed method drifts toward ~1 instead of 0. ``ambient`` is {dim: accuracy}.
    """
    from matplotlib.ticker import ScalarFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.4), sharex=True)
    amb_x = np.array(sorted(ambient))
    amb_y = np.array([ambient[d] for d in amb_x])
    for ax, mode in zip(axes, ("abs", "ratio")):
        for method in methods:
            r = agg_dims[(agg_dims.method == method) & (agg_dims.metric == knn_key)]
            r = r.sort_values("dim")
            if not len(r):
                continue
            x = r["dim"].to_numpy()
            med, lo, hi = (r[c].to_numpy() for c in ("median", "ci_lo", "ci_hi"))
            if mode == "ratio":
                denom = np.array([ambient[int(d)] for d in x]) - chance
                med, lo, hi = ((v - chance) / denom for v in (med, lo, hi))
            # toorPIA (both endpoints) solid and heavy -- the emphasized results; the generic
            # methods recede to thin dashed lines
            star = method.startswith("toorPIA")
            ax.plot(x, med, marker="o", ms=5 if star else 3.5,
                    lw=2.2 if "csvform" in method else (1.8 if star else 1.0),
                    ls="-" if star else "--", label=method, color=_color(method))
            ax.fill_between(x, lo, hi, alpha=0.15, color=_color(method))
        if mode == "abs":
            ax.plot(amb_x, amb_y, color="#999", lw=1.2, ls=":", marker=".",
                    label="ambient (raw-feature kNN)")
            ax.axhline(chance, color="#888", lw=1.0, ls="--")
            ax.annotate(f"chance = {chance:.2f}", xy=(0.02, chance),
                        xycoords=("axes fraction", "data"), fontsize=8, color="#888", va="bottom")
            ax.set_ylabel(f"2-D kNN label accuracy (k={k})", fontsize=10)
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.axhline(1.0, color="#888", lw=1.0, ls="--")
            ax.annotate("= raw-feature kNN", xy=(0.02, 1.0), xycoords=("axes fraction", "data"),
                        fontsize=8, color="#888", va="bottom")
            ax.axhline(0.0, color="#bbb", lw=0.8, ls=":")
            ax.annotate("chance", xy=(0.02, 0.0), xycoords=("axes fraction", "data"),
                        fontsize=8, color="#999", va="bottom")
            ax.set_ylabel("above-chance skill ratio vs ambient raw-feature kNN", fontsize=10)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("total dimensionality D, log10 axis  (3 signal + D-3 noise dims)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5, ncol=2)
    fig.suptitle("Curse of dimensionality (noise-dims): cluster survival, absolute and relative "
                 "to kNN on the raw features", fontsize=12)
    out = Path(out_dir) / "dimension_curve.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def population_gallery(embeddings, labels, population, dataset, snr, out_dir, fname=None):
    """2-D embedding per method for the imbalanced two-population dataset.

    Majority population = circles, minority population = larger triangles with an edge; both
    colored by cluster id (tab10, ids 0..2K-1), so the two questions are visible at a glance:
    are the two populations placed apart, and does the minority keep its internal clusters?
    Square panels, equal x/y scales. ``fname`` overrides the default snr-based filename (used by
    the minority-fraction sweep, which writes one gallery per fraction).
    """
    methods = list(embeddings)
    labels = np.asarray(labels); population = np.asarray(population)
    a = population == 0
    vmax = max(9, int(labels.max()))
    ncol = min(4, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, method in enumerate(methods):
        ax = axes.flat[i]; ax.axis("on")
        Y = embeddings[method]
        ax.scatter(Y[a, 0], Y[a, 1], c=labels[a], s=7, cmap="tab10", vmin=0, vmax=vmax,
                   marker="o", alpha=0.6, linewidths=0)
        ax.scatter(Y[~a, 0], Y[~a, 1], c=labels[~a], s=22, cmap="tab10", vmin=0, vmax=vmax,
                   marker="^", alpha=0.9, edgecolor="k", linewidth=0.3)
        _square_equal_axes(ax, Y)
        ax.set_title(method, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"2-D embeddings — {dataset} (SNR={snr_label(snr)}); circles = majority "
                 "population, triangles = minority population, colored by cluster", fontsize=11)
    out = Path(out_dir) / (fname or f"population_gallery_snr{snr_label(snr)}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def populations_sweep_curve(agg_sweep, methods, out_dir):
    """Minority-fraction sweep, same layout as ``sweep_outliers_curve``.

    Left: global Shepard ρ (all pairs — standard). Right: the SAME standard ρ restricted to the
    minority-involving pairs ([minority]-[minority] plus [minority]-[majority]) — the direct
    analog of the outliers dataset's anomaly-pair ρ. ``agg_sweep`` has columns method,
    minority_frac, metric, median, ci_lo, ci_hi."""
    panels = [("full_shepard", "global Shepard ρ  (all pairs)", "linear", None),
              ("minority_shepard__vs_ambient", "Shepard ρ — minority-involving pairs only",
               "linear", None)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    fracs = sorted(agg_sweep["minority_frac"].unique())
    for ax, (metric, title, yscale, ideal) in zip(axes, panels):
        any_line = False
        for method in methods:
            r = agg_sweep[(agg_sweep.method == method) & (agg_sweep.metric == metric)]
            r = r.sort_values("minority_frac")
            if not len(r):
                continue
            any_line = True
            x = r["minority_frac"].to_numpy()
            ax.plot(x, r["median"], marker="o", label=method, color=_color(method))
            ax.fill_between(x, r["ci_lo"], r["ci_hi"], alpha=0.15, color=_color(method))
        if ideal is not None:
            ax.axhline(ideal, color="#333", lw=1.0, ls=":")
            ax.text(fracs[0], ideal, " ideal ", fontsize=8, color="#333", va="bottom")
        ax.set_xscale("log"); ax.set_xticks(fracs)
        ax.set_xticklabels([f"{f:g}" for f in fracs])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yscale(yscale)
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlabel("minority fraction  (share of points in the small population)")
        ax.set_ylabel(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if any_line:
            ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Minority-fraction sweep (populations, SNR=1): global Shepard ρ and the "
                 "minority-pair-restricted ρ", fontsize=12)
    out = Path(out_dir) / "populations_sweep_curve.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def basemap_addplot_gallery(panels, labs_b, anchor, n_anom, out_dir):
    """Addplot gallery for the ANOMALY-FREE basemap (the basemap contains no anomalies).

    Fit bulk = normal clusters colored by cluster id (tab10, faint); added bulk controls = small
    dark dots; the added CLUSTER-ANCHORED anomalies (a near-duplicate pair per cluster) = large
    triangles colored by their SOURCE cluster. Faithful = each triangle outside the normal
    region, in its own cluster's direction, pair members adjacent; dark dots inside the bulk.
    Square panels, equal x/y scales.
    """
    methods = list(panels)
    labs_b = np.asarray(labs_b); anchor = np.asarray(anchor)
    vmax = max(9, int(labs_b.max()))
    ncol = min(4, len(methods)); nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, method in enumerate(methods):
        ax = axes.flat[i]; ax.axis("on")
        Yf, Ya = panels[method]
        ax.scatter(Yf[:, 0], Yf[:, 1], c=labs_b, s=6, cmap="tab10", vmin=0, vmax=vmax,
                   alpha=0.35, linewidths=0)
        ax.scatter(Ya[n_anom:, 0], Ya[n_anom:, 1], c="#333333", s=10, linewidths=0, zorder=4)
        ax.scatter(Ya[:n_anom, 0], Ya[:n_anom, 1], marker="^", s=110, c=anchor, cmap="tab10",
                   vmin=0, vmax=vmax, edgecolor="k", linewidth=0.8, zorder=6)
        _square_equal_axes(ax, np.vstack([Yf, Ya]))
        ax.set_title(method, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("addplot onto an ANOMALY-FREE basemap — \u25b2 = added cluster-anchored anomalies "
                 "(colored by SOURCE cluster; a near-duplicate pair per cluster), "
                 "\u00b7 = added bulk controls", fontsize=11)
    out = Path(out_dir) / "basemap_addplot_gallery.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def sweep_outliers_curve(agg_sweep, methods, out_dir):
    """Outlier-factor sweep with STANDARD/plain readings only, one CI-ribboned line per method.

    Left: global Shepard ρ (the rank correlation of the Shepard density figure — standard).
    Right: the SAME standard ρ restricted to the anomaly-involving pairs — the direct
    quantification of the outlier-related blocks of that figure. ``agg_sweep`` has columns method,
    outlier_factor, metric, median, ci_lo, ci_hi.
    """
    panels = [("full_shepard", "global Shepard ρ  (all pairs)", "linear", None),
              ("outlier_shepard__vs_ambient", "Shepard ρ — anomaly-involving pairs only", "linear",
               None)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    factors = sorted(agg_sweep["outlier_factor"].unique())
    for ax, (metric, title, yscale, ideal) in zip(axes, panels):
        any_line = False
        for method in methods:
            r = agg_sweep[(agg_sweep.method == method) & (agg_sweep.metric == metric)]
            r = r.sort_values("outlier_factor")
            if not len(r):
                continue
            any_line = True
            x = r["outlier_factor"].to_numpy()
            ax.plot(x, r["median"], marker="o", label=method, color=_color(method))
            ax.fill_between(x, r["ci_lo"], r["ci_hi"], alpha=0.15, color=_color(method))
        if ideal is not None:
            ax.axhline(ideal, color="#333", lw=1.0, ls=":")
            ax.text(factors[0], ideal, " ideal ", fontsize=8, color="#333", va="bottom")
        ax.set_xscale("log"); ax.set_xticks(factors)
        ax.set_xticklabels([f"{f:g}" for f in factors])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yscale(yscale)
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlabel("outlier factor  (distance from bulk centroid ÷ bulk radius of gyration Rg)")
        ax.set_ylabel(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        if any_line:
            ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Outlier-factor sweep (outliers, clean): global Shepard ρ and the "
                 "anomaly-pair-restricted ρ", fontsize=12)
    out = Path(out_dir) / "sweep_outliers_curve.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def score_scatter(agg, dataset, snr, methods, out_dir):
    """Per-method near-vs-global scatter: x = Shepard ρ over the first-mode near band, y = full (global).

    A method in the top-right reproduces BOTH near-neighbor and global distance structure. Uses the
    median over seeds for each axis.
    """
    pts = []
    for m in methods:
        rx = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == m)
                 & (agg.metric == "shepard_near__vs_ambient")]
        ry = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == m)
                 & (agg.metric == "full_shepard")]
        if len(rx) and len(ry):
            pts.append((m, float(rx["median"].iloc[0]), float(ry["median"].iloc[0])))
    if not pts:
        return None
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for m, x, y in pts:
        ax.scatter([x], [y], s=130, color=_color(m), edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlim(min(xs) - 0.06, max(xs) + 0.10)
    ax.set_ylim(min(ys) - 0.06, max(ys) + 0.06)
    ax.set_xlabel("near-neighbor fidelity — Shepard ρ, first-mode near band (fixed radius)")
    ax.set_ylabel("global fidelity — Shepard ρ (full)")
    ax.set_title(f"near (first-mode band) vs global (full) — {dataset} (SNR={snr_label(snr)})\n"
                 "top-right = near + global", fontsize=11)
    ax.grid(alpha=0.3)
    ax.annotate("better →", xy=(0.97, 0.02), xycoords="axes fraction", ha="right", fontsize=8,
                color="#888")
    ax.annotate("↑ better", xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                color="#888")
    out = Path(out_dir) / f"score_scatter_snr{snr_label(snr)}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def outlier_score_scatter(agg, dataset, snr, methods, out_dir):
    """Outlier-pair vs global fidelity: x = Shepard ρ over anomaly-involving pairs, y = full ρ.

    The outliers-dataset analogue of :func:`score_scatter` (near p5 vs full): top-right reproduces
    BOTH the anomaly-related distance structure and the global layout. Median over seeds per axis.
    """
    pts = []
    for m in methods:
        rx = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == m)
                 & (agg.metric == "outlier_shepard__vs_ambient")]
        ry = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == m)
                 & (agg.metric == "full_shepard")]
        if len(rx) and len(ry):
            pts.append((m, float(rx["median"].iloc[0]), float(ry["median"].iloc[0])))
    if not pts:
        return None
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for m, x, y in pts:
        ax.scatter([x], [y], s=130, color=_color(m), edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlim(min(xs) - 0.06, max(xs) + 0.10)
    ax.set_ylim(min(ys) - 0.06, max(ys) + 0.06)
    ax.set_xlabel("outlier fidelity — Shepard ρ over anomaly-involving pairs")
    ax.set_ylabel("global fidelity — Shepard ρ (full)")
    ax.set_title(f"outlier pairs vs global (full) — {dataset} (SNR={snr_label(snr)})\n"
                 "top-right = anomalies + global both right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.annotate("better →", xy=(0.97, 0.02), xycoords="axes fraction", ha="right", fontsize=8,
                color="#888")
    ax.annotate("↑ better", xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                color="#888")
    out = Path(out_dir) / f"outlier_score_scatter_snr{snr_label(snr)}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def population_score_scatter(agg, dataset, snr, methods, out_dir):
    """Minority-pair vs global fidelity: x = Shepard ρ over minority-involving pairs, y = full ρ.

    The populations-dataset analogue of :func:`outlier_score_scatter`: top-right reproduces BOTH
    the minority-related distance structure and the global layout. Median over seeds per axis.
    """
    pts = []
    for m in methods:
        rx = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == m)
                 & (agg.metric == "minority_shepard__vs_ambient")]
        ry = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == m)
                 & (agg.metric == "full_shepard")]
        if len(rx) and len(ry):
            pts.append((m, float(rx["median"].iloc[0]), float(ry["median"].iloc[0])))
    if not pts:
        return None
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for m, x, y in pts:
        ax.scatter([x], [y], s=130, color=_color(m), edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlim(min(xs) - 0.06, max(xs) + 0.10)
    ax.set_ylim(min(ys) - 0.06, max(ys) + 0.06)
    ax.set_xlabel("minority fidelity — Shepard ρ over minority-involving pairs")
    ax.set_ylabel("global fidelity — Shepard ρ (full)")
    ax.set_title(f"minority pairs vs global (full) — {dataset} (SNR={snr_label(snr)})\n"
                 "top-right = minority + global both right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.annotate("better →", xy=(0.97, 0.02), xycoords="axes fraction", ha="right", fontsize=8,
                color="#888")
    ax.annotate("↑ better", xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                color="#888")
    out = Path(out_dir) / f"population_score_scatter_snr{snr_label(snr)}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def summary_heatmap(agg, dataset, snr, methods, out_dir):
    """Heatmap methods x selected metrics (median values) for one (dataset, snr)."""
    metrics = ["shepard_near__vs_ambient", "full_shepard",
               "recall_k15", "trust_k15", "cont_k15"]
    M = np.full((len(methods), len(metrics)), np.nan)
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            r = agg[(agg.dataset == dataset) & (agg.snr == snr) & (agg.method == method)
                    & (agg.metric == metric)]
            if len(r):
                M[i, j] = r["median"].iloc[0]
    if np.all(np.isnan(M)):
        return None
    fig, ax = plt.subplots(figsize=(1.1 * len(metrics) + 2, 0.6 * len(methods) + 2))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods, fontsize=9)
    for i in range(len(methods)):
        for j in range(len(metrics)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="median (higher=better)")
    ax.set_title(f"Summary — {dataset} (SNR={snr_label(snr)})", fontsize=11)
    out = Path(out_dir) / f"summary_heatmap_snr{snr_label(snr)}.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


# --------------------------------------------------------------------------------------------------
# Explanatory figures: WHY full Shepard ρ buries local structure, and WHY recall@k is not a fair
# near-neighbor metric. These depend only on the high-D features X (no embedding needed), so they are
# dataset-level. They back the two methodological claims of the report with real data + a schematic.
# --------------------------------------------------------------------------------------------------

_NEAR = "#2ca02c"      # green — the near band / fixed-radius (fair) side
_BULK = "#9bb7e0"      # light blue — the mid/far bulk
_CUM = "#16407a"       # dark blue — cumulative curve


def distance_distribution(X_ambient, dataset, snr, out_dir, cutoffs=DEFAULT_CUTOFFS):
    """Why the FULL Shepard ρ buries near-neighbor structure — the high-D pairwise-distance profile.

    In high dimensions distances *concentrate*: almost every pair sits at a mid/far distance, and
    only the pairs inside the FIRST MODE of the distance profile are genuinely 'near' (on structured
    data that mode is the within-structure pairs). The full ρ (p=100) ranks ALL pairs together, so it
    is dominated by the mid/far bulk and near-distance accuracy is averaged away. This motivates
    restricting ρ to the structure-adaptive near band: all pairs up to the density valley where the
    first mode decays into the tail (``metrics.distances.first_mode_threshold``).

    Left  — histogram of all high-D pairwise distances (count) with the cumulative % overlaid (twin
            axis) and the percentile cutoffs p=5..100 as guides; the first-mode near band is shaded
            green with its boundary (and the equivalent percentile) marked.
    Right — the SAME point as a counting/weighting argument: the full ρ averages over every pair, of
            which only the first-mode fraction is near, so near pairs are out-voted and cannot move
            the global number.
    """
    from metrics.distances import first_mode_threshold
    d_hd = condensed_distances(X_ambient)
    n_pairs = int(d_hd.size)
    r_near, _fb = first_mode_threshold(d_hd)
    near_pct = 100.0 * float(np.mean(d_hd <= r_near))
    n_near = int(round(n_pairs * near_pct / 100.0))
    thr = band_thresholds(d_hd, cutoffs)              # {p: absolute distance threshold}
    mean_d = float(np.mean(d_hd))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.7, 1.0]})

    # ---- Left: distance histogram + cumulative %, with band cutoffs ----
    counts, edges = np.histogram(d_hd, bins=60)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    axL.bar(centers, counts, width=width, color=_BULK, edgecolor="white", linewidth=0.2,
            align="center", zorder=2)
    axL.axvspan(float(edges[0]), r_near, color=_NEAR, alpha=0.15, zorder=1,
                label=f"first-mode near band: dist ≤ {r_near:.2f} (= p{near_pct:.0f})")
    axL.axvline(r_near, color="#0a5", lw=1.4, zorder=4)
    ymax = counts.max() * 1.18
    axL.set_ylim(0, ymax)
    for p in cutoffs:                                 # band-cutoff guide lines on the distance axis
        t = thr[int(p)]
        axL.axvline(t, color="#777", lw=0.8, ls="--", alpha=0.6, zorder=3)
        axL.text(t, ymax * 0.99, f"p{p}", rotation=90, va="top", ha="right", fontsize=7, color="#444")
    axL.axvline(mean_d, color="#b00", lw=1.0, ls=":", zorder=4)
    axL.text(mean_d, ymax * 0.55, "  mean", color="#b00", fontsize=8, va="center")
    axL.set_xlabel("high-D pairwise distance")
    axL.set_ylabel("number of pairs (histogram)", color="#3a5a8a")

    axc = axL.twinx()                                 # cumulative % of pairs (the user-requested curve)
    cum = np.cumsum(counts) / counts.sum() * 100.0
    axc.plot(edges[1:], cum, color=_CUM, lw=2.0, zorder=5, label="cumulative % of pairs")
    axc.set_ylim(0, 100)
    axc.set_ylabel("cumulative % of pairs", color=_CUM)
    axc.axhline(near_pct, color=_NEAR, lw=0.8, ls="--", alpha=0.8)
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    axc.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, framealpha=0.9)
    axL.set_title(f"{dataset} (SNR={snr_label(snr)}): high-D distances concentrate — "
                  f"the near band is a thin left tail", fontsize=10)

    # ---- Right: equal-weighting dilution — how many pairs each ρ averages over ----
    axR.barh([1, 0], [n_pairs, n_near], color=[_BULK, _NEAR], edgecolor="white", zorder=2)
    axR.set_yticks([1, 0])
    axR.set_yticklabels(["full ρ (p=100)\nranks ALL pairs",
                         f"first-mode near band\n(p{near_pct:.0f}) near pairs only"],
                        fontsize=9)
    axR.set_xlabel("number of pairs the metric averages over")
    for y, v in [(1, n_pairs), (0, n_near)]:
        axR.text(v + n_pairs * 0.01, y, f"{v:,}", va="center", ha="left", fontsize=9)
    axR.set_xlim(0, n_pairs * 1.18)
    ratio = (n_pairs - n_near) / max(n_near, 1)
    axR.set_title(f"full ρ is out-voted {ratio:.0f}:1 — near errors are\naveraged away "
                  f"(≈{near_pct:.0f}% of pairs)", fontsize=10)
    axR.grid(axis="x", alpha=0.25)

    fig.suptitle("Why the full (p=100) Shepard ρ hides near-neighbor fidelity", fontsize=12)
    out = Path(out_dir) / f"distance_distribution_snr{snr_label(snr)}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def recall_bias_figure(X_ambient, dataset, snr, out_dir, k=15, cutoffs=DEFAULT_CUTOFFS, seed=0):
    """Why recall@k / trustworthiness / continuity are NOT a fair near-neighbor metric.

    Three panels:
      1. Variable radius (schematic) — with a fixed k, a point in a DENSE region encloses its k
         neighbors in a tiny radius while a point in a SPARSE region needs a huge radius, so every
         point is judged on a different distance scale. A fixed-radius band uses ONE threshold for all.
      2. Hard threshold / near-tie (schematic) — the k-th and (k+1)-th neighbors can be almost
         equidistant, yet recall@k counts one as 1 and the other as 0; a tiny wobble flips membership.
         A distance metric scores the pair by its actual distance, so it barely moves.
      3. Real data — the per-point distance to the k-th neighbor (the actual recall@k radius) varies
         widely across points, vs the single fixed-radius p5 band threshold (vertical line).
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # ---- Panel 1: variable radius (dense vs sparse), schematic ----
    dense = np.column_stack([rng.normal(0.30, 0.045, 26), rng.normal(0.62, 0.045, 26)])
    sparse = np.column_stack([rng.normal(0.74, 0.11, 8), rng.normal(0.34, 0.11, 8)])
    ax1.scatter(dense[:, 0], dense[:, 1], s=16, color="#444", zorder=3)
    ax1.scatter(sparse[:, 0], sparse[:, 1], s=16, color="#444", zorder=3)
    kk = 5

    def _kth_radius(pts, center_idx, kk):
        c = pts[center_idx]
        dd = np.sort(np.linalg.norm(pts - c, axis=1))
        return c, dd[kk]                                  # distance to kk-th neighbor (excl. self at 0)

    cA, rA = _kth_radius(dense, 0, kk)
    cB, rB = _kth_radius(sparse, 0, kk)
    ax1.add_patch(plt.Circle(cA, rA, fill=False, color=_NEAR, lw=2.0, zorder=4))
    ax1.add_patch(plt.Circle(cB, rB, fill=False, color="#d62728", lw=2.0, zorder=4))
    ax1.annotate(f"dense point\nk={kk} radius small", cA, textcoords="offset points", xytext=(6, -36),
                 fontsize=8, color=_NEAR)
    ax1.annotate(f"sparse point\nk={kk} radius large", cB, textcoords="offset points", xytext=(6, 8),
                 fontsize=8, color="#d62728")
    ax1.set_title(f"1. recall@k uses a VARIABLE radius\n(same k, radius differs ≈{rB/rA:.0f}× here)",
                  fontsize=10)
    ax1.set_xlim(0, 1.05); ax1.set_ylim(0, 1.0); ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_aspect("equal")

    # ---- Panel 2: hard threshold / near-tie, schematic ----
    q = 0.06
    nb_x = np.array([0.20, 0.32, 0.43, 0.52, 0.585, 0.615, 0.70, 0.80])  # 5th=0.585, 6th=0.615 ~tie
    ax2.scatter([q], [0.5], s=120, marker="*", color="#16407a", zorder=4)
    ax2.text(q, 0.40, "query", ha="center", fontsize=8, color="#16407a")
    cut = 0.5 * (nb_x[kk - 1] + nb_x[kk])                  # boundary between 5th and 6th neighbor
    for i, x in enumerate(nb_x):
        inside = i < kk
        ax2.scatter([x], [0.5], s=44,
                    color=(_NEAR if inside else "#bbbbbb"),
                    edgecolor=("k" if i in (kk - 1, kk) else "none"), zorder=3)
        ax2.text(x, 0.565, f"{i+1}", ha="center", fontsize=7,
                 color=("k" if i in (kk - 1, kk) else "#888"))
    ax2.axvline(cut, color="#d62728", lw=1.4, ls="--", zorder=2)
    ax2.text(cut, 0.74, f"k={kk} cutoff", color="#d62728", ha="center", fontsize=8)
    ax2.annotate("", xy=(nb_x[kk], 0.30), xytext=(nb_x[kk - 1], 0.30),
                 arrowprops=dict(arrowstyle="<->", color="#444", lw=1.0))
    ax2.text(cut, 0.245, "near-tie:\n5th counts=1, 6th counts=0", ha="center", fontsize=8, color="#444")
    ax2.text(0.5, 0.06,
             "a tiny wobble swaps 5th↔6th → recall jumps;\n"
             "Shepard ρ scores by actual distance → barely moves",
             ha="center", fontsize=8.5, color="#333",
             bbox=dict(boxstyle="round", fc="#fff8f0", ec="#f0a"))
    ax2.set_title("2. recall@k is a HARD 0/1 threshold\n(over-penalizes near-ties)", fontsize=10)
    ax2.set_xlim(0, 0.9); ax2.set_ylim(0, 0.85); ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlabel("distance from query →", fontsize=8)

    # ---- Panel 3: real data — per-point k-NN radius spread vs the fixed p5 band ----
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_ambient)
    dists, _ = nn.kneighbors(X_ambient)
    radius = dists[:, -1]                                  # distance to the k-th neighbor, per point
    d_hd = condensed_distances(X_ambient)
    fixed_r5 = float(np.percentile(d_hd, 5))
    ax3.hist(radius, bins=40, color="#bbbbbb", edgecolor="white", linewidth=0.3)
    ax3.axvline(fixed_r5, color=_NEAR, lw=2.0,
                label=f"fixed p5 radius = {fixed_r5:.2f}\n(same for every point)")
    ax3.set_title(f"3. real data ({dataset}): per-point recall@{k} radius is not one\n"
                  f"value (ranges {radius.min():.1f}–{radius.max():.1f}) — p5 is one fixed line",
                  fontsize=10)
    ax3.set_xlabel(f"per-point distance to the {k}-th neighbor (recall@{k} radius)")
    ax3.set_ylabel("number of points")
    ax3.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Why recall@k / trustworthiness / continuity are a biased near-neighbor reference",
                 fontsize=12)
    out = Path(out_dir) / f"recall_bias_snr{snr_label(snr)}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, dpi=130); plt.close(fig)
    return out
