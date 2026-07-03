"""Supplement: ADDPLOT / out-of-sample test -- monitoring on an anomaly-free basemap.

The true monitoring scenario: the basemap is fitted on NORMAL data only (the outliers dataset's
bulk, no anomaly ever seen at fit time), and new points arrive afterwards, one at a time. The
added set holds CLUSTER-ANCHORED anomalies (``synth.outliers.make_anchored_addplot``: each shares
a normal cluster's clean profile and deviates 3 Rg along new dimensions orthogonal to everything
the normal data varies in -- a near-duplicate pair per cluster) plus fresh normal points as
controls. Two questions, in order:

1. **Detection** -- does a never-seen anomaly land visibly OUTSIDE the normal region at all
   (``anomaly_radius_ratio``: its 2-D distance from the bulk centroid over the bulk's median
   radius)?
2. **Source attribution** -- is the anomaly's DIRECTION from the map centroid closest to its own
   source cluster's direction (``attribution_accuracy``, ``angle_to_own_med``), and do the two
   near-duplicates of one cluster stay co-directional (``pair_angle_med``)? The direction of an
   addplot point is information: it should say WHICH normal cluster the anomaly comes from.

The HD reference (ambient features) resolves attribution 10/10 -- the anchor signal survives the
noise -- so a faithful map can too; ``hd_*`` columns record that ceiling per run.

Which methods can do this at all: PCA and Isomap have a deterministic out-of-sample ``transform``;
UMAP has a (seeded) ``transform``; toorPIA has a server-side ``addplot`` on the fitted map. t-SNE
(sklearn), PyMDE, and PCC optimize the fit coordinates directly and expose NO out-of-sample
operation -- reported as ``supported=False`` rows rather than silently dropped (for monitoring
that is itself the finding: adding data means re-fitting, and a re-fit re-arranges the map).

toorPIA honest note: ``addplot`` uses the server-side state of the fit in the SAME session, so the
test performs a live ``fit_transform`` + ``addplot`` per seed, committed as a self-consistent
cache pair (``basemap_fit``/``basemap_add``); the benchmark's fit cache holds no server state and
the server is not bit-deterministic across sessions.

    python run/addplot_test.py --seeds 3 --dim 768 --n 1000 --snr 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

from methods.external import load_external, save_external  # noqa: E402
from methods.toorpia_method import _coords_from_result  # noqa: E402
from run.aggregate import bootstrap_median_ci  # noqa: E402
from run.benchmark import noise_rng  # noqa: E402
from synth.noise import signal_variance  # noqa: E402
from synth.outliers import make_anchored_addplot, make_outliers  # noqa: E402

UNSUPPORTED = {"t-SNE": "sklearn TSNE has no out-of-sample transform",
               "PyMDE": "pymde optimizes fit coordinates; no transform",
               "PCC": "pccdr optimizes fit coordinates; no transform"}
SUMMARY_METRICS = ["anomaly_radius_ratio_med", "anomaly_radius_ratio_min", "attribution_accuracy",
                   "angle_to_own_med", "pair_angle_med", "add_bulk_radius_ratio"]
CACHE_MODES = ("basemap_fit", "basemap_add")


def _direction_angles(Y_bulk, labs_b, points, center):
    """Angle (deg) between each point's direction from ``center`` and each cluster centroid's."""
    K = int(labs_b.max()) + 1
    v_cl = np.array([Y_bulk[labs_b == k].mean(axis=0) for k in range(K)]) - center
    v_cl /= np.linalg.norm(v_cl, axis=1, keepdims=True)
    v_p = points - center
    v_p /= np.linalg.norm(v_p, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(v_p @ v_cl.T, -1.0, 1.0)))


def anchored_metrics(Yf, Ya, Xf, Xa, labs_b, anchor, n_anom) -> dict:
    """Detection + source-attribution metrics. Added rows ``[0:n_anom]`` = anchored anomalies
    (source cluster per row in ``anchor``), the rest are bulk controls."""
    row = {}
    for tag, (Zf, Za) in (("", (Yf, Ya)), ("hd_", (Xf, Xa))):
        c = Zf.mean(axis=0)
        r_bulk = float(np.median(np.linalg.norm(Zf - c, axis=1)))
        r_anom = np.linalg.norm(Za[:n_anom] - c, axis=1) / r_bulk
        ang = _direction_angles(Zf, labs_b, Za[:n_anom], c)
        own = ang[np.arange(n_anom), anchor]
        row[f"{tag}anomaly_radius_ratio_med"] = float(np.median(r_anom))
        row[f"{tag}anomaly_radius_ratio_min"] = float(r_anom.min())
        row[f"{tag}attribution_accuracy"] = float((ang.argmin(axis=1) == anchor).mean())
        row[f"{tag}angle_to_own_med"] = float(np.median(own))
        row[f"{tag}angle_to_own_max"] = float(own.max())
        if tag == "":
            v = (Za[:n_anom] - c) / np.linalg.norm(Za[:n_anom] - c, axis=1, keepdims=True)
            pairs = []
            for k in sorted(set(int(a) for a in anchor)):
                m = np.flatnonzero(anchor == k)
                if len(m) == 2:
                    a2 = float(np.degrees(np.arccos(np.clip(v[m[0]] @ v[m[1]], -1.0, 1.0))))
                    row[f"pair_angle_c{k}"] = a2
                    pairs.append(a2)
            row["pair_angle_med"] = float(np.median(pairs))
            row["pair_angle_max"] = float(np.max(pairs))
            # added bulk controls: median radius around the bulk centroid, 2-D over HD (~1 ideal)
            rms2 = float(np.sqrt(np.mean(np.sum((Yf - Yf.mean(axis=0)) ** 2, axis=1))))
            rmsh = float(np.sqrt(np.mean(np.sum((Xf - Xf.mean(axis=0)) ** 2, axis=1))))
            r2 = float(np.median(np.linalg.norm(Ya[n_anom:] - Yf.mean(axis=0), axis=1))) / rms2
            rh = float(np.median(np.linalg.norm(Xa[n_anom:] - Xf.mean(axis=0), axis=1))) / rmsh
            row["add_bulk_radius_ratio"] = r2 / rh
    return row


def embed_pair(method, seed, Xf, Xa, tag):
    """Fit on Xf and map Xa with the method's out-of-sample operation. Returns (Yf, Ya) or None."""
    np.random.seed(int(seed))
    if method == "PCA":
        from sklearn.decomposition import PCA
        mdl = PCA(n_components=2).fit(Xf)
        return mdl.transform(Xf), mdl.transform(Xa)
    if method == "Isomap":
        from sklearn.manifold import Isomap
        mdl = Isomap(n_neighbors=15, n_components=2).fit(Xf)
        return mdl.transform(Xf), mdl.transform(Xa)
    if method == "UMAP":
        import umap
        mdl = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit(Xf)
        return np.asarray(mdl.embedding_, float), np.asarray(mdl.transform(Xa), float)
    # toorPIA: live fit + addplot in one session (cache-first, self-consistent pair)
    fit_mode, add_mode = CACHE_MODES
    Yf = load_external(ROOT, "outliers", "toorpia", fit_mode, tag, seed)
    Ya = load_external(ROOT, "outliers", "toorpia", add_mode, tag, seed)
    if Yf is not None and Ya is not None:
        return Yf, Ya
    if not os.environ.get("TOORPIA_API_KEY"):
        return None
    from toorpia import toorPIA
    cols = [f"d{i}" for i in range(Xf.shape[1])]
    client = toorPIA()
    Yf = _coords_from_result(client.fit_transform(pd.DataFrame(Xf, columns=cols), label=None,
                                                  random_seed=int(seed),
                                                  vector_normalization=False))
    # addplot one row per call — the monitoring semantics: points arrive one at a time
    Ya = np.vstack([_coords_from_result(client.addplot(pd.DataFrame(Xa[i:i + 1], columns=cols)))
                    for i in range(len(Xa))])
    save_external(Yf, ROOT, "outliers", "toorpia", fit_mode, tag, seed)
    save_external(Ya, ROOT, "outliers", "toorpia", add_mode, tag, seed)
    return Yf, Ya


def main(argv=None):
    p = argparse.ArgumentParser(description="addplot / out-of-sample test (anomaly-free basemap)")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--snr", default="1")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--n-add-bulk", type=int, default=50)
    p.add_argument("--deviation", type=float, default=3.0)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures" / "outliers"))
    args = p.parse_args(argv)

    snr = float("inf") if str(args.snr).lower() in ("inf", "clean", "none") else float(args.snr)
    base = make_outliers(n=args.n, d=args.dim, seed=args.data_seed,
                         n_add_bulk=args.n_add_bulk, add_per_direction=0)
    snr_lab = "inf" if not np.isfinite(snr) else f"{snr:g}"

    # basemap: the normal points only, noised exactly as the benchmark's fit data
    labs = base["labels"]
    bulk_mask = labs != -1
    Xf = base["clean"].copy()
    anch = make_anchored_addplot(base, deviation=args.deviation, seed=args.data_seed)
    Xa = np.vstack([anch["clean"], base["addplot"]["clean"]])
    n_anom = len(anch["anchor"])
    if np.isfinite(snr):
        std = float(np.sqrt(signal_variance(base["clean"]) / snr))
        Xf = Xf + noise_rng(args.data_seed, snr).normal(scale=std, size=Xf.shape)
        key = int(round(snr * 1000))
        rng_add = np.random.default_rng(np.random.SeedSequence([args.data_seed, key, 777]))
        Xa = Xa + rng_add.normal(scale=std, size=Xa.shape)
    Xf_B, labs_B = Xf[bulk_mask], labs[bulk_mask]
    tag = f"n{args.n}_d{args.dim}_snr{snr_lab}"
    print(f"anomaly-free basemap: fit {len(Xf_B)} pts, add {n_anom} anchored anomalies "
          f"+ {len(Xa) - n_anom} bulk controls")

    rows, panels = [], {}
    for method in ["PCA", "Isomap", "UMAP", "toorPIA", "t-SNE", "PyMDE", "PCC"]:
        if method in UNSUPPORTED:
            rows.append(dict(method=method, seed=0, supported=False, note=UNSUPPORTED[method]))
            print(f"{method:9s} unsupported: {UNSUPPORTED[method]}")
            continue
        seeds = range(args.seeds) if method in ("UMAP", "toorPIA") else [0]
        for seed in seeds:
            res = embed_pair(method, seed, Xf_B, Xa, tag)
            if res is None:
                rows.append(dict(method=method, seed=seed, supported=False,
                                 note="no cache and TOORPIA_API_KEY unset"))
                print(f"{method:9s} seed{seed}: skipped (no cache, no key)")
                continue
            Yf, Ya = (np.asarray(z, float) for z in res)
            row = dict(method=method, seed=seed, supported=True, note="")
            row.update(anchored_metrics(Yf, Ya, Xf_B, Xa, labs_B, anch["anchor"], n_anom))
            rows.append(row)
            if seed == min(seeds):
                panels[method] = (Yf, Ya)
            print(f"{method:9s} seed{seed}: r/bulk={row['anomaly_radius_ratio_med']:.2f} "
                  f"attr={row['attribution_accuracy']:.2f} "
                  f"own_angle={row['angle_to_own_med']:.1f}deg "
                  f"pair={row['pair_angle_med']:.1f}deg")

    per_run = pd.DataFrame(rows)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out_dir / "addplot_per_run.csv", index=False)

    agg = []
    sup = per_run[per_run.supported == True]  # noqa: E712
    for method, g in sup.groupby("method"):
        for metric in SUMMARY_METRICS:
            if metric not in g.columns:
                continue
            vals = g[metric].to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if not vals.size:
                continue
            lo, hi = bootstrap_median_ci(vals)
            agg.append(dict(method=method, metric=metric,
                            median=float(np.median(vals)), ci_lo=lo, ci_hi=hi,
                            n_runs=int(vals.size)))
    pd.DataFrame(agg).to_csv(out_dir / "addplot_aggregated.csv", index=False)

    from run import figures
    if panels:
        figures.basemap_addplot_gallery(panels, labs_B, anch["anchor"], n_anom, Path(args.figdir))
    print(f"\nDONE. rows={len(per_run)} -> {out_dir}/addplot_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
