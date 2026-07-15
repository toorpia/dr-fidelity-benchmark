"""Time-series (continuous change) probe: multi-series high-D random walks.

Embeds the concatenated multi-series random walk (``synth.random_walk``) with every method and
records trajectory-fidelity readouts built from the repo's standard rank statistic:

* ``within_series_shepard`` -- Spearman rho over all SAME-series pairs (the trajectory's own
  shape, local jitter through global extent).
* ``within_series_near_shepard`` -- the same restricted to small time lags (|dt| <= ``lag``):
  the saw-tooth step structure (LOCAL readout).
* ``cross_series_shepard`` -- rho over pairs from DIFFERENT series: the mutual arrangement of
  the series (GLOBAL readout -- the radial divergence).
* ``full_shepard`` -- all pairs (the standard global number).

toorPIA runs through the main benchmark's ``basemap_embedding`` endpoint, cache-first under
``external_embeddings/random_walk/``.

    python run/timeseries_probe.py --ndim 50 --npoints 500 --n-series 6
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

import torch  # noqa: E402
torch.set_num_threads(1)
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from methods import SkipMethod, default_methods, get_method  # noqa: E402
from synth.random_walk import make_random_walks  # noqa: E402


def set_seeds(seed):
    np.random.seed(int(seed)); torch.manual_seed(int(seed))


def median_tortuosity(P, series, w):
    """Median path/chord ratio over all sliding |Δt|=w windows within each series.

    ``path`` is the sum of the w consecutive step lengths, ``chord`` the straight-line distance
    between the window's endpoints. A diffusive (jagged) trajectory has path/chord ≈ √w; a
    smoothed (ballistic) rendering has path/chord ≈ 1 — the value-level signature of the
    saw-tooth that rank statistics cannot see."""
    taus = []
    for s in np.unique(series):
        seg = np.asarray(P)[series == s]
        step = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        cs = np.concatenate([[0.0], np.cumsum(step)])
        path = cs[w:] - cs[:-w]
        chord = np.linalg.norm(seg[w:] - seg[:-w], axis=1)
        ok = chord > 0
        taus.append(path[ok] / chord[ok])
    return float(np.median(np.concatenate(taus)))


def trajectory_metrics(d_hd, d_2d, series, t, lag, X=None, Y=None):
    n = len(series)
    iu = np.triu_indices(n, 1)
    same = series[iu[0]] == series[iu[1]]
    near = same & (np.abs(t[iu[0]] - t[iu[1]]) <= lag)
    out = {
        "full_shepard": float(spearmanr(d_hd, d_2d).statistic),
        "within_series_shepard": float(spearmanr(d_hd[same], d_2d[same]).statistic),
        "within_series_near_shepard": float(spearmanr(d_hd[near], d_2d[near]).statistic),
        "cross_series_shepard": float(spearmanr(d_hd[~same], d_2d[~same]).statistic),
    }
    if X is not None and Y is not None:
        # saw-tooth preservation: excess tortuosity of the 2-D trajectory over |Δt|=lag windows,
        # normalized by the high-D trajectory's own excess tortuosity (1 = the data's roughness
        # fully rendered, 0 = smoothed to a ballistic ribbon)
        tau_hd = median_tortuosity(X, series, lag)
        tau_2d = median_tortuosity(Y, series, lag)
        out["step_structure"] = float((tau_2d - 1.0) / (tau_hd - 1.0))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="time-series probe (multi-series random walks)")
    p.add_argument("--ndim", type=int, default=50)
    p.add_argument("--npoints", type=int, default=500)
    p.add_argument("--n-series", type=int, default=8)
    p.add_argument("--step", type=float, default=0.001)
    p.add_argument("--lag", type=int, default=20,
                   help="time-lag window of the LOCAL (saw-tooth) readout")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures" / "random_walk"))
    args = p.parse_args(argv)

    base = make_random_walks(ndim=args.ndim, npoints=args.npoints, n_series=args.n_series,
                             step=args.step, seed=args.data_seed)
    X, series, t = base["clean"], base["series"], base["t"]
    d_hd = pdist(X)
    tag = f"n{len(X)}_d{args.ndim}_s{args.n_series}"
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figdir); figdir.mkdir(parents=True, exist_ok=True)
    print(f"random-walk probe: {args.n_series} series x {args.npoints} steps x {args.ndim} dims "
          f"= {len(X)} points; local lag <= {args.lag}", flush=True)

    rows, gallery = [], {}
    for method in default_methods():
        m = get_method(method)
        seeds = range(args.seeds) if m.stochastic else [0]
        ctx = {"root": str(ROOT), "dataset": "random_walk", "snr": float("inf"), "tag": tag}
        k = 0
        for seed in seeds:
            set_seeds(seed)
            try:
                Y = m.embed(X, seed=seed, device=args.device, context=ctx)
            except SkipMethod as e:
                print(f"  [skip] {method}: {e}"); break
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {method} seed{seed}: {type(e).__name__}: {e}"); continue
            emb_dir = out_dir / "embeddings" / "random_walk" / method
            emb_dir.mkdir(parents=True, exist_ok=True)
            np.save(emb_dir / f"seed{seed}.npy", Y)
            if seed == min(seeds):
                gallery[method] = Y
            row = dict(dataset="random_walk", snr=float("inf"), method=method, seed=seed,
                       stochastic=m.stochastic, n=len(X), d=args.ndim)
            row.update(trajectory_metrics(d_hd, pdist(Y), series, t, args.lag, X=X, Y=Y))
            rows.append(row); k += 1
        if k:
            print(f"  {method:16s} runs={k}", flush=True)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(out_dir / "random_walk_per_run.csv", index=False)
    from run.aggregate import aggregate_runs
    agg = aggregate_runs(per_run)          # long format: median + bootstrap 95% CI per metric
    agg.to_csv(out_dir / "random_walk_aggregated.csv", index=False)
    print("\nmedian over seeds:")
    print(per_run.groupby("method", sort=False).median(numeric_only=True)
          .drop(columns=["seed", "n", "d"]).round(3).to_string())

    from run import figures
    if gallery:
        figures.trajectory_gallery(gallery, base["color_value"], series, args.npoints, figdir)
    figures.random_walk_geometry(args.ndim, args.npoints, args.n_series, args.step,
                                 args.data_seed, figdir)
    print(f"\nDONE. rows={len(per_run)} -> {out_dir}/random_walk_*.csv ; figures -> {figdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
