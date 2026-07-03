"""Regenerate all figures from saved embeddings + the aggregated metrics table.

Useful after changing `run/figures.py` (e.g. the Shepard density heatmap) so figures can be rebuilt
WITHOUT re-running the DR methods. Reconstructs each dataset deterministically (same args as the
benchmark) to recover the ambient features and coloring variable, loads the representative embedding
per method from ``results/embeddings/``, and redraws Shepard-density / embedding-panel figures plus
the aggregated curve / heatmap figures.

    python run/refigure.py --dataset all --n 1000 --dim 768 --snr inf 4 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods import default_methods  # noqa: E402  (light import; torch is lazy)
from run import figures  # noqa: E402
from synth import add_noise, list_datasets, make_dataset  # noqa: E402


def parse_snr(tokens):
    out = []
    for t in tokens:
        t = str(t).lower()
        out.append(float("inf") if t in ("inf", "clean", "none") else float(t))
    return out


def noise_rng(data_seed, snr):
    key = 10 ** 9 if not np.isfinite(snr) else int(round(snr * 1000))
    return np.random.default_rng(np.random.SeedSequence([int(data_seed), key]))


def load_representative(emb_root, dataset, snr_lab, method):
    d = Path(emb_root) / dataset / f"snr{snr_lab}" / method
    if not d.is_dir():
        return None
    files = sorted(d.glob("seed*.npy"), key=lambda p: int(p.stem.replace("seed", "")))
    return np.load(files[0]) if files else None


def main(argv=None):
    p = argparse.ArgumentParser(description="Regenerate figures from saved embeddings")
    p.add_argument("--dataset", default="all")
    p.add_argument("--methods", default="all")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--snr", nargs="+", default=["inf"])
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--figdir", default=str(ROOT / "figures"))
    args = p.parse_args(argv)

    datasets = list_datasets() if args.dataset == "all" else args.dataset.split(",")
    methods = default_methods() if args.methods == "all" else args.methods.split(",")
    snrs = parse_snr(args.snr)
    emb_root = Path(args.out) / "embeddings"
    agg_path = Path(args.out) / "metrics_aggregated.csv"
    agg = pd.read_csv(agg_path) if agg_path.exists() else None
    if agg is not None:
        agg["snr"] = agg["snr"].astype(float)

    for dataset in datasets:
        base = make_dataset(dataset, n=args.n, d=args.dim, seed=args.data_seed)
        ds_figdir = Path(args.figdir) / dataset
        ds_figdir.mkdir(parents=True, exist_ok=True)
        for snr in snrs:
            snr_lab = figures.snr_label(snr)
            X = add_noise(base["clean"], snr, noise_rng(args.data_seed, snr))
            # explanatory figures depend only on the high-D features X (no embedding needed)
            figures.distance_distribution(X, dataset, snr, ds_figdir)
            figures.recall_bias_figure(X, dataset, snr, ds_figdir)
            rep = {}
            for m in methods:
                Y = load_representative(emb_root, dataset, snr_lab, m)
                if Y is not None:
                    rep[m] = Y
            if rep:
                figures.shepard_scatter(X, rep, dataset, snr, ds_figdir)
                # the population gallery (majority/minority markers) supersedes the plain
                # embeddings panel on the populations dataset
                if base.get("population") is None:
                    figures.embedding_panels(rep, base["color_value"], base["color_name"],
                                             dataset, snr, ds_figdir)
                if base.get("outlier_idx") is not None:
                    figures.outlier_gallery(rep, base["labels"], base["outlier_idx"],
                                            dataset, snr, ds_figdir,
                                            outlier_dir=base.get("outlier_dir"))
                if base.get("population") is not None:
                    figures.population_gallery(rep, base["labels"], base["population"],
                                               dataset, snr, ds_figdir)
            if agg is not None:
                present = [m for m in methods
                           if ((agg.dataset == dataset) & (agg.method == m)).any()]
                # on outliers/populations the membership-pair-ρ scatter is the placement
                # reading; the plain near-vs-global scatter is redundant there
                if dataset == "outliers":
                    figures.outlier_score_scatter(agg, dataset, snr, present, ds_figdir)
                elif dataset == "populations":
                    figures.population_score_scatter(agg, dataset, snr, present, ds_figdir)
                else:
                    figures.score_scatter(agg, dataset, snr, present, ds_figdir)
                figures.summary_heatmap(agg, dataset, snr, present, ds_figdir)
            print(f"refigured {dataset} snr={snr_lab}  (methods: {list(rep)})")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
