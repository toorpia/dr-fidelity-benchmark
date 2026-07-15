"""Supplement: bridge-connectivity diagnostic for the transition dataset (bottleneck gaps).

The transition dataset's defining geometry has two simultaneous features: seven dense
typical-state clusters and their closed ring connectivity (0 -> 1 -> ... -> 6 -> 0). The embedding
galleries show qualitatively which methods keep the ring; this script makes the connectivity
reading quantitative and committable.

For each cyclically adjacent cluster pair (i, i+1 mod 7) the **bottleneck gap** is the smallest
radius r at which cluster i and cluster i+1 become single-linkage-connected through that segment's
own bridge points (computed exactly as the minimax edge of the pooled points' minimum spanning
tree). A drawn bridge has a bottleneck of a few point spacings; a torn bridge has a void
comparable to the inter-cluster distance. Reported per bridge:

* ``gap_ratio``     -- bottleneck / 2-D centroid distance of the pair (scale-free; headline)
* ``gap_nn``        -- bottleneck / the method's median 2-D nearest-neighbor spacing
* ``torn``          -- ``gap_ratio > 1/3`` (the observed distributions separate cleanly:
  drawn bridges sit at <= 0.24, torn ones at >= 0.40, on every method at SNR=1)

Caveat, stated where the numbers are used: connectivity is NECESSARY for the ring reading, not
sufficient. A method can connect the ring by fusing the states at a hub (PCC) or by smearing the
clusters along it (Isomap) -- the distance-fidelity tables and the gallery carry that part. The
diagnostic separates the methods that draw seven dense, separated clusters (t-SNE, UMAP, DREAMS,
toorPIA): among those, only toorPIA keeps every bridge connected.

Reads the embeddings archived by ``run/benchmark.py`` (open-source methods are byte-reproducible
from the driver; toorPIA replays from its committed cache), writes ``results/bridge_gaps.csv``.

    python run/bridge_gaps.py            # canonical: SNR=1, D=768, N=1000, all methods
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from scipy.sparse.csgraph import minimum_spanning_tree  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from methods import default_methods  # noqa: E402
from synth import make_dataset  # noqa: E402

TORN_RATIO = 1.0 / 3.0


def bottleneck_gap(A: np.ndarray, B: np.ndarray, S: np.ndarray) -> float:
    """Smallest radius connecting point sets A and B through S (single linkage, exact).

    Equals the largest edge on the minimax path between the A-component and the B-component in
    the pooled minimum spanning tree.
    """
    P = np.vstack([A, B, S]) if len(S) else np.vstack([A, B])
    na, nb = len(A), len(B)
    mst = minimum_spanning_tree(squareform(pdist(P))).toarray()
    edges = sorted((mst[i, j], i, j) for i, j in zip(*np.nonzero(mst)))
    parent = list(range(len(P)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for w, i, j in edges:
        parent[find(i)] = find(j)
        if len({find(x) for x in range(na)} & {find(x) for x in range(na, na + nb)}):
            return float(w)
    return float("inf")


def main(argv=None):
    p = argparse.ArgumentParser(description="transition bridge-connectivity bottleneck gaps")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--snr", default="1")
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--emb-root", default=str(ROOT / "results" / "embeddings" / "transition"))
    p.add_argument("--out", default=str(ROOT / "results" / "bridge_gaps.csv"))
    args = p.parse_args(argv)

    snr_lab = "inf" if str(args.snr).lower() in ("inf", "clean", "none") else f"{float(args.snr):g}"
    base = make_dataset("transition", n=args.n, d=args.dim, seed=args.data_seed)
    labs, t = base["labels"], base["color_value"]
    K = int(labs.max()) + 1

    rows = []
    for method in default_methods():
        mdir = Path(args.emb_root) / f"snr{snr_lab}" / method
        seeds = sorted(int(f.stem[4:]) for f in mdir.glob("seed*.npy")) if mdir.exists() else []
        if not seeds:
            print(f"  [skip] {method}: no archived embeddings under {mdir} "
                  f"(run run/benchmark.py first)")
            continue
        for seed in seeds:
            Y = np.load(mdir / f"seed{seed}.npy")
            D2 = squareform(pdist(Y))
            np.fill_diagonal(D2, np.inf)
            nn = float(np.median(D2.min(axis=1)))
            for i in range(K):
                j = (i + 1) % K
                seg = (labs < 0) & ((t > i / K) & (t < (i + 1) / K) if i < K - 1 else (t > i / K))
                gap = bottleneck_gap(Y[labs == i], Y[labs == j], Y[seg])
                cdist = float(np.linalg.norm(Y[labs == i].mean(0) - Y[labs == j].mean(0)))
                rows.append(dict(method=method, seed=seed, bridge=f"{i}-{j}",
                                 gap=gap, gap_nn=gap / nn, gap_ratio=gap / cdist,
                                 torn=gap / cdist > TORN_RATIO))
        sub = pd.DataFrame([r for r in rows if r["method"] == method])
        med = sub.groupby("bridge").gap_ratio.median()
        print(f"  {method:9s} max gap_ratio={med.max():.2f}  "
              f"torn bridges (median over seeds)={(med > TORN_RATIO).sum()}/7")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nDONE. rows={len(df)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
