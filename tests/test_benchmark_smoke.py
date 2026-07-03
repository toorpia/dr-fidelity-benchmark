"""Driver-level smoke test: the outliers dataset runs end-to-end through run/benchmark.py.

Uses the README smoke config (--n 300 --dim 50 --seeds 3 --snr inf) restricted to two fast methods
(one deterministic, one stochastic) with figures off and a tmp output dir, so the whole path --
generation, embedding, OSP metric block, merge, aggregation -- is exercised in seconds.
"""
import numpy as np
import pandas as pd

from run.benchmark import main as benchmark_main


def test_outliers_benchmark_smoke(tmp_path):
    rc = benchmark_main(["--dataset", "outliers", "--methods", "PCA,PyMDE",
                         "--n", "300", "--dim", "50", "--seeds", "3", "--snr", "inf",
                         "--out", str(tmp_path / "results"), "--figdir", str(tmp_path / "figures"),
                         "--no-figures", "--no-per-point", "--bootstrap", "200"])
    assert rc == 0
    per_run = pd.read_csv(tmp_path / "results" / "metrics_per_run.csv")
    assert set(per_run.method) == {"PCA", "PyMDE"}
    assert (per_run.dataset == "outliers").all()
    # the OSP block is present and finite for every run — vs-ambient only (the primary axis)
    for col in ("osp_median__vs_ambient", "osp_min__vs_ambient",
                "iso_rank_delta_mean__vs_ambient", "osp_o0__vs_ambient"):
        assert col in per_run.columns
        assert np.isfinite(per_run[col]).all()
    assert not any(c.startswith("osp") and c.endswith("__vs_truth") for c in per_run.columns)
    agg = pd.read_csv(tmp_path / "results" / "metrics_aggregated.csv")
    assert ((agg.dataset == "outliers") & (agg.metric == "osp_median__vs_ambient")).any()
    # embeddings archived per seed
    assert len(list((tmp_path / "results" / "embeddings" / "outliers").rglob("seed*.npy"))) == 4
