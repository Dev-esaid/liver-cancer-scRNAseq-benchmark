from matplotlib.pylab import seed

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_methods.bbknn import run_bbknn

def run_experiment1_baseline(adata, seeds=(0,1,2), method="bbknn"):
    results = {}

    for seed in seeds:
        set_global_seed(seed, use_torch=False)
        ad = adata.copy()

        if method == "bbknn":
            ad = run_bbknn(ad, batch_key="dataset", n_pcs=50, neighbors_within_batch=5, seed=seed)
        else:
            raise ValueError(f"Unknown method: {method}")

        results[(method, seed)] = {
            "n_cells": ad.n_obs,
            "has_umap": "X_umap" in ad.obsm
        }

    return results
