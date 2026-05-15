from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    import scvi
except Exception:
    torch = None
    scvi = None

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.plotting import (
    plot_umap_pub,
    plot_marker_dotplot_pub,
    plot_metric_summary,
    subsample_for_plotting,
)
from thesis_project.Integration.Integration_benchmark.io import save_run_artifacts
from thesis_project.Integration.Integration_benchmark.metrics import integration_metrics
from thesis_project.Integration.Integration_benchmark.graph import build_neighbors, build_umap, build_leiden


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScanviConfig:
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    counts_layer: str = "counts"
    unlabeled_category: str = "Unclassified"

    # logging/progress
    train_progress_bar: bool = True

    # exclude datasets with no counts
    exclude_datasets_no_counts: tuple = ()

    # model
    n_latent: int = 50
    n_hidden: int = 128
    n_layers: int = 2
    gene_likelihood: str = "nb"

    # training
    use_gpu: bool = True
    batch_size: int = 2048
    max_epochs: int = 200
    early_stopping: bool = True
    early_stopping_patience: int = 10
    num_workers: int = 4
    pin_memory: bool = False

    # downstream
    neighbors_k: int = 50
    umap_min_dist: float = 0.35
    umap_spread: float = 1.0
    leiden_resolution: float = 1.0

    # metrics
    n_isolated: Optional[int] = None
    lisi_subsample: Optional[int] = None
    compute_trajectory: bool = False

    # covariates
    categorical_covariates: tuple = (
        "donor_id", "technology", "platform", "tissue", "tumor_status"
    )

    # run identity
    run_tag: str = "scanvi"
    seed: int = 0

    # plotting
    plot_covariates: tuple = (
        "dataset", "major_celltype_l1", "tumor_status", "technology", "cancer_type"
    )
    plot_extra_covariates: tuple = (
        "platform", "tissue", "compartment", "disease_group", "donor_id"
    )
    plot_subsample_n: int = 200_000
    plot_subsample_stratify_by: Optional[str] = "dataset"

    # persistence
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _device(cfg: ScanviConfig) -> str:
    if torch is None:
        return "cpu"
    return "gpu" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu"


def _counts_min(ad: sc.AnnData, layer: str) -> float:
    X = ad.layers[layer]
    if hasattr(X, "data"):
        if X.data.size == 0:
            return 0.0
        return float(np.min(X.data))
    return float(np.min(X))


def _fraction_non_integer_like(
    ad: sc.AnnData, layer: str, sample_n: int = 200_000, seed: int = 0
) -> float:
    rng = np.random.default_rng(seed)
    X = ad.layers[layer]
    if hasattr(X, "data"):
        data = X.data
        if data.size == 0:
            return 0.0
        take = min(sample_n, data.size)
        idx = (
            rng.choice(data.size, size=take, replace=False)
            if take < data.size
            else np.arange(data.size)
        )
        v = data[idx]
    else:
        arr = np.asarray(X)
        flat = arr.ravel()
        if flat.size == 0:
            return 0.0
        take = min(sample_n, flat.size)
        idx = (
            rng.choice(flat.size, size=take, replace=False)
            if take < flat.size
            else np.arange(flat.size)
        )
        v = flat[idx]
    nearest = np.rint(v)
    return float(np.mean(np.abs(v - nearest) > 1e-6))


def _as_categorical_with_fill(s: pd.Series, fill_value: str) -> pd.Categorical:
    if pd.api.types.is_categorical_dtype(s):
        cat = s
        if fill_value not in cat.cat.categories:
            cat = cat.cat.add_categories([fill_value])
        cat = cat.fillna(fill_value).astype(str)
        return pd.Categorical(cat)
    else:
        filled = s.where(~pd.isna(s), other=fill_value).astype(str)
        return pd.Categorical(filled)


def _sanitize_for_scanvi(
    adata: sc.AnnData, cfg: ScanviConfig
) -> Tuple[sc.AnnData, List[str]]:
    if cfg.counts_layer not in adata.layers:
        raise ValueError(
            f"Missing counts layer: adata.layers['{cfg.counts_layer}']"
        )
    if cfg.batch_key not in adata.obs:
        raise ValueError(f"Missing obs batch_key='{cfg.batch_key}'")
    if cfg.label_key not in adata.obs:
        raise ValueError(f"Missing obs label_key='{cfg.label_key}'")

    ad = adata[
        ~adata.obs[cfg.batch_key].isin(cfg.exclude_datasets_no_counts)
    ].copy()
    ad.obs[cfg.batch_key] = ad.obs[cfg.batch_key].astype(str)

    cmin = _counts_min(ad, cfg.counts_layer)
    if cmin < -1e-8:
        raise ValueError(
            f"Counts layer '{cfg.counts_layer}' has negative values (min={cmin}). "
            "scANVI expects non-negative raw counts."
        )
    frac_nonint = _fraction_non_integer_like(ad, cfg.counts_layer, seed=cfg.seed)
    if frac_nonint > 0.05:
        warnings.warn(
            f"[scANVI] counts_layer='{cfg.counts_layer}' looks non-integer-like "
            f"(~{frac_nonint:.1%} sampled entries not near integers). "
            "If this layer is log-normalized/scaled, results will be invalid. "
            "Please ensure raw UMI counts."
        )

    raw_lab = ad.obs[cfg.label_key].copy()
    if raw_lab.notna().any():
        if (raw_lab.dropna().astype(str) == str(cfg.unlabeled_category)).any():
            warnings.warn(
                f"[scANVI] unlabeled_category='{cfg.unlabeled_category}' already "
                "appears in non-NA labels. Those cells will be treated as unlabeled."
            )
    ad.obs[cfg.label_key] = _as_categorical_with_fill(raw_lab, cfg.unlabeled_category)

    covs: List[str] = []
    for c in cfg.categorical_covariates:
        if c in ad.obs and c not in (cfg.batch_key, cfg.label_key):
            ad.obs[c] = _as_categorical_with_fill(ad.obs[c], "Unknown").astype(str)
            covs.append(c)

    return ad, covs


def _train_scanvi(
    ad: sc.AnnData, cfg: ScanviConfig, covs: List[str],
    perf: PerfLogger, outdir: str,
):
    if scvi is None:
        raise ImportError("scvi-tools not installed. Install with: pip install scvi-tools")

    accelerator = _device(cfg)

    perf.start("setup_anndata")
    scvi.model.SCANVI.setup_anndata(
        ad,
        layer=cfg.counts_layer,
        batch_key=cfg.batch_key,
        labels_key=cfg.label_key,
        unlabeled_category=cfg.unlabeled_category,
        categorical_covariate_keys=covs if covs else None,
    )
    perf.end("setup_anndata")

    perf.start("train")
    model = scvi.model.SCANVI(
        ad,
        n_latent=cfg.n_latent,
        n_hidden=cfg.n_hidden,
        n_layers=cfg.n_layers,
        gene_likelihood=cfg.gene_likelihood,
    )
    model.train(
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        accelerator=accelerator,
        devices=1,
        early_stopping=cfg.early_stopping,
        early_stopping_patience=cfg.early_stopping_patience,
        enable_progress_bar=cfg.train_progress_bar,
        datasplitter_kwargs={
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
        },
    )
    perf.end("train")

    model.save(os.path.join(outdir, "scanvi_model"), overwrite=True)
    return model


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    adata_in: sc.AnnData, outdir: str, cfg: ScanviConfig
) -> Tuple[sc.AnnData, Dict[str, Any], pd.DataFrame]:
    """
    scANVI run:
      sanitize → snapshot adata_pre → train scANVI → latent embedding
      → neighbors → UMAP/Leiden → metrics → save → plots

    Variable naming:
      ad            : sanitized + integrated AnnData
      adata_pre     : snapshot of ad after _sanitize_for_scanvi, BEFORE training.
                      Same cell set as ad (exclude_datasets_no_counts applied);
                      required for scib comparison metrics.
      key_latent    : "X_scanvi"  — scANVI latent representation
      neigh_key     : "neighbors_scanvi"
      conn_key      : ad.uns[neigh_key]["connectivities_key"]
                      → "neighbors_scanvi_connectivities"
      cluster_key   : "leiden_scanvi"

      output_type="embed": scANVI outputs a latent embedding (n_latent dims).
        kBET/LISI recompute their own kNN from key_latent internally.

      Note on adata_pre for pcr_comparison: scANVI works on raw counts, so
      adata_pre.X (or the counts layer) is not log-normalized. scib's
      pcr_comparison computes PCA on adata_pre.X; for interpretability, you
      may want to pass a log-normalized copy instead. The current implementation
      passes the post-sanitization object, which is methodologically consistent
      with how other count-based methods treat adata_pre.
    """
    os.makedirs(outdir, exist_ok=True)
    set_global_seed(cfg.seed, use_torch=True)

    perf = PerfLogger(track_gpu=True)

    # ------------------------------------------------------------------
    # 1. Sanitize (subset excluded datasets, validate counts, cast labels)
    # ------------------------------------------------------------------
    perf.start("subset_sanitize")
    ad, covs = _sanitize_for_scanvi(adata_in, cfg)
    perf.end("subset_sanitize")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← after sanitization (same cell set as ad,
    #    exclude_datasets_no_counts already applied), BEFORE training.
    #    Required for scib comparison metrics (pcr_comparison,
    #    hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Train scANVI
    # ------------------------------------------------------------------
    model = _train_scanvi(ad, cfg, covs=covs, perf=perf, outdir=outdir)

    # ------------------------------------------------------------------
    # 4. Extract latent representation
    #    key_latent = "X_scanvi"
    # ------------------------------------------------------------------
    perf.start("latent")
    key_latent = f"X_{cfg.run_tag}"     # "X_scanvi"
    ad.obsm[key_latent] = model.get_latent_representation()
    perf.end("latent")

    # ------------------------------------------------------------------
    # 5. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    perf.start("neighbors")
    neigh_key = f"neighbors_{cfg.run_tag}"   # "neighbors_scanvi"
    build_neighbors(
        ad,
        use_rep=key_latent,
        n_neighbors=cfg.neighbors_k,
        key_added=neigh_key,
        random_state=cfg.seed,
    )
    perf.end("neighbors")

    perf.start("umap")
    build_umap(
        ad,
        neighbors_key=neigh_key,
        key_umap=f"X_umap_{cfg.run_tag}",
        min_dist=cfg.umap_min_dist,
        spread=cfg.umap_spread,
        random_state=cfg.seed,
    )
    perf.end("umap")

    perf.start("leiden")
    build_leiden(
        ad,
        neighbors_key=neigh_key,
        key_leiden=f"leiden_{cfg.run_tag}",
        resolution=cfg.leiden_resolution,
        random_state=cfg.seed,
    )
    perf.end("leiden")

    # ------------------------------------------------------------------
    # 6. Standard embedding alias (must be set BEFORE metrics call)
    # ------------------------------------------------------------------
    ad.obsm["X_emb"] = np.asarray(ad.obsm[key_latent]).copy()

    # ------------------------------------------------------------------
    # 7. Metrics
    #
    #    conn_key = "neighbors_scanvi_connectivities"  (from uns)
    #    dist_key = "neighbors_scanvi_distances"        (from uns)
    #    neighbors_uns_key = "neighbors_scanvi"         (= neigh_key, explicit)
    #
    #    output_type="embed": scANVI outputs a latent embedding.
    #      kBET/LISI recompute their own kNN from key_latent internally.
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]  # "neighbors_scanvi_connectivities"
    dist_key = ad.uns[neigh_key]["distances_key"]       # "neighbors_scanvi_distances"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*pandas.value_counts.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=".*in1d.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=".*anndata2ri.*")

        metrics = integration_metrics(
            ad,
            batch_key=cfg.batch_key,
            label_key=cfg.label_key,
            cluster_key=f"leiden_{cfg.run_tag}",
            emb_key=key_latent,
            conn_key=conn_key,
            dist_key=dist_key,
            neighbors_uns_key=neigh_key,
            output_type="embed",
            adata_pre=adata_pre,
            compute_trajectory=cfg.compute_trajectory,
            n_isolated=cfg.n_isolated,
            lisi_subsample=cfg.lisi_subsample,
            organism="human",
            verbose=False,
        )
    perf.end("metrics")

    perf_df = perf.to_df()

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    save_run_artifacts(
        outdir,
        metrics=metrics,
        config=cfg,
        perf_df=perf_df,
        adata=ad,
        save_h5ad=cfg.save_h5ad,
        h5ad_name="adata_scanvi.h5ad",
    )

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    perf.start("plots")
    plot_dir = os.path.join(outdir, "plots_pub")
    os.makedirs(plot_dir, exist_ok=True)

    ad_plot = subsample_for_plotting(
        ad,
        n=cfg.plot_subsample_n,
        seed=cfg.seed,
        stratify_by=cfg.plot_subsample_stratify_by,
    )

    umap_key = f"X_umap_{cfg.run_tag}"
    title_prefix = f"scANVI — {cfg.run_tag} (n={ad_plot.n_obs:,})"

    covs_plot = list(cfg.plot_covariates) + [
        c for c in cfg.plot_extra_covariates if c not in cfg.plot_covariates
    ]
    covs_plot = [c for c in covs_plot if c in ad_plot.obs]

    for cov in covs_plot:
        plot_umap_pub(
            ad_plot, umap_key=umap_key, color=cov,
            title_prefix=title_prefix, outdir=plot_dir, alpha=0.75,
        )

    plot_umap_pub(
        ad_plot, umap_key=umap_key, color=f"leiden_{cfg.run_tag}",
        title_prefix=title_prefix, outdir=plot_dir, alpha=0.75,
    )

    if cfg.label_key in ad_plot.obs:
        plot_marker_dotplot_pub(ad_plot, groupby=cfg.label_key, outdir=plot_dir)

    plot_metric_summary(
        metrics, perf_df, outdir=plot_dir,
        title=f"scANVI metrics — {cfg.run_tag}",
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return ad, metrics, perf_df