from __future__ import annotations

import os
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime
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
# Logging helpers (unchanged)
# ---------------------------------------------------------------------------

def _make_logger(outdir: str, name: str = "scvi") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"{name}_train.log")
    logger = logging.getLogger(f"{name}_{outdir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == log_path
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path, mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    return logger


def _log(
    msg: str, *, outdir: str, cfg, logger: Optional[logging.Logger] = None
) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} | {msg}"
    if getattr(cfg, "verbose", False):
        print(line)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")
    if logger is not None:
        logger.info(msg)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScviConfig:
    # metadata
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    counts_layer: str = "counts"

    # logging / progress
    verbose: bool = True
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
    early_stopping_patience: int = 20
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
    compute_trajectory: bool = True

    # covariates
    categorical_covariates: tuple = ("platform",)

    # run identity
    run_tag: str = "scvi"
    seed: int = 0

    # plotting
    plot_covariates: tuple = (
        "dataset", "major_celltype_l1", "technology", "sequencer"
    )
    plot_extra_covariates: tuple = ("platform", "donor_id")
    plot_subsample_n: int = 200_000
    plot_subsample_stratify_by: Optional[str] = "dataset"

    # persistence
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _device(cfg: ScviConfig) -> str:
    if torch is None:
        return "cpu"
    return "gpu" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu"


def _sanitize_for_scvi(adata, cfg: ScviConfig) -> Tuple[Any, List[str]]:
    if cfg.counts_layer not in adata.layers:
        raise ValueError(f"Missing counts layer: adata.layers['{cfg.counts_layer}']")
    if cfg.batch_key not in adata.obs:
        raise ValueError(f"Missing obs batch_key='{cfg.batch_key}'")
    if cfg.label_key not in adata.obs:
        raise ValueError(f"Missing obs label_key='{cfg.label_key}'")

    ad = adata[
        ~adata.obs[cfg.batch_key].isin(cfg.exclude_datasets_no_counts)
    ].copy()

    ad.obs[cfg.batch_key] = ad.obs[cfg.batch_key].astype(str)

    lab = ad.obs[cfg.label_key]
    if pd.api.types.is_categorical_dtype(lab):
        if "Unknown" not in lab.cat.categories:
            lab = lab.cat.add_categories(["Unknown"])
        lab = lab.fillna("Unknown").astype(str)
        ad.obs[cfg.label_key] = pd.Categorical(lab)
    else:
        lab = lab.where(~pd.isna(lab), other="Unknown").astype(str)
        ad.obs[cfg.label_key] = pd.Categorical(lab)

    covs: List[str] = []
    for c in cfg.categorical_covariates:
        if c in ad.obs and c != cfg.batch_key:
            col = ad.obs[c]
            if pd.api.types.is_categorical_dtype(col):
                if "Unknown" not in col.cat.categories:
                    col = col.cat.add_categories(["Unknown"])
                col = col.fillna("Unknown").astype(str)
            else:
                col = col.where(~pd.isna(col), other="Unknown").astype(str)
            ad.obs[c] = col
            covs.append(c)

    return ad, covs


def _train_scvi(
    ad, cfg: ScviConfig, covs: List[str], perf: PerfLogger, outdir: str
):
    if scvi is None:
        raise ImportError("scvi-tools not installed. Install with: pip install scvi-tools")

    accelerator = _device(cfg)
    train_logger = _make_logger(outdir, name="scvi")

    _log(f"[scVI] accelerator={accelerator}", outdir=outdir, cfg=cfg, logger=train_logger)
    _log(
        f"[scVI] covariates={covs if covs else 'None'}",
        outdir=outdir, cfg=cfg, logger=train_logger,
    )

    perf.start("setup_anndata")
    _log("[scVI] setup_anndata: start", outdir=outdir, cfg=cfg, logger=train_logger)
    scvi.model.SCVI.setup_anndata(
        ad,
        layer=cfg.counts_layer,
        batch_key=cfg.batch_key,
        categorical_covariate_keys=covs if covs else None,
    )
    _log("[scVI] setup_anndata: done", outdir=outdir, cfg=cfg, logger=train_logger)
    perf.end("setup_anndata")

    _log(
        f"[scVI] init model: n_latent={cfg.n_latent}, n_hidden={cfg.n_hidden}, "
        f"n_layers={cfg.n_layers}, gene_likelihood={cfg.gene_likelihood}",
        outdir=outdir, cfg=cfg, logger=train_logger,
    )

    perf.start("train")
    _log("[scVI] train: start", outdir=outdir, cfg=cfg, logger=train_logger)
    model = scvi.model.SCVI(
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
    _log("[scVI] train: done", outdir=outdir, cfg=cfg, logger=train_logger)
    perf.end("train")

    perf.start("save_model")
    _log("[scVI] saving model...", outdir=outdir, cfg=cfg, logger=train_logger)
    model.save(os.path.join(outdir, "scvi_model"), overwrite=True)
    _log("[scVI] model saved", outdir=outdir, cfg=cfg, logger=train_logger)
    perf.end("save_model")

    return model


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    adata_in, outdir: str, cfg: ScviConfig
) -> Tuple[Any, Dict[str, Any], Any]:
    """
    scVI integration run:
      subset/sanitize → snapshot adata_pre → train SCVI → latent
      → neighbors → UMAP/Leiden → metrics → save → plots

    Variable naming:
      ad            : sanitized AnnData (modified in-place during training)
      adata_pre     : snapshot after _sanitize_for_scvi (same cell set as ad,
                      exclude_datasets_no_counts applied), BEFORE training.
                      Required for scib comparison metrics (pcr_comparison,
                      hvg_conservation, cell_cycle_conservation).
      key_latent    : "X_scvi"  — scVI latent representation
      neigh_key     : "neighbors_scvi"
      conn_key      : ad.uns[neigh_key]["connectivities_key"]
                      → "neighbors_scvi_connectivities"
      cluster_key   : "leiden_scvi"

      output_type="embed": scVI outputs a latent embedding (n_latent dims).
        kBET/LISI recompute their own kNN from key_latent internally.

      Note on adata_pre.X: scVI works on raw counts, so adata_pre.X is raw
      counts. scib's pcr_comparison computes PCA on adata_pre.X; for
      interpretability this is less ideal than log-normalized data, but is
      methodologically consistent with how other count-based methods treat
      adata_pre.
    """
    os.makedirs(outdir, exist_ok=True)
    set_global_seed(cfg.seed, use_torch=True)

    perf = PerfLogger(track_gpu=True)

    # ------------------------------------------------------------------
    # 1. Subset + sanitize
    # ------------------------------------------------------------------
    perf.start("subset_sanitize")
    ad, covs = _sanitize_for_scvi(adata_in, cfg)
    perf.end("subset_sanitize")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← after sanitization (same cell set, same
    #    obs structure as ad), BEFORE training.
    #    Required for scib comparison metrics (pcr_comparison,
    #    hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Train scVI
    # ------------------------------------------------------------------
    model = _train_scvi(ad, cfg, covs=covs, perf=perf, outdir=outdir)

    # ------------------------------------------------------------------
    # 4. Extract latent representation
    #    key_latent = "X_scvi"
    # ------------------------------------------------------------------
    perf.start("latent")
    key_latent = f"X_{cfg.run_tag}"    # "X_scvi"
    ad.obsm[key_latent] = model.get_latent_representation()
    perf.end("latent")

    # ------------------------------------------------------------------
    # 5. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    perf.start("neighbors")
    neigh_key = f"neighbors_{cfg.run_tag}"   # "neighbors_scvi"
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
    #    conn_key = "neighbors_scvi_connectivities"  (from uns)
    #    dist_key = "neighbors_scvi_distances"        (from uns)
    #    neighbors_uns_key = "neighbors_scvi"         (= neigh_key, explicit)
    #
    #    output_type="embed": scVI outputs a latent embedding.
    #      kBET/LISI recompute their own kNN from key_latent internally.
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]  # "neighbors_scvi_connectivities"
    dist_key = ad.uns[neigh_key]["distances_key"]       # "neighbors_scvi_distances"

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
        h5ad_name="adata_scvi.h5ad",
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
    title_prefix = f"scVI — {cfg.run_tag} (n={ad_plot.n_obs:,})"

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
        title=f"scVI metrics — {cfg.run_tag}",
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return ad, metrics, perf_df