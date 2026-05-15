from __future__ import annotations

"""
Scanorama integration benchmark module.

Key design choices
------------------
1) Prefer the native ``scanorama.correct_scanpy`` workflow, which matches the
   Scanorama/scIB style of integration more closely than the Scanpy external
   PCA-space wrapper.
2) Keep the Scanpy external backend available as an explicit fallback.
3) Make backend behaviour explicit in the config so parameter meanings stay
   consistent.
4) Support optional HVG restriction before integration and optional feature
   scaling for the native backend.
5) Preserve namespaced output keys to avoid collisions across runs.
"""

import os
import warnings
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import scanpy as sc

try:
    import anndata as ad_mod
except Exception:
    ad_mod = None

SCANORAMA_BACKEND = None
try:
    import scanorama
    SCANORAMA_BACKEND = "scanorama"
except Exception:
    scanorama = None

try:
    import scanpy.external as sce
    if SCANORAMA_BACKEND is None:
        SCANORAMA_BACKEND = "scanpy.external"
except Exception:
    sce = None

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs
from thesis_project.Integration.Integration_benchmark.preprocess import run_pca
from thesis_project.Integration.Integration_benchmark.graph import build_neighbors, build_umap, build_leiden
from thesis_project.Integration.Integration_benchmark.metrics import integration_metrics
from thesis_project.Integration.Integration_benchmark.plotting import (
    subsample_for_plotting,
    plot_umap_pub,
    plot_marker_dotplot_pub,
    plot_metric_summary,
)
from thesis_project.Integration.Integration_benchmark.io import save_run_artifacts


# ---------------------------------------------------------------------------
# Logging helpers (unchanged)
# ---------------------------------------------------------------------------

def _make_logger(outdir: str, name: str = "scanorama") -> logging.Logger:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"{name}_run.log")
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
    msg: str, *, outdir: str, verbose: bool, logger: Optional[logging.Logger] = None
) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} | {msg}"
    if verbose:
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
class ScanoramaConfig:
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # logging
    verbose: bool = True

    # backend choice
    backend: str = "scanorama"

    # optional feature restriction / preprocessing
    use_hvg_subset: bool = True
    hvg_key: str = "highly_variable"
    require_hvg: bool = False

    scale_before_pca: bool = False
    scale_max_value: float = 10.0

    # PCA controls (scanpy.external backend and fallback utility)
    n_pcs: int = 50
    pca_solver: str = "arpack"
    pca_basis_key: str = "X_pca"

    # Scanorama controls
    dimred: int = 50
    knn: int = 30
    alpha: float = 0.1
    sigma: float = 20.0
    approx: bool = True
    batch_size: int = 5000

    # downstream
    neighbors_k: int = 50
    umap_min_dist: float = 0.35
    umap_spread: float = 1.0
    leiden_resolution: float = 1.0

    # metrics
    n_isolated: Optional[int] = None
    lisi_subsample: Optional[int] = None
    compute_trajectory: bool = False

    # plotting
    plot_subsample_n: int = 200_000
    plot_subsample_stratify_by: Optional[str] = "dataset"
    plot_covariates: tuple = (
        "dataset", "major_celltype_l1", "tumor_status", "technology", "cancer_type"
    )
    plot_extra_covariates: tuple = (
        "platform", "tissue", "compartment", "disease_group", "donor_id"
    )

    # run identity
    run_tag: str = "scanorama"
    seed: int = 0

    # persistence
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _select_backend(cfg: ScanoramaConfig) -> str:
    backend = str(getattr(cfg, "backend", "scanorama")).strip().lower()
    valid = {"scanorama", "scanpy.external", "auto"}
    if backend not in valid:
        raise ValueError(
            f"Unsupported Scanorama backend '{backend}'. Expected one of {sorted(valid)}."
        )
    if backend == "scanorama":
        if scanorama is None:
            raise ImportError("scanorama backend requested but 'scanorama' is not installed.")
        return "scanorama"
    if backend == "scanpy.external":
        if sce is None:
            raise ImportError("scanpy.external backend requested but not available.")
        return "scanpy.external"
    if scanorama is not None:
        return "scanorama"
    if sce is not None:
        return "scanpy.external"
    raise ImportError("No Scanorama backend available. Install scanorama or scanpy[external].")


def _pca_basis_key(cfg: ScanoramaConfig) -> str:
    return getattr(cfg, "pca_basis_key", "X_pca")


def _subset_hvgs_if_requested(
    ad, cfg: ScanoramaConfig, *, outdir: str, logger: logging.Logger
):
    if not bool(getattr(cfg, "use_hvg_subset", True)):
        return ad
    hvg_key = str(getattr(cfg, "hvg_key", "highly_variable"))
    if hvg_key not in ad.var:
        msg = (
            f"[HVG] '{hvg_key}' not found in ad.var; continuing without HVG subsetting."
        )
        if bool(getattr(cfg, "require_hvg", False)):
            raise KeyError(msg)
        _log(msg, outdir=outdir, verbose=cfg.verbose, logger=logger)
        return ad
    mask = np.asarray(ad.var[hvg_key]).astype(bool)
    if mask.sum() == 0:
        raise ValueError(f"ad.var['{hvg_key}'] contains no True entries.")
    if int(mask.sum()) == int(ad.n_vars):
        _log(
            f"[HVG] Input already HVG-restricted ({int(mask.sum())} genes).",
            outdir=outdir, verbose=cfg.verbose, logger=logger,
        )
        return ad
    ad_hvg = ad[:, mask].copy()
    _log(
        f"[HVG] Restricted from {ad.n_vars} to {ad_hvg.n_vars} genes "
        f"using ad.var['{hvg_key}'].",
        outdir=outdir, verbose=cfg.verbose, logger=logger,
    )
    return ad_hvg


def _ensure_pca_basis(
    ad, cfg: ScanoramaConfig, *, outdir: str, logger: logging.Logger
) -> None:
    basis_key = _pca_basis_key(cfg)
    if basis_key in ad.obsm:
        return
    candidates = [k for k in ad.obsm.keys() if k.lower().startswith("x_pca")]
    if candidates:
        src = candidates[0]
        ad.obsm[basis_key] = np.array(ad.obsm[src], copy=True)
        _log(
            f"[PCA] Aliased ad.obsm['{src}'] -> ad.obsm['{basis_key}'].",
            outdir=outdir, verbose=cfg.verbose, logger=logger,
        )
        return
    _log(
        f"[PCA] '{basis_key}' missing. Computing PCA fallback.",
        outdir=outdir, verbose=cfg.verbose, logger=logger,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        sc.pp.pca(ad, n_comps=int(cfg.n_pcs), svd_solver=str(cfg.pca_solver))
    if basis_key != "X_pca" and "X_pca" in ad.obsm:
        ad.obsm[basis_key] = np.array(ad.obsm["X_pca"], copy=True)
    if basis_key not in ad.obsm:
        raise RuntimeError(
            f"Failed to create PCA basis '{basis_key}'. "
            f"obsm keys: {list(ad.obsm.keys())[:30]}..."
        )


def _validate_embedding(ad, key_emb: str) -> np.ndarray:
    if key_emb not in ad.obsm:
        raise RuntimeError(
            f"Integration produced no embedding in ad.obsm['{key_emb}']."
        )
    Z = np.asarray(ad.obsm[key_emb])
    if Z.ndim != 2 or Z.shape[0] != ad.n_obs:
        raise RuntimeError(
            f"Bad embedding shape for {key_emb}: {Z.shape}, expected (n_obs, dim)."
        )
    if not np.isfinite(Z).all():
        bad = int(np.sum(~np.isfinite(Z)))
        raise RuntimeError(
            f"Embedding {key_emb} contains non-finite values: {bad} entries."
        )
    return Z


def _split_by_batch(ad, batch_key: str) -> Tuple[List[Any], List[str]]:
    batch_series = ad.obs[batch_key].astype(str)
    batch_names = list(pd.unique(batch_series))
    split = [ad[batch_series.values == b].copy() for b in batch_names]
    return split, batch_names


def _merge_scanorama_batches(split_corrected: List[Any], original_obs_names) -> Any:
    if ad_mod is None:
        raise ImportError("anndata is required to merge Scanorama-corrected batches.")
    merged = ad_mod.concat(
        split_corrected, join="inner", merge="same", index_unique=None
    )
    merged = merged[list(original_obs_names)].copy()
    return merged


def _run_scanorama_native(
    ad, cfg: ScanoramaConfig, *, perf, outdir: str, logger: logging.Logger
) -> Tuple[Any, str]:
    if scanorama is None:
        raise ImportError("scanorama not installed. pip install scanorama")

    ad_work = _subset_hvgs_if_requested(ad, cfg, outdir=outdir, logger=logger)

    if cfg.scale_before_pca:
        _log(
            f"[Scale] Scaling before native Scanorama (max_value={cfg.scale_max_value}).",
            outdir=outdir, verbose=cfg.verbose, logger=logger,
        )
        sc.pp.scale(ad_work, max_value=float(cfg.scale_max_value))

    split, batch_names = _split_by_batch(ad_work, cfg.batch_key)

    if perf:
        perf.start("scanorama_integrate")

    _log(
        f"[Scanorama] backend=scanorama | batches={len(batch_names)} | "
        f"dimred={cfg.dimred}, knn={cfg.knn}, alpha={cfg.alpha}, "
        f"sigma={cfg.sigma}, approx={cfg.approx}, batch_size={cfg.batch_size}",
        outdir=outdir, verbose=cfg.verbose, logger=logger,
    )

    corrected = scanorama.correct_scanpy(
        split,
        return_dimred=True,
        dimred=int(cfg.dimred),
        knn=int(cfg.knn),
        sigma=float(cfg.sigma),
        approx=bool(cfg.approx),
        alpha=float(cfg.alpha),
        batch_size=int(cfg.batch_size),
    )

    ad_corrected = _merge_scanorama_batches(corrected, ad_work.obs_names)

    for layer_key in ad.layers.keys():
        if layer_key in ad.layers:
            ad_corrected.layers[layer_key] = ad.layers[layer_key][
                ad.obs_names.get_indexer(ad_corrected.obs_names)
            ]
    ad_corrected.obs[cfg.batch_key] = ad_corrected.obs[cfg.batch_key].astype(str)
    if cfg.label_key in ad_corrected.obs:
        ad_corrected.obs[cfg.label_key] = ad_corrected.obs[cfg.label_key].astype(str)

    key_out = f"X_scanorama_{cfg.run_tag}"
    if "X_scanorama" not in ad_corrected.obsm:
        raise RuntimeError(
            "Native Scanorama completed but did not write ad.obsm['X_scanorama']."
        )
    ad_corrected.obsm[key_out] = np.array(ad_corrected.obsm["X_scanorama"], copy=True)
    if key_out != "X_scanorama":
        del ad_corrected.obsm["X_scanorama"]

    Z = _validate_embedding(ad_corrected, key_out)
    _log(
        f"[Scanorama] embedding '{key_out}' shape={Z.shape}",
        outdir=outdir, verbose=cfg.verbose, logger=logger,
    )

    if perf:
        perf.end("scanorama_integrate")

    return ad_corrected, key_out


def _run_scanorama_external(
    ad, cfg: ScanoramaConfig, *, perf, outdir: str, logger: logging.Logger
) -> Tuple[Any, str]:
    if sce is None:
        raise ImportError("scanpy.external not available.")

    ad_work = _subset_hvgs_if_requested(ad, cfg, outdir=outdir, logger=logger)

    if perf:
        perf.start("pca")
    run_pca(
        ad_work,
        n_pcs=cfg.n_pcs,
        scale=cfg.scale_before_pca,
        scale_max_value=cfg.scale_max_value,
        solver=cfg.pca_solver,
    )
    if perf:
        perf.end("pca")

    _ensure_pca_basis(ad_work, cfg, outdir=outdir, logger=logger)
    basis_key = _pca_basis_key(cfg)
    key_out = f"X_scanorama_{cfg.run_tag}"

    if perf:
        perf.start("scanorama_integrate")

    _log(
        f"[Scanorama] backend=scanpy.external | basis={basis_key} | "
        f"knn={cfg.knn}, alpha={cfg.alpha}, sigma={cfg.sigma}, "
        f"approx={cfg.approx}, batch_size={cfg.batch_size}",
        outdir=outdir, verbose=cfg.verbose, logger=logger,
    )

    batch_series = ad_work.obs[cfg.batch_key].astype(str)
    stable_order = np.argsort(batch_series.to_numpy(), kind="stable")
    ad_sorted = ad_work[stable_order].copy()

    sce.pp.scanorama_integrate(
        ad_sorted,
        key=cfg.batch_key,
        basis=basis_key,
        adjusted_basis=key_out,
        knn=int(cfg.knn),
        sigma=float(cfg.sigma),
        approx=bool(cfg.approx),
        alpha=float(cfg.alpha),
        batch_size=int(cfg.batch_size),
    )

    if key_out not in ad_sorted.obsm:
        raise RuntimeError(
            "scanpy.external Scanorama completed but did not write the adjusted embedding."
        )

    Z_sorted = pd.DataFrame(
        np.asarray(ad_sorted.obsm[key_out]), index=ad_sorted.obs_names
    )
    ad_work.obsm[key_out] = Z_sorted.loc[ad_work.obs_names].to_numpy(dtype=np.float32)

    Z = _validate_embedding(ad_work, key_out)
    _log(
        f"[Scanorama] embedding '{key_out}' shape={Z.shape}",
        outdir=outdir, verbose=cfg.verbose, logger=logger,
    )

    if perf:
        perf.end("scanorama_integrate")

    return ad_work, key_out


def _run_scanorama(
    ad, cfg: ScanoramaConfig, *, perf, outdir: str, logger: logging.Logger
) -> Tuple[Any, str]:
    backend = _select_backend(cfg)
    if backend == "scanorama":
        return _run_scanorama_native(ad, cfg, perf=perf, outdir=outdir, logger=logger)
    return _run_scanorama_external(ad, cfg, perf=perf, outdir=outdir, logger=logger)


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    adata_in, outdir: str, cfg: ScanoramaConfig
) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Scanorama run:
      subset → snapshot adata_pre → integration
      → neighbors → UMAP/Leiden → metrics → save → plots

    Variable naming:
      ad (pre-integration)  : subsetted AnnData before _run_scanorama
      ad (post-integration) : returned by _run_scanorama — may be a new object
                              (native backend creates ad_corrected via concat)
      adata_pre             : snapshot of pre-integration ad (after subset),
                              BEFORE _run_scanorama is called.
                              Critical: the native backend may return a new
                              AnnData object (different identity from input ad),
                              so adata_pre MUST be snapshotted from ad before
                              the integration call, not from the returned object.
      key_emb               : "X_scanorama_scanorama"
      neigh_key             : "neighbors_scanorama"
      conn_key              : ad.uns[neigh_key]["connectivities_key"]
      cluster_key           : "leiden_scanorama"

      output_type="embed": Scanorama outputs a low-dimensional embedding
        (dimred-dimensional Scanorama space). kBET/LISI recompute their own
        kNN from key_emb internally.
    """
    os.makedirs(outdir, exist_ok=True)
    logger = _make_logger(outdir, name="scanorama")

    set_global_seed(cfg.seed, use_torch=False)
    perf = PerfLogger(track_gpu=False)

    # ------------------------------------------------------------------
    # 1. Subset
    # ------------------------------------------------------------------
    perf.start("subset")
    ad, _ = subset_and_cast_obs(
        adata_in, cfg.batch_key, cfg.label_key, cfg.exclude_datasets
    )
    perf.end("subset")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← after subset, BEFORE _run_scanorama.
    #    The native backend returns a NEW ad_corrected object (via concat),
    #    not the same object as input ad. Snapshotting here ensures adata_pre
    #    has the same cell set as the final integrated ad and is unmodified
    #    by HVG restriction or scaling done inside _run_scanorama.
    #    Required for scib comparison metrics (pcr_comparison,
    #    hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Scanorama integration
    #    ad may be replaced by a new object (native backend).
    # ------------------------------------------------------------------
    ad, key_emb = _run_scanorama(ad, cfg, perf=perf, outdir=outdir, logger=logger)

    # ------------------------------------------------------------------
    # 4. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    perf.start("neighbors")
    neigh_key = f"neighbors_{cfg.run_tag}"   # "neighbors_scanorama"
    build_neighbors(
        ad,
        use_rep=key_emb,
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
    # 5. Standard embedding alias (must be set BEFORE metrics call)
    # ------------------------------------------------------------------
    ad.obsm["X_emb"] = np.asarray(ad.obsm[key_emb]).copy()

    # ------------------------------------------------------------------
    # 6. Metrics
    #
    #    conn_key = "neighbors_scanorama_connectivities"  (from uns)
    #    dist_key = "neighbors_scanorama_distances"        (from uns)
    #    neighbors_uns_key = "neighbors_scanorama"         (= neigh_key, explicit)
    #
    #    output_type="embed": Scanorama outputs a dimred-dimensional embedding.
    #      kBET/LISI recompute their own kNN from key_emb internally.
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]  # "neighbors_scanorama_connectivities"
    dist_key = ad.uns[neigh_key]["distances_key"]       # "neighbors_scanorama_distances"

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
            emb_key=key_emb,
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

    metrics["scanorama_backend"] = _select_backend(cfg)
    metrics["scanorama_dimred"] = cfg.dimred
    metrics["scanorama_knn"] = cfg.knn
    metrics["scanorama_alpha"] = cfg.alpha
    metrics["scanorama_sigma"] = cfg.sigma
    metrics["scanorama_approx"] = cfg.approx
    metrics["scanorama_batch_size"] = cfg.batch_size
    metrics["scanorama_scale_before_pca"] = cfg.scale_before_pca
    metrics["scanorama_use_hvg_subset"] = cfg.use_hvg_subset

    perf.end("metrics")
    perf_df = perf.to_df()

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    save_run_artifacts(
        outdir,
        metrics=metrics,
        config=cfg,
        perf_df=perf_df,
        adata=ad,
        save_h5ad=cfg.save_h5ad,
        h5ad_name="adata_scanorama.h5ad",
    )

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    perf.start("plots")
    plot_dir = os.path.join(outdir, "plots_pub")
    os.makedirs(plot_dir, exist_ok=True)

    ad_plot = subsample_for_plotting(
        ad, n=cfg.plot_subsample_n, seed=cfg.seed,
        stratify_by=cfg.plot_subsample_stratify_by,
    )

    umap_key = f"X_umap_{cfg.run_tag}"
    covs = list(cfg.plot_covariates) + [
        c for c in cfg.plot_extra_covariates if c not in cfg.plot_covariates
    ]
    covs = [c for c in covs if c in ad_plot.obs]

    title_prefix = f"Scanorama — {cfg.run_tag} (n={ad_plot.n_obs:,})"
    for cov in covs:
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
        title=f"Scanorama metrics — {cfg.run_tag}",
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return ad, metrics, perf_df