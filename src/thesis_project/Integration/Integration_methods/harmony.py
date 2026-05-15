from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import scanpy as sc

try:
    import harmonypy
except Exception:
    harmonypy = None

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs
from thesis_project.Integration.Integration_benchmark.preprocess import run_pca_sparse_safe
from thesis_project.Integration.Integration_benchmark.graph import build_neighbors, build_umap, build_leiden
from thesis_project.Integration.Integration_benchmark.metrics import integration_metrics
from thesis_project.Integration.Integration_benchmark.plotting import (
    subsample_for_plotting,
    plot_umap_pub,
    plot_marker_dotplot_pub,
    plot_metric_summary,
)
from thesis_project.Integration.Integration_benchmark.io import save_run_artifacts


@dataclass
class HarmonyConfig:
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # sparse-safe PCA
    n_pcs: int = 50
    clip_nonzero_max: Optional[float] = 10.0
    clip_strategy: str = "absolute"
    clip_quantile: float = 0.999
    pca_solver: str = "arpack"

    # harmony
    harmony_theta: Optional[float] = None
    harmony_max_iter_harmony: int = 20

    # downstream
    neighbors_k: int = 25
    umap_min_dist: float = 0.4
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
    run_tag: str = "harmony"
    seed: int = 0

    # persistence
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _ensure_obs_column_strict(obs_df, key: str) -> None:
    if key not in obs_df.columns:
        raise KeyError(f"Required obs column '{key}' not found.")
    if obs_df[key].isna().any():
        n = int(obs_df[key].isna().sum())
        raise ValueError(
            f"obs['{key}'] contains {n} missing values; fill/drop before Harmony."
        )
    obs_df[key] = obs_df[key].astype(str)


def _assert_embedding_ok(
    X: np.ndarray, *, name: str, n_obs: int, n_pcs: int
) -> None:
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    if X.shape[0] != n_obs:
        raise ValueError(f"{name} row count mismatch: {X.shape[0]} != n_obs {n_obs}")
    if X.shape[1] != n_pcs:
        raise ValueError(f"{name} col count mismatch: {X.shape[1]} != n_pcs {n_pcs}")
    if not np.isfinite(X).all():
        raise ValueError(f"{name} contains non-finite values (NaN/Inf).")


def _assert_not_degenerate(X: np.ndarray, *, name: str) -> None:
    v = float(np.nanvar(X))
    if not np.isfinite(v) or v <= 1e-12:
        raise ValueError(f"{name} appears degenerate (global variance={v:.3g}).")


def _run_harmonypy_and_fix(
    X_pca: np.ndarray,
    obs_df,
    batch_key: str,
    theta,
    max_iter_harmony: int,
) -> np.ndarray:
    if harmonypy is None:
        raise ImportError("harmonypy not installed. Install with: pip install harmonypy")

    _ensure_obs_column_strict(obs_df, batch_key)

    X_pca = np.asarray(X_pca)
    if X_pca.ndim != 2:
        raise ValueError(f"Expected X_pca 2D, got shape {X_pca.shape}")
    if not np.isfinite(X_pca).all():
        raise ValueError("X_pca contains non-finite values (NaN/Inf).")

    meta = obs_df[[batch_key]].copy()

    try:
        ho = harmonypy.run_harmony(
            X_pca,
            meta_data=meta,
            vars_use=[batch_key],
            theta=theta,
            max_iter_harmony=max_iter_harmony,
        )
    except TypeError:
        ho = harmonypy.run_harmony(
            X_pca,
            meta_data=meta,
            vars_use=batch_key,
            theta=theta,
            max_iter_harmony=max_iter_harmony,
        )

    Zc = np.asarray(ho.Z_corr)

    if Zc.shape[0] == X_pca.shape[1] and Zc.shape[1] == X_pca.shape[0]:
        Z = Zc.T
    elif Zc.shape == X_pca.shape:
        Z = Zc
    elif Zc.ndim == 2 and Zc.shape[1] == X_pca.shape[0] and Zc.shape[0] == X_pca.shape[1]:
        Z = Zc.T
    else:
        raise ValueError(
            f"Unexpected harmony output shape {Zc.shape} for X_pca {X_pca.shape}"
        )

    _assert_embedding_ok(Z, name="X_harmony", n_obs=X_pca.shape[0], n_pcs=X_pca.shape[1])
    _assert_not_degenerate(Z, name="X_harmony")
    return Z


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(adata_in, outdir: str, cfg: HarmonyConfig) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Harmony run:
      subset → snapshot adata_pre → sparse-safe PCA → Harmony correction
      → neighbors → UMAP/Leiden → metrics → save → plots

    Variable naming:
      ad            : integrated AnnData (modified in-place)
      adata_pre     : snapshot after subset, BEFORE PCA or Harmony
                      (same cell set; required for scib comparison metrics)
      key_emb       : "X_harmony_harmony"  — Harmony-corrected PCA embedding
      neigh_key     : "neighbors_harmony"
      conn_key      : adata.uns[neigh_key]["connectivities_key"]
                      → "neighbors_harmony_connectivities"
      cluster_key   : "leiden_harmony"

      output_type="embed": Harmony outputs a corrected PCA-space embedding.
        kBET/LISI recompute their own kNN from key_emb internally.
        graph_connectivity reads conn_key (aliased to obsp['connectivities']).
        neighbors_uns_key passed explicitly to avoid wrong auto-derivation.
    """
    os.makedirs(outdir, exist_ok=True)
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

    if cfg.batch_key not in ad.obs:
        raise KeyError(f"batch_key '{cfg.batch_key}' missing from ad.obs")
    if cfg.label_key not in ad.obs:
        raise KeyError(f"label_key '{cfg.label_key}' missing from ad.obs")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← after subset, before PCA or Harmony.
    #    Same cell set as ad; required for scib comparison metrics
    #    (pcr_comparison, hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Sparse-safe PCA
    # ------------------------------------------------------------------
    perf.start("pca_sparse_safe")
    run_pca_sparse_safe(
        ad,
        n_pcs=cfg.n_pcs,
        clip_nonzero_max=cfg.clip_nonzero_max,
        clip_strategy=cfg.clip_strategy,
        clip_quantile=cfg.clip_quantile,
        solver=cfg.pca_solver,
    )
    perf.end("pca_sparse_safe")

    if "X_pca" not in ad.obsm:
        raise KeyError("run_pca_sparse_safe did not write ad.obsm['X_pca']")
    X_pca = np.asarray(ad.obsm["X_pca"])
    _assert_embedding_ok(X_pca, name="X_pca", n_obs=ad.n_obs, n_pcs=cfg.n_pcs)
    _assert_not_degenerate(X_pca, name="X_pca")

    # ------------------------------------------------------------------
    # 4. Harmony correction
    # ------------------------------------------------------------------
    perf.start("harmony")
    key_emb = f"X_harmony_{cfg.run_tag}"    # "X_harmony_harmony"
    X_h = _run_harmonypy_and_fix(
        X_pca,
        ad.obs,
        cfg.batch_key,
        cfg.harmony_theta,
        cfg.harmony_max_iter_harmony,
    ).astype(np.float32)
    ad.obsm[key_emb] = X_h
    perf.end("harmony")

    # ------------------------------------------------------------------
    # 5. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    perf.start("neighbors")
    neigh_key = f"neighbors_{cfg.run_tag}"   # "neighbors_harmony"
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
    # 6. Standard embedding alias (must be set BEFORE metrics call)
    # ------------------------------------------------------------------
    ad.obsm["X_emb"] = np.asarray(ad.obsm[key_emb]).copy()

    # ------------------------------------------------------------------
    # 7. Metrics
    #
    #    conn_key = "neighbors_harmony_connectivities"  (from uns)
    #    dist_key = "neighbors_harmony_distances"        (from uns)
    #    neighbors_uns_key = "neighbors_harmony"         (= neigh_key, explicit)
    #
    #    output_type="embed": Harmony outputs a corrected PCA-space embedding.
    #      kBET/LISI recompute their own kNN from key_emb internally.
    #      graph_connectivity uses conn_key (aliased to obsp['connectivities']).
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]  # "neighbors_harmony_connectivities"
    dist_key = ad.uns[neigh_key]["distances_key"]       # "neighbors_harmony_distances"

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

    metrics["harmony_theta"] = cfg.harmony_theta
    metrics["harmony_max_iter_harmony"] = int(cfg.harmony_max_iter_harmony)
    metrics["harmony_n_pcs"] = int(cfg.n_pcs)
    try:
        metrics["harmonypy_version"] = getattr(harmonypy, "__version__", "unknown")
    except Exception:
        metrics["harmonypy_version"] = "unknown"

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
        h5ad_name="adata_harmony.h5ad",
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
    covs = list(cfg.plot_covariates) + [
        c for c in cfg.plot_extra_covariates if c not in cfg.plot_covariates
    ]
    covs = [c for c in covs if c in ad_plot.obs]

    title_prefix = f"Harmony — {cfg.run_tag} (n={ad_plot.n_obs:,})"
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
        title=f"Harmony metrics — {cfg.run_tag}",
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return ad, metrics, perf_df