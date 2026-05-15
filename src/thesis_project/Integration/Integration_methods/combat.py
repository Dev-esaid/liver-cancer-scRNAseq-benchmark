from __future__ import annotations

import copy
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from pandas.api.types import is_bool_dtype, is_numeric_dtype

try:
    import anndata as ad
except Exception:  # pragma: no cover - optional dependency guard
    ad = None

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


@dataclass
class CombatConfig:
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # ------------------------------------------------------------------
    # Integration space
    #   "expr" => run ComBat on corrected expression, then PCA (recommended)
    #   "pca"  => legacy mode: run ComBat directly on PCA coordinates
    # ------------------------------------------------------------------
    combat_space: str = "expr"  # {"expr", "pca"}

    # Expression input
    input_layer: Optional[str] = None

    # HVG restriction for expression-space ComBat
    require_hvg: bool = True
    hvg_key: str = "highly_variable"
    max_hvgs: Optional[int] = 4000
    hvg_flavor: str = "seurat_v3"

    # PCA (computed AFTER ComBat when combat_space="expr")
    n_pcs: int = 50
    scale_before_pca: bool = False
    scale_max_value: float = 10.0
    pca_solver: str = "arpack"

    # ComBat options
    covariates: Optional[tuple] = None

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
    run_tag: str = "combat"
    seed: int = 0

    # persistence
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Validation helpers (unchanged)
# ---------------------------------------------------------------------------

def _validate_cfg(cfg: CombatConfig) -> None:
    if cfg.combat_space not in {"expr", "pca"}:
        raise ValueError("CombatConfig.combat_space must be one of {'expr', 'pca'}.")
    if int(cfg.n_pcs) <= 0:
        raise ValueError(f"n_pcs must be positive, got {cfg.n_pcs!r}.")
    if int(cfg.neighbors_k) <= 0:
        raise ValueError(f"neighbors_k must be positive, got {cfg.neighbors_k!r}.")
    if float(cfg.umap_min_dist) < 0:
        raise ValueError(f"umap_min_dist must be non-negative, got {cfg.umap_min_dist!r}.")
    if float(cfg.umap_spread) <= 0:
        raise ValueError(f"umap_spread must be positive, got {cfg.umap_spread!r}.")
    if float(cfg.leiden_resolution) <= 0:
        raise ValueError(f"leiden_resolution must be positive, got {cfg.leiden_resolution!r}.")
    if cfg.max_hvgs is not None and int(cfg.max_hvgs) <= 0:
        raise ValueError(f"max_hvgs must be positive when provided, got {cfg.max_hvgs!r}.")

    valid_hvg_flavors = {"seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"}
    if cfg.hvg_flavor not in valid_hvg_flavors:
        raise ValueError(
            f"Unsupported hvg_flavor '{cfg.hvg_flavor}'. Expected one of {sorted(valid_hvg_flavors)}."
        )
    if cfg.require_hvg and cfg.hvg_flavor in {"seurat_v3", "seurat_v3_paper"} and cfg.max_hvgs is None:
        raise ValueError(
            "max_hvgs must be set when hvg_flavor is 'seurat_v3' or 'seurat_v3_paper'."
        )


def _matrix_to_dense_float32(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape!r}.")
    if not np.isfinite(arr).all():
        raise ValueError("Matrix contains non-finite values.")
    return arr


def _embedding_to_float32(Z, *, name: str) -> np.ndarray:
    arr = np.asarray(Z, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape!r}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _warn_if_counts_like(X, *, context: str) -> None:
    if hasattr(X, "data") and hasattr(X, "toarray"):
        vals = np.asarray(X.data)
    else:
        vals = np.asarray(X).ravel()
    if vals.size == 0:
        return
    if vals.size > 100_000:
        rng = np.random.default_rng(0)
        vals = vals[rng.choice(vals.size, size=100_000, replace=False)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if np.min(vals) >= 0 and np.allclose(vals, np.round(vals)):
        warnings.warn(
            f"{context} appears to contain non-negative integer-like values. "
            "ComBat is typically applied to log-normalized expression, not raw counts.",
            RuntimeWarning,
            stacklevel=2,
        )


def _validate_batch_structure(obs: pd.DataFrame, batch_key: str) -> None:
    if batch_key not in obs:
        raise ValueError(f"batch_key '{batch_key}' not found in obs.")
    n_batches = obs[batch_key].nunique(dropna=False)
    if n_batches < 2:
        raise ValueError(
            f"ComBat requires at least 2 batches in obs['{batch_key}']; found {n_batches}."
        )
    counts = obs[batch_key].value_counts(dropna=False)
    small = counts[counts < 2].index.tolist()
    if small:
        raise ValueError(
            f"Batches {small!r} have fewer than 2 cells. "
            "ComBat requires at least 2 cells per batch."
        )


def _prepare_covariates_inplace(
    obs: pd.DataFrame,
    *,
    batch_key: str,
    covariates: Optional[Sequence[str]],
) -> Optional[List[str]]:
    if not covariates:
        return None
    cov_list = list(covariates)
    if len(cov_list) != len(set(cov_list)):
        raise ValueError("Covariates must be unique.")
    if batch_key in cov_list:
        raise ValueError("Batch key and covariates cannot overlap.")
    missing = [c for c in cov_list if c not in obs.columns]
    if missing:
        raise ValueError(f"Could not find the covariate(s) {missing!r} in adata.obs.")

    kept: List[str] = []
    dropped_constant: List[str] = []
    for c in cov_list:
        s = obs[c]
        if is_numeric_dtype(s) and not is_bool_dtype(s):
            if s.isna().any():
                raise ValueError(f"Numeric covariate '{c}' contains missing values.")
            arr = pd.to_numeric(s, errors="raise").to_numpy(dtype=float, copy=False)
            if not np.isfinite(arr).all():
                raise ValueError(f"Numeric covariate '{c}' contains non-finite values.")
            if pd.Series(arr).nunique(dropna=True) < 2:
                dropped_constant.append(c)
                continue
            obs[c] = arr
            kept.append(c)
            continue
        s_cat = s.where(~s.isna(), other="Unknown").astype(str)
        if s_cat.nunique(dropna=False) < 2:
            dropped_constant.append(c)
            continue
        obs[c] = s_cat
        kept.append(c)

    if dropped_constant:
        warnings.warn(
            f"Dropping constant covariate(s) for ComBat: {dropped_constant!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return kept if kept else None


def _subset_hvgs_or_raise(adata: sc.AnnData, cfg: CombatConfig) -> sc.AnnData:
    if not cfg.require_hvg:
        return adata
    if cfg.hvg_key not in adata.var:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(cfg.max_hvgs) if cfg.max_hvgs is not None else None,
            flavor=cfg.hvg_flavor,
            batch_key=cfg.batch_key if cfg.batch_key in adata.obs else None,
            subset=False,
            inplace=True,
        )
    mask = adata.var[cfg.hvg_key].fillna(False).astype(bool).to_numpy()
    if not np.any(mask):
        raise ValueError(f"ad.var['{cfg.hvg_key}'] exists but contains no True entries.")
    selected = adata.var_names[mask]
    max_hvgs = int(cfg.max_hvgs) if cfg.max_hvgs is not None else None
    if max_hvgs is not None and selected.size > max_hvgs:
        rank_key = "highly_variable_rank"
        if rank_key in adata.var and adata.var.loc[selected, rank_key].notna().any():
            ranks = adata.var.loc[selected, rank_key].astype(float)
            selected = ranks.sort_values(kind="mergesort").index[:max_hvgs]
        else:
            raise ValueError(
                f"{cfg.hvg_key!r} marks {selected.size} genes, exceeding max_hvgs={max_hvgs}, "
                "but no 'highly_variable_rank' column is available for deterministic truncation."
            )
    return adata[:, list(selected)].copy()


def _ensure_X_from_layer(adata: sc.AnnData, cfg: CombatConfig) -> None:
    if cfg.input_layer is None:
        return
    if cfg.input_layer not in adata.layers:
        raise ValueError(f"input_layer='{cfg.input_layer}' not found in adata.layers.")
    X = adata.layers[cfg.input_layer]
    adata.X = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)


def _call_scanpy_combat(
    adata: sc.AnnData,
    *,
    batch_key: str,
    covariates: Optional[Sequence[str]],
) -> None:
    try:
        sig = inspect.signature(sc.pp.combat)
        params = sig.parameters
    except (TypeError, ValueError):
        params = {}
    kwargs: Dict[str, Any] = {"key": batch_key}
    if covariates is not None:
        if not params or "covariates" in params:
            kwargs["covariates"] = list(covariates)
        else:
            warnings.warn(
                "Installed scanpy.pp.combat does not expose a 'covariates' parameter. "
                "Proceeding without covariate adjustment.",
                RuntimeWarning,
                stacklevel=2,
            )
    if not params or "inplace" in params:
        kwargs["inplace"] = True
        result = sc.pp.combat(adata, **kwargs)
        if result is not None:
            adata.X = _matrix_to_dense_float32(result)
    else:
        result = sc.pp.combat(adata, **kwargs)
        if result is None:
            raise RuntimeError("scanpy.pp.combat returned None but does not expose 'inplace'.")
        adata.X = _matrix_to_dense_float32(result)


def _combat_on_expression(
    adata: sc.AnnData,
    *,
    batch_key: str,
    covariates: Optional[Sequence[str]],
) -> None:
    if batch_key not in adata.obs:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs.")
    _validate_batch_structure(adata.obs, batch_key)
    kept_covariates = _prepare_covariates_inplace(
        adata.obs, batch_key=batch_key, covariates=covariates
    )
    _call_scanpy_combat(adata, batch_key=batch_key, covariates=kept_covariates)
    adata.X = _matrix_to_dense_float32(adata.X)


def _combat_on_pca(
    adata_in: sc.AnnData,
    *,
    batch_key: str,
    covariates: Optional[Sequence[str]],
    X_pca: np.ndarray,
) -> np.ndarray:
    if ad is None:
        raise ImportError("anndata is required for the ComBat PCA wrapper.")
    if batch_key not in adata_in.obs:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs.")
    _validate_batch_structure(adata_in.obs, batch_key)
    X_pca = _embedding_to_float32(X_pca, name="X_pca")
    var_names = [f"PC{i + 1}" for i in range(X_pca.shape[1])]
    tmp = ad.AnnData(X=X_pca.copy(), obs=adata_in.obs.copy())
    tmp.var_names = var_names
    kept_covariates = _prepare_covariates_inplace(
        tmp.obs, batch_key=batch_key, covariates=covariates
    )
    _call_scanpy_combat(tmp, batch_key=batch_key, covariates=kept_covariates)
    X_corr = _embedding_to_float32(tmp.X, name="ComBat-corrected PCA")
    if X_corr.shape != X_pca.shape:
        raise ValueError(
            f"Unexpected ComBat output shape {X_corr.shape}, expected {X_pca.shape}."
        )
    return X_corr


def _check_pca_feasibility(adata: sc.AnnData, *, n_pcs: int, solver: str) -> None:
    max_components = min(int(adata.n_obs), int(adata.n_vars))
    if solver == "arpack":
        max_components -= 1
    if max_components < 1:
        raise ValueError(
            f"PCA not feasible for shape ({adata.n_obs}, {adata.n_vars}) with solver={solver!r}."
        )
    if int(n_pcs) > max_components:
        comparator = "<" if solver == "arpack" else "<="
        raise ValueError(
            f"n_pcs={n_pcs} too large for shape ({adata.n_obs}, {adata.n_vars}) "
            f"with solver={solver!r}. Set n_pcs {comparator} {max_components}."
        )


def _namespace_pca_output(adata: sc.AnnData, *, run_tag: str) -> str:
    if "X_pca" not in adata.obsm:
        raise KeyError("Expected ad.obsm['X_pca'] after PCA, but it was not found.")
    key_emb = f"X_pca_{run_tag}"
    adata.obsm[key_emb] = _embedding_to_float32(adata.obsm["X_pca"], name="X_pca")
    del adata.obsm["X_pca"]
    if "pca" in adata.uns:
        adata.uns[f"pca_{run_tag}"] = copy.deepcopy(adata.uns.pop("pca"))
    if "PCs" in adata.varm:
        adata.varm[f"PCs_{run_tag}"] = np.asarray(adata.varm["PCs"]).copy()
        del adata.varm["PCs"]
    return key_emb


def _drop_default_pca_artifacts(adata: sc.AnnData) -> None:
    if "X_pca" in adata.obsm:
        del adata.obsm["X_pca"]
    if "pca" in adata.uns:
        del adata.uns["pca"]
    if "PCs" in adata.varm:
        del adata.varm["PCs"]


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(adata_in, outdir: str, cfg: CombatConfig) -> Tuple[Any, Dict[str, Any], Any]:
    """
    ComBat run.

    Variable naming (for reference when reading the metrics call below):
      adata         : integrated AnnData (modified in-place through the pipeline)
      adata_pre     : snapshot of adata after subsetting, BEFORE any modification.
                      Same cell set as adata; required for scib comparison metrics
                      (pcr_comparison, hvg_conservation, cell_cycle_conservation).
      key_emb       : "X_pca_combat"  — namespaced PCA embedding (final representation)
      neigh_key     : "neighbors_combat"  — uns key for graph metadata
      conn_key      : adata.uns[neigh_key]["connectivities_key"]
                      → "neighbors_combat_connectivities"  (obsp key)
      cluster_key   : "leiden_combat"

      output_type="full": ComBat outputs a corrected feature matrix.
        kBET recomputes its own kNN from X_pca internally.
        iLISI/cLISI recompute from use_rep=key_emb.
        graph_connectivity reads from conn_key (aliased to obsp['connectivities']).
        neighbors_uns_key passed explicitly because the scib naming convention
        differs from the scanpy key_added convention used here.
    """
    _validate_cfg(cfg)
    os.makedirs(outdir, exist_ok=True)
    set_global_seed(cfg.seed, use_torch=False)

    if cfg.combat_space == "pca":
        warnings.warn(
            "ComBat is running in legacy PCA-space mode. Expression-space ComBat "
            "followed by PCA is preferred for fair benchmarking.",
            RuntimeWarning,
            stacklevel=2,
        )

    perf = PerfLogger(track_gpu=False)

    # ------------------------------------------------------------------
    # 1. Subset
    # ------------------------------------------------------------------
    perf.start("subset")
    adata, _ = subset_and_cast_obs(
        adata_in, cfg.batch_key, cfg.label_key, cfg.exclude_datasets
    )
    perf.end("subset")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← MUST be here: after subset, before any
    #    modification. Ensures same cell set as adata for scib comparison
    #    metrics (pcr_comparison, hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = adata.copy()

    # ------------------------------------------------------------------
    # 3. ComBat integration
    # ------------------------------------------------------------------
    if cfg.combat_space == "expr":
        perf.start("prep_expr")
        _ensure_X_from_layer(adata, cfg)
        if not cfg.require_hvg:
            warnings.warn(
                "Running expression-space ComBat without HVG restriction will densify "
                "the full matrix. This can be prohibitively memory-intensive at atlas scale.",
                RuntimeWarning,
                stacklevel=2,
            )
        _warn_if_counts_like(adata.X, context="ComBat input matrix")
        adata = _subset_hvgs_or_raise(adata, cfg)
        adata.X = _matrix_to_dense_float32(adata.X)
        perf.end("prep_expr")

        perf.start("combat_expr")
        _combat_on_expression(
            adata, batch_key=cfg.batch_key, covariates=cfg.covariates
        )
        perf.end("combat_expr")

        perf.start("pca")
        _check_pca_feasibility(adata, n_pcs=cfg.n_pcs, solver=cfg.pca_solver)
        run_pca(
            adata,
            n_pcs=cfg.n_pcs,
            scale=cfg.scale_before_pca,
            scale_max_value=cfg.scale_max_value,
            solver=cfg.pca_solver,
        )
        perf.end("pca")

        key_emb = _namespace_pca_output(adata, run_tag=cfg.run_tag)
        title_method = "ComBat"

    else:  # pca mode
        perf.start("pca")
        _check_pca_feasibility(adata, n_pcs=cfg.n_pcs, solver=cfg.pca_solver)
        run_pca(
            adata,
            n_pcs=cfg.n_pcs,
            scale=cfg.scale_before_pca,
            scale_max_value=cfg.scale_max_value,
            solver=cfg.pca_solver,
        )
        perf.end("pca")

        perf.start("combat_pca")
        X_pca = _embedding_to_float32(adata.obsm["X_pca"], name="X_pca")
        X_corr = _combat_on_pca(
            adata,
            batch_key=cfg.batch_key,
            covariates=cfg.covariates,
            X_pca=X_pca,
        )
        perf.end("combat_pca")

        key_emb = f"X_pca_{cfg.run_tag}"
        adata.obsm[key_emb] = X_corr
        _drop_default_pca_artifacts(adata)
        title_method = "ComBat(PCA)"

    if key_emb not in adata.obsm:
        raise RuntimeError(f"Expected embedding '{key_emb}' not found in adata.obsm.")

    # ------------------------------------------------------------------
    # 4. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    perf.start("neighbors")
    neigh_key = f"neighbors_{cfg.run_tag}"   # "neighbors_combat"
    build_neighbors(
        adata,
        use_rep=key_emb,
        n_neighbors=cfg.neighbors_k,
        key_added=neigh_key,
        random_state=cfg.seed,
    )
    perf.end("neighbors")

    perf.start("umap")
    build_umap(
        adata,
        neighbors_key=neigh_key,
        key_umap=f"X_umap_{cfg.run_tag}",
        min_dist=cfg.umap_min_dist,
        spread=cfg.umap_spread,
        random_state=cfg.seed,
    )
    perf.end("umap")

    perf.start("leiden")
    build_leiden(
        adata,
        neighbors_key=neigh_key,
        key_leiden=f"leiden_{cfg.run_tag}",
        resolution=cfg.leiden_resolution,
        random_state=cfg.seed,
    )
    perf.end("leiden")

    # ------------------------------------------------------------------
    # 5. Standard embedding alias (must be set BEFORE metrics call)
    # ------------------------------------------------------------------
    adata.obsm["X_emb"] = np.asarray(adata.obsm[key_emb]).copy()

    # ------------------------------------------------------------------
    # 6. Metrics
    #
    #    conn_key   = adata.uns[neigh_key]["connectivities_key"]
    #               = "neighbors_combat_connectivities"  (actual obsp key)
    #    dist_key   = adata.uns[neigh_key]["distances_key"]
    #               = "neighbors_combat_distances"
    #    neighbors_uns_key = neigh_key = "neighbors_combat"
    #      ↑ passed explicitly because the scanpy key_added convention
    #        ("neighbors_combat") differs from the scib auto-derivation
    #        convention ("neighbors_" + run_tag_from_conn_key) which would
    #        incorrectly produce "neighbors_neighbors_combat".
    #
    #    output_type = "full":
    #      ComBat outputs a corrected feature matrix (not an embedding or
    #      kNN graph natively). kBET and LISI recompute their own graphs
    #      internally from X_pca. graph_connectivity uses conn_key.
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = adata.uns[neigh_key]["connectivities_key"]   # "neighbors_combat_connectivities"
    dist_key = adata.uns[neigh_key]["distances_key"]        # "neighbors_combat_distances"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=".*pandas.value_counts.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=".*in1d.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=".*anndata2ri.*")

        metrics = integration_metrics(
            adata,
            batch_key=cfg.batch_key,
            label_key=cfg.label_key,
            cluster_key=f"leiden_{cfg.run_tag}",
            emb_key=key_emb,
            conn_key=conn_key,
            dist_key=dist_key,
            neighbors_uns_key=neigh_key,
            output_type="full",
            adata_pre=adata_pre,
            compute_trajectory=cfg.compute_trajectory,
            n_isolated=cfg.n_isolated,
            lisi_subsample=cfg.lisi_subsample,
            organism="human",
            verbose=False,
        )

    metrics["combat_space"] = cfg.combat_space
    metrics["combat_input_layer"] = cfg.input_layer if cfg.input_layer is not None else "X"
    metrics["combat_hvg_key"] = cfg.hvg_key if cfg.require_hvg else "none"
    metrics["combat_max_hvgs"] = cfg.max_hvgs if cfg.require_hvg else None
    metrics["combat_hvg_flavor"] = cfg.hvg_flavor if cfg.require_hvg else None
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
        adata=adata,
        save_h5ad=cfg.save_h5ad,
        h5ad_name="adata_combat.h5ad",
    )

    # ------------------------------------------------------------------
    # 8. Plots (subsampled)
    # ------------------------------------------------------------------
    perf.start("plots")
    plot_dir = os.path.join(outdir, "plots_pub")
    os.makedirs(plot_dir, exist_ok=True)

    ad_plot = subsample_for_plotting(
        adata,
        n=cfg.plot_subsample_n,
        seed=cfg.seed,
        stratify_by=cfg.plot_subsample_stratify_by,
    )

    umap_key = f"X_umap_{cfg.run_tag}"
    covs = list(cfg.plot_covariates) + [
        c for c in cfg.plot_extra_covariates if c not in cfg.plot_covariates
    ]
    covs = [c for c in covs if c in ad_plot.obs]

    title_prefix = f"{title_method} — {cfg.run_tag} (n={ad_plot.n_obs:,})"
    for cov in covs:
        plot_umap_pub(
            ad_plot,
            umap_key=umap_key,
            color=cov,
            title_prefix=title_prefix,
            outdir=plot_dir,
            alpha=0.75,
        )

    plot_umap_pub(
        ad_plot,
        umap_key=umap_key,
        color=f"leiden_{cfg.run_tag}",
        title_prefix=title_prefix,
        outdir=plot_dir,
        alpha=0.75,
    )

    if cfg.label_key in ad_plot.obs:
        plot_marker_dotplot_pub(ad_plot, groupby=cfg.label_key, outdir=plot_dir)    
    plot_metric_summary(
        metrics, perf_df, outdir=plot_dir, title=f"{title_method} metrics — {cfg.run_tag}"
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return adata, metrics, perf_df