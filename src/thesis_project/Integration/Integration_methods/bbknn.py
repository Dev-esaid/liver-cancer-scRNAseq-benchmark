from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import scanpy as sc

try:
    import bbknn
except Exception:  # pragma: no cover - optional dependency guard
    bbknn = None

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs
from thesis_project.Integration.Integration_benchmark.preprocess import run_pca
from thesis_project.Integration.Integration_benchmark.graph import (
    namespace_existing_graph,
    build_umap,
    build_leiden,
)
from thesis_project.Integration.Integration_benchmark.metrics import integration_metrics
from thesis_project.Integration.Integration_benchmark.plotting import (
    subsample_for_plotting,
    plot_umap_pub,
    plot_marker_dotplot_pub,
    plot_metric_summary,
)
from thesis_project.Integration.Integration_benchmark.io import save_run_artifacts


@dataclass
class BBKNNConfig:
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # PCA
    n_pcs: int = 50
    scale_before_pca: bool = False
    scale_max_value: float = 10.0
    pca_solver: str = "arpack"

    # BBKNN
    neighbors_within_batch: int = 5
    trim: Optional[int] = None
    set_op_mix_ratio: float = 1.0
    local_connectivity: int = 1

    # Optional explicit value for tools that require params['n_neighbors'] on the
    # stored graph metadata (for example sc.tl.diffmap). No default is imposed
    # here because BBKNN's graph is batch-balanced and does not always map cleanly
    # to a single global k. Set this explicitly in the benchmark configuration if
    # your Scanpy version omits it from the BBKNN metadata.
    diffmap_n_neighbors: Optional[int] = 50

    # downstream
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
    run_tag: str = "bbknn"
    seed: int = 0

    # persistence
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_bbknn_graph(
    ad, cfg: BBKNNConfig, perf: Optional[PerfLogger] = None
) -> Tuple[str, str]:
    """Run BBKNN and namespace the resulting graph.

    Returns
    -------
    neighbors_key : str  — e.g. "neighbors_bbknn"
    conn_key      : str  — e.g. "bbknn_connectivities"
    """
    if bbknn is None:
        raise ImportError("bbknn not installed. Install with: pip install bbknn")

    if perf:
        perf.start("bbknn")
    bbknn.bbknn(
        ad,
        batch_key=cfg.batch_key,
        neighbors_within_batch=cfg.neighbors_within_batch,
        n_pcs=cfg.n_pcs,
        trim=cfg.trim,
        set_op_mix_ratio=cfg.set_op_mix_ratio,
        local_connectivity=cfg.local_connectivity,
    )
    if perf:
        perf.end("bbknn")

    # namespace_existing_graph stores:
    #   ad.obsp["bbknn_connectivities"]
    #   ad.obsp["bbknn_distances"]
    #   ad.uns["neighbors_bbknn"]
    neighbors_key, conn_key, _ = namespace_existing_graph(
        ad,
        run_tag=cfg.run_tag,
        n_neighbors=cfg.diffmap_n_neighbors,
        delete_default_slots=True,
    )
    return neighbors_key, conn_key


def _build_diffmap_embedding(
    ad,
    *,
    neighbors_key: str,
    out_key: str,
    n_comps: int = 50,
    random_state: int = 0,
) -> str:
    """Build a diffusion map embedding from the BBKNN-integrated neighbor graph.

    Intentionally strict: fails with a clear error if the graph metadata does
    not expose params['n_neighbors'], rather than inferring from graph sparsity.
    """
    if neighbors_key not in ad.uns:
        raise KeyError(f"neighbors_key '{neighbors_key}' not found in ad.uns")

    neigh = ad.uns[neighbors_key]
    if not isinstance(neigh, dict):
        raise TypeError(
            f"ad.uns['{neighbors_key}'] must be a dict, got {type(neigh).__name__}."
        )

    params = neigh.get("params")
    if not isinstance(params, dict) or "n_neighbors" not in params:
        raise ValueError(
            f"ad.uns['{neighbors_key}']['params']['n_neighbors'] is required to build "
            "a diffusion map. For BBKNN, set BBKNNConfig.diffmap_n_neighbors explicitly "
            "if your Scanpy/BBKNN metadata does not already provide it. "
            "No sparsity-based inference is performed."
        )

    sc.tl.diffmap(
        ad,
        neighbors_key=neighbors_key,
        n_comps=int(n_comps),
        random_state=int(random_state),
    )

    if "X_diffmap" not in ad.obsm:
        raise RuntimeError(
            "sc.tl.diffmap completed but did not write ad.obsm['X_diffmap'] as expected."
        )

    ad.obsm[out_key] = ad.obsm["X_diffmap"].copy()
    del ad.obsm["X_diffmap"]

    if "diffmap_evals" in ad.uns:
        ad.uns[f"diffmap_evals_{out_key}"] = ad.uns.pop("diffmap_evals")

    return out_key


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(adata_in, outdir: str, cfg: BBKNNConfig) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Full BBKNN benchmarking run:
      subset → snapshot adata_pre → PCA → BBKNN graph
      → diffmap embedding → UMAP → Leiden → metrics → save → plots

    Variable naming (for reference when reading the metrics call below):
      ad            : integrated AnnData (modified in-place through the pipeline)
      adata_pre     : snapshot of ad after subsetting, BEFORE PCA / integration
                      (same cell set as ad; required for scib comparison metrics)
      neighbors_key : "neighbors_bbknn"      (uns key for graph metadata)
      conn_key      : "bbknn_connectivities" (obsp key for connectivity matrix)
      emb_key       : "X_diffmap_bbknn"      (obsm key for diffmap embedding)
      cluster_key   : "leiden_bbknn"         (obs key for Leiden clusters)

      dist_key and neighbors_uns_key are NOT passed explicitly — they are
      auto-derived inside _alias_graph() from conn_key:
        dist_key          → "bbknn_distances"   (conn_key with _connectivities → _distances)
        neighbors_uns_key → "neighbors_bbknn"   (= neighbors_key)
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

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← MUST be taken here, after subsetting but
    #    before any PCA or integration modifies ad.
    #    This ensures adata_pre and ad have identical cell sets so that
    #    scib's pcr_comparison, hvg_overlap, and cell_cycle metrics can
    #    compare pre vs post integration without a shape mismatch.
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. PCA  (runs on ad; adata_pre is untouched)
    # ------------------------------------------------------------------
    perf.start("pca")
    run_pca(
        ad,
        n_pcs=cfg.n_pcs,
        scale=cfg.scale_before_pca,
        scale_max_value=cfg.scale_max_value,
        solver=cfg.pca_solver,
    )
    perf.end("pca")

    # ------------------------------------------------------------------
    # 4. BBKNN graph
    #    After this call:
    #      ad.obsp["bbknn_connectivities"]   — corrected connectivity matrix
    #      ad.obsp["bbknn_distances"]        — corrected distance matrix
    #      ad.uns["neighbors_bbknn"]         — graph metadata (n_neighbors etc.)
    # ------------------------------------------------------------------
    neighbors_key, conn_key = _run_bbknn_graph(ad, cfg, perf=perf)
    # neighbors_key = "neighbors_bbknn"
    # conn_key      = "bbknn_connectivities"

    # ------------------------------------------------------------------
    # 5. Diffmap embedding (derived from the BBKNN graph)
    #    emb_key = "X_diffmap_bbknn"
    # ------------------------------------------------------------------
    perf.start("diffmap")
    emb_key = _build_diffmap_embedding(
        ad,
        neighbors_key=neighbors_key,
        out_key=f"X_diffmap_{cfg.run_tag}",
        n_comps=cfg.n_pcs,
        random_state=cfg.seed,
    )
    perf.end("diffmap")

    # ------------------------------------------------------------------
    # 6. UMAP
    # ------------------------------------------------------------------
    perf.start("umap")
    build_umap(
        ad,
        neighbors_key=neighbors_key,
        key_umap=f"X_umap_{cfg.run_tag}",
        min_dist=cfg.umap_min_dist,
        spread=cfg.umap_spread,
        random_state=cfg.seed,
    )
    perf.end("umap")

    # ------------------------------------------------------------------
    # 7. Leiden clustering
    #    cluster_key = "leiden_bbknn"
    # ------------------------------------------------------------------
    perf.start("leiden")
    build_leiden(
        ad,
        neighbors_key=neighbors_key,
        key_leiden=f"leiden_{cfg.run_tag}",
        resolution=cfg.leiden_resolution,
        random_state=cfg.seed,
    )
    perf.end("leiden")

    # ------------------------------------------------------------------
    # 8. Standard embedding alias expected by downstream scib calls
    # ------------------------------------------------------------------
    import numpy as np
    ad.obsm["X_emb"] = np.asarray(ad.obsm[emb_key]).copy()

    # ------------------------------------------------------------------
    # 9. Metrics
    #
    #    output_type = 'knn':
    #      - kBET, iLISI, cLISI, graph_connectivity all operate in knn
    #        graph mode, reading ad.obsp['connectivities'] (aliased from
    #        conn_key = "bbknn_connectivities" for the duration of each call)
    #      - dist_key  auto-derived → "bbknn_distances"
    #      - neighbors_uns_key auto-derived → "neighbors_bbknn"
    #
    #    emb_key = "X_diffmap_bbknn":
    #      - enables embedding-based metrics (cell_type_ASW, batch_ASW,
    #        isolated_label_ASW, isolated_label_F1) on the diffmap space
    #
    #    cluster_key = "leiden_bbknn":
    #      - NMI and ARI compare these clusters against cfg.label_key
    #
    #    adata_pre = snapshot from step 2 (same cells, pre-integration):
    #      - pcr_comparison, hvg_conservation, cell_cycle_conservation
    # ------------------------------------------------------------------
    perf.start("metrics")
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
            emb_key=emb_key,                    # "X_diffmap_bbknn"
            conn_key=conn_key,                  # "bbknn_connectivities"
            # dist_key auto-derived    → "bbknn_distances"
            # neighbors_uns_key auto-derived → "neighbors_bbknn"
            output_type="knn",
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
    # 10. Save artifacts
    # ------------------------------------------------------------------
    save_run_artifacts(
        outdir,
        metrics=metrics,
        config=cfg,
        perf_df=perf_df,
        adata=ad,
        save_h5ad=cfg.save_h5ad,
        h5ad_name="adata_bbknn.h5ad",
    )

    # ------------------------------------------------------------------
    # 11. Plots
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

    title_prefix = f"BBKNN — {cfg.run_tag} (n={ad_plot.n_obs:,})"
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
        metrics, perf_df, outdir=plot_dir, title=f"BBKNN metrics — {cfg.run_tag}"
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return ad, metrics, perf_df


# Backwards-compatible alias for older runner imports.
run_bbknn = run