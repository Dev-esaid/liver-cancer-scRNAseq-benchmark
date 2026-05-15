from __future__ import annotations

"""
Classic MNN integration runner.

This module runs batchelor::mnnCorrect in R on a concatenated AnnData object,
then imports a PCA embedding of the corrected expression back into Python for
neighbors/UMAP/Leiden/metrics.

Key design choices
------------------
* This implements *classic* MNN correction (mnnCorrect), not fastMNN.
* When HVGs are requested, the selected gene list is passed to R and used as
  ``subset.row`` in mnnCorrect.
* The corrected expression assay is reduced to PCA inside R using
  BiocSingular::runPCA with an approximate SVD.
* Runtime is controlled with two official levers exposed by batchelor:
  - ``BPPARAM`` for parallelized PCA / NN search.
  - ``BNPARAM`` for exact or approximate nearest-neighbor search via
    BiocNeighbors.
* All outputs are namespaced by ``run_tag``.
"""

import os
import selectors
import subprocess
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs
from thesis_project.Integration.Integration_benchmark.graph import build_neighbors, build_umap, build_leiden
from thesis_project.Integration.Integration_benchmark.metrics import integration_metrics
from thesis_project.Integration.Integration_benchmark.io import save_run_artifacts
from thesis_project.Integration.Integration_benchmark.plotting import (
    subsample_for_plotting,
    plot_umap_pub,
    plot_marker_dotplot_pub,
    plot_metric_summary,
)


@dataclass
class MNNConfig:
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # Expression input
    input_layer: Optional[str] = None

    # Feature selection
    require_hvg: bool = True
    hvg_key: str = "highly_variable"
    max_hvgs: Optional[int] = 4000

    # Classic MNN parameters
    k: int = 20
    sigma: float = 0.1
    cos_norm_in: bool = True
    cos_norm_out: bool = True
    svd_dim: int = 0
    var_adj: bool = True
    auto_merge: bool = False

    # Neighbor search / parallelization
    nn_method: str = "hnsw"
    bp_workers: int = 4
    restrict_n_per_batch: Optional[int] = None

    # Downstream PCA / graph / UMAP / Leiden
    n_pcs: int = 50
    neighbors_k: int = 50
    umap_min_dist: float = 0.35
    umap_spread: float = 1.0
    leiden_resolution: float = 1.0

    # R execution
    rscript_exe: str = "Rscript"
    r_script_path: Optional[str] = None
    r_timeout_s: Optional[int] = 1800

    # metrics
    n_isolated: Optional[int] = None
    lisi_subsample: Optional[int] = None
    compute_trajectory: bool = False

    # plotting
    plot_subsample_n: int = 200_000
    plot_subsample_stratify_by: Optional[str] = "dataset"
    plot_covariates: tuple = (
        "dataset", "major_celltype_l1", "tumor_status", "technology", "cancer_type",
    )
    plot_extra_covariates: tuple = (
        "platform", "tissue", "compartment", "disease_group", "donor_id",
    )

    # run identity / persistence
    run_tag: str = "mnnclassic"
    seed: int = 0
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Validation (unchanged)
# ---------------------------------------------------------------------------

def _validate_cfg(cfg: MNNConfig) -> None:
    if not isinstance(cfg.batch_key, str) or not cfg.batch_key.strip():
        raise ValueError("batch_key must be a non-empty string.")
    if not isinstance(cfg.label_key, str) or not cfg.label_key.strip():
        raise ValueError("label_key must be a non-empty string.")
    if cfg.input_layer is not None and (
        not isinstance(cfg.input_layer, str) or not cfg.input_layer.strip()
    ):
        raise ValueError("input_layer must be None or a non-empty string.")
    if not isinstance(cfg.hvg_key, str) or not cfg.hvg_key.strip():
        raise ValueError("hvg_key must be a non-empty string.")
    if int(cfg.k) <= 0:
        raise ValueError(f"k must be positive, got {cfg.k!r}.")
    if float(cfg.sigma) <= 0:
        raise ValueError(f"sigma must be positive, got {cfg.sigma!r}.")
    if int(cfg.svd_dim) < 0:
        raise ValueError(f"svd_dim must be non-negative, got {cfg.svd_dim!r}.")
    if cfg.nn_method not in {"kmknn", "hnsw", "annoy"}:
        raise ValueError("nn_method must be one of {'kmknn', 'hnsw', 'annoy'}.")
    if int(cfg.bp_workers) <= 0:
        raise ValueError(f"bp_workers must be positive, got {cfg.bp_workers!r}.")
    if cfg.restrict_n_per_batch is not None and int(cfg.restrict_n_per_batch) <= 0:
        raise ValueError(
            f"restrict_n_per_batch must be positive when provided, got {cfg.restrict_n_per_batch!r}."
        )
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
    if cfg.r_timeout_s is not None and int(cfg.r_timeout_s) <= 0:
        raise ValueError(
            f"r_timeout_s must be positive when provided, got {cfg.r_timeout_s!r}."
        )
    if not isinstance(cfg.run_tag, str) or not cfg.run_tag.strip():
        raise ValueError("run_tag must be a non-empty string.")


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _resolve_r_script_path(cfg: MNNConfig) -> str:
    if cfg.r_script_path is not None:
        path = Path(cfg.r_script_path).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent / "R" / "run_mnncorrect.R"
    if not path.is_file():
        raise FileNotFoundError(f"R script not found: {path}")
    return str(path)


def _stream_subprocess(
    cmd, *, cwd=None, env=None, timeout_s: Optional[int] = None
) -> None:
    start = time.time()
    print(f"[Python] Launching R subprocess at {time.strftime('%H:%M:%S')}")
    print("[Python] Command:", " ".join(str(x) for x in cmd))

    proc = subprocess.Popen(
        [str(x) for x in cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=cwd, env=env, bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError("Failed to capture R subprocess stdout.")

    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    try:
        while True:
            events = selector.select(timeout=1.0)
            if events:
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        print("[R] " + line.rstrip())
            else:
                if proc.poll() is not None:
                    break
                if timeout_s is not None and (time.time() - start) > int(timeout_s):
                    proc.kill()
                    raise TimeoutError(
                        f"R subprocess exceeded timeout of {timeout_s} seconds."
                    )
            if proc.poll() is not None and not events:
                break
    finally:
        try:
            selector.unregister(proc.stdout)
        except Exception:
            pass
        selector.close()
        try:
            proc.stdout.close()
        except Exception:
            pass

    ret = proc.wait()
    elapsed = time.time() - start
    print(f"[Python] R subprocess finished in {elapsed / 60:.2f} min")
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def _load_pca_csv_to_obsm(
    ad: sc.AnnData, pca_csv: str, key: str, expected_rank: Optional[int] = None
) -> None:
    if not os.path.isfile(pca_csv):
        raise FileNotFoundError(f"PCA CSV not found: {pca_csv}")
    print(f"[Python] Loading PCA CSV: {pca_csv}")
    df = pd.read_csv(pca_csv, index_col=0)
    df.index = df.index.astype(str)

    obs_names = ad.obs_names.astype(str)
    missing = obs_names.difference(df.index)
    if len(missing):
        raise RuntimeError(
            f"Imported PCA file is missing {len(missing)} cells present in AnnData. "
            f"Example missing cell: {missing[0]!r}"
        )

    emb = df.loc[obs_names].to_numpy(dtype=np.float32, copy=True)
    if emb.ndim != 2 or emb.shape[0] != ad.n_obs:
        raise RuntimeError(
            f"Loaded embedding has invalid shape {emb.shape}; expected ({ad.n_obs}, rank)."
        )
    if expected_rank is not None and emb.shape[1] != int(expected_rank):
        raise RuntimeError(
            f"Loaded embedding has {emb.shape[1]} PCs, expected {expected_rank}."
        )
    if not np.isfinite(emb).all():
        raise RuntimeError("Imported embedding contains non-finite values.")
    ad.obsm[key] = emb
    print(f"[Python] Loaded embedding shape: {emb.shape}")


def _prepare_input_adata(
    adata_in: sc.AnnData, cfg: MNNConfig
) -> Tuple[sc.AnnData, Optional[List[str]]]:
    ad, _ = subset_and_cast_obs(
        adata_in, cfg.batch_key, cfg.label_key, cfg.exclude_datasets
    )
    if cfg.input_layer is not None:
        if cfg.input_layer not in ad.layers:
            raise ValueError(
                f"input_layer='{cfg.input_layer}' not found in ad.layers"
            )
        ad = ad.copy()
        ad.X = ad.layers[cfg.input_layer]

    subset_genes = None
    if cfg.require_hvg:
        if cfg.hvg_key not in ad.var:
            raise ValueError(
                f"require_hvg=True but ad.var['{cfg.hvg_key}'] is missing. "
                "Provide the HVG flag column or set require_hvg=False."
            )
        mask = np.asarray(ad.var[cfg.hvg_key]).astype(bool)
        genes = ad.var_names[mask].astype(str).tolist()
        if not genes:
            raise ValueError(f"ad.var['{cfg.hvg_key}'] contains no True entries.")
        if cfg.max_hvgs is not None:
            genes = genes[: int(cfg.max_hvgs)]
        subset_genes = genes

    return ad, subset_genes


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    adata_in: sc.AnnData, outdir: str, cfg: MNNConfig
) -> Tuple[sc.AnnData, Dict[str, Any], pd.DataFrame]:
    """
    MNN (mnnCorrect) run:
      subset → snapshot adata_pre → write temp h5ad → call R
      → load corrected PCA embedding → neighbors/UMAP/Leiden → metrics → save → plots

    Variable naming:
      ad            : integrated AnnData (modified in-place)
      adata_pre     : snapshot after _prepare_input_adata, BEFORE writing h5ad or R call.
                      Same cell set as ad; required for scib comparison metrics.
                      Snapshotted after _prepare_input_adata because that function may
                      create a copy (input_layer swap) and modifies obs columns.
      emb_key       : "X_mnn_mnnclassic"  — PCA of mnnCorrect-corrected expression (from R)
      neigh_key     : "neighbors_mnnclassic"
      conn_key      : ad.uns[neigh_key]["connectivities_key"]
                      → "neighbors_mnnclassic_connectivities"
      cluster_key   : "leiden_mnnclassic"

      output_type="full": mnnCorrect is a feature-space correction method.
        kBET/LISI recompute their own kNN from X (corrected expression) internally,
        consistent with scib's categorisation of classic MNN as a "full" output type.
        Embedding-based metrics (silhouette, isolated label ASW/F1) use emb_key.
        graph_connectivity reads conn_key (aliased to obsp['connectivities']).
        neighbors_uns_key passed explicitly to avoid wrong auto-derivation.
    """
    _validate_cfg(cfg)
    os.makedirs(outdir, exist_ok=True)
    set_global_seed(cfg.seed, use_torch=False)
    perf = PerfLogger(track_gpu=False)

    print(f"[Python] Starting MNN run: {cfg.run_tag}")
    print(f"[Python] Input cells: {adata_in.n_obs}, genes: {adata_in.n_vars}")

    # ------------------------------------------------------------------
    # 1. Subset + optional layer swap + HVG selection
    # ------------------------------------------------------------------
    perf.start("subset")
    ad, subset_genes = _prepare_input_adata(adata_in, cfg)
    perf.end("subset")

    if ad.n_obs < 2:
        raise ValueError(f"Need at least 2 cells after subsetting, found {ad.n_obs}.")
    if ad.n_vars < 2:
        raise ValueError(f"Need at least 2 genes after subsetting, found {ad.n_vars}.")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← after _prepare_input_adata (which may copy
    #    and modify ad), BEFORE writing h5ad or calling R.
    #    Same cell set as ad; required for scib comparison metrics
    #    (pcr_comparison, hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Write temp h5ad for R
    # ------------------------------------------------------------------
    temp_h5ad = os.path.join(outdir, f"adata_for_mnn_{cfg.run_tag}.h5ad")
    perf.start("write_h5ad")
    ad.write_h5ad(temp_h5ad)
    perf.end("write_h5ad")

    # ------------------------------------------------------------------
    # 4. HVG gene list for R (subset.row)
    # ------------------------------------------------------------------
    subset_genes_csv = None
    if subset_genes is not None:
        subset_genes_csv = os.path.join(outdir, f"mnn_genes_{cfg.run_tag}.csv")
        pd.Series(subset_genes, dtype="string").to_csv(
            subset_genes_csv, index=False, header=False
        )
        print(
            f"[Python] HVGs exported for mnnCorrect subset.row: {len(subset_genes)}"
        )
    else:
        print(
            "[Python] No HVG subset requested; mnnCorrect will use all "
            "genes present in the input object."
        )

    # ------------------------------------------------------------------
    # 5. Call R mnnCorrect
    # ------------------------------------------------------------------
    r_script = _resolve_r_script_path(cfg)
    out_prefix = os.path.join(outdir, cfg.run_tag)
    cmd = [
        cfg.rscript_exe, r_script,
        temp_h5ad,
        cfg.batch_key,
        out_prefix,
        str(int(cfg.k)),
        (subset_genes_csv if subset_genes_csv is not None else "NA"),
        str(int(cfg.n_pcs)),
        str(int(cfg.seed)),
        str(float(cfg.sigma)),
        str(bool(cfg.cos_norm_in)).upper(),
        str(bool(cfg.cos_norm_out)).upper(),
        str(int(cfg.svd_dim)),
        str(bool(cfg.var_adj)).upper(),
        str(bool(cfg.auto_merge)).upper(),
        cfg.nn_method,
        str(int(cfg.bp_workers)),
        (str(int(cfg.restrict_n_per_batch))
         if cfg.restrict_n_per_batch is not None else "NA"),
    ]

    perf.start("r_mnncorrect")
    _stream_subprocess(cmd, cwd=outdir, timeout_s=cfg.r_timeout_s)
    perf.end("r_mnncorrect")

    # ------------------------------------------------------------------
    # 6. Load PCA of corrected expression from R
    #    emb_key = "X_mnn_mnnclassic"
    # ------------------------------------------------------------------
    pca_csv = out_prefix + "_mnn_pca.csv"
    emb_key = f"X_mnn_{cfg.run_tag}"

    perf.start("load_embedding")
    _load_pca_csv_to_obsm(ad, pca_csv, emb_key)
    perf.end("load_embedding")

    # ------------------------------------------------------------------
    # 7. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    neigh_key = f"neighbors_{cfg.run_tag}"   # "neighbors_mnnclassic"

    perf.start("neighbors")
    build_neighbors(
        ad,
        use_rep=emb_key,
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
    # 8. Standard embedding alias (must be set BEFORE metrics call)
    # ------------------------------------------------------------------
    ad.obsm["X_emb"] = np.asarray(ad.obsm[emb_key]).copy()

    # ------------------------------------------------------------------
    # 9. Metrics
    #
    #    conn_key = "neighbors_mnnclassic_connectivities"  (from uns)
    #    dist_key = "neighbors_mnnclassic_distances"        (from uns)
    #    neighbors_uns_key = "neighbors_mnnclassic"         (= neigh_key, explicit)
    #
    #    output_type="full": mnnCorrect is a feature-space correction method.
    #      kBET/LISI recompute kNN from X (corrected expression) internally,
    #      consistent with scib's "full" categorisation of classic MNN.
    #      Embedding-based metrics use emb_key (PCA of corrected expression).
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]  # "neighbors_mnnclassic_connectivities"
    dist_key = ad.uns[neigh_key]["distances_key"]       # "neighbors_mnnclassic_distances"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning,
                                message=r".*pandas.value_counts.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=r".*in1d.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning,
                                message=r".*anndata2ri.*")

        metrics = integration_metrics(
            ad,
            batch_key=cfg.batch_key,
            label_key=cfg.label_key,
            cluster_key=f"leiden_{cfg.run_tag}",
            emb_key=emb_key,
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

    metrics["mnn_k"] = int(cfg.k)
    metrics["mnn_sigma"] = float(cfg.sigma)
    metrics["mnn_cos_norm_in"] = bool(cfg.cos_norm_in)
    metrics["mnn_cos_norm_out"] = bool(cfg.cos_norm_out)
    metrics["mnn_svd_dim"] = int(cfg.svd_dim)
    metrics["mnn_var_adj"] = bool(cfg.var_adj)
    metrics["mnn_auto_merge"] = bool(cfg.auto_merge)
    metrics["mnn_nn_method"] = cfg.nn_method
    metrics["mnn_bp_workers"] = int(cfg.bp_workers)
    metrics["mnn_restrict_n_per_batch"] = cfg.restrict_n_per_batch
    metrics["mnn_input_layer"] = cfg.input_layer if cfg.input_layer is not None else "X"
    metrics["mnn_hvg_key"] = cfg.hvg_key if cfg.require_hvg else "none"
    metrics["mnn_max_hvgs"] = cfg.max_hvgs if cfg.require_hvg else None
    perf.end("metrics")

    perf_df = perf.to_df()

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    save_run_artifacts(
        outdir,
        metrics=metrics,
        config=cfg,
        perf_df=perf_df,
        adata=ad,
        save_h5ad=cfg.save_h5ad,
        h5ad_name=f"adata_mnn_{cfg.run_tag}.h5ad",
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

    title_prefix = f"mnnCorrect — {cfg.run_tag} (n={ad_plot.n_obs:,})"
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
        title=f"mnnCorrect metrics — {cfg.run_tag}",
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    print("[Python] MNN run completed.")
    return ad, metrics, perf_df


run_mnncorrect_via_r = run