from __future__ import annotations

import os
import shlex
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

from thesis_project.Integration.Integration_benchmark.graph import (
    build_leiden,
    build_neighbors,
    build_umap,
)
from thesis_project.Integration.Integration_benchmark.io import save_run_artifacts
from thesis_project.Integration.Integration_benchmark.metrics import integration_metrics
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.plotting import (
    plot_marker_dotplot_pub,
    plot_metric_summary,
    plot_umap_pub,
    subsample_for_plotting,
)
from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs


@dataclass
class LigerConfig:
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    input_layer_raw: Optional[str] = "counts"

    require_hvg: bool = True
    hvg_key: str = "highly_variable"
    max_hvgs: Optional[int] = None

    k_factors: int = 50
    lambda_reg: float = 3.0
    n_iters: int = 50
    n_cores: int = 2
    align_method: str = "centroidAlign"

    neighbors_k: int = 50
    umap_min_dist: float = 0.4
    umap_spread: float = 1.0
    leiden_resolution: float = 1.0

    # metrics
    n_isolated: Optional[int] = None
    lisi_subsample: Optional[int] = None
    compute_trajectory: bool = False

    run_tag: str = "liger"
    seed: int = 0

    plot_subsample_n: int = 200_000
    plot_subsample_stratify_by: Optional[str] = "dataset"
    plot_covariates: Tuple[str, ...] = (
        "dataset", "major_celltype_l1", "tumor_status", "technology", "cancer_type",
    )
    plot_extra_covariates: Tuple[str, ...] = (
        "platform", "tissue", "compartment", "disease_group", "donor_id",
    )

    save_h5ad: bool = True
    save_rds: bool = True
    keep_temp_files: bool = False
    r_script_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.k_factors <= 1:
            raise ValueError("k_factors must be > 1")
        if self.lambda_reg <= 0:
            raise ValueError("lambda_reg must be > 0")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be > 0")
        if self.n_cores <= 0:
            raise ValueError("n_cores must be > 0")
        if self.max_hvgs is not None and self.max_hvgs <= 0:
            raise ValueError("max_hvgs must be None or > 0")
        if self.align_method not in {"centroidAlign", "quantileNorm"}:
            raise ValueError("align_method must be one of {'centroidAlign', 'quantileNorm'}")


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _stream_subprocess(cmd, cwd=None, env=None) -> None:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=cwd, env=env, bufsize=1,
    )
    captured: List[str] = []
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if line:
                line = line.rstrip("\n")
                captured.append(line)
                print("[R] " + line, flush=True)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    ret = proc.wait()
    if ret != 0:
        tail = "\n".join(captured[-80:])
        raise RuntimeError(
            "LIGER R script failed.\n"
            f"Command: {' '.join(shlex.quote(x) for x in cmd)}\n"
            f"Exit status: {ret}\n"
            "Last R output:\n"
            f"{tail}"
        )


def _resolve_r_script(cfg: LigerConfig) -> Path:
    if cfg.r_script_path is not None:
        path = Path(cfg.r_script_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"run_liger.R not found at: {path}")
        return path

    here = Path(__file__).resolve()
    candidates = [
        here.with_name("run_liger.R"),
        here.parent / "R" / "run_liger.R",
        here.parents[1] / "R" / "run_liger.R",
        here.parents[2] / "R" / "run_liger.R",
        Path("/data1/esraa/Thesis-Project/src/thesis_project/Integration"
             "/Integration_methods/R/run_liger.R"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "run_liger.R could not be located. Set LigerConfig.r_script_path explicitly."
    )


def _get_raw_counts_matrix(ad: sc.AnnData, cfg: LigerConfig):
    if cfg.input_layer_raw is None:
        print("[Python] Using raw counts from ad.X")
        return ad.X
    if cfg.input_layer_raw not in ad.layers:
        raise KeyError(
            f"cfg.input_layer_raw='{cfg.input_layer_raw}' not found in "
            f"ad.layers: {list(ad.layers.keys())}"
        )
    print(f"[Python] Using raw counts from layer '{cfg.input_layer_raw}'")
    return ad.layers[cfg.input_layer_raw]


def _cell_sums_from_anndata_matrix(X) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X).sum(axis=1)


def _validate_raw_counts_matrix(X) -> None:
    if sparse.issparse(X):
        values = X.data
        min_val = float(values.min()) if values.size else 0.0
        sample = values[: min(values.size, 100_000)]
    else:
        arr = np.asarray(X)
        min_val = float(arr.min()) if arr.size else 0.0
        flat = arr.ravel()
        sample = flat[: min(flat.size, 100_000)]

    if min_val < 0:
        raise ValueError(
            f"Negative values found in input matrix (min={min_val:.6g}). "
            "LIGER requires non-negative raw counts."
        )
    if sample.size and not np.allclose(sample, np.round(sample), atol=1e-6):
        warnings.warn(
            "Input values are non-negative but not integer-like. "
            "This is only appropriate if raw counts were stored as float dtype.",
            RuntimeWarning,
        )
    cell_sums = _cell_sums_from_anndata_matrix(X)
    zero_cells = int(np.sum(cell_sums <= 0))
    if zero_cells > 0:
        raise ValueError(
            f"Found {zero_cells} cells with zero total raw counts. "
            "Filter or drop zero-count cells before running LIGER."
        )
    print(f"[Python] Raw-count validation passed (min={min_val:.3f}, zero_count_cells=0)")


def _drop_zero_count_cells(ad: sc.AnnData, cfg: LigerConfig) -> Tuple[sc.AnnData, int]:
    X = _get_raw_counts_matrix(ad, cfg)
    cell_sums = _cell_sums_from_anndata_matrix(X)
    keep = cell_sums > 0
    n_drop = int((~keep).sum())
    if n_drop > 0:
        dropped = ad.obs_names[~keep].tolist()
        preview = ", ".join(dropped[:10])
        print(f"[Python] Dropping {n_drop} cells with zero total raw counts before LIGER")
        if preview:
            print(f"[Python] Example dropped cells: {preview}")
        ad = ad[keep].copy()
    return ad, n_drop


def _write_temp_h5ad_for_r(ad: sc.AnnData, outdir: str, cfg: LigerConfig) -> str:
    if not ad.obs_names.is_unique:
        raise ValueError(
            "ad.obs_names must be unique so the exported LIGER embedding "
            "can be mapped back exactly."
        )
    if cfg.batch_key not in ad.obs:
        raise KeyError(f"batch_key '{cfg.batch_key}' not found in ad.obs")

    raw_counts = _get_raw_counts_matrix(ad, cfg)
    _validate_raw_counts_matrix(raw_counts)

    obs = ad.obs[[cfg.batch_key]].copy()
    var = pd.DataFrame(index=ad.var_names.copy())

    ad_r = AnnData(X=raw_counts, obs=obs, var=var)
    ad_r.obs_names = ad.obs_names.copy()
    ad_r.var_names = ad.var_names.copy()
    ad_r.uns["X_name"] = "counts"

    temp_h5ad = os.path.join(outdir, f"adata_for_liger_{cfg.run_tag}.h5ad")
    print(f"[Python] Writing compact temporary H5AD for R: {temp_h5ad}")
    ad_r.write_h5ad(temp_h5ad)
    return temp_h5ad


def _select_hvgs(ad: sc.AnnData, cfg: LigerConfig) -> List[str]:
    if cfg.hvg_key not in ad.var:
        raise KeyError(
            f"require_hvg=True but ad.var lacks '{cfg.hvg_key}'. "
            "This wrapper expects externally precomputed HVGs."
        )
    mask = ad.var[cfg.hvg_key].astype(bool).values
    genes = ad.var_names[mask].tolist()
    if not genes:
        raise ValueError(f"No genes were marked True in ad.var['{cfg.hvg_key}']")
    if cfg.max_hvgs is not None and len(genes) > cfg.max_hvgs:
        genes = genes[: cfg.max_hvgs]
        print(f"[Python] Using first {len(genes)} externally supplied HVGs")
    else:
        print(f"[Python] Using all {len(genes)} externally supplied HVGs")
    return genes


def _write_hvg_text(genes: List[str], outdir: str, cfg: LigerConfig) -> str:
    out_path = os.path.join(outdir, f"liger_genes_{cfg.run_tag}.txt")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(genes))
        handle.write("\n")
    print(f"[Python] HVG list written: {out_path} (n={len(genes)})")
    return out_path


def _load_embedding_csv(ad: sc.AnnData, csv_path: str, key: str) -> None:
    df = pd.read_csv(csv_path, index_col=0)
    if list(df.index) == list(ad.obs_names):
        emb = df.to_numpy(dtype=np.float32, copy=False)
    elif all(cell in df.index for cell in ad.obs_names):
        emb = df.loc[ad.obs_names].to_numpy(dtype=np.float32, copy=False)
    else:
        emb = df.to_numpy(dtype=np.float32, copy=False)
        if emb.shape[0] != ad.n_obs:
            raise RuntimeError(
                f"[LIGER] Embedding row count mismatch: csv={emb.shape[0]} "
                f"vs ad.n_obs={ad.n_obs}. "
                "This usually means row names were changed during the R run."
            )
        warnings.warn(
            "Embedding row names could not be matched to ad.obs_names exactly; "
            "falling back to raw row order.",
            RuntimeWarning,
        )
    ad.obsm[key] = emb


def _make_plots(
    ad: sc.AnnData, *, outdir: str, cfg: LigerConfig,
    title_prefix: str, umap_key: str,
) -> None:
    plot_dir = os.path.join(outdir, "plots_pub")
    os.makedirs(plot_dir, exist_ok=True)

    ad_plot = subsample_for_plotting(
        ad, n=cfg.plot_subsample_n, seed=cfg.seed,
        stratify_by=cfg.plot_subsample_stratify_by,
    )

    covs = [
        c for c in (list(cfg.plot_covariates) + list(cfg.plot_extra_covariates))
        if c in ad_plot.obs
    ]
    for must in [cfg.batch_key, cfg.label_key]:
        if must in ad_plot.obs and must not in covs:
            covs.insert(0, must)
    seen: set = set()
    covs = [c for c in covs if not (c in seen or seen.add(c))]

    for cov in covs:
        try:
            plot_umap_pub(
                ad_plot, umap_key=umap_key, color=cov,
                title_prefix=title_prefix, outdir=plot_dir, alpha=0.85,
            )
        except Exception as exc:
            print(f"[plot] UMAP failed for '{cov}': {exc}")

    if cfg.label_key in ad_plot.obs:
        try:
            plot_marker_dotplot_pub(ad_plot, groupby=cfg.label_key, outdir=plot_dir)
        except Exception as exc:
            print(f"[plot] dotplot failed: {exc}")


def _cleanup_temp_files(paths: List[Optional[str]]) -> None:
    for path in paths:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            print(f"[cleanup] Could not remove {path}: {exc}")


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    adata_in: sc.AnnData, outdir: str, cfg: LigerConfig
) -> Tuple[sc.AnnData, Dict[str, Any], pd.DataFrame]:
    """
    LIGER run:
      subset → drop zero-count cells → snapshot adata_pre → write temp h5ad
      → call R → load corrected factors → neighbors/UMAP/Leiden → metrics → save → plots

    Variable naming:
      ad            : integrated AnnData (modified in-place)
      adata_pre     : snapshot after subset + zero-count drop, BEFORE writing h5ad or R call.
                      Same cell set as ad; required for scib comparison metrics.
                      Snapshotted AFTER _drop_zero_count_cells so cell counts match exactly.
      key_emb       : "X_liger_liger"  — LIGER cell factor loadings from R
      neigh_key     : "neighbors_liger"
      conn_key      : ad.uns[neigh_key]["connectivities_key"]
                      → "neighbors_liger_connectivities"
      cluster_key   : "leiden_liger"

      output_type="embed": LIGER outputs cell factor loadings (a low-dim embedding).
        kBET/LISI recompute their own kNN from key_emb internally.
        graph_connectivity reads conn_key (aliased to obsp['connectivities']).
        neighbors_uns_key passed explicitly to avoid wrong auto-derivation.
    """
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)

    set_global_seed(cfg.seed, use_torch=False)
    perf = PerfLogger(track_gpu=False)

    # ------------------------------------------------------------------
    # 1. Subset + drop zero-count cells
    # ------------------------------------------------------------------
    perf.start("subset")
    ad, _ = subset_and_cast_obs(
        adata_in, cfg.batch_key, cfg.label_key, cfg.exclude_datasets
    )
    ad, n_zero_dropped = _drop_zero_count_cells(ad, cfg)
    perf.end("subset")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← AFTER subset and zero-count drop so cell
    #    counts match ad exactly. BEFORE writing h5ad or calling R.
    #    Required for scib comparison metrics (pcr_comparison,
    #    hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Write minimal h5ad for R
    # ------------------------------------------------------------------
    perf.start("write_h5ad")
    temp_h5ad = _write_temp_h5ad_for_r(ad, outdir, cfg)
    perf.end("write_h5ad")

    # ------------------------------------------------------------------
    # 4. HVG list for R
    # ------------------------------------------------------------------
    perf.start("prepare_hvgs")
    subset_genes_txt: Optional[str] = None
    hvg_genes: Optional[List[str]] = None
    if cfg.require_hvg:
        hvg_genes = _select_hvgs(ad, cfg)
        subset_genes_txt = _write_hvg_text(hvg_genes, outdir, cfg)
    perf.end("prepare_hvgs")

    # ------------------------------------------------------------------
    # 5. Call R LIGER
    # ------------------------------------------------------------------
    r_script = _resolve_r_script(cfg)
    out_prefix = os.path.join(outdir, cfg.run_tag)
    cmd = [
        "Rscript", "--vanilla", str(r_script),
        temp_h5ad,
        cfg.batch_key,
        out_prefix,
        str(cfg.k_factors),
        str(cfg.lambda_reg),
        str(cfg.n_iters),
        (subset_genes_txt if subset_genes_txt is not None else "NA"),
        str(cfg.seed),
        cfg.align_method,
        str(cfg.n_cores),
        ("1" if cfg.save_rds else "0"),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["NUMBA_NUM_THREADS"] = "1"
    env["LIGER_N_CORES"] = str(cfg.n_cores)

    print("[Python] Calling R LIGER with:", " ".join(shlex.quote(x) for x in cmd))
    perf.start("r_liger")
    _stream_subprocess(cmd, cwd=outdir, env=env)
    perf.end("r_liger")

    # ------------------------------------------------------------------
    # 6. Load corrected factor embedding from R
    #    key_emb = "X_liger_liger"
    # ------------------------------------------------------------------
    emb_csv = out_prefix + "_liger_factors.csv"
    if not os.path.exists(emb_csv):
        raise RuntimeError(f"Expected LIGER embedding CSV not found: {emb_csv}")

    perf.start("load_embedding")
    key_emb = f"X_liger_{cfg.run_tag}"     # "X_liger_liger"
    _load_embedding_csv(ad, emb_csv, key_emb)
    perf.end("load_embedding")

    # ------------------------------------------------------------------
    # 7. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    neigh_key = f"neighbors_{cfg.run_tag}"  # "neighbors_liger"

    perf.start("neighbors")
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
    # 8. Standard embedding alias (must be set BEFORE metrics call)
    # ------------------------------------------------------------------
    ad.obsm["X_emb"] = np.asarray(ad.obsm[key_emb]).copy()

    # ------------------------------------------------------------------
    # 9. Metrics
    #
    #    conn_key = "neighbors_liger_connectivities"  (from uns)
    #    dist_key = "neighbors_liger_distances"        (from uns)
    #    neighbors_uns_key = "neighbors_liger"         (= neigh_key, explicit)
    #
    #    output_type="embed": LIGER outputs cell factor loadings.
    #      kBET/LISI recompute their own kNN from key_emb internally.
    #      graph_connectivity uses conn_key (aliased to obsp['connectivities']).
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]  # "neighbors_liger_connectivities"
    dist_key = ad.uns[neigh_key]["distances_key"]       # "neighbors_liger_distances"

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
    perf.end("metrics")

    metrics["liger_r_script"] = str(r_script)
    metrics["liger_embedding_csv"] = emb_csv
    metrics["liger_k_factors"] = cfg.k_factors
    metrics["liger_lambda"] = cfg.lambda_reg
    metrics["liger_n_iters"] = cfg.n_iters
    metrics["liger_n_cores"] = cfg.n_cores
    metrics["liger_align_method"] = cfg.align_method
    metrics["liger_used_raw_counts"] = True
    metrics["liger_normalized_internally"] = True
    metrics["liger_hvg_selection_recomputed"] = False
    metrics["liger_n_hvgs_input"] = len(hvg_genes) if hvg_genes is not None else 0
    metrics["liger_used_full_gene_panel_in_object"] = True
    metrics["liger_zero_count_cells_dropped"] = n_zero_dropped

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
        h5ad_name=f"adata_liger_{cfg.run_tag}.h5ad",
    )

    # ------------------------------------------------------------------
    # 11. Plots
    # ------------------------------------------------------------------
    perf.start("plots")
    title_prefix = f"LIGER (external HVGs, full raw panel) — {cfg.run_tag}"
    try:
        _make_plots(
            ad, outdir=outdir, cfg=cfg,
            title_prefix=title_prefix, umap_key=f"X_umap_{cfg.run_tag}",
        )
    except Exception as exc:
        print(f"[plot] plotting failed: {exc}")

    try:
        plot_metric_summary(
            metrics=metrics,
            perf_df=perf_df,
            outdir=os.path.join(outdir, "plots_pub"),
            title=title_prefix,
        )
    except Exception as exc:
        print(f"[plot] plot_metric_summary failed: {exc}")
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))

    # ------------------------------------------------------------------
    # 12. Cleanup temp files
    # ------------------------------------------------------------------
    if not cfg.keep_temp_files:
        temp_paths: List[Optional[str]] = [temp_h5ad, subset_genes_txt]
        if not cfg.save_rds:
            temp_paths.append(out_prefix + "_liger_integrated.rds")
        _cleanup_temp_files(temp_paths)

    return ad, metrics, perf_df