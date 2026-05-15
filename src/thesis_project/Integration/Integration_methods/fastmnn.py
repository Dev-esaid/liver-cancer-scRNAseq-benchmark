from __future__ import annotations

import warnings
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
import pandas as pd

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs
from thesis_project.Integration.Integration_benchmark.graph import (
    build_neighbors,
    build_umap,
    build_leiden,
)
from thesis_project.Integration.Integration_benchmark.metrics import integration_metrics
from thesis_project.Integration.Integration_benchmark.io import save_run_artifacts
from thesis_project.Integration.Integration_benchmark.plotting import (
    subsample_for_plotting,
    plot_umap_pub,
    plot_marker_dotplot_pub,
    plot_metric_summary,
)


@dataclass
class FastMNNConfig:
    # required metadata
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # HVG handling (recommended)
    require_hvg: bool = True
    hvg_key: str = "highly_variable"
    max_hvgs: Optional[int] = 4000

    # fastMNN parameters (R)
    k: int = 50
    d: int = 50
    cos_norm: bool = False
    ndist: float = 1.0
    assay_type: str = "X"

    # downstream graph
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
    plot_covariates: Tuple[str, ...] = (
        "dataset",
        "major_celltype_l1",
        "tumor_status",
        "technology",
        "cancer_type",
    )
    plot_extra_covariates: Tuple[str, ...] = (
        "platform",
        "tissue",
        "compartment",
        "disease_group",
        "donor_id",
    )

    # run identity
    run_tag: str = "fastmnn"
    seed: int = 0

    # persistence
    save_h5ad: bool = True

    # I/O optimisation
    write_minimal_h5ad: bool = True
    minimal_obs_extra: Tuple[str, ...] = ("donor_id",)

    # guard thresholds
    max_zero_signal_fraction: float = 0.05
    max_zero_signal_cells_report: int = 10

    def __post_init__(self):
        if self.k <= 0:
            raise ValueError("k must be > 0")
        if self.d <= 0:
            raise ValueError("d must be > 0")
        if self.max_hvgs is not None and self.max_hvgs <= 0:
            raise ValueError("max_hvgs must be None or > 0")
        if self.assay_type not in ("X", "logcounts", "log1p_norm"):
            raise ValueError("assay_type must be 'X', 'logcounts', or 'log1p_norm'")
        if not (0 <= float(self.max_zero_signal_fraction) <= 1):
            raise ValueError("max_zero_signal_fraction must be in [0, 1]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_hvg_genes(ad, cfg: FastMNNConfig) -> Optional[List[str]]:
    if cfg.require_hvg and cfg.hvg_key not in ad.var:
        raise ValueError(
            f"cfg.require_hvg=True but ad.var['{cfg.hvg_key}'] is missing. "
            "Either annotate HVGs or set require_hvg=False."
        )
    if cfg.hvg_key not in ad.var:
        return None

    hv_mask = ad.var[cfg.hvg_key].values
    if hv_mask.sum() == 0 and cfg.require_hvg:
        raise ValueError(f"ad.var['{cfg.hvg_key}'] exists but has zero TRUE entries.")

    if "highly_variable_rank" in ad.var.columns:
        hv = ad.var.loc[hv_mask].sort_values("highly_variable_rank", ascending=True)
        genes = hv.index.astype(str).tolist()
    else:
        genes = ad.var_names[hv_mask].astype(str).tolist()

    if cfg.max_hvgs is not None and len(genes) > int(cfg.max_hvgs):
        genes = genes[: int(cfg.max_hvgs)]

    if cfg.require_hvg and len(genes) == 0:
        raise ValueError("HVG selection resulted in empty gene list.")

    return genes


def _write_gene_list_csv(
    genes: Optional[List[str]], cfg: FastMNNConfig, outdir: Path
) -> Optional[Path]:
    if genes is None:
        return None
    gene_csv = outdir / f"fastmnn_genes_{cfg.run_tag}.csv"
    pd.DataFrame({"gene": genes}).to_csv(gene_csv, index=False, header=False)
    return gene_csv


def _make_minimal_anndata_for_r(ad, cfg: FastMNNConfig, genes: Optional[List[str]]):
    keep_obs = [cfg.batch_key, cfg.label_key]
    for c in cfg.minimal_obs_extra:
        if c in ad.obs and c not in keep_obs:
            keep_obs.append(c)

    if genes is not None:
        present = [g for g in genes if g in ad.var_names]
        if cfg.require_hvg and len(present) == 0:
            raise ValueError("None of the selected HVGs were found in ad.var_names.")
        ad_min = ad[:, present].copy()
    else:
        ad_min = ad.copy()

    ad_min.obs = ad_min.obs.loc[:, [c for c in keep_obs if c in ad_min.obs]].copy()
    ad_min.obs[cfg.batch_key] = ad_min.obs[cfg.batch_key].astype(str)
    ad_min.obs[cfg.label_key] = ad_min.obs[cfg.label_key].astype(str)
    return ad_min


def _get_assay_matrix(ad, assay_type: str):
    """
    Return the matrix corresponding to the configured assay.
    - 'X' uses ad.X
    - otherwise uses ad.layers[assay_type]
    """
    if assay_type == "X":
        return ad.X
    if assay_type not in ad.layers:
        raise ValueError(
            f"Requested assay_type='{assay_type}' not found in ad.layers. "
            f"Available layers: {list(ad.layers.keys())}"
        )
    return ad.layers[assay_type]


def _check_fastmnn_assay_has_signal(
    ad,
    cfg: FastMNNConfig,
    genes: Optional[List[str]],
) -> None:
    """
    Guard against launching fastMNN on an assay that is empty or effectively empty
    across the selected genes.

    Raises early if too many cells have zero total signal across the selected genes.
    """
    if genes is not None:
        present = [g for g in genes if g in ad.var_names]
        if len(present) == 0:
            raise ValueError(
                "No selected HVG genes are present in ad.var_names for fastMNN."
            )
        ad_chk = ad[:, present]
    else:
        ad_chk = ad

    Xchk = _get_assay_matrix(ad_chk, cfg.assay_type)

    try:
        if hasattr(Xchk, "data"):
            vals = Xchk.data
            if vals is not None and vals.size > 0 and not np.all(np.isfinite(vals)):
                raise ValueError(
                    f"fastMNN input assay '{cfg.assay_type}' contains non-finite "
                    "values (NA/Inf) in stored entries."
                )
        else:
            arr = np.asarray(Xchk)
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"fastMNN input assay '{cfg.assay_type}' contains non-finite "
                    "values (NA/Inf)."
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Unable to validate finiteness of fastMNN assay '{cfg.assay_type}': {exc}"
        ) from exc

    try:
        if hasattr(Xchk, "power"):
            cell_signal = np.asarray(Xchk.power(2).sum(axis=1)).ravel()
        else:
            arr = np.asarray(Xchk)
            cell_signal = np.sum(arr ** 2, axis=1)
    except Exception as exc:
        raise ValueError(
            f"Unable to compute per-cell signal for fastMNN assay '{cfg.assay_type}': {exc}"
        ) from exc

    if cell_signal.ndim != 1 or cell_signal.shape[0] != ad_chk.n_obs:
        raise ValueError(
            "Internal fastMNN guard error: per-cell signal vector has unexpected shape."
        )

    if not np.all(np.isfinite(cell_signal)):
        raise ValueError(
            f"fastMNN input assay '{cfg.assay_type}' produced non-finite per-cell "
            "signal across the selected genes."
        )

    zero_mask = cell_signal <= 0
    zero_cells = int(np.sum(zero_mask))
    frac_zero = float(zero_cells / ad_chk.n_obs) if ad_chk.n_obs > 0 else 0.0

    if zero_cells > 0:
        zero_names = ad_chk.obs_names[np.where(zero_mask)[0]]
        preview = list(map(str, zero_names[: int(cfg.max_zero_signal_cells_report)]))
        warnings.warn(
            f"[fastMNN guard] assay '{cfg.assay_type}' has {zero_cells}/{ad_chk.n_obs} "
            f"cells ({frac_zero:.1%}) with zero signal across the selected genes. "
            f"First few cell ids: {preview}"
        )

    if frac_zero > float(cfg.max_zero_signal_fraction):
        raise ValueError(
            f"Selected fastMNN assay '{cfg.assay_type}' has {zero_cells}/{ad_chk.n_obs} "
            f"cells ({frac_zero:.1%}) with zero signal across the selected genes. "
            "Refusing to run fastMNN on this assay. "
            "Use the correct normalized matrix (for this pipeline, typically assay_type='X')."
        )


def _resolve_fastmnn_embedding_csv(out_prefix: Path) -> Path:
    """
    Resolve the embedding CSV written by the R fastMNN script.

    Supported filenames:
    - <out_prefix>_fastmnn_embedding.csv   (current R output in your run)
    - <out_prefix>_fastmnn.csv             (older/alternate convention)
    """
    candidates = [
        Path(str(out_prefix) + "_fastmnn_embedding.csv"),
        Path(str(out_prefix) + "_fastmnn.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Expected fastMNN embedding CSV not found. Checked:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(adata_in, outdir: str, cfg: Optional[FastMNNConfig] = None):
    if cfg is None:
        cfg = FastMNNConfig()
    return run_fastmnn_via_r(adata_in=adata_in, outdir=outdir, cfg=cfg)


def run_fastmnn_via_r(
    adata_in, outdir: str, cfg: FastMNNConfig
) -> Tuple[Any, Dict[str, Any], Any]:
    """
    Full fastMNN benchmark run (R batchelor::fastMNN):
      subset → snapshot adata_pre → write temp h5ad → call R
      → read corrected embedding → neighbors/umap/leiden → metrics → save → plots
    """
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

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
        raise ValueError(f"Missing batch_key='{cfg.batch_key}' in ad.obs")
    if cfg.label_key not in ad.obs:
        raise ValueError(f"Missing label_key='{cfg.label_key}' in ad.obs")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. HVG selection
    # ------------------------------------------------------------------
    genes = _select_hvg_genes(ad, cfg)

    # ------------------------------------------------------------------
    # 3b. Guard: validate that the selected assay has real signal
    # ------------------------------------------------------------------
    _check_fastmnn_assay_has_signal(ad, cfg, genes)

    gene_csv = _write_gene_list_csv(genes, cfg, outdir_p)

    # ------------------------------------------------------------------
    # 4. Write minimal h5ad for R
    # ------------------------------------------------------------------
    perf.start("write_h5ad")
    temp_h5ad = outdir_p / f"adata_for_fastmnn_{cfg.run_tag}.h5ad"

    if cfg.write_minimal_h5ad:
        ad_for_r = _make_minimal_anndata_for_r(ad, cfg, genes)
    else:
        ad_for_r = ad

    try:
        Xmin = ad_for_r.X.min()
        if float(Xmin) < -1e-6:
            warnings.warn(
                f"ad.X has negative values (min={float(Xmin):.3g}); "
                "fastMNN typically expects log-normalized non-negative expression."
            )
    except Exception:
        pass

    ad_for_r.write_h5ad(str(temp_h5ad))
    perf.end("write_h5ad")

    # ------------------------------------------------------------------
    # 5. Call R
    # ------------------------------------------------------------------
    r_script = Path(
        "/data1/esraa/Thesis-Project/src/thesis_project/Integration"
        "/Integration_methods/R/run_fastmnn.R"
    )
    if not r_script.exists():
        raise FileNotFoundError(f"run_fastmnn.R not found at: {r_script}")

    out_prefix = outdir_p / cfg.run_tag
    subset_csv_arg = str(gene_csv) if gene_csv is not None else "NA"

    rcmd = [
        "Rscript",
        str(r_script),
        str(temp_h5ad),
        cfg.batch_key,
        str(out_prefix),
        str(int(cfg.d)),
        str(int(cfg.k)),
        "TRUE" if cfg.cos_norm else "FALSE",
        str(float(cfg.ndist)),
        cfg.assay_type,
        subset_csv_arg,
        str(int(cfg.seed)),
    ]

    perf.start("r_fastmnn")
    proc = subprocess.run(rcmd, text=True, capture_output=True)
    perf.end("r_fastmnn")

    if proc.returncode != 0:
        raise RuntimeError(
            "R fastMNN failed.\n\n"
            f"Command:\n  {' '.join(rcmd)}\n\n"
            f"STDOUT:\n{proc.stdout}\n\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    # ------------------------------------------------------------------
    # 6. Read corrected embedding back from R
    # ------------------------------------------------------------------
    perf.start("read_embedding")
    embed_csv = _resolve_fastmnn_embedding_csv(out_prefix)

    emb_df = pd.read_csv(embed_csv, index_col=0, low_memory=False)
    emb_df.index = emb_df.index.astype(str)

    if emb_df.shape[0] != ad.n_obs:
        raise RuntimeError(
            f"Embedding has {emb_df.shape[0]} cells but AnnData has {ad.n_obs} cells."
        )

    if list(emb_df.index) == list(ad.obs_names):
        emb = emb_df.values
    else:
        if set(ad.obs_names).issubset(set(emb_df.index)):
            emb = emb_df.loc[ad.obs_names].values
        else:
            raise RuntimeError(
                "fastMNN embedding rownames do not match ad.obs_names. "
                "Refusing to assume identical ordering (publication safety)."
            )

    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[1] != int(cfg.d):
        raise RuntimeError(
            f"fastMNN embedding has shape {emb.shape}, expected (n_cells, d={cfg.d})."
        )

    key_emb = f"X_fastmnn_{cfg.run_tag}"
    ad.obsm[key_emb] = emb
    perf.end("read_embedding")

    # ------------------------------------------------------------------
    # 7. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    perf.start("neighbors")
    neigh_key = f"neighbors_{cfg.run_tag}"
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
    # 8. Standard embedding alias
    # ------------------------------------------------------------------
    ad.obsm["X_emb"] = np.asarray(ad.obsm[key_emb]).copy()

    # ------------------------------------------------------------------
    # 9. Metrics
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]
    dist_key = ad.uns[neigh_key]["distances_key"]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*pandas.value_counts.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*in1d.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*anndata2ri.*",
        )

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

    metrics["fastmnn_r_script"] = str(r_script)
    metrics["fastmnn_assay_type"] = cfg.assay_type
    metrics["fastmnn_d"] = int(cfg.d)
    metrics["fastmnn_k"] = int(cfg.k)
    metrics["fastmnn_cos_norm"] = bool(cfg.cos_norm)
    metrics["fastmnn_ndist"] = float(cfg.ndist)
    metrics["fastmnn_seed"] = int(cfg.seed)
    metrics["fastmnn_hvg_used"] = bool(gene_csv is not None)
    metrics["fastmnn_gene_csv"] = str(gene_csv) if gene_csv is not None else None
    metrics["fastmnn_wrote_minimal_h5ad"] = bool(cfg.write_minimal_h5ad)
    metrics["fastmnn_minimal_obs_extra"] = list(cfg.minimal_obs_extra)
    metrics["fastmnn_embedding_csv"] = str(embed_csv)

    perf.end("metrics")
    perf_df = perf.to_df()

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    save_run_artifacts(
        str(outdir_p),
        metrics=metrics,
        config=cfg,
        perf_df=perf_df,
        adata=ad,
        save_h5ad=cfg.save_h5ad,
        h5ad_name=f"adata_fastmnn_{cfg.run_tag}.h5ad",
    )

    # ------------------------------------------------------------------
    # 11. Plots
    # ------------------------------------------------------------------
    perf.start("plots")
    plot_dir = outdir_p / "plots_pub"
    plot_dir.mkdir(exist_ok=True)

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

    title_prefix = f"fastMNN — {cfg.run_tag} (n={ad_plot.n_obs:,})"
    for cov in covs:
        plot_umap_pub(
            ad_plot,
            umap_key=umap_key,
            color=cov,
            title_prefix=title_prefix,
            outdir=str(plot_dir),
            alpha=0.75,
        )

    plot_umap_pub(
        ad_plot,
        umap_key=umap_key,
        color=f"leiden_{cfg.run_tag}",
        title_prefix=title_prefix,
        outdir=str(plot_dir),
        alpha=0.75,
    )

    if cfg.label_key in ad_plot.obs:
        plot_marker_dotplot_pub(ad_plot, groupby=cfg.label_key, outdir=str(plot_dir))

    plot_metric_summary(
        metrics,
        perf_df,
        outdir=str(plot_dir),
        title=f"fastMNN metrics — {cfg.run_tag}",
    )
    perf.end("plots")

    perf_df = perf.save_csv(str(outdir_p / "perf_log.csv"))
    return ad, metrics, perf_df