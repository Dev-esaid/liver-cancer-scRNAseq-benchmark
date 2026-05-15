from __future__ import annotations

import os
import shlex
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc

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
class SeuratConfig:
    # required columns
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # input expression export
    # If set, export this layer into X for R.
    # If None, export ad.X as-is.
    input_layer_counts: Optional[str] = None

    # HVG restriction
    require_hvg: bool = True
    hvg_key: str = "highly_variable"
    max_hvgs: Optional[int] = 4000

    # Seurat integration mode
    mode: str = "rpca"
    k_anchor: int = 20
    dims: int = 30
    n_pcs: int = 50

    # downstream graph/embedding
    neighbors_k: int = 25
    umap_min_dist: float = 0.4
    umap_spread: float = 1.0
    leiden_resolution: float = 1.0

    # metrics
    n_isolated: Optional[int] = None
    lisi_subsample: Optional[int] = None
    compute_trajectory: bool = False

    # run identity
    run_tag: str = "seurat_rpca"
    seed: int = 0

    # plotting
    plot_subsample_n: int = 200_000
    plot_subsample_stratify_by: Optional[str] = "dataset"
    plot_covariates: tuple = (
        "dataset",
        "major_celltype_l1",
        "tumor_status",
        "technology",
        "cancer_type",
    )
    plot_extra_covariates: tuple = (
        "platform",
        "tissue",
        "compartment",
        "disease_group",
        "donor_id",
    )

    # persistence
    save_h5ad: bool = True

    def __post_init__(self):
        self.mode = str(self.mode).lower().strip()
        if self.mode not in ("cca", "rpca"):
            raise ValueError("mode must be 'cca' or 'rpca'")
        if self.k_anchor <= 0:
            raise ValueError("k_anchor must be > 0")
        if self.dims <= 0:
            raise ValueError("dims must be > 0")
        if self.n_pcs <= 0:
            raise ValueError("n_pcs must be > 0")
        if self.max_hvgs is not None and self.max_hvgs <= 0:
            raise ValueError("max_hvgs must be None or > 0")


def _stream_subprocess(cmd, cwd=None, env=None):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        env=env,
    )
    captured = []
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            if line:
                stripped = line.rstrip()
                captured.append(stripped)
                print("[R] " + stripped)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    ret = proc.wait()
    if ret != 0:
        tail = "\n".join(captured[-80:])
        raise RuntimeError(
            f"R Seurat script failed (exit status {ret}).\n\n"
            f"Command:\n  {' '.join(shlex.quote(str(x)) for x in cmd)}\n\n"
            f"Last R output:\n{tail}"
        )


def _load_csv_embedding(ad: sc.AnnData, csv_path: str, key: str) -> None:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(str)

    if list(df.index) == list(ad.obs_names):
        emb = df.values.astype(np.float32)
    elif all(x in df.index for x in ad.obs_names):
        emb = df.loc[ad.obs_names].values.astype(np.float32)
    else:
        emb = df.values.astype(np.float32)
        if emb.shape[0] != ad.n_obs:
            raise RuntimeError(
                f"Embedding rows mismatch: csv rows={emb.shape[0]} vs ad.n_obs={ad.n_obs}. "
                "Likely missing/incorrect rownames in R export."
            )

    ad.obsm[key] = emb


def _make_plots(
    ad: sc.AnnData,
    *,
    outdir: str,
    cfg: SeuratConfig,
    title_prefix: str,
    umap_key: str,
):
    plots_dir = os.path.join(outdir, "plots_pub")
    os.makedirs(plots_dir, exist_ok=True)

    adp = subsample_for_plotting(
        ad,
        n=cfg.plot_subsample_n,
        seed=cfg.seed,
        stratify_by=cfg.plot_subsample_stratify_by,
    )

    covs = []
    for c in list(cfg.plot_covariates) + list(cfg.plot_extra_covariates):
        if c in adp.obs.columns:
            covs.append(c)

    for must in [cfg.batch_key, cfg.label_key]:
        if must in adp.obs.columns and must not in covs:
            covs.insert(0, must)

    seen = set()
    covs = [c for c in covs if not (c in seen or seen.add(c))]

    for c in covs:
        try:
            plot_umap_pub(
                adp,
                umap_key=umap_key,
                color=c,
                title_prefix=title_prefix,
                outdir=plots_dir,
                alpha=0.85,
            )
        except Exception as e:
            print(f"[plot] UMAP failed for '{c}': {e}")

    if cfg.label_key in ad.obs.columns:
        try:
            plot_marker_dotplot_pub(
                ad,
                groupby=cfg.label_key,
                outdir=plots_dir,
                title=f"{title_prefix} — marker dotplot (grouped by {cfg.label_key})",
            )
        except Exception as e:
            print(f"[plot] dotplot failed: {e}")


def run(adata_in: sc.AnnData, outdir: str, cfg: SeuratConfig):
    return run_seurat_via_r(adata_in=adata_in, outdir=outdir, cfg=cfg)


def run_seurat_via_r(
    adata_in: sc.AnnData,
    outdir: str,
    cfg: SeuratConfig,
) -> Tuple[sc.AnnData, Dict[str, Any], pd.DataFrame]:
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)

    set_global_seed(cfg.seed, use_torch=False)
    perf = PerfLogger(track_gpu=False)

    # ------------------------------------------------------------------
    # 1. Subset
    # ------------------------------------------------------------------
    perf.start("subset")
    ad, _ = subset_and_cast_obs(
        adata_in,
        cfg.batch_key,
        cfg.label_key,
        cfg.exclude_datasets,
    )
    perf.end("subset")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre BEFORE export mutation
    #    Required for scib comparison metrics.
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Export matrix to X for R if requested
    # ------------------------------------------------------------------
    if cfg.input_layer_counts is not None:
        if cfg.input_layer_counts not in ad.layers:
            raise KeyError(
                f"cfg.input_layer_counts='{cfg.input_layer_counts}' not in ad.layers"
            )
        ad = ad.copy()
        ad.X = ad.layers[cfg.input_layer_counts]
        print(f"[Python] Exported '{cfg.input_layer_counts}' as X for R.")
    else:
        print("[Python] Exporting ad.X as-is for R.")

    # ------------------------------------------------------------------
    # 4. Write H5AD for R
    # ------------------------------------------------------------------
    temp_h5ad = os.path.join(outdir, f"adata_for_seurat_{cfg.run_tag}.h5ad")
    print(f"[Python] Writing temporary h5ad for R: {temp_h5ad}")
    perf.start("write_h5ad")
    ad.write_h5ad(temp_h5ad)
    perf.end("write_h5ad")

    # ------------------------------------------------------------------
    # 5. Optional HVG list
    # ------------------------------------------------------------------
    subset_genes_csv: Optional[str] = None
    if cfg.require_hvg and (cfg.hvg_key in ad.var):
        genes = ad.var_names[ad.var[cfg.hvg_key].values].tolist()
        if cfg.max_hvgs is not None:
            genes = genes[: cfg.max_hvgs]
        subset_genes_csv = os.path.join(outdir, f"seurat_genes_{cfg.run_tag}.csv")
        pd.DataFrame(genes).to_csv(subset_genes_csv, index=False, header=False)
        print(f"[Python] Wrote HVG list: {subset_genes_csv} (n={len(genes)})")
    elif cfg.require_hvg:
        print(
            f"[Python] Warning: cfg.require_hvg=True but ad.var lacks '{cfg.hvg_key}'. "
            "Proceeding without HVG restriction."
        )

    # ------------------------------------------------------------------
    # 6. Find R script
    # ------------------------------------------------------------------
    r_script = Path(
        "/data1/esraa/Thesis-Project/src/thesis_project/Integration/Integration_methods/R/run_seurat_integration.R"
    )
    if not r_script.exists():
        raise FileNotFoundError(f"run_seurat_integration.R not found at {r_script}.")

    out_prefix = os.path.join(outdir, cfg.run_tag)

    # Current R script signature:
    # run_seurat_integration.R <input_h5ad> <batch_key> <out_prefix> <mode>
    #   [k_anchor] [dims] [n_pcs] [subset_genes_csv] [seed]
    cmd = [
        "Rscript",
        str(r_script),
        temp_h5ad,
        cfg.batch_key,
        out_prefix,
        cfg.mode,
        str(cfg.k_anchor),
        str(cfg.dims),
        str(cfg.n_pcs),
        (subset_genes_csv if subset_genes_csv is not None else "NA"),
        str(cfg.seed),
    ]

    print("[Python] Calling R Seurat with:", " ".join(shlex.quote(x) for x in cmd))
    perf.start("r_seurat")
    _stream_subprocess(cmd, cwd=outdir)
    perf.end("r_seurat")

    # ------------------------------------------------------------------
    # 7. Load embedding from R
    # ------------------------------------------------------------------
    pca_csv = out_prefix + "_seurat_pca.csv"
    if not os.path.exists(pca_csv):
        raise RuntimeError(f"Expected Seurat PCA CSV not found: {pca_csv}")

    perf.start("load_embedding")
    key_emb_method = f"X_emb_{cfg.run_tag}"
    _load_csv_embedding(ad, pca_csv, key_emb_method)
    ad.obsm["X_emb"] = np.asarray(ad.obsm[key_emb_method]).copy()
    perf.end("load_embedding")

    # ------------------------------------------------------------------
    # 8. Neighbors / UMAP / Leiden
    # ------------------------------------------------------------------
    neigh_key = f"neighbors_{cfg.run_tag}"
    use_rep = "X_emb"

    perf.start("neighbors")
    build_neighbors(
        ad,
        use_rep=use_rep,
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
            emb_key=use_rep,
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

    metrics["seurat_r_script"] = str(r_script)
    metrics["seurat_mode"] = cfg.mode
    metrics["seurat_pca_csv"] = pca_csv

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
        h5ad_name=f"adata_seurat_{cfg.run_tag}.h5ad",
    )

    # ------------------------------------------------------------------
    # 11. Plots
    # ------------------------------------------------------------------
    perf.start("plots")
    title_prefix = f"Seurat ({cfg.mode.upper()}) — {cfg.run_tag}"

    try:
        _make_plots(
            ad,
            outdir=outdir,
            cfg=cfg,
            title_prefix=title_prefix,
            umap_key=f"X_umap_{cfg.run_tag}",
        )
    except Exception as e:
        print(f"[plot] plotting stage failed: {e}")

    try:
        plot_metric_summary(
            metrics=metrics,
            perf_df=perf_df,
            outdir=os.path.join(outdir, "plots_pub"),
            title=title_prefix,
        )
    except Exception as e:
        print(f"[plot] plot_metric_summary failed: {e}")

    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return ad, metrics, perf_df