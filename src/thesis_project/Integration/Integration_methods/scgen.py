"""
scGen integration benchmark module (label-aware batch correction in latent space).

Why this implementation:
- scgen.SCGEN.batch_removal() can crash in some scgen+scvi-tools combos with:
    AttributeError: 'NoneType' object has no attribute 'sqrt'
  (due to an internal latent sampling path).
- We avoid that by:
    1) training scGen normally
    2) extracting a deterministic latent representation (mean of q(z|x))
    3) applying label-aware batch mean-shift correction in latent space:
         z_corrected = z - mean(z | label,batch) + mean(z | label,target)

Pipeline:
  subset → sanitize → (optional HVG subset) → snapshot adata_pre
  → scGen setup/train → latent extraction (mean) → label-aware batch correction
  → neighbors → UMAP/Leiden → integration_metrics → save_run_artifacts → plots

Assumptions:
- .X (or input_layer) is log-normalized expression for scGen training
- batch_key and label_key exist in adata.obs
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import scanpy as sc  # noqa: F401

try:
    import torch
except Exception:
    torch = None

import sys
import types
import importlib

_SCGEN_IMPORT_ERROR: Exception | None = None


def _ensure_scvi_compat_for_scgen():
    if "scvi._compat" in sys.modules:
        return
    mod = types.ModuleType("scvi._compat")
    try:
        from typing import Literal
    except Exception:
        from typing_extensions import Literal
    mod.Literal = Literal
    sys.modules["scvi._compat"] = mod


try:
    import scgen
except ModuleNotFoundError as e:
    if "scvi._compat" in str(e):
        try:
            _ensure_scvi_compat_for_scgen()
            import scgen  # type: ignore
        except Exception as e2:
            _SCGEN_IMPORT_ERROR = e2
            scgen = None
    else:
        _SCGEN_IMPORT_ERROR = e
        scgen = None
except Exception as e:
    _SCGEN_IMPORT_ERROR = e
    scgen = None

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.perf import PerfLogger
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs
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
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScgenConfig:
    # metadata
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple = ()

    # input expression (should be log-normalized)
    input_layer: Optional[str] = None

    # HVGs
    require_hvg: bool = True
    hvg_key: str = "highly_variable"
    max_hvgs: Optional[int] = 4000

    # model
    n_latent: int = 30
    n_hidden: int = 256
    n_layers: int = 2
    dropout_rate: float = 0.1

    # training
    use_gpu: bool = True
    batch_size: int = 512
    max_epochs: int = 200
    early_stopping: bool = True
    early_stopping_patience: int = 25
    num_workers: int = 4
    pin_memory: bool = False
    enable_progress_bar: bool = True

    # latent correction
    reference_batch: Optional[str] = None

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
    run_tag: str = "scgen"
    seed: int = 0

    # persistence
    save_h5ad: bool = True


# ---------------------------------------------------------------------------
# Internal helpers (unchanged)
# ---------------------------------------------------------------------------

def _require_scgen():
    global scgen, _SCGEN_IMPORT_ERROR
    if scgen is not None:
        return
    try:
        scgen = importlib.import_module("scgen")
        _SCGEN_IMPORT_ERROR = None
        return
    except Exception as e:
        _SCGEN_IMPORT_ERROR = e
    err = _SCGEN_IMPORT_ERROR
    err_txt = f"{type(err).__name__}: {err}" if err is not None else "(unknown error)"
    raise ImportError(
        "scgen is not available in this Python environment.\n"
        f"Python: {sys.executable}\n"
        f"Import error: {err_txt}\n"
        "Install/verify scgen with:\n"
        "    pip install scgen\n"
        "    python -c \"import scgen; print(scgen.__version__)\"\n"
    )


def _get_scgen_class():
    _require_scgen()
    if hasattr(scgen, "SCGEN"):
        return getattr(scgen, "SCGEN")
    try:
        from scgen._scgen import SCGEN  # type: ignore
        return SCGEN
    except Exception:
        pass
    mod_path = getattr(scgen, "__file__", None)
    mod_ver = getattr(scgen, "__version__", None)
    raise ImportError(
        "Could not locate SCGEN class in your installed `scgen` module.\n"
        f"scgen.__file__={mod_path}\n"
        f"scgen.__version__={mod_ver}\n"
    )


def _device(cfg: ScgenConfig) -> str:
    if torch is None:
        return "cpu"
    return "gpu" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu"


def _get_X(ad, cfg: ScgenConfig):
    if cfg.input_layer is None:
        return ad.X
    if cfg.input_layer not in ad.layers:
        raise ValueError(f"input_layer='{cfg.input_layer}' not found in adata.layers")
    return ad.layers[cfg.input_layer]


def _subset_hvgs_or_raise(ad, cfg: ScgenConfig):
    if not cfg.require_hvg and cfg.hvg_key not in ad.var:
        return ad
    if cfg.hvg_key not in ad.var:
        raise ValueError(
            f"ScgenConfig.require_hvg=True but ad.var['{cfg.hvg_key}'] is missing. "
            "Either annotate HVGs or set require_hvg=False."
        )
    mask = ad.var[cfg.hvg_key].values
    genes = ad.var_names[mask].tolist()
    if len(genes) == 0:
        raise ValueError(f"ad.var['{cfg.hvg_key}'] exists but contains no True entries.")
    if cfg.max_hvgs is not None and len(genes) > int(cfg.max_hvgs):
        genes = genes[: int(cfg.max_hvgs)]
    return ad[:, genes].copy()


def _sanitize_for_scgen(adata_in, cfg: ScgenConfig):
    ad, _ = subset_and_cast_obs(
        adata_in, cfg.batch_key, cfg.label_key, cfg.exclude_datasets
    )
    if cfg.batch_key not in ad.obs:
        raise ValueError(f"Missing batch_key='{cfg.batch_key}' in ad.obs")
    if cfg.label_key not in ad.obs:
        raise ValueError(f"Missing label_key='{cfg.label_key}' in ad.obs")

    ad.obs[cfg.batch_key] = ad.obs[cfg.batch_key].astype(str).fillna("UnknownBatch")
    ad.obs[cfg.label_key] = (
        ad.obs[cfg.label_key].astype(str).fillna("Unknown").astype("category")
    )

    ad = _subset_hvgs_or_raise(ad, cfg)

    X = _get_X(ad, cfg)
    try:
        if hasattr(X, "dtype") and str(X.dtype) != "float32":
            if cfg.input_layer is None:
                ad.X = X.astype(np.float32)
            else:
                ad.layers[cfg.input_layer] = X.astype(np.float32)
    except Exception:
        pass

    return ad


def _train_scgen(ad, cfg: ScgenConfig, perf: PerfLogger, outdir: str):
    SCGEN = _get_scgen_class()
    accelerator = _device(cfg)

    perf.start("scgen_setup_anndata")
    if pd.api.types.is_categorical_dtype(ad.obs[cfg.label_key]):
        cat = ad.obs[cfg.label_key]
        ad.obs[cfg.label_key] = pd.Categorical(
            cat.values,
            categories=cat.cat.categories.to_numpy(),
            ordered=cat.cat.ordered,
        )
    setup_kwargs = dict(batch_key=cfg.batch_key, labels_key=cfg.label_key)
    if cfg.input_layer is not None:
        setup_kwargs["layer"] = cfg.input_layer
    SCGEN.setup_anndata(ad, **setup_kwargs)
    perf.end("scgen_setup_anndata")

    perf.start("scgen_init")
    model = SCGEN(
        ad,
        n_hidden=cfg.n_hidden,
        n_latent=cfg.n_latent,
        n_layers=cfg.n_layers,
        dropout_rate=cfg.dropout_rate,
    )
    perf.end("scgen_init")

    perf.start("scgen_train")
    model.train(
        max_epochs=cfg.max_epochs,
        batch_size=cfg.batch_size,
        accelerator=accelerator,
        devices=1,
        early_stopping=cfg.early_stopping,
        early_stopping_patience=cfg.early_stopping_patience,
        enable_progress_bar=cfg.enable_progress_bar,
        datasplitter_kwargs={
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
        },
    )
    perf.end("scgen_train")

    perf.start("scgen_save_model")
    model.save(os.path.join(outdir, "scgen_model"), overwrite=True)
    perf.end("scgen_save_model")

    return model


def _get_latent_mean(model, ad) -> np.ndarray:
    """Extract deterministic latent (mean of q(z|x)) via direct encoder access."""
    import torch

    print("[scGen] Using direct encoder access to avoid sqrt(None) bug.")
    if not hasattr(model, "module"):
        raise RuntimeError("Model does not have 'module' attribute.")
    if not hasattr(model.module, "z_encoder"):
        raise RuntimeError("Model.module does not have 'z_encoder' attribute.")

    model.module.eval()

    try:
        adata_manager = model.get_anndata_manager(ad, required=True)
    except Exception as e:
        print(f"[scGen] Could not get adata_manager: {e}")
        adata_manager = model.adata_manager

    try:
        from scvi.dataloaders import AnnDataLoader
        dataloader = AnnDataLoader(
            adata_manager, shuffle=False, batch_size=512, drop_last=False
        )
    except Exception as e:
        print(f"[scGen] AnnDataLoader failed: {e}, using manual DataLoader.")
        from torch.utils.data import DataLoader, TensorDataset
        X_data = ad.X
        if hasattr(X_data, "toarray"):
            X_data = X_data.toarray()
        X_tensor = torch.tensor(np.asarray(X_data, dtype=np.float32))
        dataloader = DataLoader(
            TensorDataset(X_tensor), batch_size=512, shuffle=False, drop_last=False
        )

    latents = []
    device = next(model.module.parameters()).device

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            if isinstance(batch_data, dict):
                x = batch_data.get("X", batch_data.get("x", None))
                if x is None:
                    x = next(iter(batch_data.values()))
            elif isinstance(batch_data, (list, tuple)):
                x = batch_data[0]
            else:
                x = batch_data

            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device)

            encoder_output = model.module.z_encoder(x)
            if isinstance(encoder_output, (tuple, list)):
                qz_m = encoder_output[0]
            else:
                qz_m = encoder_output

            latents.append(qz_m.cpu().numpy())

    Z = np.concatenate(latents, axis=0)
    if Z.shape[0] != ad.n_obs:
        raise RuntimeError(
            f"Latent shape mismatch: {Z.shape[0]} rows vs {ad.n_obs} cells."
        )
    print(f"[scGen] Latent shape: {Z.shape}")
    return Z


def _latent_label_batch_mean_shift(
    Z: np.ndarray,
    batch: np.ndarray,
    labels: np.ndarray,
    reference_batch: Optional[str] = None,
) -> np.ndarray:
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (cells x latent_dim).")

    batch = batch.astype(str)
    labels = labels.astype(str)
    Zc = Z.astype(np.float32, copy=True)
    unique_labels = pd.unique(labels)

    label_global_mean: Dict[str, np.ndarray] = {}
    label_ref_mean: Dict[str, np.ndarray] = {}

    for lab in unique_labels:
        mask_lab = (labels == lab)
        if not np.any(mask_lab):
            continue
        label_global_mean[lab] = Z[mask_lab].mean(axis=0)
        if reference_batch is not None:
            mask_ref = mask_lab & (batch == str(reference_batch))
            if np.any(mask_ref):
                label_ref_mean[lab] = Z[mask_ref].mean(axis=0)

    for lab in unique_labels:
        mask_lab = (labels == lab)
        if not np.any(mask_lab):
            continue
        target = label_global_mean[lab]
        if reference_batch is not None and lab in label_ref_mean:
            target = label_ref_mean[lab]
        for b in pd.unique(batch[mask_lab]):
            mask_lb = mask_lab & (batch == b)
            if not np.any(mask_lb):
                continue
            mu_lb = Z[mask_lb].mean(axis=0)
            Zc[mask_lb] = (Zc[mask_lb] - mu_lb.astype(np.float32)) + target.astype(np.float32)

    return Zc


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(
    adata_in, outdir: str, cfg: ScgenConfig
) -> Tuple[Any, Dict[str, Any], Any]:
    """
    scGen run:
      subset → sanitize → snapshot adata_pre → train scGen
      → latent extraction → label-aware batch correction
      → neighbors → UMAP/Leiden → metrics → save → plots

    Variable naming:
      ad            : sanitized + HVG-subsetted AnnData (modified in-place)
      adata_pre     : snapshot after _sanitize_for_scgen (same cell set, same
                      gene set as ad), BEFORE training.
                      Required for scib comparison metrics (pcr_comparison,
                      hvg_conservation, cell_cycle_conservation).
                      Note: _sanitize_for_scgen applies HVG subsetting, so
                      adata_pre has fewer genes than adata_in.
      key_latent    : "X_scgen"  — label-aware mean-shift corrected latent
      neigh_key     : "neighbors_scgen"
      conn_key      : ad.uns[neigh_key]["connectivities_key"]
      cluster_key   : "leiden_scgen"

      output_type="embed": scGen outputs a corrected latent embedding.
        kBET/LISI recompute their own kNN from key_latent internally.
    """
    os.makedirs(outdir, exist_ok=True)
    set_global_seed(cfg.seed, use_torch=True)

    perf = PerfLogger(track_gpu=True)

    # ------------------------------------------------------------------
    # 1. Subset + sanitize + HVG restriction
    # ------------------------------------------------------------------
    perf.start("subset_sanitize")
    ad = _sanitize_for_scgen(adata_in, cfg)
    perf.end("subset_sanitize")

    # ------------------------------------------------------------------
    # 2. Snapshot adata_pre  ← after _sanitize_for_scgen (which applies
    #    subset_and_cast_obs AND HVG subsetting), BEFORE training.
    #    Same cells AND same gene set as ad; required for scib comparison
    #    metrics (pcr_comparison, hvg_conservation, cell_cycle_conservation).
    # ------------------------------------------------------------------
    adata_pre = ad.copy()

    # ------------------------------------------------------------------
    # 3. Train scGen
    # ------------------------------------------------------------------
    model = _train_scgen(ad, cfg, perf=perf, outdir=outdir)

    # ------------------------------------------------------------------
    # 4. Extract latent + label-aware batch correction
    #    key_latent = "X_scgen"
    # ------------------------------------------------------------------
    perf.start("latent_mean")
    Z = _get_latent_mean(model, ad).astype(np.float32)
    perf.end("latent_mean")

    perf.start("latent_label_batch_correction")
    Z_corr = _latent_label_batch_mean_shift(
        Z=Z,
        batch=ad.obs[cfg.batch_key].values,
        labels=ad.obs[cfg.label_key].astype(str).values,
        reference_batch=cfg.reference_batch,
    )
    perf.end("latent_label_batch_correction")

    key_latent = f"X_{cfg.run_tag}"    # "X_scgen"
    ad.obsm[key_latent] = Z_corr

    # ------------------------------------------------------------------
    # 5. Neighbors → UMAP → Leiden
    # ------------------------------------------------------------------
    perf.start("neighbors")
    neigh_key = f"neighbors_{cfg.run_tag}"   # "neighbors_scgen"
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
    #    conn_key = "neighbors_scgen_connectivities"  (from uns)
    #    dist_key = "neighbors_scgen_distances"        (from uns)
    #    neighbors_uns_key = "neighbors_scgen"         (= neigh_key, explicit)
    #
    #    output_type="embed": scGen outputs a corrected latent embedding.
    #      kBET/LISI recompute their own kNN from key_latent internally.
    # ------------------------------------------------------------------
    perf.start("metrics")
    conn_key = ad.uns[neigh_key]["connectivities_key"]  # "neighbors_scgen_connectivities"
    dist_key = ad.uns[neigh_key]["distances_key"]       # "neighbors_scgen_distances"

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

    metrics["scgen_backend"] = "scgen (theislab/scgen)"
    metrics["scgen_device"] = _device(cfg)
    metrics["scgen_input_layer"] = (
        cfg.input_layer if cfg.input_layer is not None else "X(lognorm)"
    )
    metrics["scgen_hvg_key"] = cfg.hvg_key
    metrics["scgen_max_hvgs"] = cfg.max_hvgs
    metrics["scgen_reference_batch"] = (
        cfg.reference_batch if cfg.reference_batch is not None else "global-per-label"
    )
    metrics["scgen_note"] = (
        "Used deterministic latent + label-aware mean-shift correction "
        "(no model.batch_removal)."
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
        h5ad_name="adata_scgen.h5ad",
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

    title_prefix = f"scGen — {cfg.run_tag} (n={ad_plot.n_obs:,})"
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
        title=f"scGen metrics — {cfg.run_tag}",
    )
    perf.end("plots")

    perf_df = perf.save_csv(os.path.join(outdir, "perf_log.csv"))
    return ad, metrics, perf_df