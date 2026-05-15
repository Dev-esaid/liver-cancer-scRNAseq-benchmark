from __future__ import annotations

"""
unintegrated_reference.py
-------------------------
Build a publication-style unintegrated reference UMAP from a concatenated,
log-normalized single-cell atlas AnnData.


- Uses the same downstream PCA / neighbors / UMAP / Leiden settings as the
  integration benchmark, but applies no integration method.
- Defaults to the HVG-only concatenated object, because the requested reference
  should match the HVG-based benchmark input.
- Includes all datasets by default (exclude_datasets=()), which intentionally
  differs from the integration method runners that may exclude one dataset.
  This matches the explicit requirement to keep all 9 datasets in the
  unintegrated reference panel.
- Uses unscaled PCA by default, consistent with the provided benchmark helpers.

Outputs
-------
Saves two UMAP figures to the configured output directory:
- one coloured by the biological label key (default: major_celltype_l1)
- one coloured by the batch key (default: dataset)

No h5ad is saved by default.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import scanpy as sc
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_SRC = _THIS_FILE.parents[3]
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))

from thesis_project.Integration.Integration_benchmark.seeds import set_global_seed
from thesis_project.Integration.Integration_benchmark.subset import subset_and_cast_obs
from thesis_project.Integration.Integration_benchmark.preprocess import run_pca
from thesis_project.Integration.Integration_benchmark.graph import (
    build_neighbors,
    build_umap,
    build_leiden,
)
from thesis_project.Integration.Integration_benchmark.plotting import (
    subsample_for_plotting,
    plot_umap_pub,
)


DEFAULT_OUTPUT_DIR = "/data1/esraa/Thesis-Project/Results/unintegrated_reference"
ROOT_DIR = "/data1/esraa/Thesis-Project/Data/Processed_data/post_HVG_intersection/concatenated_hvg.h5ad"

@dataclass
class UnintegratedReferenceConfig:
    input_h5ad_path: str = ROOT_DIR
    outdir: str = DEFAULT_OUTPUT_DIR

    # keys
    batch_key: str = "dataset"
    label_key: str = "major_celltype_l1"
    exclude_datasets: tuple[str, ...] = ()

    # downstream preprocessing / graph settings
    n_pcs: int = 50
    scale_before_pca: bool = False
    scale_max_value: float = 10.0
    pca_solver: str = "arpack"
    neighbors_k: int = 50
    umap_min_dist: float = 0.35
    umap_spread: float = 1.0
    leiden_resolution: float = 1.0

    # plotting
    plot_subsample_n: Optional[int] = 200000
    plot_subsample_stratify_by: Optional[str] = "dataset"

    # run identity / reproducibility
    run_tag: str = "unintegrated_hvg"
    seed: int = 0


def _require_file(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("input_h5ad_path must be a non-empty string.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input AnnData file not found: {path}")
    return path


def _validate_config(cfg: UnintegratedReferenceConfig) -> None:
    _require_file(cfg.input_h5ad_path)

    if not isinstance(cfg.outdir, str) or not cfg.outdir.strip():
        raise ValueError("outdir must be a non-empty string.")
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
    if cfg.scale_before_pca and float(cfg.scale_max_value) <= 0:
        raise ValueError(
            f"scale_max_value must be positive when scaling is enabled, got {cfg.scale_max_value!r}."
        )


def _validate_pca_feasibility(ad, cfg: UnintegratedReferenceConfig) -> None:
    if ad.n_obs < 2:
        raise ValueError(f"Need at least 2 cells for PCA/UMAP, found n_obs={ad.n_obs}.")
    if ad.n_vars < 2:
        raise ValueError(f"Need at least 2 genes/features for PCA/UMAP, found n_vars={ad.n_vars}.")

    n_pcs = int(cfg.n_pcs)
    max_dim = min(int(ad.n_obs), int(ad.n_vars))
    if cfg.pca_solver == "arpack":
        if n_pcs >= max_dim:
            raise ValueError(
                f"With pca_solver='arpack', n_pcs must be strictly smaller than min(n_obs, n_vars). "
                f"Got n_pcs={n_pcs}, min(n_obs, n_vars)={max_dim}."
            )
    else:
        if n_pcs > max_dim:
            raise ValueError(
                f"n_pcs must be <= min(n_obs, n_vars). Got n_pcs={n_pcs}, min(n_obs, n_vars)={max_dim}."
            )


def _load_and_prepare_adata(cfg: UnintegratedReferenceConfig):
    adata = sc.read_h5ad(cfg.input_h5ad_path)
    adata, _ = subset_and_cast_obs(
        adata,
        batch_key=cfg.batch_key,
        label_key=cfg.label_key,
        exclude=cfg.exclude_datasets,
    )
    return adata


def _plot_outputs(ad, cfg: UnintegratedReferenceConfig) -> Dict[str, str]:
    os.makedirs(cfg.outdir, exist_ok=True)

    if cfg.plot_subsample_n is None:
        ad_plot = ad
    else:
        ad_plot = subsample_for_plotting(
            ad,
            n=int(cfg.plot_subsample_n),
            seed=cfg.seed,
            stratify_by=cfg.plot_subsample_stratify_by,
        )

    umap_key = f"X_umap_{cfg.run_tag}"
    title_prefix = f"Unintegrated (HVG) — {cfg.run_tag} (n={ad_plot.n_obs})"

    plot_umap_pub(
        ad_plot,
        umap_key=umap_key,
        color=cfg.label_key,
        title_prefix=title_prefix,
        outdir=cfg.outdir,
        alpha=0.75,
    )
    plot_umap_pub(
        ad_plot,
        umap_key=umap_key,
        color=cfg.batch_key,
        title_prefix=title_prefix,
        outdir=cfg.outdir,
        alpha=0.75,
    )

    return {
        "celltype_png": os.path.join(cfg.outdir, f"umap_{cfg.label_key}.png"),
        "celltype_pdf": os.path.join(cfg.outdir, f"umap_{cfg.label_key}.pdf"),
        "dataset_png": os.path.join(cfg.outdir, f"umap_{cfg.batch_key}.png"),
        "dataset_pdf": os.path.join(cfg.outdir, f"umap_{cfg.batch_key}.pdf"),
    }


def run(cfg: UnintegratedReferenceConfig):
    """
    Build an unintegrated benchmark reference from the concatenated HVG AnnData.

    Pipeline:
        load h5ad -> subset/cast obs -> PCA (unscaled by default) -> neighbors
        -> UMAP -> Leiden -> save two UMAP figures

    Returns
    -------
    adata : AnnData
        AnnData with computed X_pca, namespaced neighbors/UMAP/Leiden outputs.
    figure_paths : dict
        Paths to the saved dataset- and celltype-coloured UMAP figures.
    """
    _validate_config(cfg)
    os.makedirs(cfg.outdir, exist_ok=True)
    set_global_seed(cfg.seed, use_torch=False)

    adata = _load_and_prepare_adata(cfg)
    _validate_pca_feasibility(adata, cfg)

    run_pca(
        adata,
        n_pcs=cfg.n_pcs,
        scale=cfg.scale_before_pca,
        scale_max_value=cfg.scale_max_value,
        solver=cfg.pca_solver,
    )

    neigh_key = f"neighbors_{cfg.run_tag}"
    build_neighbors(
        adata,
        use_rep="X_pca",
        n_neighbors=cfg.neighbors_k,
        key_added=neigh_key,
        random_state=cfg.seed,
    )

    umap_key = f"X_umap_{cfg.run_tag}"
    build_umap(
        adata,
        neighbors_key=neigh_key,
        key_umap=umap_key,
        min_dist=cfg.umap_min_dist,
        spread=cfg.umap_spread,
        random_state=cfg.seed,
    )

    build_leiden(
        adata,
        neighbors_key=neigh_key,
        key_leiden=f"leiden_{cfg.run_tag}",
        resolution=cfg.leiden_resolution,
        random_state=cfg.seed,
    )

    figure_paths = _plot_outputs(adata, cfg)
    return adata, figure_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an unintegrated reference UMAP from a concatenated HVG AnnData "
            "using the same downstream PCA/neighbors/UMAP settings as the integration benchmark."
        )
    )
    parser.add_argument(
        "--input-h5ad-path",
        default=ROOT_DIR,
        help="Full path to the concatenated HVG AnnData (e.g. concatenated_hvg.h5ad).",
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where the UMAP figures will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--batch-key", default="dataset")
    parser.add_argument("--label-key", default="major_celltype_l1")
    parser.add_argument(
        "--exclude-datasets",
        nargs="*",
        default=(),
        help=(
            "Optional list of dataset labels to exclude. Default is empty so all 9 datasets are kept."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-tag", default="unintegrated_hvg")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--pca-solver", default="arpack")
    parser.add_argument("--neighbors-k", type=int, default=50)
    parser.add_argument("--umap-min-dist", type=float, default=0.35)
    parser.add_argument("--umap-spread", type=float, default=1.0)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument(
        "--plot-subsample-n",
        type=int,
        default=200000,
        help=(
            "Optional plotting subsample size. Use 0 to disable subsampling and plot all cells. "
            "Default matches the integration method plotting modules."
        ),
    )
    parser.add_argument(
        "--plot-subsample-stratify-by",
        default="dataset",
        help="Observation column used to stratify the plotting subsample. Default: dataset",
    )
    parser.add_argument(
        "--scale-before-pca",
        action="store_true",
        help="Enable scaling before PCA. Disabled by default to match the requested unscaled PCA reference.",
    )
    parser.add_argument(
        "--scale-max-value",
        type=float,
        default=10.0,
        help="Maximum absolute value used only when --scale-before-pca is enabled.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    plot_subsample_n = None if int(args.plot_subsample_n) == 0 else int(args.plot_subsample_n)

    cfg = UnintegratedReferenceConfig(
        input_h5ad_path=args.input_h5ad_path,
        outdir=args.outdir,
        batch_key=args.batch_key,
        label_key=args.label_key,
        exclude_datasets=tuple(args.exclude_datasets),
        n_pcs=int(args.n_pcs),
        scale_before_pca=bool(args.scale_before_pca),
        scale_max_value=float(args.scale_max_value),
        pca_solver=str(args.pca_solver),
        neighbors_k=int(args.neighbors_k),
        umap_min_dist=float(args.umap_min_dist),
        umap_spread=float(args.umap_spread),
        leiden_resolution=float(args.leiden_resolution),
        plot_subsample_n=plot_subsample_n,
        plot_subsample_stratify_by=args.plot_subsample_stratify_by,
        run_tag=str(args.run_tag),
        seed=int(args.seed),
    )

    _, figure_paths = run(cfg)
    print("Saved unintegrated reference figures:")
    for label, path in figure_paths.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
