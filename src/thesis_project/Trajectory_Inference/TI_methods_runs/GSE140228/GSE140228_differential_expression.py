"""
GSE140228 — Differential Expression Analysis
========================================================
Runs Wilcoxon rank-sum differential expression on all cell subtypes
defined in adata.obs['celltype_sub'], with outputs structured for
direct use in TI benchmarking marker programs.

Assumes input data is already log-normalised.
No raw count handling is performed.

Input path (hardcoded):
    /data1/esraa/Thesis-Project/Data/Processed_data/GSE140228/adata_140228.h5ad

Output path (hardcoded):
    /data1/esraa/Thesis-Project/src/thesis_project/Trajectory_Inference/
    priors_registry/GSE140228/

Usage
-----
    # Run with defaults (uses hardcoded paths):
    python GSE140228_differential_expression.py

    # Override any parameter:
    python GSE140228_differential_expression.py \
        [--layer   log1p]       # specific layer to use as expression matrix \
        [--n_top   15]          # top N DEGs per cluster to export \
        [--min_pct 0.1]         # min fraction of cells expressing gene \
        [--logfc   0.25]        # min log2FC threshold \
        [--tasks_only]          # export task JSONs only (skip full tables + plots)

Outputs
-------
priors_registry/GSE140228/
  ├── full_deg_results.csv          # all DEGs, all clusters, all stats
  ├── top{N}_per_cluster.csv        # top N DEGs per cluster (filtered)
  ├── deg_summary.txt               # human-readable per-cluster summary
  ├── volcano_plots/                # one volcano plot per cluster
  │   ├── Mono-C1-CD14_volcano.png
  │   └── ...
  ├── dotplots/                     # one dotplot per task
  ├── task_jsons/                   # task JSONs with DEG-derived markers
  │   ├── task3_myeloid_monocyte_to_TAM.json
  │   ├── task4_CD8_exhaustion.json
  │   ├── task5_CD4_differentiation.json
  │   └── task6_NK_maturation.json
  └── logs/
      └── de_run.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Optional imports with graceful degradation ────────────────────────────────
try:
    import anndata as ad
except ImportError:
    print("ERROR: anndata not installed. Run: pip install anndata")
    sys.exit(1)

try:
    import scanpy as sc
except ImportError:
    print("ERROR: scanpy not installed. Run: pip install scanpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available — plots will be skipped.")

try:
    from scipy import sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Hardcoded paths
# =============================================================================

DEFAULT_ADATA_PATH = (
    "/data1/esraa/Thesis-Project/Data/Processed_data/post_norm_log1p/adata_140228.h5ad"
)

DEFAULT_OUT_DIR = (
    "/data1/esraa/Thesis-Project/src/thesis_project/"
    "Trajectory_Inference/priors_registry/GSE140228"
)

# =============================================================================
# Task definitions — exact cluster names as they appear in celltype_sub
# =============================================================================

TASKS: Dict[str, Dict] = {
    "task3_myeloid_monocyte_to_TAM": {
        "description": (
            "Monocyte-to-TAM differentiation trajectory. "
            "Models circulating monocyte recruitment through tissue entry "
            "to five terminal TAM fates in the liver TME."
        ),
        "root_population": "Mono-C1-CD14",
        "include_populations": [
            "Mono-C1-CD14",
            "Mono-C2-FCGR3A",
            "Mφ-C5-VCAN",
            "Mφ-C1-THBS1",
            "Mφ-C2-C1QA",
            "Mφ-C3-APOE",
            "Mφ-C4-GPX3",
            "Mφ-C6-MARCO",
        ],
        "pan_lineage_genes": ["CST3", "LYZ", "CD68", "CD14"],
    },
    "task4_CD8_exhaustion": {
        "description": (
            "CD8 T cell exhaustion continuum. "
            "Models naive/progenitor through effector memory to "
            "terminal exhaustion (PDCD1+) and TEMRA (CX3CR1+) fates."
        ),
        "root_population": "CD4/CD8-C1-CCR7",
        "include_populations": [
            "CD4/CD8-C1-CCR7",
            "CD8-C5-SELL",
            "CD8-C3-IL7R",
            "CD8-C6-GZMK",
            "CD8-C7-KLRD1",
            "CD8-C8-PDCD1",
            "CD8-C4-CX3CR1",
        ],
        "pan_lineage_genes": ["CD3D", "CD3E", "CD3G", "CD8A", "CD8B"],
    },
    "task5_CD4_differentiation": {
        "description": (
            "CD4 T cell differentiation trajectory. "
            "Models TCF7+ naive/stem through memory and effector states "
            "to Tfh/exhausted (CXCL13+) and Treg (FOXP3+) terminal fates."
        ),
        "root_population": "CD4-C5-TCF7",
        "include_populations": [
            "CD4-C5-TCF7",
            "CD4/CD8-C1-CCR7",
            "CD4-C4-IL7R",
            "CD4-C3-ANXA1",
            "CD4-C6-CXCL13",
            "CD4-C7-FOXP3",
        ],
        "pan_lineage_genes": ["CD3D", "CD3E", "CD3G", "CD4"],
    },
    "task6_NK_maturation": {
        "description": (
            "NK cell maturation trajectory (conservative). "
            "Models SELL+ naive through tissue-resident (CD69+) "
            "and mature cytotoxic (FCGR3A+, CD160+) to activated effector (IFNG+)."
        ),
        "root_population": "NK-C2-SELL",
        "include_populations": [
            "NK-C2-SELL",
            "NK-C5-CD69",
            "NK-C1-FCGR3A",
            "NK-C7-CD160",
            "NK-C3-IFNG",
        ],
        "pan_lineage_genes": ["NKG7", "GNLY", "PRF1"],
    },
}

# Global genes to exclude from all marker lists
GLOBAL_EXCLUDE = {
    # Ribosomal
    *[f"RPS{i}" for i in range(1, 30)],
    *[f"RPL{i}" for i in range(1, 45)],
    # Mitochondrial
    *[f"MT-{g}" for g in [
        "ND1", "ND2", "ND3", "ND4", "ND4L", "ND5", "ND6",
        "CO1", "CO2", "CO3", "ATP6", "ATP8", "CYB",
    ]],
    # Housekeeping / technical
    "ACTB", "ACTG1", "B2M", "GAPDH", "MALAT1", "NEAT1",
    "TMSB4X", "TMSB10", "FTL", "FTH1", "VIM",
    # Cell cycle (confound trajectory)
    "MKI67", "TOP2A", "CDK1", "CCNB1", "CCNA2", "PCNA",
    "STMN1", "HMGB1", "HMGB2", "HIST1H4C", "UBE2C",
}


# =============================================================================
# Load and validate
# =============================================================================

def load_and_validate(
    adata_path: str,
    layer: Optional[str],
) -> Tuple["ad.AnnData", Optional[str]]:
    """
    Load AnnData from path, confirm celltype_sub exists,
    optionally select expression layer.

    The data is assumed to be log-normalised. No transformation is applied.
    """
    path = Path(adata_path)
    if not path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")

    logger.info(f"Loading {path} ...")
    adata = ad.read_h5ad(str(path))
    logger.info(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Validate celltype_sub column
    if "celltype_sub" not in adata.obs.columns:
        available = list(adata.obs.columns)
        raise KeyError(
            f"'celltype_sub' not found in adata.obs.\n"
            f"Available columns: {available}"
        )

    n_types = adata.obs["celltype_sub"].nunique()
    logger.info(f"celltype_sub: {n_types} unique subtypes")
    logger.info(f"Subtypes found: {sorted(adata.obs['celltype_sub'].unique().tolist())}")

    # Confirm data looks log-normalised
    if HAS_SCIPY and sparse.issparse(adata.X):
        sample = np.asarray(adata.X[:100].todense())
    else:
        sample = np.asarray(adata.X[:100])
    max_val = float(np.max(sample))

    if max_val > 100:
        logger.warning(
            f"adata.X max value = {max_val:.1f} — this looks like raw counts, "
            f"not log-normalised data. DE results may be unreliable. "
            f"Consider passing --layer with a log-normalised layer name."
        )
    else:
        logger.info(
            f"adata.X max value = {max_val:.3f} — confirmed log-normalised. "
            f"Proceeding without transformation."
        )

    # Determine expression layer
    layer_used: Optional[str] = None
    if layer and layer in adata.layers:
        layer_used = layer
        logger.info(f"Using expression layer: '{layer}'")

        # Confirm the requested layer is also log-normalised
        if HAS_SCIPY and sparse.issparse(adata.layers[layer]):
            lsample = np.asarray(adata.layers[layer][:100].todense())
        else:
            lsample = np.asarray(adata.layers[layer][:100])
        lmax = float(np.max(lsample))
        if lmax > 100:
            logger.warning(
                f"Layer '{layer}' max value = {lmax:.1f} — looks like raw counts. "
                f"Proceeding, but results may be unreliable."
            )
        else:
            logger.info(f"Layer '{layer}' max value = {lmax:.3f} — confirmed log-normalised.")

    elif layer:
        logger.warning(
            f"Requested layer '{layer}' not found. "
            f"Available layers: {list(adata.layers.keys())}. "
            f"Falling back to adata.X."
        )
    else:
        logger.info(f"No layer specified — using adata.X directly.")

    return adata, layer_used


def set_expression_layer(
    adata: "ad.AnnData",
    layer: Optional[str],
) -> "ad.AnnData":
    """
    If a specific layer is requested, copy it to adata.X.
    Otherwise leave adata.X untouched.
    Returns a copy to avoid modifying the original object.
    """
    adata = adata.copy()
    if layer:
        logger.info(f"Setting adata.X = adata.layers['{layer}']")
        adata.X = adata.layers[layer].copy()
    return adata


# =============================================================================
# Differential expression
# =============================================================================

def run_de(
    adata: "ad.AnnData",
    groupby: str = "celltype_sub",
    method: str = "wilcoxon",
    n_genes: int = 200,
    pts: bool = True,
) -> "ad.AnnData":
    """
    Run scanpy rank_genes_groups (Wilcoxon rank-sum) on all subtypes.

    Parameters
    ----------
    adata    : AnnData with log-normalised expression in X.
    groupby  : obs column to group cells by.
    method   : DE method — 'wilcoxon' recommended for scRNA-seq.
    n_genes  : Number of top-ranked genes to store per group.
    pts      : Compute fraction of expressing cells per group and rest.
    """
    logger.info(
        f"Running rank_genes_groups: groupby='{groupby}', "
        f"method='{method}', n_genes={n_genes}, pts={pts}"
    )

    adata.obs[groupby] = adata.obs[groupby].astype("category")

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        pts=pts,
        use_raw=False,
        key_added="rank_genes_groups",
    )
    logger.info("rank_genes_groups complete.")
    return adata


# =============================================================================
# Result extraction
# =============================================================================

def extract_de_results(
    adata: "ad.AnnData",
    key: str = "rank_genes_groups",
    min_logfc: float = 0.25,
    min_pct: float = 0.10,
    max_pct_rest: float = 0.98,
) -> pd.DataFrame:
    """
    Unpack scanpy DE results into a tidy long-format DataFrame.

    Columns
    -------
    cluster, rank, gene, score, logfoldchange, pval, pval_adj,
    pct_group, pct_rest, specificity
    """
    logger.info("Extracting DE results into tidy DataFrame ...")
    result = adata.uns[key]
    groups = result["names"].dtype.names

    rows = []
    for group in groups:
        genes    = result["names"][group]
        scores   = result["scores"][group]
        logfc    = result["logfoldchanges"][group]
        pvals    = result["pvals"][group]
        padj     = result["pvals_adj"][group]
        pts_grp  = result.get("pts",      {}).get(group, [np.nan] * len(genes))
        pts_rest = result.get("pts_rest", {}).get(group, [np.nan] * len(genes))

        # pts may be a dict keyed by gene name in newer scanpy versions
        if isinstance(pts_grp, dict):
            pts_grp  = [pts_grp.get(g,  np.nan) for g in genes]
            pts_rest = [pts_rest.get(g, np.nan) for g in genes]

        for rank, (g, sc_, lfc, pv, pa, pg, pr) in enumerate(
            zip(genes, scores, logfc, pvals, padj, pts_grp, pts_rest), start=1
        ):
            rows.append({
                "cluster":       str(group),
                "rank":          int(rank),
                "gene":          str(g),
                "score":         float(sc_),
                "logfoldchange": float(lfc),
                "pval":          float(pv),
                "pval_adj":      float(pa),
                "pct_group":     float(pg) if pg is not None else np.nan,
                "pct_rest":      float(pr) if pr is not None else np.nan,
            })

    df = pd.DataFrame(rows)
    n_total = len(df)

    # ── Filters ───────────────────────────────────────────────────────────────
    df = df[
        (df["logfoldchange"] >= min_logfc) &
        (df["pct_group"]     >= min_pct)   &
        (df["pct_rest"]      <= max_pct_rest)
    ].copy()
    logger.info(
        f"After filters (logFC≥{min_logfc}, pct_group≥{min_pct}, "
        f"pct_rest≤{max_pct_rest}): {len(df):,} / {n_total:,} rows retained"
    )

    # ── Global gene exclusions ────────────────────────────────────────────────
    before_excl = len(df)
    df = df[~df["gene"].isin(GLOBAL_EXCLUDE)].copy()
    logger.info(
        f"After global exclusions (ribosomal/MT/housekeeping/cell-cycle): "
        f"{len(df):,} / {before_excl:,} rows retained"
    )

    # ── Specificity score ─────────────────────────────────────────────────────
    df["specificity"] = df["pct_group"] - df["pct_rest"]

    df = df.sort_values(["cluster", "rank"]).reset_index(drop=True)
    return df


def get_top_n(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return top N genes per cluster ranked by original scanpy rank."""
    return (
        df.groupby("cluster", group_keys=False)
        .apply(lambda x: x.nsmallest(n, "rank"))
        .reset_index(drop=True)
    )


# =============================================================================
# Marker selection for task JSONs
# =============================================================================

def select_markers_for_cluster(
    df: pd.DataFrame,
    cluster: str,
    pan_genes: List[str],
    n_markers: int = 15,
    min_specificity: float = 0.01,
) -> List[str]:
    """
    Select best n_markers genes for a given cluster for TI metrics.

    Selection criteria (in priority order):
    1. pval_adj < 0.05
    2. Not in pan-lineage gene list for this task
    3. Not in global exclusion set
    4. specificity (pct_group - pct_rest) >= min_specificity
    5. Ranked by specificity descending, then logfoldchange descending
    """
    sub = df[df["cluster"] == cluster].copy()
    if sub.empty:
        logger.warning(f"  No DEGs found for cluster '{cluster}'")
        return []

    sub = sub[sub["pval_adj"] < 0.05].copy()
    sub = sub[~sub["gene"].isin(pan_genes)].copy()
    sub = sub[~sub["gene"].isin(GLOBAL_EXCLUDE)].copy()
    sub = sub[sub["specificity"] >= min_specificity].copy()

    sub = sub.sort_values(
        ["specificity", "logfoldchange"],
        ascending=[False, False],
    )

    return sub["gene"].head(n_markers).tolist()


# =============================================================================
# Overlap check between adjacent populations
# =============================================================================

def check_marker_overlap(
    task_name: str,
    markers_dict: Dict[str, List[str]],
    max_allowed_shared: int = 3,
) -> None:
    """
    Warn if two populations in the same task share more than
    max_allowed_shared markers — these shared genes will not
    discriminate states for pseudotime validation.
    """
    from itertools import combinations
    pops = list(markers_dict.keys())
    any_issue = False
    for p1, p2 in combinations(pops, 2):
        shared = set(markers_dict[p1]) & set(markers_dict[p2])
        if len(shared) > max_allowed_shared:
            logger.warning(
                f"  OVERLAP WARNING [{task_name}]: "
                f"'{p1}' and '{p2}' share {len(shared)} markers: "
                f"{sorted(shared)}"
            )
            any_issue = True
    if not any_issue:
        logger.info(f"  Overlap check passed for {task_name}")


# =============================================================================
# Plotting
# =============================================================================

def plot_volcano(
    df: pd.DataFrame,
    cluster: str,
    out_dir: Path,
    top_label: int = 15,
) -> None:
    """Volcano plot: logfoldchange vs -log10(pval_adj) for one cluster."""
    if not HAS_MATPLOTLIB:
        return

    sub = df[df["cluster"] == cluster].copy()
    if sub.empty:
        return

    sub["-log10_padj"] = -np.log10(sub["pval_adj"].clip(lower=1e-300))

    sig    = (sub["pval_adj"] < 0.05) & (sub["logfoldchange"] >= 0.5)
    nonsig = ~sig

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        sub.loc[nonsig, "logfoldchange"], sub.loc[nonsig, "-log10_padj"],
        s=6, alpha=0.4, color="#aaaaaa", linewidths=0, label="ns",
    )
    ax.scatter(
        sub.loc[sig, "logfoldchange"], sub.loc[sig, "-log10_padj"],
        s=8, alpha=0.7, color="#2166ac", linewidths=0,
        label="padj<0.05, logFC≥0.5",
    )

    # Label top-specificity significant genes
    top_genes = (
        sub[sig].nlargest(top_label, "specificity")
        if "specificity" in sub.columns
        else sub[sig].nsmallest(top_label, "pval_adj")
    )
    for _, row in top_genes.iterrows():
        ax.annotate(
            row["gene"],
            xy=(row["logfoldchange"], row["-log10_padj"]),
            fontsize=6.5, xytext=(3, 1), textcoords="offset points",
            color="#1a1a1a",
        )

    ax.axvline(0.5,  color="#d62728", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(-np.log10(0.05), color="#ff7f0e", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.set_xlabel("log2 Fold Change", fontsize=10)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=10)
    ax.set_title(f"DEGs — {cluster}", fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    safe = cluster.replace("/", "_").replace(" ", "_").replace("+", "pos")
    out_path = out_dir / f"{safe}_volcano.png"
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_dotplot(
    adata: "ad.AnnData",
    markers_per_cluster: Dict[str, List[str]],
    out_dir: Path,
    task_name: str,
    groupby: str = "celltype_sub",
    max_genes_per_cluster: int = 5,
) -> None:
    """Scanpy dotplot showing top markers per cluster for one task."""
    if not HAS_MATPLOTLIB:
        return

    seen: set = set()
    gene_list: List[str] = []
    for genes in markers_per_cluster.values():
        for g in genes[:max_genes_per_cluster]:
            if g not in seen and g in adata.var_names:
                gene_list.append(g)
                seen.add(g)

    if not gene_list:
        logger.warning(f"No valid genes for dotplot: {task_name}")
        return

    clusters = [
        c for c in markers_per_cluster
        if c in adata.obs[groupby].cat.categories
    ]
    if not clusters:
        return

    adata_sub = adata[adata.obs[groupby].isin(clusters)].copy()
    adata_sub.obs[groupby] = (
        adata_sub.obs[groupby].cat.remove_unused_categories()
    )

    try:
        fig = sc.pl.dotplot(
            adata_sub,
            var_names=gene_list,
            groupby=groupby,
            standard_scale="var",
            color_map="Blues",
            show=False,
            return_fig=True,
            title=task_name,
            figsize=(
                max(8, len(gene_list) * 0.45),
                max(4, len(clusters) * 0.55),
            ),
        )
        safe = task_name.replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"{safe}_dotplot.png"
        fig.savefig(str(out_path), dpi=200, facecolor="white", bbox_inches="tight")
        plt.close("all")
        logger.info(f"Dotplot saved: {out_path.name}")
    except Exception as e:
        logger.warning(f"Dotplot failed for {task_name}: {e}")


# =============================================================================
# Summary text report
# =============================================================================

def write_summary(
    df_top: pd.DataFrame,
    task_markers: Dict[str, Dict[str, List[str]]],
    out_path: Path,
) -> None:
    lines = [
        "GSE140228 — Differential Expression Summary",
        "=" * 70,
        "Generated by GSE140228_differential_expression.py",
        "Expression input: log-normalised (no transformation applied)",
        "",
        f"Total clusters with DEGs : {df_top['cluster'].nunique()}",
        f"Total DEGs (top-N table) : {len(df_top)}",
        "",
        "Per-Cluster DEG Counts (top-N, after all filters)",
        "-" * 60,
    ]
    for cluster, grp in df_top.groupby("cluster"):
        top5 = ", ".join(grp.head(5)["gene"].tolist())
        lines.append(f"  {cluster:<38} {len(grp):3d} genes  |  {top5}")

    lines += [
        "",
        "Task-Specific Marker Programs (DEG-derived, specificity-ranked)",
        "=" * 70,
    ]
    for task_name, markers_dict in task_markers.items():
        lines.append(f"\n{task_name}")
        lines.append("-" * len(task_name))
        for pop, genes in markers_dict.items():
            lines.append(f"  {pop:<38} ({len(genes)} markers)")
            lines.append(f"    {', '.join(genes) if genes else 'NONE — check cluster name'}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary written: {out_path.name}")


# =============================================================================
# Task JSON builder
# =============================================================================

def build_task_json(
    task_name: str,
    task_def: Dict,
    deg_markers: Dict[str, List[str]],
    mmc2_markers: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """
    Build task JSON with:
    - markers      : DEG-derived, specificity-ranked (primary source for metrics)
    - study_markers: mmc2 Table S3 representative genes (secondary, for reference)
    """
    marker_programs = []
    for pop in task_def["include_populations"]:
        deg_genes   = deg_markers.get(pop, [])
        study_genes = mmc2_markers.get(pop, []) if mmc2_markers else []
        marker_programs.append({
            "population":      pop,
            "role":            "root" if pop == task_def["root_population"]
                               else "intermediate_or_terminal",
            "markers":         deg_genes,
            "study_markers":   study_genes,
            "n_deg_markers":   len(deg_genes),
            "n_study_markers": len(study_genes),
        })

    return {
        "task_name":           task_name,
        "description":         task_def["description"],
        "root_population":     task_def["root_population"],
        "include_populations": task_def["include_populations"],
        "marker_programs":     marker_programs,
        "metadata": {
            "dataset":           "GSE140228",
            "marker_source":     "wilcoxon_rank_genes_groups (scanpy)",
            "expression_input":  "log-normalised (adata.X or specified layer)",
            "filter_logfc":      ">=0.25",
            "filter_pct_group":  ">=0.10",
            "filter_padj":       "<0.05",
            "filter_pct_rest":  "<=0.98  (relaxed for 40-cluster dataset)",
            "ranking":           "specificity (pct_group - pct_rest) desc, then logFC desc",
            "global_exclusions": (
                "ribosomal (RPS/RPL), mitochondrial (MT-), "
                "housekeeping (ACTB/GAPDH/B2M/MALAT1/...), "
                "cell-cycle (MKI67/TOP2A/CDK1/...)"
            ),
        },
    }


# =============================================================================
# Optional mmc2 loader
# =============================================================================

def load_mmc2_markers(mmc2_path: Optional[str]) -> Dict[str, List[str]]:
    """Load representative markers from mmc2.xlsx Table S3 if path provided."""
    if not mmc2_path:
        return {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(mmc2_path, read_only=True, data_only=True)
        ws = wb["TableS3"]
        rows = list(ws.iter_rows(values_only=True))
        result: Dict[str, List[str]] = {}
        for r in rows[3:]:
            if r[0] and r[1]:
                name  = str(r[0]).strip()
                genes = [g.strip() for g in str(r[1]).split(",") if g.strip()]
                result[name] = genes
        logger.info(f"Loaded mmc2 markers for {len(result)} clusters")
        return result
    except Exception as e:
        logger.warning(f"Could not load mmc2 markers from '{mmc2_path}': {e}")
        return {}


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> None:

    # ── Resolve paths ─────────────────────────────────────────────────────────
    adata_path = args.adata if args.adata else DEFAULT_ADATA_PATH
    out_dir    = Path(args.out_dir if args.out_dir else DEFAULT_OUT_DIR)

    # ── Output directory structure ────────────────────────────────────────────
    dirs = {
        "root":     out_dir,
        "volcano":  out_dir / "volcano_plots",
        "tasks":    out_dir / "task_jsons",
        "logs":     out_dir / "logs",
        "dotplots": out_dir / "dotplots",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # File handler for persistent logging
    fh = logging.FileHandler(dirs["logs"] / "de_run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 65)
    logger.info("GSE140228 Differential Expression Pipeline")
    logger.info("=" * 65)
    logger.info(f"Input AnnData : {adata_path}")
    logger.info(f"Output dir    : {out_dir}")
    logger.info(f"Layer         : {args.layer or 'adata.X (default)'}")
    logger.info(f"Top N per cluster : {args.n_top}")
    logger.info(f"Min pct_group     : {args.min_pct}")
    logger.info(f"Min logFC         : {args.logfc}")
    logger.info(f"Expression input  : log-normalised (no transformation)")

    # ── Load optional mmc2 study markers ─────────────────────────────────────
    mmc2_markers = load_mmc2_markers(getattr(args, "mmc2", None))

    # ── Load and validate ─────────────────────────────────────────────────────
    adata, layer_used = load_and_validate(adata_path, args.layer)

    # ── If a specific layer requested, copy to X ──────────────────────────────
    adata = set_expression_layer(adata, layer_used)

    # ── Verify task populations exist in data ─────────────────────────────────
    present_types = set(adata.obs["celltype_sub"].astype(str).unique())
    logger.info("\nTask population presence check:")
    all_task_pops: set = set()
    for task_name, task_def in TASKS.items():
        for pop in task_def["include_populations"]:
            all_task_pops.add(pop)
            status = "✓" if pop in present_types else "✗ MISSING"
            logger.info(f"  [{status}]  {pop:<35}  ({task_name})")

    missing = all_task_pops - present_types
    if missing:
        logger.warning(
            f"\n{len(missing)} task population(s) not found in celltype_sub — "
            f"they will produce empty marker lists:\n"
            + "\n".join(f"  • {p}" for p in sorted(missing))
        )

    # ── Run Wilcoxon DE ───────────────────────────────────────────────────────
    adata = run_de(
        adata,
        groupby="celltype_sub",
        method="wilcoxon",
        n_genes=max(args.n_top * 4, args.n_genes_de),
        pts=True,
    )

    # ── Extract and filter results ────────────────────────────────────────────
    df_full = extract_de_results(
        adata,
        min_logfc=args.logfc,
        min_pct=args.min_pct,
        max_pct_rest=0.98,
    )

    df_top = get_top_n(df_full, n=args.n_top)

    # ── Save full tables ──────────────────────────────────────────────────────
    if not args.tasks_only:
        fp = dirs["root"] / "full_deg_results.csv"
        df_full.to_csv(fp, index=False)
        logger.info(f"Saved: {fp.name}  ({len(df_full):,} rows)")

        tp = dirs["root"] / f"top{args.n_top}_per_cluster.csv"
        df_top.to_csv(tp, index=False)
        logger.info(f"Saved: {tp.name}  ({len(df_top):,} rows)")

    # ── Volcano plots ─────────────────────────────────────────────────────────
    if HAS_MATPLOTLIB and not args.tasks_only:
        clusters_with_degs = df_full["cluster"].unique()
        logger.info(f"Generating {len(clusters_with_degs)} volcano plots ...")
        for cluster in clusters_with_degs:
            plot_volcano(df_full, cluster, dirs["volcano"])
        logger.info("Volcano plots complete.")

    # ── Task marker programs ──────────────────────────────────────────────────
    logger.info("\nBuilding task marker programs ...")
    all_task_markers: Dict[str, Dict[str, List[str]]] = {}

    for task_name, task_def in TASKS.items():
        pan        = task_def.get("pan_lineage_genes", [])
        task_mkrs: Dict[str, List[str]] = {}

        for pop in task_def["include_populations"]:
            if pop not in present_types:
                task_mkrs[pop] = []
                continue

            markers = select_markers_for_cluster(
                df_full,
                cluster=pop,
                pan_genes=pan,
                n_markers=15,
                min_specificity=0.01,
            )
            task_mkrs[pop] = markers
            top3 = ", ".join(markers[:3]) if markers else "—"
            logger.info(
                f"  {task_name} | {pop:<33} "
                f"{len(markers):2d} markers  [{top3}]"
            )

        # Overlap check between populations within this task
        check_marker_overlap(task_name, task_mkrs)

        all_task_markers[task_name] = task_mkrs

        # Dotplot
        if HAS_MATPLOTLIB and not args.tasks_only:
            plot_dotplot(adata, task_mkrs, dirs["dotplots"], task_name)

        # Write task JSON
        task_json  = build_task_json(
            task_name, task_def, task_mkrs,
            mmc2_markers=mmc2_markers or None,
        )
        json_path  = dirs["tasks"] / f"{task_name}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(task_json, fh, indent=2, ensure_ascii=False)
        logger.info(f"  → {json_path.name}")

    # ── Summary report ────────────────────────────────────────────────────────
    write_summary(
        df_top, all_task_markers,
        dirs["root"] / "deg_summary.txt",
    )

    # ── Done ──────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 65)
    logger.info(f"Output directory  : {out_dir}")
    logger.info(f"full_deg_results  : {len(df_full):,} filtered DEGs across "
                f"{df_full['cluster'].nunique()} clusters")
    logger.info(f"top{args.n_top}_per_cluster : {len(df_top):,} rows")
    logger.info(f"volcano_plots/    : {df_full['cluster'].nunique()} PNGs")
    logger.info(f"task_jsons/       : {len(TASKS)} JSONs")
    logger.info(f"logs/de_run.log   : full run log")


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "GSE140228 Differential Expression — marker gene extraction "
            "for TI benchmarking tasks.\n"
            "Expects log-normalised expression in adata.X (or specified layer).\n"
            f"Default input : {DEFAULT_ADATA_PATH}\n"
            f"Default output: {DEFAULT_OUT_DIR}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--adata", default=None,
        help=(
            f"Path to AnnData .h5ad file. "
            f"Default: {DEFAULT_ADATA_PATH}"
        ),
    )
    p.add_argument(
        "--out_dir", default=None,
        help=(
            f"Output directory. "
            f"Default: {DEFAULT_OUT_DIR}"
        ),
    )
    p.add_argument(
        "--layer", default=None,
        help=(
            "Log-normalised expression layer to use as adata.X "
            "(e.g. 'log1p', 'lognorm'). "
            "If omitted, uses adata.X directly."
        ),
    )
    p.add_argument(
        "--n_top", type=int, default=50,
        help="Top N DEGs per cluster to include in output tables. Default: 15",
    )
    p.add_argument(
        "--min_pct", type=float, default=0.10,
        help="Min fraction of cells in cluster expressing gene. Default: 0.10",
    )
    p.add_argument(
        "--logfc", type=float, default=0.25,
        help="Min log2 fold-change. Default: 0.25",
    )
    p.add_argument(
        "--n_genes_de", type=int, default=500,
        help=(
            "Number of genes requested from rank_genes_groups per cluster "
            "before downstream filtering. Default: 200"
        ),
    )
    p.add_argument(
        "--mmc2", default=None,
        help=(
            "Optional path to mmc2.xlsx. "
            "If provided, study_markers field is populated in task JSONs."
        ),
    )
    p.add_argument(
        "--tasks_only", action="store_true",
        help="Skip full DEG tables, volcano plots, and dotplots — task JSONs only.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()