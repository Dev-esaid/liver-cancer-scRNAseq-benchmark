# TI_benchmark/plotting.py
"""
Trajectory Inference Benchmarking: Plotting (plotting.py)

Fix history
-----------
2026-04-06 (canonical group/node ordering):
  Standardises the cell-type ordering across all methods so cross-method
  comparison plots are directly comparable.  Results are NOT affected —
  only the visual ordering of groups in pseudotime-by-group boxplots and
  the integer node labels in topology graphs are fixed.

  New helper: _canonical_group_order(series)
    Returns a stable list of group names:
    - pandas Categorical   → honours .cat.categories
    - plain object/string  → sorted(unique values) — alphabetical

  _plot_boxplot_by_group : new param group_order=None
    When provided, the x-axis uses this fixed order (missing groups are
    silently dropped).  Fallback: sort by median (original behaviour).

  _plot_topology_on_embedding : new param canonical_node_order=None
    node_to_id assigned in canonical order so the integer label for each
    cell type is identical across all methods on the same task.

  _plot_topology_graph_curved : new param canonical_node_order=None
    Same invariant: node_to_id from canonical_node_order first.

  generate_plot_suite
    Computes canonical_group_order once and threads it through all calls.

2026-03-19 (pseudotime by group + stability histogram):
  [unchanged — see original file]

2026-03-18 (shared-UMAP hardening):
  [unchanged — see original file]

Key design goals:
* Publication-ready layout and typography.
* Shared embedding within each run.
* Undirected curved lines for trajectory topology.
* Every saved figure carries a caption written to figure_captions.txt.
"""

from __future__ import annotations

import textwrap
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator
except Exception as e:
    raise ImportError("plotting.py requires matplotlib to be installed.") from e

try:
    import anndata as ad
except Exception as e:
    raise ImportError("plotting.py requires anndata to be installed.") from e

try:
    from scipy.stats import gaussian_kde
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import networkx as nx
except Exception:
    nx = None

from .shared_types import TIOutput
from .utils import merge_json_shallow

logger = logging.getLogger(__name__)


# =============================================================================
# Global RC and constants
# =============================================================================

_DEFAULT_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

_CAPTION_FONTSIZE = 7.5
_CAPTION_COLOR    = "#444444"
_CAPTION_WRAP     = 115
_CAPTION_BOTTOM   = 0.08

# Annotation text colour
_ANNO_GREY = "#4a4a4a"


def _with_rc_context():
    return plt.rc_context(_DEFAULT_RC)


def _wrap_caption(text: str) -> str:
    return "\n".join(
        textwrap.fill(line, width=_CAPTION_WRAP)
        for line in text.splitlines()
    )


def _save_fig(fig: "plt.Figure", out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


def _save_fig_with_caption(
    fig: "plt.Figure",
    out_path: Path,
    caption: Optional[str],
    *,
    extra_bottom: float = 0.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)


# =============================================================================
# Caption builders
# =============================================================================

def _cap_umap_categorical(key: str, n_cats: int) -> str:
    return (
        f"UMAP embedding coloured by '{key}' ({n_cats} categories). "
        "Each point represents a single cell projected onto the two-dimensional "
        "UMAP manifold computed from the top principal components of the "
        "highly variable gene expression matrix. "
        "Categories are assigned distinct colours; cells with missing annotation "
        "are shown as 'NA'."
    )

def _cap_umap_gene(gene: str) -> str:
    return (
        f"UMAP embedding coloured by normalised expression of '{gene}'. "
        "Expression values are min-max normalised to [0, 1] across all cells "
        "and mapped to the viridis colour scale (dark purple: low; yellow: high). "
        "Log1p-normalised counts are used."
    )

def _cap_pseudotime_embedding(method: str) -> str:
    return (
        f"Pseudotime inferred by {method} projected onto the shared UMAP embedding. "
        "Pseudotime values are rescaled to [0, 1] for visualisation (viridis scale; "
        "dark purple: root / earliest cells; yellow: most differentiated / latest cells). "
        "The root cell was selected based on the biologically defined progenitor population."
    )

def _cap_pseudotime_hist(method: str) -> str:
    return (
        f"Distribution of {method} pseudotime values across all cells in this lineage. "
        "The x-axis shows raw pseudotime in the method's native scale. "
        "A broad, unimodal distribution indicates a smooth, well-resolved trajectory; "
        "sharp peaks or bimodality may indicate disconnected cell clusters or a "
        "poorly resolved branching point."
    )

def _cap_pseudotime_by_group(group_key: str, method: str, n_groups: int) -> str:
    return (
        f"Violin + box plot of {method} pseudotime stratified by '{group_key}' "
        f"({n_groups} groups, ordered left to right by the canonical task-level "
        f"cell-type order so all methods share the same x-axis layout). "
        "Violins show the full distribution; boxes span the IQR; "
        "the black line marks the median. "
        "Groups with <500 cells show individual cell jitter. "
        "Y-axis is fixed to [0, 1]."
    )

def _cap_topology_on_embedding(method: str) -> str:
    return (
        f"Trajectory topology inferred by {method} overlaid on the shared UMAP embedding. "
        "Numbered circles mark cluster centroids computed as the mean UMAP coordinate "
        "of all cells belonging to that cluster in the SHARED embedding "
        "(adata.obsm['X_umap']), ensuring all methods are visualised in the same "
        "coordinate space regardless of any internal coordinate system used by the method. "
        "Node integer IDs are assigned from a single canonical ordering shared across "
        "all methods on this task, so the same cell type always carries the same ID. "
        "Curved lines connect clusters linked by a trajectory edge; line weight is "
        "proportional to edge weight. "
        "Edges are undirected. "
        "The legend key on the right maps numeric IDs to cluster names."
    )

def _cap_topology_graph(method: str) -> str:
    return (
        f"Abstract topology graph of the trajectory inferred by {method}. "
        "Nodes represent trajectory waypoints or cell clusters; edges represent "
        "inferred developmental connections. "
        "Node position is determined by a force-directed (spring) layout with "
        "strong inter-node repulsion, so spatial distance here does not correspond "
        "to expression distance. "
        "Edge width is proportional to edge weight. "
        "Node integer IDs are assigned from a single canonical ordering shared across "
        "all methods on this task, so the same cell type always carries the same ID. "
        "All edges are undirected."
    )

def _cap_topology_heatmap(method: str) -> str:
    return (
        f"Adjacency weight matrix of the trajectory topology inferred by {method}. "
        "Rows and columns correspond to trajectory nodes (clusters or waypoints). "
        "Colour intensity indicates the connection weight between each pair of nodes "
        "(darker = stronger). "
        "Symmetric matrix entries indicate undirected edges."
    )

def _cap_terminal_prob(col: str, method: str) -> str:
    return (
        f"Estimated terminal-state probability for fate '{col}' inferred by {method}, "
        "projected onto the shared UMAP embedding. "
        "Values are rescaled to [0, 1] (viridis scale). "
        "High values (yellow) indicate cells with a strong commitment to this terminal fate; "
        "low values (dark purple) indicate cells with low commitment or commitment to "
        "alternative fates."
    )

def _cap_branch_labels(method: str) -> str:
    return (
        f"Branch assignment labels inferred by {method} projected onto the shared UMAP "
        "embedding. "
        "Each colour represents a distinct trajectory branch. "
        "Cells are assigned to branches based on their pseudotime and topology; "
        "cells near branch points may receive ambiguous or transitional labels."
    )

def _cap_stability_spearman(n_replicates: int) -> str:
    n_pairs = n_replicates * (n_replicates - 1) // 2
    return (
        f"Distribution of pairwise Spearman |rho| correlations between pseudotime "
        f"vectors computed across {n_replicates} bootstrap replicates ({n_pairs} pairs). "
        "Each replicate is generated by sampling 80% of cells with stratification "
        "by cell type. "
        "The absolute value accounts for direction ambiguity. "
        "X-axis fixed to [0, 1] for cross-method comparability. "
        "Values close to 1.0 indicate high pseudotime stability; "
        "values below 0.7 suggest poor reproducibility."
    )

def _cap_stability_jaccard(n_replicates: int) -> str:
    n_pairs = n_replicates * (n_replicates - 1) // 2
    return (
        f"Distribution of pairwise edge Jaccard indices across {n_replicates} bootstrap "
        f"replicates ({n_pairs} pairs). "
        "The Jaccard index measures the overlap of edge sets (edges restricted "
        "to nodes shared between both replicates). "
        "X-axis fixed to [0, 1] for cross-method comparability. "
        "Values close to 1.0 indicate that the same topology is recovered consistently; "
        "values below 0.5 suggest unstable branching structure. "
        "Fixed-edge methods produce discrete bars: J = k/(2n-k)."
    )


# =============================================================================
# Core utilities
# =============================================================================

def _get_umap(adata: "ad.AnnData", umap_key: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if umap_key not in adata.obsm:
        return None, f"missing obsm['{umap_key}']"
    arr = np.asarray(adata.obsm[umap_key])
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None, f"obsm['{umap_key}'] is not (n,2+)"
    return arr[:, :2].astype(float, copy=False), None


def _capture_shared_umap(
    adata: "ad.AnnData",
    umap_key: str,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    arr, err = _get_umap(adata, umap_key)
    if arr is None:
        return None, err
    umap_copy = arr.copy()
    logger.info(
        "Shared UMAP captured: shape=%s, x=[%.2f, %.2f], y=[%.2f, %.2f]",
        umap_copy.shape,
        float(umap_copy[:, 0].min()), float(umap_copy[:, 0].max()),
        float(umap_copy[:, 1].min()), float(umap_copy[:, 1].max()),
    )
    return umap_copy, None


def _warn_if_umap_changed(
    adata: "ad.AnnData",
    umap_key: str,
    umap_fixed: np.ndarray,
    context: str = "",
) -> None:
    if umap_key not in adata.obsm:
        return
    current = np.asarray(adata.obsm[umap_key])
    if current.shape != umap_fixed.shape:
        logger.warning(
            "%sShared UMAP shape changed: expected %s, got %s. "
            "A TI adapter may have overwritten adata.obsm['%s']. "
            "Plots will use the captured umap_fixed — results are unaffected.",
            f"[{context}] " if context else "",
            umap_fixed.shape, current.shape, umap_key,
        )
        return
    if not np.allclose(current[:, :2], umap_fixed, atol=1e-4, rtol=0.0):
        xmin_orig = float(umap_fixed[:, 0].min())
        xmax_orig = float(umap_fixed[:, 0].max())
        xmin_curr = float(current[:, 0].min())
        xmax_curr = float(current[:, 0].max())
        logger.warning(
            "%sShared UMAP values changed: original x=[%.2f, %.2f], current x=[%.2f, %.2f]. "
            "A TI adapter overwrote adata.obsm['%s']. "
            "Plots use the captured umap_fixed — visual comparison to other methods is preserved.",
            f"[{context}] " if context else "",
            xmin_orig, xmax_orig, xmin_curr, xmax_curr, umap_key,
        )


def _as_numeric_series(x: Any, index: pd.Index) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.reindex(index)
    else:
        s = pd.Series(x, index=index)
    return pd.to_numeric(s, errors="coerce").astype(float)


def _normalize_01(values: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    v = np.asarray(values, dtype=float)
    v_finite = v[np.isfinite(v)]
    if v_finite.size == 0:
        return values, "no finite values"
    vmin = float(np.nanmin(v_finite))
    vmax = float(np.nanmax(v_finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return values, "cannot normalize (degenerate range)"
    out = (np.asarray(values, dtype=float) - vmin) / (vmax - vmin)
    return out, None


def _categorical_palette(n: int) -> List[Tuple[float, float, float, float]]:
    cmaps = ["tab20", "tab20b", "tab20c"]
    cols: List[Tuple[float, float, float, float]] = []
    for name in cmaps:
        try:
            cmap = plt.colormaps.get_cmap(name)
        except AttributeError:
            cmap = plt.cm.get_cmap(name)
        cols.extend([cmap(i) for i in range(cmap.N)])
        if len(cols) >= n:
            break
    if len(cols) < n:
        try:
            cmap = plt.colormaps.get_cmap("hsv")
        except AttributeError:
            cmap = plt.cm.get_cmap("hsv")
        cols.extend([cmap(i / max(1, n)) for i in range(n - len(cols))])
    return cols[:n]


def _wrap_label(s: str, width: int = 26) -> str:
    s = str(s)
    if len(s) <= width:
        return s
    parts = s.split(" ")
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in parts:
        add = len(p) + (1 if cur else 0)
        if cur_len + add <= width:
            cur.append(p)
            cur_len += add
        else:
            lines.append(" ".join(cur))
            cur = [p]
            cur_len = len(p)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


# =============================================================================
# NEW HELPER — canonical group ordering
# =============================================================================

def _canonical_group_order(series: pd.Series) -> List[str]:
    """Return a stable, task-level list of group names for cross-method plots.

    Priority:
    1. If *series* is a ``pandas.Categorical``, honour its ``.cat.categories``
       order (set once by the task/dataset loader and shared by all methods).
    2. Otherwise fall back to ``sorted(unique_values)`` — deterministic,
       alphabetical, independent of any single method's results.

    The returned list is used as the fixed x-axis order for
    ``_plot_boxplot_by_group`` and the fixed node-ID assignment in topology
    graphs.  **Results (pseudotime values, edge weights, etc.) are untouched.**
    """
    try:
        # Prefer Categorical order — set by dataset/task, not method-specific.
        if hasattr(series, "cat") and hasattr(series.cat, "categories"):
            cats = [str(c) for c in series.cat.categories]
            if cats:
                return cats
        # Fallback: sorted unique non-null string values.
        unique_vals = sorted(
            str(v) for v in series.dropna().unique()
        )
        return unique_vals
    except Exception:
        return []


# =============================================================================
# Embedding plots
# =============================================================================

def _plot_embedding_categorical(
    umap: np.ndarray,
    labels: pd.Series,
    *,
    title: str,
    out_path: Path,
    caption: Optional[str] = None,
    legend_max: int = 20,
) -> Tuple[str, Optional[str]]:
    try:
        with _with_rc_context():
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            labs = labels.astype("string").fillna("NA")
            cats = sorted(labs.unique().tolist())
            palette = _categorical_palette(len(cats))
            color_map = {c: palette[i] for i, c in enumerate(cats)}
            colors = labs.map(color_map).tolist()
            ax.scatter(
                umap[:, 0], umap[:, 1],
                s=18, c=colors,
                linewidths=0.25, edgecolors="black",
                alpha=0.90,
            )
            ax.set_title(title)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal", adjustable="datalim")
            if len(cats) <= legend_max:
                handles = [
                    ax.scatter([], [], s=40, c=[color_map[c]],
                               edgecolors="black", linewidths=0.4, label=str(c))
                    for c in cats
                ]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                    title=str(labels.name) if labels.name else None,
                )
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None
    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


def _plot_embedding_continuous(
    umap: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    out_path: Path,
    caption: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "value",
) -> Tuple[str, Optional[str]]:
    try:
        with _with_rc_context():
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            norm = Normalize(vmin=vmin, vmax=vmax)
            ax.scatter(
                umap[:, 0], umap[:, 1],
                s=18, c=values, cmap=cmap, norm=norm,
                linewidths=0.15, edgecolors="black",
                alpha=0.92,
            )
            ax.set_title(title)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal", adjustable="datalim")
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                                ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None
    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Histogram — pseudotime distribution
# =============================================================================

def _plot_hist(
    values: Iterable[float],
    *,
    title: str,
    xlabel: str,
    out_path: Path,
    caption: Optional[str] = None,
    kde_color: str = "#c0392b",
) -> Tuple[str, Optional[str]]:
    try:
        v = np.asarray(list(values), dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "skipped", "no finite values"
        with _with_rc_context():
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            counts, _, _ = ax.hist(
                v, bins=30, color="#4c72b0", edgecolor="white", linewidth=0.4,
            )
            y_peak = float(counts.max()) if counts.max() > 0 else 1.0

            if _HAS_SCIPY and v.size >= 10:
                try:
                    kde = gaussian_kde(v, bw_method="scott")
                    xk = np.linspace(float(v.min()), float(v.max()), 500)
                    kv = kde(xk)
                    kv_scaled = kv * (y_peak / (kv.max() + 1e-12))
                    ax.plot(xk, kv_scaled, color=kde_color,
                            linewidth=2.0, alpha=0.85, zorder=5)
                except Exception:
                    pass

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
            mu = float(np.mean(v))
            ax.axvline(mu, color="#c0392b", linewidth=1.1, linestyle="--",
                       label=f"mean = {mu:.3f}")
            ax.legend(frameon=False, fontsize=8)
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None
    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Stability histogram — fixed [0,1] axis, adaptive bins, zoom inset
# =============================================================================

def _plot_stability_hist(
    values: Iterable[float],
    *,
    title: str,
    xlabel: str,
    out_path: Path,
    caption: Optional[str] = None,
    x_fixed_range: Tuple[float, float] = (0.0, 1.0),
    n_bins: int = 50,
    mean_color: str = "#c0392b",
    bar_color: str = "#4c72b0",
    kde_color: str = "#1a3a6b",
) -> Tuple[str, Optional[str]]:
    try:
        v = np.asarray(list(values), dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return "skipped", "no finite values"

        xmin, xmax = float(x_fixed_range[0]), float(x_fixed_range[1])
        v_clipped = np.clip(v, xmin, xmax)

        n_pairs = int(v.size)
        mu  = float(np.mean(v_clipped))
        med = float(np.median(v_clipped))
        sd  = float(np.std(v_clipped))
        v_min   = float(v_clipped.min())
        v_max   = float(v_clipped.max())
        v_span  = v_max - v_min

        raw_bw        = v_span / 35.0 if v_span > 0 else 0.02
        bw_hist       = float(np.clip(raw_bw, 0.004, 0.02))
        n_bins_actual = max(int(np.ceil(1.0 / bw_hist)), 50)
        bin_edges     = np.linspace(xmin, xmax, n_bins_actual + 1)

        tight = v_span < 0.12

        kde = None
        if _HAS_SCIPY and v.size >= 10:
            try:
                v_for_kde = v_clipped.copy()
                data_std  = float(v_for_kde.std(ddof=1))
                if data_std < 1e-6:
                    rng = np.random.default_rng(42)
                    v_for_kde = v_for_kde + rng.normal(0, 0.005,
                                                       size=v_for_kde.size)
                    v_for_kde = np.clip(v_for_kde, xmin, xmax)
                    data_std  = float(v_for_kde.std(ddof=1))
                scott_abs = data_std * float(v.size ** -0.2)
                abs_bw    = float(np.clip(scott_abs * 0.65, 0.012, 0.06))
                factor    = abs_bw / max(data_std, 1e-9)
                kde       = gaussian_kde(v_for_kde, bw_method=factor)
            except Exception:
                kde = None

        rc_pub = {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }

        with plt.rc_context({**_DEFAULT_RC, **rc_pub}):
            fig, ax = plt.subplots(figsize=(6.5, 4.4))

            counts, _, _ = ax.hist(
                v_clipped, bins=bin_edges,
                color=bar_color, edgecolor="white",
                linewidth=0.3, alpha=0.85, zorder=2,
            )
            y_max = float(counts.max()) if counts.max() > 0 else 1.0

            if kde is not None:
                try:
                    xk       = np.linspace(xmin, xmax, 1000)
                    kde_vals = kde(xk)
                    kde_scale = y_max / (kde_vals.max() + 1e-12)
                    ax.plot(xk, kde_vals * kde_scale,
                            color=kde_color, linewidth=1.5, alpha=0.85, zorder=5)
                except Exception:
                    pass

            ax.axvline(mu, color=mean_color, linewidth=1.6,
                       linestyle="--", zorder=6)
            mean_ha = "left" if mu < 0.85 else "right"
            mean_dx = 0.012 if mu < 0.85 else -0.012
            ax.text(
                mu + mean_dx, y_max * 0.97,
                f"mean = {mu:.3f}\n± {sd:.3f}",
                ha=mean_ha, va="top",
                fontsize=8.5, color=mean_color,
                fontweight="bold", linespacing=1.35, zorder=8,
            )

            if abs(mu - med) > 0.005:
                ax.axvline(med, color="#5d4037", linewidth=1.1,
                           linestyle=(0, (5, 3)), zorder=5)
                med_ha = "left" if med < mu else "right"
                med_dx = 0.012 if med < mu else -0.012
                ax.text(
                    med + med_dx, y_max * 0.75,
                    f"median = {med:.3f}",
                    ha=med_ha, va="top",
                    fontsize=8, color="#5d4037",
                    linespacing=1.3, zorder=8,
                )

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(0, y_max * 1.12)
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
            ax.tick_params(axis="x", which="major", length=4.0, width=0.8, labelsize=9)
            ax.tick_params(axis="x", which="minor", length=2.5, width=0.5)
            ax.tick_params(axis="y", which="major", length=4.0, width=0.8, labelsize=9)
            ax.set_title(title, pad=10, fontweight="normal")
            ax.set_xlabel(xlabel, labelpad=6)
            ax.set_ylabel("Count", labelpad=6)
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)
            ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.3, zorder=0)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))

            ax.annotate(
                f"n = {n_pairs:,} pairs",
                xy=(0.98, 0.04), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8.5, color=_ANNO_GREY,
            )

            if tight:
                pad   = max(v_span * 0.15, 0.002)
                z_lo  = max(xmin, v_min - pad)
                z_hi  = min(xmax, v_max + pad)
                z_span = z_hi - z_lo

                z_bw    = max(v_span / 30.0, 0.0005)
                z_nbins = max(int(np.ceil(z_span / z_bw)), 15)
                z_edges = np.linspace(z_lo, z_hi, z_nbins + 1)

                ax_ins = ax.inset_axes([0.03, 0.55, 0.36, 0.40])

                cnt_z, _, _ = ax_ins.hist(
                    v_clipped, bins=z_edges,
                    color=bar_color, edgecolor="white",
                    linewidth=0.2, alpha=0.85, zorder=2,
                )
                cy = float(cnt_z.max()) if cnt_z.max() > 0 else 1.0

                if kde is not None:
                    try:
                        xz  = np.linspace(z_lo, z_hi, 400)
                        kz  = kde(xz)
                        ax_ins.plot(xz, kz / (kz.max() + 1e-12) * cy,
                                    color=kde_color, linewidth=1.2,
                                    alpha=0.85, zorder=5)
                    except Exception:
                        pass

                ax_ins.axvline(mu, color=mean_color, linewidth=1.3,
                               linestyle="--", zorder=7)
                if abs(mu - med) > 0.005:
                    ax_ins.axvline(med, color="#5d4037", linewidth=0.9,
                                   linestyle=(0, (4, 2)), zorder=6)

                ax_ins.set_xlim(z_lo, z_hi)
                ax_ins.set_ylim(0, cy * 1.12)
                z_ticks = np.round(np.linspace(z_lo, z_hi, 4), 3)
                ax_ins.set_xticks(z_ticks)
                ax_ins.tick_params(labelsize=6.5, length=2.5, width=0.6, pad=1.5)
                ax_ins.set_xlabel("Zoomed", fontsize=6.5, labelpad=2)
                ax_ins.set_ylabel("Count", fontsize=6.5, labelpad=2)
                ax_ins.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=4))
                ax_ins.grid(axis="y", linestyle=":", linewidth=0.3,
                            color="#cccccc", alpha=0.6, zorder=0)
                ax_ins.set_axisbelow(True)
                ax_ins.spines["top"].set_visible(False)
                ax_ins.spines["right"].set_visible(False)
                ax_ins.spines["left"].set_linewidth(0.6)
                ax_ins.spines["bottom"].set_linewidth(0.6)

                ax.indicate_inset_zoom(
                    ax_ins, edgecolor="#999999", linewidth=0.7, alpha=0.7,
                )

            fig.tight_layout()
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None
    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Pseudotime by group — boxplot, VERTICAL, coolwarm gradient
# =============================================================================

def _plot_boxplot_by_group(
    group: pd.Series,
    values: pd.Series,
    *,
    title: str,
    ylabel: str,
    out_path: Path,
    caption: Optional[str] = None,
    orientation: str = "vertical",
    wrap_label_width: int = 28,
    group_order: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """
    Publication-quality boxplot of pseudotime by group — vertical layout.

    Parameters
    ----------
    group_order : list of str, optional
        **Canonical task-level ordering** computed once in ``generate_plot_suite``
        and shared across all methods.  When supplied, the x-axis follows this
        fixed order (groups present in the data but absent from *group_order* are
        appended alphabetically at the right; groups in *group_order* absent from
        the data are silently skipped).
        When *None* (fallback), groups are ordered by ascending median pseudotime
        — the original per-method behaviour.

    Notes
    -----
    Only the *display order* of groups on the x-axis is affected.
    Pseudotime values, box positions, colours, and all other visual properties
    are computed from the data and are untouched.
    """
    try:
        df = pd.DataFrame({
            "group": group.astype("string").fillna("NA").str.strip(),
            "value": pd.to_numeric(values, errors="coerce").astype(float),
        })
        df = df[np.isfinite(df["value"].values)]
        if df.empty:
            return "skipped", "no finite values"

        present_groups = set(df["group"].unique().tolist())

        if group_order is not None:
            # Use canonical order, keeping only groups present in this run.
            groups = [str(g) for g in group_order if str(g) in present_groups]
            # Append any groups in the data but missing from canonical order
            # (e.g. a method introduces an 'NA' group not in the original set).
            extras = sorted(g for g in present_groups if g not in set(groups))
            groups = groups + extras
        else:
            # Fallback: sort by median pseudotime (original behaviour).
            med = df.groupby("group")["value"].median().sort_values(ascending=True)
            groups = med.index.astype("string").tolist()

        if not groups:
            return "skipped", "no groups after ordering"

        n_grp = len(groups)
        data = [df.loc[df["group"] == g, "value"].values for g in groups]
        tick_labels = [_wrap_label(str(g), width=int(wrap_label_width)) for g in groups]

        try:
            grad_cmap = plt.colormaps.get_cmap("coolwarm")
        except AttributeError:
            grad_cmap = plt.cm.get_cmap("coolwarm")
        box_colors = [grad_cmap(i / max(1, n_grp - 1)) for i in range(n_grp)]

        with _with_rc_context():
            fig_w = max(5.5, 0.55 * n_grp + 1.8)
            fig, ax = plt.subplots(figsize=(fig_w, 5.4))

            box_w = min(0.2, 3.0 / max(n_grp, 1))

            bp = ax.boxplot(
                data, labels=tick_labels,
                widths=box_w,
                showfliers=False, patch_artist=True,
                medianprops={"linewidth": 1.4, "color": "#222222"},
                boxprops={"linewidth": 0.9},
                whiskerprops={"linewidth": 0.9},
                capprops={"linewidth": 0.9, "solid_capstyle": "round"},
            )

            for box, col in zip(bp["boxes"], box_colors):
                r, g_c, b, a = col
                box.set_facecolor((r, g_c, b, 0.55))
                box.set_edgecolor("black")

            # Diamond marker at median
            for i, (g_name, col) in enumerate(zip(groups, box_colors), start=1):
                vals = df.loc[df["group"] == g_name, "value"].values
                if vals.size:
                    ax.scatter(i, np.median(vals), marker="D", s=28, zorder=5,
                               color=col, edgecolors="#222222", linewidths=0.6)

            ax.set_title(title, pad=10)
            ax.set_ylabel(ylabel, labelpad=6)
            ax.set_xlabel("")
            ax.set_xticklabels(tick_labels, rotation=45, ha="right",
                               fontsize=8.5, rotation_mode="anchor")
            ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ylim = ax.get_ylim()
            ymax = ylim[1]
            for i, g_name in enumerate(groups, start=1):
                n_cells = int((df["group"] == g_name).sum())
                ax.text(i, ymax * 1.01, f"n={n_cells:,}",
                        va="bottom", ha="center", fontsize=6.5, color="#555555")

            fig.tight_layout(pad=0.8)
            fig.subplots_adjust(bottom=0.25)
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None

    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Topology helpers
# =============================================================================

def _stable_rad(source: str, target: str) -> float:
    a, b = (source, target) if source < target else (target, source)
    h = (hash(a) ^ (hash(b) << 1)) & 0xFFFFFFFF
    r = ((h / 0xFFFFFFFF) * 0.6) - 0.30
    if abs(r) < 0.07:
        r = 0.10 if r >= 0 else -0.10
    return float(r)


def _draw_undirected_curve(
    ax: "plt.Axes",
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    *,
    rad: float,
    lw: float,
    alpha: float,
    color: str = "black",
    shrinkA: float = 0.0,
    shrinkB: float = 0.0,
    zorder: int = 5,
) -> None:
    patch = FancyArrowPatch(
        p0, p1,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _compute_group_centroids(
    umap: np.ndarray,
    group_labels: pd.Series,
    node_names: List[str],
) -> Dict[str, Tuple[float, float]]:
    labels_str = group_labels.astype("string").fillna("NA")
    pos: Dict[str, Tuple[float, float]] = {}
    for name in node_names:
        mask = (labels_str == name).values
        if mask.sum() == 0:
            continue
        pos[name] = (float(umap[mask, 0].mean()), float(umap[mask, 1].mean()))
    return pos


def _infer_graph_node_mode(
    adata: "ad.AnnData",
    edge_list: pd.DataFrame,
    group_key: Optional[str],
    umap_key: str = "X_umap",
    umap_fixed: Optional[np.ndarray] = None,
) -> Tuple[str, Dict[str, Tuple[float, float]], Optional[str]]:
    nodes = pd.unique(
        pd.concat(
            [edge_list["source"].astype(str), edge_list["target"].astype(str)],
            ignore_index=True,
        )
    )
    nodes = [str(x) for x in nodes if x is not None]
    if umap_fixed is not None:
        umap = umap_fixed
    else:
        umap, err = _get_umap(adata, umap_key)
        if umap is None:
            return "none", {}, f"missing umap: {err}"
    obs_set = set(map(str, adata.obs_names))
    in_obs = sum((n in obs_set) for n in nodes)
    if in_obs / max(1, len(nodes)) >= 0.85:
        idx = {str(k): i for i, k in enumerate(map(str, adata.obs_names))}
        pos = {n: (float(umap[idx[n], 0]), float(umap[idx[n], 1]))
               for n in nodes if n in idx}
        return "cell", pos, None
    if group_key and group_key in adata.obs.columns:
        groups = adata.obs[group_key].astype("string").fillna("NA")
        df = pd.DataFrame({"g": groups.values, "x": umap[:, 0], "y": umap[:, 1]})
        cent = df.groupby("g")[["x", "y"]].mean()
        pos = {
            str(g): (float(cent.loc[g, "x"]), float(cent.loc[g, "y"]))
            for g in cent.index.tolist()
        }
        pos = {n: pos[n] for n in nodes if n in pos}
        if pos:
            return "group", pos, None
    return "none", {}, "nodes not mappable to obs_names or group_key"


# =============================================================================
# Helper — build canonical node_to_id mapping
# =============================================================================

def _build_node_to_id(
    nodes_present: List[str],
    canonical_node_order: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Assign integer IDs to topology nodes in a stable, cross-method order.

    Parameters
    ----------
    nodes_present : list of str
        All node names that actually appear in this method's edge_list.
    canonical_node_order : list of str, optional
        The task-level canonical group ordering (from ``_canonical_group_order``).
        Nodes from this list are assigned IDs 1, 2, 3, … in order.
        Nodes present in *nodes_present* but absent from *canonical_node_order*
        (e.g. intermediate waypoints added by a method) are appended after the
        canonical set, in sorted order.

    Returns
    -------
    dict mapping node name → integer ID (1-based).
    """
    if canonical_node_order:
        # IDs 1..k for canonical groups that appear in this edge_list.
        ordered = [n for n in canonical_node_order if n in set(nodes_present)]
        # Remaining nodes (method-specific waypoints / unlabelled clusters)
        # sorted for determinism.
        remaining = sorted(n for n in nodes_present if n not in set(ordered))
        full_order = ordered + remaining
    else:
        # Fallback: alphabetical
        full_order = sorted(nodes_present)

    return {name: idx + 1 for idx, name in enumerate(full_order)}


# =============================================================================
# Plot Type 1: Topology on embedding
# =============================================================================

def _plot_topology_on_embedding(
    adata: "ad.AnnData",
    *,
    umap_key: str,
    group_key: Optional[str],
    edge_list: pd.DataFrame,
    title: str,
    out_path: Path,
    caption: Optional[str] = None,
    max_edges: int = 500,
    umap_fixed: Optional[np.ndarray] = None,
    canonical_node_order: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Topology overlaid on UMAP with canonical, cross-method node IDs.

    ``canonical_node_order`` is the task-level group ordering computed once in
    ``generate_plot_suite``.  Passing it here ensures that, e.g., node 3 always
    refers to the same cell type regardless of which method produced the graph.
    """
    try:
        if umap_fixed is not None:
            umap = umap_fixed
        else:
            umap, err = _get_umap(adata, umap_key)
            if umap is None:
                return "skipped", err
        if edge_list is None or edge_list.empty:
            return "skipped", "empty edge_list"
        df = edge_list.copy()
        if not {"source", "target"}.issubset(df.columns):
            return "skipped", "edge_list missing source/target"
        df["source"] = df["source"].astype(str)
        df["target"] = df["target"].astype(str)
        if "weight" in df.columns:
            w = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(float)
        else:
            w = pd.Series(np.ones(len(df), dtype=float))
        if len(df) > max_edges:
            keep = np.argsort(-w.values)[:max_edges]
            df = df.iloc[keep].reset_index(drop=True)
            w = w.iloc[keep].reset_index(drop=True)
        all_nodes = sorted(set(df["source"].tolist()) | set(df["target"].tolist()))
        if group_key and group_key in adata.obs.columns:
            pos = _compute_group_centroids(umap, adata.obs[group_key], all_nodes)
        else:
            obs_set = set(map(str, adata.obs_names))
            idx_map = {str(k): i for i, k in enumerate(adata.obs_names)}
            pos = {
                n: (float(umap[idx_map[n], 0]), float(umap[idx_map[n], 1]))
                for n in all_nodes if n in obs_set
            }
        if not pos:
            return "skipped", "could not map any node names to UMAP positions"

        # ── Canonical node-to-ID (cross-method stable) ────────────────────
        nodes_with_pos = list(pos.keys())
        node_to_id = _build_node_to_id(nodes_with_pos, canonical_node_order)
        node_list = sorted(pos.keys(), key=lambda n: node_to_id.get(n, 99999))

        with _with_rc_context():
            fig, ax = plt.subplots(figsize=(8.5, 5.8))
            fig.subplots_adjust(right=0.72)
            if group_key and group_key in adata.obs.columns:
                labs = adata.obs[group_key].astype("string").fillna("NA")
                cats = sorted(labs.unique().tolist())
                palette = _categorical_palette(len(cats))
                cmap_cat = {c: palette[i] for i, c in enumerate(cats)}
                colors = labs.map(cmap_cat).tolist()
                ax.scatter(umap[:, 0], umap[:, 1], s=22, c=colors,
                           linewidths=0.25, edgecolors="black", alpha=0.90, zorder=1)
            else:
                ax.scatter(umap[:, 0], umap[:, 1], s=18,
                           c=[(0.82, 0.82, 0.82, 1.0)],
                           linewidths=0.15, edgecolors="black", alpha=0.85, zorder=1)
            wv = w.values.astype(float)
            finite_w = wv[np.isfinite(wv)]
            wmax = float(np.max(finite_w)) if finite_w.size else 1.0
            wmax = max(wmax, 1e-12)
            edges_to_draw = [
                (s, t, float(ww))
                for s, t, ww in zip(df["source"].tolist(), df["target"].tolist(), wv.tolist())
                if s in pos and t in pos and s != t
            ]
            for s, t, ww in edges_to_draw:
                lw_core = 0.8 + 0.9 * ww / wmax
                _draw_undirected_curve(ax, pos[s], pos[t],
                                       rad=_stable_rad(s, t), lw=lw_core + 1.4,
                                       alpha=0.90, color="white", zorder=4)
            for s, t, ww in edges_to_draw:
                lw_core = 0.8 + 0.9 * ww / wmax
                _draw_undirected_curve(ax, pos[s], pos[t],
                                       rad=_stable_rad(s, t), lw=lw_core,
                                       alpha=0.85, color="#1a1a1a", zorder=5)
            node_xy = np.asarray([pos[n] for n in node_list], dtype=float)
            ax.scatter(node_xy[:, 0], node_xy[:, 1], s=180, c="white",
                       edgecolors="#1a1a1a", linewidths=1.0, zorder=7)
            for n in node_list:
                x, y = pos[n]
                ax.text(x, y, str(node_to_id[n]),
                        fontsize=7, fontweight="normal",
                        ha="center", va="center", color="#1a1a1a", zorder=8)
            lines = [f"{node_to_id[n]}.  {_wrap_label(n, width=28)}" for n in node_list]
            fig.text(0.74, 0.50, "\n\n".join(lines),
                     ha="left", va="center", fontsize=8.5, fontweight="normal")
            ax.set_title(title, fontweight="normal")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal", adjustable="datalim")
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None
    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Plot Type 4: Abstract topology graph
# =============================================================================

def _plot_topology_graph_curved(
    edge_list: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    caption: Optional[str] = None,
    seed: int = 0,
    max_edges: int = 250,
    canonical_node_order: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Abstract topology graph with canonical, cross-method node IDs.

    ``canonical_node_order`` is the task-level group ordering.  Passing it here
    ensures the integer label inside every node circle refers to the same cell
    type regardless of method.
    """
    if nx is None:
        return "skipped", "networkx not installed"
    try:
        if edge_list is None or edge_list.empty:
            return "skipped", "empty edge_list"
        if not {"source", "target"}.issubset(edge_list.columns):
            return "skipped", "edge_list missing source/target"
        df = edge_list.copy()
        df["source"] = df["source"].astype(str)
        df["target"] = df["target"].astype(str)
        if "weight" in df.columns:
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(float)
        else:
            df["weight"] = 1.0
        if len(df) > max_edges:
            df = df.sort_values("weight", ascending=False).head(max_edges).copy()
        G = nx.Graph()
        for _, r in df.iterrows():
            s, t = str(r["source"]), str(r["target"])
            if s == t:
                continue
            if G.has_edge(s, t):
                G[s][t]["weight"] = max(G[s][t]["weight"], float(r["weight"]))
            else:
                G.add_edge(s, t, weight=float(r["weight"]))
        if G.number_of_nodes() == 0:
            return "skipped", "graph has 0 nodes"

        # ── Canonical node-to-ID (cross-method stable) ────────────────────
        node_list = list(G.nodes())
        node_to_id = _build_node_to_id(node_list, canonical_node_order)
        # Re-order node_list so the legend follows canonical ID order.
        node_list = sorted(node_list, key=lambda n: node_to_id.get(n, 99999))

        n = G.number_of_nodes()
        k_val = 3.5 / math.sqrt(max(1, n))
        pos = nx.spring_layout(G, seed=seed, k=k_val, iterations=800, scale=3.0)
        deg = dict(G.degree())
        deg_vals = np.asarray([deg.get(v, 0) for v in node_list], dtype=float)
        if deg_vals.size and deg_vals.max() > deg_vals.min():
            sizes = 600 + 700 * (deg_vals - deg_vals.min()) / (deg_vals.max() - deg_vals.min())
        else:
            sizes = np.full(len(node_list), 800.0)
        edge_weights = np.asarray(
            [G.edges[e].get("weight", 1.0) for e in G.edges()], dtype=float
        )
        edge_weights = edge_weights[np.isfinite(edge_weights)]
        wmax = float(np.max(edge_weights)) if edge_weights.size else 1.0
        wmax = max(wmax, 1e-12)
        fig_w = max(9.5, 1.1 * n)
        fig_h = max(6.5, 0.7 * n)
        with _with_rc_context():
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            fig.subplots_adjust(right=0.70)
            for (s, t) in G.edges():
                p0 = (float(pos[s][0]), float(pos[s][1]))
                p1 = (float(pos[t][0]), float(pos[t][1]))
                ww = float(G.edges[(s, t)].get("weight", 1.0))
                lw = 0.9 + 2.0 * ww / wmax
                _draw_undirected_curve(ax, p0, p1,
                                       rad=0.55 * _stable_rad(str(s), str(t)),
                                       lw=lw, alpha=0.88, color="black",
                                       shrinkA=16.0, shrinkB=16.0, zorder=2)
            xy = np.asarray([pos[v] for v in node_list], dtype=float)
            ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c="white",
                       edgecolors="black", linewidths=1.2, zorder=3)
            for v in node_list:
                x, y = pos[v]
                ax.text(x, y, str(node_to_id[v]),
                        fontsize=10, fontweight="normal",
                        ha="center", va="center", zorder=4)
            lines = [f"{node_to_id[v]}.  {_wrap_label(v, width=30)}" for v in node_list]
            fig.text(0.72, 0.50, "\n\n".join(lines),
                     ha="left", va="center", fontsize=9, fontweight="normal")
            ax.set_title(title, fontweight="normal")
            ax.set_axis_off()
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None
    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Topology matrix heatmap
# =============================================================================

def _plot_topology_matrix_heatmap(
    topology: Any,
    out_path: Path,
    *,
    title: str = "Topology adjacency matrix",
    caption: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    try:
        if topology is None:
            return "skipped", "no topology_matrix"
        if isinstance(topology, pd.DataFrame):
            mat = topology.to_numpy(dtype=float, copy=False)
            labels = list(map(str, topology.index.tolist()))
        else:
            arr = np.asarray(topology)
            if arr.ndim != 2:
                return "skipped", "topology_matrix is not 2D"
            mat = arr.astype(float)
            labels = [str(i) for i in range(mat.shape[0])]
        if mat.size == 0:
            return "skipped", "empty matrix"
        with _with_rc_context():
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            im = ax.imshow(mat, aspect="auto", cmap="Blues")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("edge weight")
            ax.set_title(title)
            if len(labels) <= 40:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_yticklabels(labels)
            _save_fig_with_caption(fig, out_path, caption)
        return "ok", None
    except Exception as e:
        plt.close("all")
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Caption file writer
# =============================================================================

def _write_captions_file(
    captions: Dict[str, str],
    figures_dir: Path,
) -> Tuple[str, Optional[str]]:
    if not captions:
        return "skipped", "no captions to write"
    try:
        out_path = figures_dir / "figure_captions.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sep = "-" * 72
        lines: List[str] = [
            "Figure Captions",
            "=" * 72,
            "Generated by TI_benchmark/plotting.py",
            "",
        ]
        for i, (fname, cap) in enumerate(captions.items(), start=1):
            lines.append(f"Figure {i}: {fname}")
            lines.append(sep)
            wrapped = "\n".join(
                textwrap.fill(sentence.strip(), width=80)
                for sentence in cap.splitlines()
                if sentence.strip()
            )
            lines.append(wrapped)
            lines.append("")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return "ok", None
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"


# =============================================================================
# Public: generate_plot_suite
# =============================================================================

def generate_plot_suite(
    adata: "ad.AnnData",
    *,
    run_dir: Path,
    figures_dir: Path,
    umap_key: str,
    group_key: Optional[str],
    color_keys: List[str],
    ti: Optional[TIOutput],
    stability_manifest: Optional[Dict[str, Any]],
    run_config_path: Path,
    canonical_group_order: Optional[List[str]] = None,
) -> None:
    """
    Central plotting hook. Never raises; logs all results to run_config.json.

    Canonical ordering
    ------------------
    ``canonical_group_order`` fixes the cell-type order used in:

    * Pseudotime-by-group boxplot x-axis
    * Topology graph / topology-on-embedding node integer IDs

    **Priority** (highest → lowest):

    1. *Explicit*: caller passes ``canonical_group_order`` (preferred).
       In ``method_runner.run_benchmark`` this is set to ``spec.include_values``
       — the comma-separated list from the shell script's ``INCLUDE_VALUES`` /
       ``TASK_INCLUDE``.  That list already encodes the researcher's intended
       biological ordering (root first, terminal states last) and is identical
       across all methods on the same task.

    2. *Categorical auto-detect*: if ``adata.obs[group_key]`` is a
       ``pandas.Categorical`` and its ``.cat.categories`` order was set by the
       dataset loader, use that.

    3. *Alphabetical fallback*: ``sorted(unique_values)``.

    Results (pseudotime values, edge weights, metrics) are completely unaffected.
    """
    plot_log: Dict[str, Any] = {"status": "ok", "written": [], "skipped": [], "errors": []}
    captions: Dict[str, str] = {}

    umap_fixed, umap_err = _capture_shared_umap(adata, umap_key)
    if umap_fixed is None:
        plot_log["status"] = "error"
        plot_log["errors"].append({"plot": "umap_required", "error": umap_err})
        merge_json_shallow(run_config_path, {"plotting": plot_log})
        return

    merge_json_shallow(run_config_path, {
        "plotting": {
            "umap_fixed": {
                "shape": list(umap_fixed.shape),
                "x_range": [float(umap_fixed[:, 0].min()), float(umap_fixed[:, 0].max())],
                "y_range": [float(umap_fixed[:, 1].min()), float(umap_fixed[:, 1].max())],
                "note": "Captured before TI execution; used for all plots.",
            }
        }
    })

    method_name = (
        getattr(ti, "method_name", None) or "the trajectory inference method"
        if ti is not None else "the trajectory inference method"
    )
    _warn_if_umap_changed(adata, umap_key, umap_fixed,
                          context=f"post-TI [{method_name}]")

    # ── Canonical group order — single source of truth for all plot calls ────
    # Priority:
    #   1. Explicit caller-supplied list (spec.include_values from the shell
    #      script INCLUDE_VALUES / TASK_INCLUDE) → biological order, root first.
    #   2. pandas Categorical .cat.categories (dataset-loader order).
    #   3. sorted(unique) alphabetical fallback.
    _order_source: str
    if canonical_group_order:
        canonical_group_order = [str(g).strip() for g in canonical_group_order if str(g).strip()]
        _order_source = "explicit_include_values"
    elif group_key and group_key in adata.obs.columns:
        canonical_group_order = _canonical_group_order(adata.obs[group_key])
        _order_source = (
            "cat.categories"
            if hasattr(adata.obs[group_key], "cat")
            else "sorted_unique"
        )
    else:
        canonical_group_order = []
        _order_source = "none"

    merge_json_shallow(run_config_path, {
        "plotting": {
            "canonical_group_order": {
                "group_key": group_key,
                "order": canonical_group_order,
                "n_groups": len(canonical_group_order),
                "source": _order_source,
            }
        }
    })

    cap: Optional[str] = None

    # UMAP color keys
    keys_to_plot: List[str] = []
    if group_key:
        keys_to_plot.append(group_key)
    for k in (color_keys or []):
        if k and k not in keys_to_plot:
            keys_to_plot.append(k)

    for k in keys_to_plot:
        out = figures_dir / f"umap_by_{k}.png"
        cap = None
        try:
            if k in adata.obs.columns:
                n_cats = int(adata.obs[k].nunique(dropna=True))
                cap = _cap_umap_categorical(k, n_cats)
                st, msg = _plot_embedding_categorical(
                    umap_fixed, adata.obs[k],
                    title=f"UMAP coloured by {k}",
                    out_path=out, caption=cap,
                )
            elif k in adata.var_names:
                x = adata[:, k].X
                if hasattr(x, "toarray"):
                    x = x.toarray()
                x = np.asarray(x, dtype=float).reshape(-1)
                x_norm, _ = _normalize_01(x)
                cap = _cap_umap_gene(k)
                st, msg = _plot_embedding_continuous(
                    umap_fixed, x_norm,
                    title=f"UMAP — expression of {k} (normalised)",
                    out_path=out, caption=cap,
                    cmap="viridis", vmin=0.0, vmax=1.0,
                    cbar_label=f"{k} (norm)",
                )
            else:
                st, msg = "skipped", f"key '{k}' not in obs or var"
        except Exception as e:
            st, msg = "error", f"{type(e).__name__}: {e}"
        _log_plot(plot_log, st, msg, str(out), f"umap_by_{k}",
                  captions=captions, caption=cap if st == "ok" else None)

    # TI-dependent pseudotime plots
    pt_obs_key = "ti_pseudotime"
    has_pseudotime = ti is not None and isinstance(
        getattr(ti, "pseudotime", None), pd.Series
    )

    if has_pseudotime:
        try:
            pt = _as_numeric_series(ti.pseudotime, adata.obs_names)
            adata.obs[pt_obs_key] = pt
        except Exception:
            pass

        # Plot Type 2: pseudotime on embedding
        try:
            pt_vals = pd.to_numeric(
                adata.obs.get(pt_obs_key), errors="coerce"
            ).astype(float).values
            pt_norm, nmsg = _normalize_01(pt_vals)
            out = figures_dir / "pseudotime_on_embedding.png"
            cap = _cap_pseudotime_embedding(method_name)
            st, msg = _plot_embedding_continuous(
                umap_fixed, pt_norm,
                title=f"Pseudotime on embedding — {method_name}",
                out_path=out, caption=cap,
                cmap="viridis", vmin=0.0, vmax=1.0,
                cbar_label="pseudotime (0-1)",
            )
            _log_plot(plot_log, st, msg or nmsg, str(out),
                      "pseudotime_on_embedding", captions=captions, caption=cap)
        except Exception as e:
            plot_log["errors"].append({"plot": "pseudotime_on_embedding",
                                       "error": f"{type(e).__name__}: {e}"})

        # Pseudotime histogram
        out = figures_dir / "pseudotime_distribution.png"
        cap = _cap_pseudotime_hist(method_name)
        st, msg = _plot_hist(
            pd.to_numeric(
                adata.obs.get(pt_obs_key), errors="coerce"
            ).astype(float).values,
            title=f"Pseudotime distribution — {method_name}",
            xlabel="pseudotime",
            out_path=out, caption=cap,
        )
        _log_plot(plot_log, st, msg, str(out), "pseudotime_distribution",
                  captions=captions, caption=cap)

        # Plot Type 3: pseudotime by group (vertical violin+box, canonical x-order)
        pt_group_key: Optional[str] = None
        if group_key and group_key in adata.obs.columns:
            pt_group_key = group_key
        else:
            for k in keys_to_plot:
                if not k or k not in adata.obs.columns:
                    continue
                n_unique = int(pd.Series(adata.obs[k]).nunique(dropna=True))
                if 2 <= n_unique <= 60:
                    pt_group_key = k
                    break

        if pt_group_key:
            grp = adata.obs[pt_group_key]
            n_groups = int(grp.nunique(dropna=True))
            cap = _cap_pseudotime_by_group(pt_group_key, method_name, n_groups)
            out = figures_dir / "pseudotime_by_group.png"
            pt_raw = pd.to_numeric(
                adata.obs.get(pt_obs_key), errors="coerce"
            ).astype(float)
            pt_norm_grp, _ = _normalize_01(pt_raw.values)
            pt_norm_series = pd.Series(pt_norm_grp, index=adata.obs_names)
            # Use canonical_group_order for this key; fall back to the global
            # canonical list only if pt_group_key == group_key.
            order_for_plot = (
                canonical_group_order
                if pt_group_key == group_key
                else _canonical_group_order(adata.obs[pt_group_key])
            )
            st, msg = _plot_boxplot_by_group(
                grp, pt_norm_series,
                title=f"Pseudotime by {pt_group_key} — {method_name}",
                ylabel="Pseudotime",
                out_path=out, caption=cap,
                orientation="vertical", wrap_label_width=30,
                group_order=order_for_plot,   # ← canonical order injected here
            )
            _log_plot(plot_log, st, msg, str(out), "pseudotime_by_group",
                      captions=captions, caption=cap)

        # Branch labels
        if getattr(ti, "branch_labels", None) is not None:
            try:
                bl_key = "ti_branch_labels"
                adata.obs[bl_key] = ti.branch_labels.reindex(
                    adata.obs_names
                ).astype("string")
                out = figures_dir / "branch_labels_umap.png"
                cap = _cap_branch_labels(method_name)
                st, msg = _plot_embedding_categorical(
                    umap_fixed, adata.obs[bl_key],
                    title=f"Branch labels — {method_name}",
                    out_path=out, caption=cap, legend_max=25,
                )
                _log_plot(plot_log, st, msg, str(out), "branch_labels_umap",
                          captions=captions, caption=cap)
            except Exception as e:
                plot_log["errors"].append({"plot": "branch_labels_umap",
                                           "error": f"{type(e).__name__}: {e}"})

        # Terminal probabilities
        tp = getattr(ti, "terminal_probabilities", None)
        if isinstance(tp, pd.DataFrame) and tp.shape[1] > 0:
            for c in list(tp.columns[:12]):
                try:
                    vals = pd.to_numeric(
                        tp[c].reindex(adata.obs_names), errors="coerce"
                    ).astype(float).values
                    vals_norm, _ = _normalize_01(vals)
                    out = figures_dir / f"terminal_prob_{c}.png"
                    cap = _cap_terminal_prob(str(c), method_name)
                    st, msg = _plot_embedding_continuous(
                        umap_fixed, vals_norm,
                        title=f"Terminal probability: {c} — {method_name}",
                        out_path=out, caption=cap,
                        cmap="viridis", vmin=0.0, vmax=1.0,
                        cbar_label="probability (0-1)",
                    )
                    _log_plot(plot_log, st, msg, str(out), f"terminal_prob_{c}",
                              captions=captions, caption=cap)
                except Exception as e:
                    plot_log["errors"].append(
                        {"plot": f"terminal_prob_{c}",
                         "error": f"{type(e).__name__}: {e}"}
                    )

    # Topology plots — canonical_node_order passed to both topology functions
    if ti is not None:
        edge_list = getattr(ti, "edge_list", None)
        if edge_list is not None:
            out = figures_dir / "trajectory_topology_on_embedding.png"
            cap = _cap_topology_on_embedding(method_name)
            st, msg = _plot_topology_on_embedding(
                adata,
                umap_key=umap_key,
                group_key=group_key,
                edge_list=edge_list,
                title=f"Trajectory topology on embedding — {method_name}",
                out_path=out, caption=cap, max_edges=600,
                umap_fixed=umap_fixed,
                canonical_node_order=canonical_group_order,  # ← canonical IDs
            )
            _log_plot(plot_log, st, msg, str(out),
                      "trajectory_topology_on_embedding",
                      captions=captions, caption=cap)

            out = figures_dir / "topology_graph_curved.png"
            cap = _cap_topology_graph(method_name)
            st, msg = _plot_topology_graph_curved(
                edge_list,
                title=f"Topology graph — {method_name}",
                out_path=out, caption=cap, seed=0, max_edges=250,
                canonical_node_order=canonical_group_order,  # ← canonical IDs
            )
            _log_plot(plot_log, st, msg, str(out), "topology_graph_curved",
                      captions=captions, caption=cap)

        topology_matrix = getattr(ti, "topology_matrix", None)
        if topology_matrix is not None:
            out = figures_dir / "topology_matrix_heatmap.png"
            cap = _cap_topology_heatmap(method_name)
            st, msg = _plot_topology_matrix_heatmap(
                topology_matrix, out,
                title=f"Topology adjacency matrix — {method_name}",
                caption=cap,
            )
            _log_plot(plot_log, st, msg, str(out), "topology_matrix_heatmap",
                      captions=captions, caption=cap)

    # Stability histograms
    if isinstance(stability_manifest, dict):
        _pw = stability_manifest.get("pairwise_scores")
        if not isinstance(_pw, dict):
            _pw = stability_manifest

        abs_rhos = (
            _pw.get("pairwise_pseudotime_spearman_abs")
            or _pw.get("pseudotime_spearman_abs_values")
            or _pw.get("spearman_abs_values")
            or stability_manifest.get("pairwise_pseudotime_spearman_abs")
        )
        jacs = (
            _pw.get("pairwise_edge_jaccard")
            or _pw.get("edge_jaccard_values")
            or _pw.get("jaccard_values")
            or stability_manifest.get("pairwise_edge_jaccard")
        )

        def _n_from_pairs(pairs: int) -> int:
            return max(2, int(round((1 + math.sqrt(1 + 8 * pairs)) / 2)))

        if isinstance(abs_rhos, list) and len(abs_rhos) > 0:
            n_rep = _n_from_pairs(len(abs_rhos))
            out = figures_dir / "stability_hist_pseudotime_spearman_abs.png"
            cap = _cap_stability_spearman(n_rep)
            st, msg = _plot_stability_hist(
                [float(x) for x in abs_rhos if x is not None],
                title=f"Bootstrap stability: pseudotime Spearman |rho| — {method_name}",
                xlabel="Spearman |rho| (cell intersection across replicates)",
                out_path=out, caption=cap,
                x_fixed_range=(0.0, 1.0), n_bins=50,
                bar_color="#4c72b0",
            )
            _log_plot(plot_log, st, msg, str(out),
                      "stability_hist_pseudotime_spearman_abs",
                      captions=captions, caption=cap)

        if isinstance(jacs, list) and len(jacs) > 0:
            n_rep = _n_from_pairs(len(jacs))
            out = figures_dir / "stability_hist_edge_jaccard.png"
            cap = _cap_stability_jaccard(n_rep)
            st, msg = _plot_stability_hist(
                [float(x) for x in jacs if x is not None],
                title=f"Bootstrap stability: edge Jaccard — {method_name}",
                xlabel="Jaccard index (edges restricted to common nodes)",
                out_path=out, caption=cap,
                x_fixed_range=(0.0, 1.0), n_bins=50,
                bar_color="#4c72b0",
                kde_color="#1a3a6b",
            )
            _log_plot(plot_log, st, msg, str(out), "stability_hist_edge_jaccard",
                      captions=captions, caption=cap)

    # Caption file
    cap_status, cap_err = _write_captions_file(captions, figures_dir)
    if cap_status == "ok":
        plot_log["captions_file"] = str(figures_dir / "figure_captions.txt")
    else:
        plot_log["captions_file_error"] = cap_err
    merge_json_shallow(run_config_path, {"plotting": plot_log})


# =============================================================================
# Internal logging helper
# =============================================================================

def _log_plot(
    log: Dict[str, Any],
    status: str,
    message: Optional[str],
    path: str,
    name: str,
    captions: Optional[Dict[str, str]] = None,
    caption: Optional[str] = None,
) -> None:
    if status == "ok":
        log["written"].append(path)
        if captions is not None and caption:
            fname = Path(path).name
            captions[fname] = caption
    elif status == "skipped":
        log["skipped"].append({"plot": name, "reason": message})
    else:
        log["errors"].append({"plot": name, "error": message})

__all__ = ["generate_plot_suite"]