import os
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import scanpy as sc


# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------
def set_pub_style():
    sc.settings.set_figure_params(
        dpi=200,
        facecolor="white",
        frameon=False,
        fontsize=10,
        vector_friendly=True,
    )
    plt.rcParams["figure.figsize"] = (6.8, 5.6)
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.transparent"] = False


_UMAP_RC = {
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


def _with_umap_rc_context():
    return plt.rc_context(_UMAP_RC)


def _rasterize(fig):
    try:
        for ax in getattr(fig, "axes", []):
            for col in getattr(ax, "collections", []):
                try:
                    col.set_rasterized(True)
                except Exception:
                    pass
    except Exception:
        pass


def save_pubfig(fig, out_png: str, out_pdf: str, dpi_png: int = 450):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    _rasterize(fig)
    fig.savefig(out_png, dpi=dpi_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Fixed color maps for publication consistency
# -----------------------------------------------------------------------------
# Keep dataset colors stable across all methods, even when some datasets are
# absent from a specific result (e.g. LIGER dropping a zero-count dataset).
FIXED_DATASET_COLORS: Dict[str, str] = {
    "GSE115469": "#BD0319",  # strong red
    "GSE124395": "#0011FF",  # dark blue
    "GSE125449": "#7CFC00",  # neon green
    "GSE136103": "#FF7A00",  # orange
    "GSE138709": "#FFD500",  # yellow
    "GSE140228": "#00E5FF",  # cyan
    "GSE146409": "#FF00E5",  # magenta
    "GSE149614": "#731EEB",  # purple
    "GSE151530": "#042A68",  # deep blue
}


def _fixed_color_map_for_labels(labels: pd.Series, legend_title: str) -> Optional[Dict[str, str]]:
    """
    Return a fixed category->color mapping when the plotted variable has a
    benchmark-wide canonical color assignment.

    Currently this is enforced for dataset-level plots so colors remain stable
    even when one or more datasets are absent from a given method output.
    """
    labs = labels.astype("string").fillna("NA")
    cats = sorted(labs.unique().tolist())

    if str(legend_title).strip().lower() == "dataset":
        color_map: Dict[str, str] = {}
        fallback_palette = _categorical_palette(max(len(cats), 1))
        fallback_iter = iter(fallback_palette)

        for cat in cats:
            if cat in FIXED_DATASET_COLORS:
                color_map[cat] = FIXED_DATASET_COLORS[cat]
            else:
                color_map[cat] = next(fallback_iter)
        return color_map

    return None


# -----------------------------------------------------------------------------
# General plotting utilities
# -----------------------------------------------------------------------------
def subsample_for_plotting(ad, n: int, seed: int = 0, stratify_by: Optional[str] = None):
    """Slice AnnData safely (avoids numpy-index IndexError)."""
    if n <= 0 or ad.n_obs <= n:
        return ad
    rng = np.random.default_rng(seed)

    if stratify_by is None or stratify_by not in ad.obs:
        idx = rng.choice(ad.n_obs, size=n, replace=False)
        return ad[idx].copy()

    groups = ad.obs[stratify_by].astype(str).values
    uniq, counts = np.unique(groups, return_counts=True)
    props = counts / counts.sum()
    take_per = np.maximum(1, (props * n).astype(int))

    picked = []
    for u, k in zip(uniq, take_per):
        idx_u = np.where(groups == u)[0]
        k = min(k, idx_u.size)
        picked.append(rng.choice(idx_u, size=k, replace=False))

    idx = np.unique(np.concatenate(picked))
    if idx.size > n:
        idx = rng.choice(idx, size=n, replace=False)
    return ad[idx].copy()


# -----------------------------------------------------------------------------
# UMAP + marker dotplot
# -----------------------------------------------------------------------------
def _categorical_palette(n: int):
    """
    High-saturation 'phosphoric' palette for scRNA UMAPs.
    Inspired by scanpy + napari + cytometry palettes.
    """
    base = [
        "#BD0319",  # strong red
        "#0011FF",  # dark blue
        "#7CFC00",  # neon green
        "#FF7A00",  # orange
        "#FFD500",  # yellow
        "#00E5FF",  # cyan
        "#FF00E5",  # magenta
        "#731EEB",  # purple
        "#042A68",  # kuhly blue
        "#0094FF",  # blue
        "#A998FF",  # violet
        "#FF3B3B",  # red
        "#00FF85",  # mint
        "#F178B4",  # pink
        "#00FFD0",  # aqua
        "#FF9F1C",  # warm orange
        "#00BBA9",  # teal
        "#FF006E",  # neon pink
    ]

    if n <= len(base):
        return base[:n]

    cmap = plt.get_cmap("hsv")
    return [cmap(i / n) for i in range(n)]


def _get_umap_coords(ad, umap_key: str) -> np.ndarray:
    if umap_key not in ad.obsm:
        raise ValueError(f"umap_key='{umap_key}' not found in ad.obsm")
    arr = np.asarray(ad.obsm[umap_key])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"ad.obsm['{umap_key}'] is not shaped (n_obs, 2+)")
    return arr[:, :2].astype(float, copy=False)


def _is_categorical_obs(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def _looks_like_integration_key(color: str) -> bool:
    c = str(color).strip().lower()
    tokens = {
        "dataset", "batch", "sample", "donor", "patient", "study",
        "origin", "source", "library", "site", "cohort", "technology",
        "platform", "slide", "pool"
    }
    if c in tokens:
        return True
    return any(tok in c for tok in tokens)


def _umap_point_style_integration(n_obs: int, integration_view: bool) -> Tuple[float, float]:
    """
    Tuned for large integration maps (~300k cells):
    - very small points
    - no marker edge
    - lower alpha for batch/dataset views to avoid false segregation
    """
    return (1.10, 0.3) if integration_view else (1.10, 0.26)


def _format_umap_axes(ax, title: str):
    ax.set_title(title, pad=10, fontsize=14, fontweight="normal")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.015)
    ax.tick_params(axis="both", which="major", length=4.0, width=0.8, labelsize=9)
    ax.grid(False)


def _random_plot_order(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


def _plot_umap_categorical(
    ax,
    umap: np.ndarray,
    labels: pd.Series,
    *,
    title: str,
    legend_title: str,
    legend_max: int = 35,
    seed: int = 0,
) -> bool:
    labs = labels.astype("string").fillna("NA")
    cats = sorted(labs.unique().tolist())

    fixed_map = _fixed_color_map_for_labels(labels, legend_title)
    if fixed_map is not None:
        color_map = fixed_map
    else:
        palette = _categorical_palette(len(cats))
        color_map = {c: palette[i] for i, c in enumerate(cats)}

    colors = np.asarray(labs.map(color_map).tolist(), dtype=object)

    integration_view = _looks_like_integration_key(legend_title)
    perm = _random_plot_order(umap.shape[0], seed=seed)
    umap_plot = umap[perm]
    colors_plot = colors[perm]
    s, alpha = _umap_point_style_integration(umap.shape[0], integration_view)

    ax.scatter(
        umap_plot[:, 0],
        umap_plot[:, 1],
        s=s,
        c=colors_plot.tolist(),
        linewidths=0.0,
        edgecolors="none",
        alpha=alpha,
        rasterized=True,
    )
    _format_umap_axes(ax, title)

    if len(cats) <= legend_max:
        handles = [
            ax.scatter(
                [], [],
                s=55,
                c=[color_map[c]],
                edgecolors="none",
                linewidths=0.0,
                label=str(c),
                alpha=0.95,
            )
            for c in cats
        ]
        ax.legend(
            handles=handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            title=legend_title,
            borderaxespad=0.0,
            handletextpad=0.35,
            labelspacing=0.35,
            scatterpoints=1,
        )
        return True
    return False


def _plot_umap_continuous(
    fig,
    ax,
    umap: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    seed: int = 0,
):
    vals = np.asarray(values, dtype=float).reshape(-1)
    if vals.shape[0] != umap.shape[0]:
        raise ValueError("continuous colour vector length does not match n_obs")

    perm = _random_plot_order(umap.shape[0], seed=seed)
    umap_plot = umap[perm]
    vals_plot = vals[perm]

    integration_view = _looks_like_integration_key(cbar_label)
    s, alpha = _umap_point_style_integration(umap.shape[0], integration_view)
    finite = np.isfinite(vals_plot)

    ax.scatter(
        umap_plot[:, 0],
        umap_plot[:, 1],
        s=s,
        c="#d9d9d9",
        linewidths=0.0,
        edgecolors="none",
        alpha=min(alpha, 0.10),
        rasterized=True,
    )

    if not finite.any():
        raise ValueError(f"colour '{cbar_label}' has no finite values to plot")

    v = vals_plot[finite]
    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError(f"colour '{cbar_label}' has no finite values to plot")
    if vmax <= vmin:
        vmax = vmin + 1e-12

    norm = Normalize(vmin=vmin, vmax=vmax)
    ax.scatter(
        umap_plot[finite, 0],
        umap_plot[finite, 1],
        s=s,
        c=v,
        cmap=cmap,
        norm=norm,
        linewidths=0.0,
        edgecolors="none",
        alpha=max(alpha, 0.40),
        rasterized=True,
    )

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_label(cbar_label)
    _format_umap_axes(ax, title)


def plot_umap_pub(ad, umap_key: str, color: str, title_prefix: str, outdir: str, alpha: float = 0.75):
    set_pub_style()
    os.makedirs(outdir, exist_ok=True)

    if umap_key not in ad.obsm:
        raise ValueError(f"umap_key='{umap_key}' not found in ad.obsm")

    umap = _get_umap_coords(ad, umap_key)
    ad.obsm["X_umap"] = umap.copy()

    title = f"{title_prefix} — {color.replace('_', ' ')}"

    if color in ad.obs:
        series = ad.obs[color]
        is_cat = _is_categorical_obs(series)
    elif color in ad.var_names:
        series = None
        is_cat = False
    else:
        raise ValueError(f"color='{color}' not found in ad.obs or ad.var_names")

    with _with_umap_rc_context():
        if is_cat:
            labels = ad.obs[color]
            labs = labels.astype("string").fillna("NA")
            n_cat = int(labs.nunique(dropna=False))
            show_legend = n_cat <= 35

            fig, ax = plt.subplots(figsize=(8.8, 5.8) if show_legend else (6.6, 5.6))
            _plot_umap_categorical(
                ax,
                umap,
                labels,
                title=title,
                legend_title=color,
                legend_max=35,
                seed=0,
            )

            if show_legend:
                fig.tight_layout(pad=0.7, rect=(0.0, 0.0, 0.80, 1.0))
            else:
                fig.tight_layout(pad=0.7)

        else:
            if color in ad.obs:
                values = pd.to_numeric(ad.obs[color], errors="coerce").astype(float).values
            else:
                x = ad[:, color].X
                if hasattr(x, "toarray"):
                    x = x.toarray()
                values = np.asarray(x, dtype=float).reshape(-1)

            fig, ax = plt.subplots(figsize=(6.8, 5.8))
            _plot_umap_continuous(
                fig,
                ax,
                umap,
                values,
                title=title,
                cbar_label=color.replace("_", " "),
                cmap="viridis",
                seed=0,
            )
            fig.tight_layout(pad=0.7)

        save_pubfig(
            fig,
            os.path.join(outdir, f"umap_{color}.png"),
            os.path.join(outdir, f"umap_{color}.pdf"),
        )


DEFAULT_LIVER_MARKERS_L1 = {
    "T cell": ["TRAC", "CD3D", "CD3E", "IL7R", "LTB"],
    "NK cell": ["NKG7", "GNLY", "KLRD1", "FCGR3A"],
    "B cell": ["MS4A1", "CD79A", "CD74", "HLA-DRA"],
    "Plasma cell": ["MZB1", "XBP1", "JCHAIN", "SDC1"],
    "Myeloid": ["LYZ", "S100A8", "S100A9", "FCN1", "LGALS3"],
    "Hepatocyte": ["ALB", "APOA1", "TTR", "CYP3A4"],
    "Cholangiocyte": ["KRT19", "KRT8", "KRT18", "EPCAM"],
    "Endothelial": ["PECAM1", "VWF", "KDR", "EMCN"],
    "LSEC": ["CLEC4G", "FCGR2B", "STAB2", "KDR"],
    "Fibroblast/CAF": ["COL1A1", "DCN", "LUM", "COL3A1"],
    "Stellate cell": ["RBP1", "COL3A1", "LUM", "DES"],
    "Pericyte": ["RGS5", "PDGFRB", "CSPG4", "MCAM"],
    "vSMC": ["ACTA2", "TAGLN", "MYH11", "CNN1"],
    "Proliferating": ["MKI67", "TOP2A", "HMGB2", "TYMS"],
    "Erythroid": ["HBB", "HBA1", "HBA2", "ALAS2"],
}


def plot_marker_dotplot_pub(ad, groupby: str, outdir: str, markers: Optional[Dict[str, list]] = None, title: Optional[str] = None):
    set_pub_style()
    os.makedirs(outdir, exist_ok=True)

    markers = DEFAULT_LIVER_MARKERS_L1 if markers is None else markers
    markers_present = {k: [g for g in v if g in ad.var_names] for k, v in markers.items()}
    markers_present = {k: v for k, v in markers_present.items() if len(v) > 0}
    if not markers_present:
        return

    dp = sc.pl.dotplot(
        ad,
        var_names=markers_present,
        groupby=groupby,
        swap_axes=True,
        mean_only_expressed=True,
        standard_scale=None,
        dot_max=0.6,
        dot_min=0.05,
        show=False,
        return_fig=True,
    )
    try:
        dp.make_figure()
    except Exception:
        pass

    if title is None:
        title = f"Marker dotplot — grouped by {groupby.replace('_', ' ')}"

    out_png = os.path.join(outdir, f"dotplot_markers_by_{groupby}.png")
    out_pdf = os.path.join(outdir, f"dotplot_markers_by_{groupby}.pdf")

    if hasattr(dp, "savefig"):
        dp.savefig(out_png, dpi=450)
        dp.savefig(out_pdf)
        plt.close("all")
        return

    fig = getattr(dp, "fig", None) or getattr(dp, "figure", None) or getattr(dp, "_fig", None)
    if fig is None:
        return
    fig.suptitle(title, y=0.98, fontsize=14)
    save_pubfig(fig, out_png, out_pdf)


# -----------------------------------------------------------------------------
# spider / radar plot across methods (means only)
# -----------------------------------------------------------------------------
def plot_radar_means_across_methods(
    summary_df: pd.DataFrame,
    *,
    method_col: str = "method",
    outdir: str,
    title: str = "Integration metrics (means)",
    metrics: Optional[List[str]] = None,
):
    """
    summary_df: one row per method, columns include mean metrics.
    Creates a radar/spider plot (publication-friendly) using means only.
    """
    set_pub_style()
    os.makedirs(outdir, exist_ok=True)

    if metrics is None:
        metrics = ["kBET", "iLISI_batch", "cLISI_label", "batch_ASW", "cell_type_ASW", "graph_connectivity"]

    df = summary_df.copy()
    df = df.dropna(subset=[method_col])
    if df.empty:
        return

    metrics = [m for m in metrics if m in df.columns]
    if len(metrics) < 3:
        return

    M = df[metrics].astype(float)
    mins = np.nanmin(M.values, axis=0)
    maxs = np.nanmax(M.values, axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    M01 = (M.values - mins) / denom
    M01 = np.clip(M01, 0.0, 1.0)

    labels = metrics
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = plt.subplot(111, polar=True)
    ax.set_title(title, y=1.08, fontsize=13)

    for row in M01:
        vals = row.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, alpha=0.9)
        ax.fill(angles, vals, alpha=0.06)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.grid(True, alpha=0.3)

    methods = df[method_col].astype(str).tolist()
    ax.legend(methods, loc="upper left", bbox_to_anchor=(1.05, 1.05), frameon=False, fontsize=8)

    plt.tight_layout()
    save_pubfig(fig, os.path.join(outdir, "radar_means.png"), os.path.join(outdir, "radar_means.pdf"))


# -----------------------------------------------------------------------------
# summary plotting (kept)
# -----------------------------------------------------------------------------
def plot_metric_summary(metrics: dict, perf_df: pd.DataFrame, outdir: str, title: str):
    os.makedirs(outdir, exist_ok=True)
    keys = ["kBET", "iLISI_batch", "cLISI_label", "isolated_label_F1", "rare_knn_purity_mean"]
    vals = [metrics.get(k, np.nan) for k in keys]

    plt.figure(figsize=(10, 4))
    plt.bar(keys, vals)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "metrics_barplot.png"), dpi=300)
    plt.close()

    if perf_df is not None and len(perf_df):
        df_end = perf_df[perf_df["event"] == "end"].copy()
        if len(df_end):
            plt.figure(figsize=(10, 4))
            plt.bar(df_end["step"].astype(str), df_end["time_s"].astype(float))
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Seconds")
            plt.title("Runtime by step")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "runtime_by_step.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.plot(perf_df["rss_gb"].astype(float).values, marker="o")
            plt.ylabel("RSS memory (GB)")
            plt.title("Memory trace across steps")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "memory_trace.png"), dpi=300)
            plt.close()
