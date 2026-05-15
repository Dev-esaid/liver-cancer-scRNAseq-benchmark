#!/usr/bin/env python3
"""
make_cross_method_heatmaps_per_task_v5.py

Layout (left -> right):
  [Method header] [Raw scores header] [Col-wise z-scores header]   <- slim coloured headers
  [Method names ] [Heatmap A        ] [Heatmap B               ] [Score cbar] [Z-score cbar]

Changes in v5:
  - No A / B panel letters
  - Half-column gap between the two heatmaps
  - Colorbars shifted apart (CBAR_SEP) and made narrower + shorter to prevent overlap
  - Reduced header height
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})


# =============================================================================
# Configuration
# =============================================================================

BASE_RESULTS_DIR = Path(
    "/data1/esraa/Thesis-Project/Results/Trajectory_Inference"
)
OUTPUT_DIR = BASE_RESULTS_DIR / "tier1_Cross_Method_Heatmaps_Per_Task"

TASK_DATASET_MAP: Dict[str, str] = {
    "task1_macrophage_TAM":          "GSE149614",
    "task2_Tcell_continuum":         "GSE149614",
    "task3_myeloid_monocyte_to_TAM": "GSE140228",
    "task4_CD8_exhaustion":          "GSE140228",
    "task5_CD4_differentiation":     "GSE140228",
    "task6_NK_maturation":           "GSE140228",
}

TASK_TITLE_MAP: Dict[str, str] = {
    "task1_macrophage_TAM":          "Macrophage \u2192 TAM",
    "task2_Tcell_continuum":         "T-cell continuum",
    "task3_myeloid_monocyte_to_TAM": "Myeloid monocyte \u2192 TAM",
    "task4_CD8_exhaustion":          "CD8 exhaustion",
    "task5_CD4_differentiation":     "CD4 differentiation",
    "task6_NK_maturation":           "NK maturation",
}

DEFAULT_TASKS: List[str] = list(TASK_DATASET_MAP.keys())

DEFAULT_METHODS: List[str] = [
    "cellrank", "elpigraph", "monocle3", "paga",
    "scorpius", "slingshot", "tscan", "via",
]

METHOD_LABELS: Dict[str, str] = {
    "cellrank":  "CellRank",
    "elpigraph": "ElPiGraph",
    "monocle3":  "Monocle3",
    "paga":      "PAGA",
    "scorpius":  "SCORPIUS",
    "slingshot": "Slingshot",
    "tscan":     "TSCAN",
    "via":       "VIA",
}

METRIC_COLS: List[str] = [
    "Marker concordance",
    "Marker monotonicity",
    "kNN smoothness",
    "Root purity",
    "Topology consistency",
]

ROW_COLORS = ["#E8E8E8", "#FFFFFF"]

COL_METHOD = "#AAAAAA"
COL_RAW    = "#fb6f92"
COL_ZSCORE = "#809bce"


# =============================================================================
# Geometry  (inches)
# =============================================================================

CELL_SIZE    = 0.20

LEFT_PAD     = 0.06
LABEL_COL_W  = 0.76
LABEL_GAP    = CELL_SIZE * 0.5   # gap between label column and heatmap A
PANEL_GAP    = CELL_SIZE * 0.5   # half-column gap between the two heatmaps
CBAR_GAP     = 0.12              # heatmap B right edge -> first colorbar
CBAR_SEP     = 0.40              # gap between the two colorbars
CBAR_W       = 0.055             # narrow colorbar
CBAR_H_FRAC  = 0.58              # colorbar height as fraction of panel height
CBAR_B_FRAC  = 0.21              # colorbar bottom offset
RIGHT_PAD    = 0.14

HEADER_H     = 0.14              # slim header
TOP_PAD      = 0.06
BOTTOM_PAD   = 1.20
TITLE_BLOCK  = 0.34

FONT_METHOD   = 6.0
FONT_METRIC   = 6.2
FONT_HEADER   = 6.0
FONT_TITLE    = 9.5
FONT_SUBTITLE = 7.0
FONT_CBAR     = 5.5
FONT_FOOT     = 5.6


# =============================================================================
# Colormaps
# =============================================================================

def _raw_cmap() -> LinearSegmentedColormap:
    stops = [(0.00,"#FFE8ED"),(0.28,"#FFB3C0"),(0.55,"#FF7A8A"),
             (0.78,"#FF4466"),(1.00,"#E00030")]
    return LinearSegmentedColormap.from_list("pink_red", stops, N=256)


def _zscore_cmap() -> LinearSegmentedColormap:
    stops = [(0.00,"#3549C2"),(0.18,"#6B7ED8"),(0.36,"#B5BCE8"),(0.50,"#F5F6FF"),
             (0.64,"#FFB3C0"),(0.82,"#FF7A8A"),(1.00,"#E00030")]
    return LinearSegmentedColormap.from_list("blue_white_red", stops, N=512)


# =============================================================================
# Utilities
# =============================================================================

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def _clip01(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(np.clip(x, 0.0, 1.0))


def _mean_of_key(items: Any, key: str) -> Optional[float]:
    if not isinstance(items, list):
        return None
    vals = [_safe_float(r.get(key)) for r in items if isinstance(r, dict)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metrics_path(method: str, task: str) -> Path:
    dataset = TASK_DATASET_MAP.get(task, "UNKNOWN")
    return (BASE_RESULTS_DIR / dataset / method / task
            / "tables" / "metrics_summary.json")


def _pretty_task(task: str) -> str:
    return TASK_TITLE_MAP.get(task, task.replace("_", " "))


# =============================================================================
# Metric extraction
# =============================================================================

def aggregate_metrics(j: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in METRIC_COLS}

    mc = j.get("marker_concordance", {})
    if mc.get("status") == "ok":
        rho = _mean_of_key(mc.get("value"), "spearman_rho_aligned")
        if rho is not None:
            out["Marker concordance"] = _clip01((rho + 1.0) / 2.0)

    mm = j.get("marker_monotonicity", {})
    if mm.get("status") == "ok":
        out["Marker monotonicity"] = _clip01(
            _mean_of_key(mm.get("value"), "binned_monotonicity"))

    ks = j.get("knn_smoothness", {})
    if ks.get("status") == "ok" and isinstance(ks.get("value"), dict):
        mad = _safe_float(ks["value"].get("mean_abs_deviation_to_neighbor_mean"))
        if mad is not None:
            out["kNN smoothness"] = _clip01(1.0 / (1.0 + mad))

    rp = j.get("root_purity", {})
    if rp.get("status") == "ok" and isinstance(rp.get("value"), dict):
        out["Root purity"] = _clip01(_safe_float(rp["value"].get("root_purity")))

    tc = j.get("topology_pseudotime_consistency", {})
    if tc.get("status") == "ok" and isinstance(tc.get("value"), dict):
        v = _safe_float(tc["value"].get("spearman_rho_abs"))
        if v is None:
            raw = _safe_float(tc["value"].get("spearman_rho"))
            v = abs(raw) if raw is not None else None
        out["Topology consistency"] = _clip01(v)

    return out


# =============================================================================
# Data loading
# =============================================================================

def build_task_dataframe(
    task: str, methods: Sequence[str]
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    rows: List[Dict[str, Any]] = []
    tie_vals: Dict[str, float] = {}
    missing: List[str] = []

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        p = _metrics_path(method, task)
        if not p.exists():
            missing.append(str(p))
            rows.append({"Method": label, **{k: np.nan for k in METRIC_COLS}})
            tie_vals[label] = np.nan
            continue

        js  = _load_json(p)
        agg = aggregate_metrics(js)
        ps  = js.get("pseudotime_sanity_raw", {})
        tie = None
        if ps.get("status") == "ok" and isinstance(ps.get("value"), dict):
            tie = _safe_float(ps["value"].get("tie_fraction"))

        row: Dict[str, Any] = {"Method": label}
        row.update({k: (v if v is not None else np.nan) for k, v in agg.items()})
        rows.append(row)
        tie_vals[label] = tie if tie is not None else np.nan

    df = pd.DataFrame(rows).set_index("Method").astype(float)
    return df, pd.Series(tie_vals, name="tie_fraction"), missing


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(float)
    for col in out.columns:
        vals = out[col].to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() < 2:
            out[col] = np.where(mask, 0.0, np.nan)
            continue
        mu = float(np.nanmean(vals))
        sd = float(np.nanstd(vals, ddof=1))
        out[col] = (out[col] - mu) / sd if sd > 1e-12 else np.where(mask, 0.0, np.nan)
    return out


# =============================================================================
# Drawing helpers
# =============================================================================

def _draw_header(fig: plt.Figure, rect: List[float],
                 color: str, text: str) -> None:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("auto")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none"))
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=FONT_HEADER, fontweight="normal", color="white")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def _draw_label_column(fig: plt.Figure, rect: List[float],
                       method_names: List[str], n_rows: int) -> None:
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(n_rows - 0.5, -0.5); ax.set_aspect("auto")
    for i, name in enumerate(method_names):
        ax.add_patch(Rectangle((0, i - 0.5), 1, 1,
                                facecolor=ROW_COLORS[i % 2], edgecolor="none", zorder=1))
        ax.text(0.06, i, name, ha="left", va="center",
                fontsize=FONT_METHOD, fontweight="normal", color="#1A1A1A", zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def _draw_heatmap(ax: plt.Axes, df: pd.DataFrame, *,
                  cmap, norm, n_rows: int, n_cols: int) -> None:
    """Heatmap panel — no panel letter."""
    data = df.to_numpy(dtype=float)
    for i in range(n_rows):
        ax.add_patch(Rectangle((-0.5, i - 0.5), n_cols, 1,
                                facecolor=ROW_COLORS[i % 2], edgecolor="none", zorder=0))
        for j in range(n_cols):
            v  = data[i, j]
            fc = cmap(norm(v)) if np.isfinite(v) else "#DCDCDC"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    facecolor=fc, edgecolor="white",
                                    linewidth=0.45, zorder=2))
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(df.columns.tolist(), fontsize=FONT_METRIC,
                       fontweight="normal", rotation=90, ha="right", va="top")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", which="both", length=0, pad=3)
    ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def _add_colorbar(fig: plt.Figure, rect: List[float], cmap, norm,
                  *, label: str, ticks: List[float],
                  ticklabels: List[str]) -> None:
    cax = fig.add_axes(rect)
    sm  = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cb.ax.yaxis.set_major_formatter(FixedFormatter(ticklabels))
    cb.ax.tick_params(labelsize=FONT_CBAR, length=2.0, width=0.5, pad=2.5)
    cb.outline.set_linewidth(0.4)
    cb.set_label(label, fontsize=FONT_CBAR, labelpad=4, fontweight="normal")


# =============================================================================
# Main figure builder
# =============================================================================

def make_task_figure(
    df_raw: pd.DataFrame, df_z: pd.DataFrame, tie_series: pd.Series,
    *, task: str, dataset: str, n_missing: int, out_path: Path,
) -> None:
    n_rows, n_cols = df_raw.shape
    panel_w = n_cols * CELL_SIZE
    panel_h = n_rows * CELL_SIZE

    fig_w = (LEFT_PAD + LABEL_COL_W + LABEL_GAP
             + panel_w + PANEL_GAP + panel_w
             + CBAR_GAP + CBAR_W + CBAR_SEP + CBAR_W
             + RIGHT_PAD)
    fig_h = TITLE_BLOCK + HEADER_H + TOP_PAD + panel_h + BOTTOM_PAD

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")

    raw_b = BOTTOM_PAD / fig_h
    raw_h = panel_h    / fig_h
    raw_w = panel_w    / fig_w
    hdr_b = (BOTTOM_PAD + panel_h + TOP_PAD) / fig_h
    hdr_h = HEADER_H  / fig_h
    cb_h  = raw_h * CBAR_H_FRAC
    cb_b  = raw_b + raw_h * CBAR_B_FRAC
    cb_w  = CBAR_W / fig_w

    lbl_l    = LEFT_PAD / fig_w
    lbl_w    = LABEL_COL_W / fig_w
    raw_l    = (LEFT_PAD + LABEL_COL_W + LABEL_GAP) / fig_w
    z_l      = raw_l + raw_w + PANEL_GAP / fig_w
    raw_cb_l = z_l + raw_w + CBAR_GAP / fig_w
    z_cb_l   = raw_cb_l + cb_w + CBAR_SEP / fig_w

    # headers
    _draw_header(fig, [lbl_l, hdr_b, lbl_w, hdr_h], COL_METHOD, "Method")
    _draw_header(fig, [raw_l, hdr_b, raw_w, hdr_h], COL_RAW, "Raw scores")
    _draw_header(fig, [z_l,   hdr_b, raw_w, hdr_h], COL_ZSCORE, "Column wise z-scores")

    # label column
    _draw_label_column(fig, [lbl_l, raw_b, lbl_w, raw_h],
                       df_raw.index.tolist(), n_rows)

    # heatmaps
    raw_cmap = _raw_cmap(); z_cmap = _zscore_cmap()
    raw_norm = Normalize(vmin=0.0, vmax=1.0)
    fin_z    = df_z.to_numpy(dtype=float); fin_z = fin_z[np.isfinite(fin_z)]
    vmax_z   = max(2.0, float(np.nanmax(np.abs(fin_z)))) if fin_z.size else 2.0
    z_norm   = TwoSlopeNorm(vmin=-vmax_z, vcenter=0.0, vmax=vmax_z)

    raw_ax = fig.add_axes([raw_l, raw_b, raw_w, raw_h])
    z_ax   = fig.add_axes([z_l,   raw_b, raw_w, raw_h])
    _draw_heatmap(raw_ax, df_raw, cmap=raw_cmap, norm=raw_norm,
                  n_rows=n_rows, n_cols=n_cols)
    _draw_heatmap(z_ax,   df_z,   cmap=z_cmap,  norm=z_norm,
                  n_rows=n_rows, n_cols=n_cols)

    # colorbars
    _add_colorbar(fig, [raw_cb_l, cb_b, cb_w, cb_h], raw_cmap, raw_norm,
                  label="Score",
                  ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
                  ticklabels=["0", ".25", ".5", ".75", "1"])
    z_ticks = [-vmax_z, -vmax_z / 2, 0.0, vmax_z / 2, vmax_z]
    _add_colorbar(fig, [z_cb_l, cb_b, cb_w, cb_h], z_cmap, z_norm,
                  label="z-score",
                  ticks=z_ticks,
                  ticklabels=[f"{t:.1f}" for t in z_ticks])

    # title block
    title    = _pretty_task(task)
    miss_txt = f"  \u00b7  {n_missing} missing" if n_missing else ""
    subtitle = (f"{dataset}  \u00b7  {n_rows} methods  "
                f"\u00b7  {n_cols} metrics{miss_txt}")
    fig.text(0.5, 1.0 - TITLE_BLOCK * 0.22 / fig_h,
             title, ha="center", va="top",
             fontsize=FONT_TITLE, fontweight="normal", color="#111111")
    fig.text(0.5, 1.0 - TITLE_BLOCK * 0.62 / fig_h,
             subtitle, ha="center", va="top",
             fontsize=FONT_SUBTITLE, fontweight="normal",
             color="#777777", style="italic")

    # footer
    fig.text(lbl_l, 0.012,
             "Raw scores [0, 1], marker concordance = (\u03c1 + 1) / 2   "
             "|   Column-wise z-scores",
             ha="left", va="bottom",
             fontsize=FONT_FOOT, fontweight="normal", color="#888888", style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  [saved] {out_path.name}")


# =============================================================================
# CLI & main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compact publication-quality cross-method heatmaps (v5)."
    )
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--tasks",   nargs="+", default=DEFAULT_TASKS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_missing: List[str] = []

    for task in list(args.tasks):
        try:
            print(f"\n[task] {task}")
            dataset = TASK_DATASET_MAP.get(task, "?")
            df_raw, tie_series, missing = build_task_dataframe(
                task, list(args.methods))
            df_z = zscore_columns(df_raw)
            all_missing.extend(missing)

            make_task_figure(
                df_raw, df_z, tie_series,
                task=task, dataset=dataset, n_missing=len(missing),
                out_path=OUTPUT_DIR / f"{task}_cross_method_heatmap.png",
            )
            df_raw.to_csv(OUTPUT_DIR / f"{task}_metrics_raw.csv")
            df_z.to_csv(  OUTPUT_DIR / f"{task}_metrics_zscore.csv")

        except Exception as exc:
            print(f"  [ERROR] {task}: {exc}")
            traceback.print_exc()

    if all_missing:
        log = OUTPUT_DIR / "missing_files.txt"
        log.write_text("\n".join(all_missing) + "\n", encoding="utf-8")
        print(f"\n[warning] {len(all_missing)} missing -> {log}")

    print(f"\nOutput directory:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()