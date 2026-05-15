#!/usr/bin/env python3
"""
make_quantitative_metric_tables_per_method.py

Generate publication-style Tier 3 supplementary tables.
One figure per method, six task panels per figure.
Each panel shows per-cluster concordance and monotonicity
plus a compact run-level scalar summary above the table.

Usage
-----
    python make_quantitative_metric_tables_per_method.py --method cellrank
    python make_quantitative_metric_tables_per_method.py --method via
    python make_quantitative_metric_tables_per_method.py --method paga
    # ... repeat for all 8 methods

    # or all at once:
    for method in cellrank paga elpigraph scorpius slingshot tscan monocle3 via; do
        python make_quantitative_metric_tables_per_method.py --method $method
    done

    # specific tasks only:
    python make_quantitative_metric_tables_per_method.py \
        --method cellrank \
        --tasks task1_macrophage_TAM task2_Tcell_continuum

Output
------
    /data1/esraa/Thesis-Project/Results/Trajectory_Inference/
        tier3_Quantitive_Metrics_Per_Method/
            cellrank_cluster_level_metrics.png
            paga_cluster_level_metrics.png
            ...

Expected input path per task
-----------------------------
    {BASE_RESULTS_DIR}/{dataset}/{method}/{task}/tables/metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

BASE_RESULTS_DIR = Path(
    "/data1/esraa/Thesis-Project/Results/Trajectory_Inference"
)
OUTPUT_DIR = BASE_RESULTS_DIR / "tier3_Quantitive_Metrics_Per_Method"

# Maps each task to its dataset — tasks 1-2 are GSE149614, tasks 3-6 are GSE140228
TASK_DATASET_MAP: Dict[str, str] = {
    "task1_macrophage_TAM":          "GSE149614",
    "task2_Tcell_continuum":         "GSE149614",
    "task3_myeloid_monocyte_to_TAM": "GSE140228",
    "task4_CD8_exhaustion":          "GSE140228",
    "task5_CD4_differentiation":     "GSE140228",
    "task6_NK_maturation":           "GSE140228",
}

DEFAULT_TASKS: List[str] = list(TASK_DATASET_MAP.keys())


# =============================================================================
# Utility helpers
# =============================================================================

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def _fmt(x: Any, ndigits: int = 3, na: str = "NA") -> str:
    v = _safe_float(x)
    return na if v is None else f"{v:.{ndigits}f}"


def _fmt_int(x: Any, na: str = "NA") -> str:
    try:
        return na if x is None else f"{int(x):,}"
    except Exception:
        return na


def _sig_stars(p: Any) -> str:
    v = _safe_float(p)
    if v is None:
        return ""
    if v < 0.001:
        return "***"
    if v < 0.01:
        return "**"
    if v < 0.05:
        return "*"
    return "ns"


def _task_pretty_name(task_name: str) -> str:
    pretty = task_name.replace("_", " ")
    parts = pretty.split(" ", 1)
    return parts[1].strip() if len(parts) == 2 else pretty


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Metrics extraction
# =============================================================================

def extract_cluster_rows(
    metrics_json: Dict[str, Any],
) -> Tuple[List[str], List[List[str]]]:
    """
    Build per-cluster table rows from metrics_summary.json.

    Columns: Cluster | Concordance rho | Sig. | Monotonicity | n cells | Root?

    Rows are sorted: root cluster first, then by concordance descending.
    """
    col_headers = [
        "Cluster",
        "Concordance \u03c1",
        "Sig.",
        "Monotonicity",
        "n cells",
        "Root?",
    ]

    # monotonicity lookup: cluster -> binned_monotonicity
    mono_lookup: Dict[str, float] = {}
    mm = metrics_json.get("marker_monotonicity", {})
    if mm.get("status") == "ok":
        for item in mm.get("value", []):
            prog = str(item.get("program", ""))
            val = _safe_float(item.get("binned_monotonicity"))
            if val is not None:
                mono_lookup[prog] = val

    total_n = metrics_json.get("meta", {}).get("n_cells")

    rows: List[List[str]] = []
    mc = metrics_json.get("marker_concordance", {})
    if mc.get("status") == "ok":
        for item in mc.get("value", []):
            cluster  = str(item.get("program", "?"))
            rho      = _fmt(item.get("spearman_rho_aligned"), ndigits=3)
            sig      = _sig_stars(item.get("spearman_p"))
            mono     = _fmt(mono_lookup.get(cluster), ndigits=3)
            n        = _fmt_int(item.get("n_used", total_n))
            root     = "yes" if item.get("used_for_root") else "\u2014"
            rows.append([cluster, rho, sig, mono, n, root])

    # sort: root first, then concordance descending
    rows.sort(
        key=lambda r: (
            0 if r[5] == "yes" else 1,
            -float(r[1]) if r[1] not in ("NA", "") else 0.0,
        )
    )
    return col_headers, rows


def extract_run_scalars(metrics_json: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Compact run-level scalars displayed above the cluster table.
    Returns list of (label, value) pairs.
    """
    scalars: List[Tuple[str, str]] = []

    # cell count
    meta = metrics_json.get("meta", {})
    scalars.append(("Total cells", _fmt_int(meta.get("n_cells"))))

    # pseudotime sanity
    ps = metrics_json.get("pseudotime_sanity_raw", {})
    if ps.get("status") == "ok" and isinstance(ps.get("value"), dict):
        v = ps["value"]
        tie = _safe_float(v.get("tie_fraction"))
        tie_str = _fmt(tie)
        if tie is not None and tie > 0.5:
            tie_str += " \u2605"          # ★ flag
        scalars.append(("Tie fraction", tie_str))
        scalars.append(("n unique PT", _fmt_int(v.get("n_unique"))))

    # root purity
    rp = metrics_json.get("root_purity", {})
    if rp.get("status") == "ok" and isinstance(rp.get("value"), dict):
        v = rp["value"]
        scalars.append(("Root purity", _fmt(v.get("root_purity"))))
        scalars.append(("Root fold enrichment", _fmt(v.get("fold_enrichment"))))

    # kNN smoothness — label direction explicitly
    ks = metrics_json.get("knn_smoothness", {})
    if ks.get("status") == "ok" and isinstance(ks.get("value"), dict):
        v = ks["value"]
        scalars.append((
            "kNN smoothness MAD (\u2193=better)",
            _fmt(v.get("mean_abs_deviation_to_neighbor_mean")),
        ))

    # topology consistency
    tc = metrics_json.get("topology_pseudotime_consistency", {})
    if tc.get("status") == "ok" and isinstance(tc.get("value"), dict):
        v = tc["value"]
        scalars.append(("Topology |\u03c1|", _fmt(v.get("spearman_rho_abs"))))

    return scalars


# =============================================================================
# Rendering
# =============================================================================

def draw_task_panel(
    ax: plt.Axes,
    task_pretty: str,
    col_headers: List[str],
    data_rows: List[List[str]],
    run_scalars: List[Tuple[str, str]],
    n_loaded: int,
    n_total: int,
) -> None:
    """
    Draw one task panel onto ax using publication-style table formatting:
    - Three horizontal rules only (top, below header, bottom)
    - No vertical lines
    - No colored header backgrounds
    - Subtle alternating row shading
    - Bold root cluster row
    - Scalar summary split into two lines to avoid overflow
    """
    ax.axis("off")

    # ── scalar summary: split into two lines to prevent overflow ──
    half = math.ceil(len(run_scalars) / 2)
    line1 = "  |  ".join(f"{k}: {v}" for k, v in run_scalars[:half])
    line2 = "  |  ".join(f"{k}: {v}" for k, v in run_scalars[half:])

    ax.text(
        0.5, 1.065, line1,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=7.5, color="#4B5563", style="italic",
    )
    ax.text(
        0.5, 1.025, line2,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=7.5, color="#4B5563", style="italic",
    )

    # ── table ──
    col_widths = [0.32, 0.14, 0.08, 0.14, 0.12, 0.10]

    tbl = ax.table(
        cellText=data_rows,
        colLabels=col_headers,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)

    n_data_rows = len(data_rows)
    n_cols      = len(col_headers)

    for (r, c), cell in tbl.get_celld().items():
        # reset all borders and fills
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_color("#111827")
        cell.PAD = 0.05

        if r == 0:
            # ── header row ──
            cell.get_text().set_weight("bold")
            cell.get_text().set_ha("center")
            cell.get_text().set_fontsize(9)
            cell.set_facecolor("white")
            # thick bottom rule under header
            cell.visible_edges = "B"
            cell.set_linewidth(0.9)
            cell.set_edgecolor("#111827")

        else:
            is_root = (
                len(data_rows[r - 1]) > 5
                and data_rows[r - 1][5] == "yes"
            )

            # alternating very light shading — no colored fills
            cell.set_facecolor("#F5F5F5" if r % 2 == 0 else "white")
            cell.visible_edges = ""     # no cell borders in body

            # cluster name column: left-aligned, bold if root
            if c == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_weight(
                    "bold" if is_root else "normal"
                )
            else:
                cell.get_text().set_ha("center")

            # thin top rule on first data row (separates from header)
            if r == 1:
                cell.visible_edges = "T"
                cell.set_linewidth(0.4)
                cell.set_edgecolor("#9CA3AF")

    # thick bottom rule on last data row
    for c in range(n_cols):
        cell = tbl[n_data_rows, c]
        cell.visible_edges = "B"
        cell.set_linewidth(0.9)
        cell.set_edgecolor("#111827")

    # ── panel title ──
    status = "" if n_loaded == n_total else f" ({n_loaded}/{n_total} tasks loaded)"
    ax.set_title(
        f"{task_pretty}{status}",
        fontsize=11,
        fontweight="bold",
        color="#111827",
        pad=40,         # generous pad so scalar lines sit above without overlap
    )

    # ── footnote ──
    ax.text(
        0.0, -0.04,
        "\u2605 tie_fraction > 0.5 \u2014 concordance/monotonicity scores may be inflated",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#6B7280",
        style="italic",
    )


# =============================================================================
# Figure assembly
# =============================================================================

def make_figure(
    method: str,
    task_data: Dict[str, Any],
    out_path: Path,
) -> None:
    tasks  = list(task_data.keys())
    n      = len(tasks)
    ncols  = 2
    nrows  = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(16, 6.5 * nrows),
    )
    axes = np.array(axes).reshape(-1)

    for ax, task in zip(axes, tasks):
        col_headers, data_rows = task_data[task]["cluster_rows"]
        run_scalars            = task_data[task]["run_scalars"]
        draw_task_panel(
            ax,
            _task_pretty_name(task),
            col_headers,
            data_rows,
            run_scalars,
            n_loaded=n,
            n_total=len(DEFAULT_TASKS),
        )

    # label unused panels
    for j in range(n, len(axes)):
        axes[j].axis("off")
        axes[j].text(
            0.5, 0.5, "no data for this task",
            ha="center", va="center",
            fontsize=10, color="#9CA3AF",
            transform=axes[j].transAxes,
        )

    dataset_str = " + ".join(
        sorted({TASK_DATASET_MAP.get(t, "?") for t in tasks})
    )
    fig.suptitle(
        f"Cluster-level metrics \u2014 {method}  ({dataset_str})",
        fontsize=16,
        fontweight="bold",
        y=0.998,
        color="#111827",
    )
    fig.subplots_adjust(
        left=0.03, right=0.97,
        top=0.94,  bottom=0.03,
        hspace=0.60,
        wspace=0.14,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] saved: {out_path}")


# =============================================================================
# Path resolution
# =============================================================================

def resolve_metrics_path(method: str, task: str) -> Path:
    dataset = TASK_DATASET_MAP.get(task, "UNKNOWN")
    return (
        BASE_RESULTS_DIR
        / dataset
        / method
        / task
        / "tables"
        / "metrics_summary.json"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Tier 3 cluster-level metric tables per method."
    )
    p.add_argument(
        "--method",
        required=True,
        help="Method name, e.g. cellrank",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task names (default: all 6 tasks).",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    method = str(args.method)
    tasks: List[str] = list(args.tasks)

    task_data: Dict[str, Any] = {}
    missing:   List[str]      = []

    for task in tasks:
        path = resolve_metrics_path(method, task)
        if not path.exists():
            missing.append(f"  - {task}: {path}")
            continue
        j = _load_json(path)
        task_data[task] = {
            "cluster_rows": extract_cluster_rows(j),
            "run_scalars":  extract_run_scalars(j),
        }

    if not task_data:
        raise FileNotFoundError(
            "No metrics_summary.json files were found.\n"
            + "\n".join(missing)
        )

    out_path = OUTPUT_DIR / f"{method}_cluster_level_metrics.png"
    make_figure(method, task_data, out_path)

    if missing:
        print(f"\n[WARNING] {len(missing)} task(s) not found:")
        for m in missing:
            print(m)


if __name__ == "__main__":
    main()