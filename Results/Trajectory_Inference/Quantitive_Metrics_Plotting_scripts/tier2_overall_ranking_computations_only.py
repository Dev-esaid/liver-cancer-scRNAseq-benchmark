#!/usr/bin/env python3
"""
tier2_overall_ranking_computations_only.py  –  Tier 2 overall method ranking (data only)
==============================================================================

Computes aggregated metrics and Friedman ranks, then saves three CSVs:

    tier2_mean_scores.csv      — mean score per method per metric (across 6 tasks)
    tier2_std_scores.csv       — std  score per method per metric (across 6 tasks)
    tier2_friedman_ranks.csv   — Friedman rank per method per metric + composite
    tier2_score_table.csv      — mean ± sd display table, sorted by composite rank
    tier2_dotplot_data.csv     — tidy long-format table for R/ggplot2

Usage
-----
    python tier2_overall_ranking_computations_only.py
"""

from __future__ import annotations

import json, math, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

BASE_RESULTS_DIR = Path(
    "/data1/esraa/Thesis-Project/Results/Trajectory_Inference"
)
OUTPUT_DIR = BASE_RESULTS_DIR / "Quantitive_Metrics_Plotting_scripts"

TASK_DATASET_MAP: Dict[str, str] = {
    "task1_macrophage_TAM":          "GSE149614",
    "task2_Tcell_continuum":         "GSE149614",
    "task3_myeloid_monocyte_to_TAM": "GSE140228",
    "task4_CD8_exhaustion":          "GSE140228",
    "task5_CD4_differentiation":     "GSE140228",
    "task6_NK_maturation":           "GSE140228",
}
TASKS: List[str] = list(TASK_DATASET_MAP.keys())

METHODS_ORDER: List[str] = [
    "cellrank", "elpigraph", "monocle3", "paga",
    "scorpius", "slingshot", "tscan", "via",
]
METHOD_LABELS: Dict[str, str] = {
    "cellrank":  "CellRank",
    "elpigraph": "ElPiGraph",
    "monocle3":  "Monocle3",
    "paga":      "PAGA + DPT",
    "scorpius":  "SCORPIUS",
    "slingshot": "Slingshot",
    "tscan":     "TSCAN",
    "via":       "VIA",
}

METHOD_COLORS: Dict[str, str] = {
    "cellrank":  "#4E9EC2",
    "elpigraph": "#E07B3F",
    "monocle3":  "#3DA85A",
    "paga":      "#C94040",
    "scorpius":  "#8E6BBE",
    "slingshot": "#D4A017",
    "tscan":     "#D9629B",
    "via":       "#5BB8A8",
}

_LABEL_TO_KEY: Dict[str, str] = {v: k for k, v in METHOD_LABELS.items()}

METRIC_COLS: List[str] = [
    "Marker concordance",
    "Marker monotonicity",
    "kNN smoothness",
    "Root purity",
    "Topology consistency",
]

TIE_THRESHOLD = 0.5


# =============================================================================
# Data loading helpers
# =============================================================================

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


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
    dataset = TASK_DATASET_MAP[task]
    return (BASE_RESULTS_DIR / dataset / method / task
            / "tables" / "metrics_summary.json")


def _extract_metrics(j: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in METRIC_COLS}
    mc = j.get("marker_concordance", {})
    if mc.get("status") == "ok":
        out["Marker concordance"] = _mean_of_key(mc.get("value"), "spearman_rho_aligned")
    mm = j.get("marker_monotonicity", {})
    if mm.get("status") == "ok":
        out["Marker monotonicity"] = _mean_of_key(mm.get("value"), "binned_monotonicity")
    ks = j.get("knn_smoothness", {})
    if ks.get("status") == "ok" and isinstance(ks.get("value"), dict):
        mad = _safe_float(ks["value"].get("mean_abs_deviation_to_neighbor_mean"))
        if mad is not None:
            out["kNN smoothness"] = 1.0 / (1.0 + mad)
    rp = j.get("root_purity", {})
    if rp.get("status") == "ok" and isinstance(rp.get("value"), dict):
        out["Root purity"] = _safe_float(rp["value"].get("root_purity"))
    tc = j.get("topology_pseudotime_consistency", {})
    if tc.get("status") == "ok" and isinstance(tc.get("value"), dict):
        out["Topology consistency"] = _safe_float(tc["value"].get("spearman_rho_abs"))
    return out


def _extract_tie_fraction(j: Dict[str, Any]) -> Optional[float]:
    ps = j.get("pseudotime_sanity_raw", {})
    if ps.get("status") == "ok" and isinstance(ps.get("value"), dict):
        return _safe_float(ps["value"].get("tie_fraction"))
    return None


# =============================================================================
# Build aggregated dataset
# =============================================================================

def build_aggregated_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Returns
    -------
    df_mean    : DataFrame[method × metric]  — mean score across tasks
    df_std     : DataFrame[method × metric]  — std  score across tasks
    tie_series : Series[method_label → mean tie_fraction]
    """
    all_scores: Dict[str, Dict[str, Dict[str, float]]] = {}
    all_ties:   Dict[str, List[float]] = {}

    for method in METHODS_ORDER:
        label = METHOD_LABELS[method]
        all_scores[label] = {}
        all_ties[label]   = []
        for task in TASKS:
            p = _metrics_path(method, task)
            if not p.exists():
                continue
            j       = _load_json(p)
            metrics = _extract_metrics(j)
            tie     = _extract_tie_fraction(j)
            all_scores[label][task] = {
                k: (v if v is not None else np.nan)
                for k, v in metrics.items()
            }
            if tie is not None:
                all_ties[label].append(tie)

    mean_rows: List[Dict] = []
    std_rows:  List[Dict] = []

    for method in METHODS_ORDER:
        label       = METHOD_LABELS[method]
        task_scores = all_scores[label]
        mean_row    = {"Method": label}
        std_row     = {"Method": label}
        for metric in METRIC_COLS:
            vals = [
                task_scores[t][metric]
                for t in task_scores
                if metric in task_scores[t] and np.isfinite(task_scores[t][metric])
            ]
            mean_row[metric] = float(np.mean(vals)) if vals else np.nan
            std_row[metric]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        mean_rows.append(mean_row)
        std_rows.append(std_row)

    df_mean = pd.DataFrame(mean_rows).set_index("Method").astype(float)
    df_std  = pd.DataFrame(std_rows).set_index("Method").astype(float)

    tie_series = pd.Series(
        {METHOD_LABELS[m]: (float(np.mean(all_ties[METHOD_LABELS[m]]))
                            if all_ties[METHOD_LABELS[m]] else np.nan)
         for m in METHODS_ORDER},
        name="mean_tie_fraction",
    )
    return df_mean, df_std, tie_series


# =============================================================================
# Friedman ranking
# =============================================================================

def compute_friedman_ranks(df_mean: pd.DataFrame) -> pd.DataFrame:
    """
    Rank methods within each metric (rank 1 = best = highest score).
    Returns DataFrame[method × metric] of ranks (1..8),
    plus a 'Composite rank' column = mean rank across all metrics.
    """
    ranks = df_mean.copy() * np.nan
    for metric in METRIC_COLS:
        col   = df_mean[metric]
        valid = col.dropna()
        ranks[metric] = valid.rank(ascending=False, method="average")
    ranks["Composite rank"] = ranks[METRIC_COLS].mean(axis=1)
    return ranks


# =============================================================================
# CSV exports
# =============================================================================

def export_raw_scores(df_mean: pd.DataFrame,
                      df_std: pd.DataFrame,
                      out_dir: Path) -> None:
    """Save mean and std score tables as CSV."""
    # Methods follow METHODS_ORDER (as built in build_aggregated_data)
    # No sorting applied — preserved in original METHODS_ORDER
    df_mean.to_csv(out_dir / "tier2_mean_scores.csv")
    print(f"  [saved] tier2_mean_scores.csv")

    df_std.to_csv(out_dir / "tier2_std_scores.csv")
    print(f"  [saved] tier2_std_scores.csv")


def export_friedman_ranks(df_mean: pd.DataFrame,
                          out_dir: Path) -> None:
    """Save Friedman rank table (per metric + composite) as CSV."""
    ranks_df = compute_friedman_ranks(df_mean)
    # No sorting applied — preserved in original METHODS_ORDER
    # ranks_df_sorted = ranks_df.sort_values("Composite rank")
    ranks_df.to_csv(out_dir / "tier2_friedman_ranks.csv")
    print(f"  [saved] tier2_friedman_ranks.csv")


def export_score_table(df_mean: pd.DataFrame,
                       df_std: pd.DataFrame,
                       out_dir: Path) -> None:
    """
    Save display-style table (mean ± sd).
    Columns: Method, 5 metrics, Composite rank.
    """
    ranks_df = compute_friedman_ranks(df_mean)

    # No sorting applied — use original METHODS_ORDER
    # sorted_methods = ranks_df["Composite rank"].dropna().sort_values().index.tolist()
    unsorted_methods: List[str] = [METHOD_LABELS[m] for m in METHODS_ORDER]

    rows = []
    for method in unsorted_methods:
        row: Dict[str, Any] = {"Method": method}
        for metric in METRIC_COLS:
            mean_v = df_mean.loc[method, metric] if method in df_mean.index else np.nan
            std_v  = df_std.loc[method, metric]  if method in df_std.index  else np.nan
            if np.isfinite(mean_v) and np.isfinite(std_v):
                row[metric] = f"{mean_v:.3f} ± {std_v:.3f}"
            elif np.isfinite(mean_v):
                row[metric] = f"{mean_v:.3f}"
            else:
                row[metric] = "—"
        comp = ranks_df.loc[method, "Composite rank"] if method in ranks_df.index else np.nan
        row["Composite rank"] = f"{comp:.2f}" if np.isfinite(comp) else "—"
        rows.append(row)

    df_tbl = pd.DataFrame(rows).set_index("Method")
    df_tbl.to_csv(out_dir / "tier2_score_table.csv")
    print(f"  [saved] tier2_score_table.csv")


def export_dotplot_data(df_mean: pd.DataFrame,
                        df_std: pd.DataFrame,
                        tie_series: pd.Series,
                        out_dir: Path) -> None:
    """
    Save tidy long-format table for R/ggplot2.
    One row per method × metric combination.
    """
    ranks_df        = compute_friedman_ranks(df_mean)
    n_methods_total = len(METHODS_ORDER)

    # No sorting applied — use original METHODS_ORDER
    # sorted_methods = ranks_df["Composite rank"].dropna().sort_values().index.tolist()
    unsorted_methods: List[str] = [METHOD_LABELS[m] for m in METHODS_ORDER]

    score_ranges: Dict[str, Tuple[float, float]] = {}
    for metric in METRIC_COLS:
        vals = df_mean[metric].dropna()
        score_ranges[metric] = (
            float(vals.min()), float(vals.max())
        ) if len(vals) > 1 else (0.0, 1.0)

    GAMMA = 0.6
    rows  = []
    for method in unsorted_methods:
        key     = _LABEL_TO_KEY.get(method)
        mcolor  = METHOD_COLORS.get(key, "#333333") if key else "#333333"
        comp_r  = (ranks_df.loc[method, "Composite rank"]
                   if method in ranks_df.index else np.nan)
        tf      = tie_series.get(method, np.nan)
        flagged = isinstance(tf, float) and np.isfinite(tf) and tf > TIE_THRESHOLD

        for col in METRIC_COLS + ["Composite rank"]:
            is_comp  = (col == "Composite rank")
            rank_val = (ranks_df.loc[method, col]
                        if method in ranks_df.index else np.nan)

            if not np.isfinite(rank_val):
                rows.append({
                    "method": method, "metric_key": col,
                    "is_composite": is_comp,
                    "friedman_rank": np.nan,
                    "mean_score": np.nan,
                    "std_score": np.nan,
                    "size_norm": np.nan,
                    "rank_label": "NA",
                    "composite_rank": comp_r,
                    "mean_tie_fraction": tf,
                    "is_flagged": flagged,
                    "method_color": mcolor,
                })
                continue

            mean_v = np.nan
            std_v  = np.nan
            if not is_comp and method in df_mean.index:
                mean_v = df_mean.loc[method, col]
            if not is_comp and method in df_std.index:
                std_v = df_std.loc[method, col]

            if is_comp:
                v = 1.0 - (rank_val - 1) / max(n_methods_total - 1, 1)
            elif np.isfinite(mean_v):
                lo, hi = score_ranges[col]
                span   = hi - lo if hi > lo else 1.0
                v      = (mean_v - lo) / span
            else:
                v = 0.5
            v         = float(np.clip(v, 0.0, 1.0))
            size_norm = v ** GAMMA

            rows.append({
                "method": method,
                "metric_key": col,
                "is_composite": is_comp,
                "friedman_rank": round(rank_val, 4),
                "mean_score":    round(mean_v, 5) if np.isfinite(mean_v) else np.nan,
                "std_score":     round(std_v,  5) if np.isfinite(std_v)  else np.nan,
                "size_norm":     round(size_norm, 5),
                "rank_label":    f"{rank_val:.1f}" if is_comp else str(int(round(rank_val))),
                "composite_rank": round(comp_r, 4) if np.isfinite(comp_r) else np.nan,
                "mean_tie_fraction": round(tf, 5) if np.isfinite(tf) else np.nan,
                "is_flagged":    flagged,
                "method_color":  mcolor,
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "tier2_dotplot_data.csv", index=False)
    print(f"  [saved] tier2_dotplot_data.csv")


def export_tie_fractions(tie_series: pd.Series, out_dir: Path) -> None:
    """Save mean tie fraction per method as CSV."""
    tie_series.to_frame().to_csv(out_dir / "tier2_tie_fractions.csv")
    print(f"  [saved] tier2_tie_fractions.csv")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building aggregated dataset…")

    try:
        df_mean, df_std, tie_series = build_aggregated_data()
        print(f"  methods  : {list(df_mean.index)}")
        print(f"  NaN cells: {int(df_mean.isna().sum().sum())}")

        print("\nSaving raw score tables…")
        export_raw_scores(df_mean, df_std, OUTPUT_DIR)

        print("\nSaving Friedman rank table…")
        export_friedman_ranks(df_mean, OUTPUT_DIR)

        print("\nSaving display score table (mean ± sd)…")
        export_score_table(df_mean, df_std, OUTPUT_DIR)

        print("\nSaving tidy long-format table for R/ggplot2…")
        export_dotplot_data(df_mean, df_std, tie_series, OUTPUT_DIR)

        print("\nSaving tie fractions…")
        export_tie_fractions(tie_series, OUTPUT_DIR)

        print(f"\nAll CSVs saved to:\n  {OUTPUT_DIR}")

    except Exception as exc:
        print(f"[ERROR] {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()