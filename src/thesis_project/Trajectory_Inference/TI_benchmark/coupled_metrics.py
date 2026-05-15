"""
coupled_metrics.py
==================
Evaluation metrics for the coupled integration × TI benchmark (Chapter 6).

These metrics operate across the integration conditions for a fixed TI
method and lineage task — not within a single run. They quantify how sensitive
a TI method's output is to upstream integration choice.

Metrics implemented
-------------------

1. Kendall's W (coefficient of concordance)
   For a fixed TI method across k integration methods, compute the degree
   of agreement across all k pseudotime rankings simultaneously.
   W = 1: all integration methods produce identical pseudotime orderings.
   W ≈ 0: the ordering is essentially random with respect to integration choice.

2. Topology sensitivity
   2a. Pairwise topology Jaccard (mean and std over all valid pairs)
       Compare recovered topology edge sets across all pairs of integration
       conditions. Restricted to common nodes as in Chapter 4 stability.
   2b. Branch count variance
       Across all integration conditions, how much does the number of recovered
       leaves and branchpoints change? High CV signals topology destabilisation.

3. Root placement consistency
   Across all integration conditions, what fraction of runs correctly anchor
   the trajectory root in the biologically designated root population?
   Reported both:
     - over all runs (penalises failures / non-evaluable runs)
     - over evaluable runs only

4. Integration Sensitivity Score (ISS)
   A composite scalar per (TI method, task) pair combining:
     - Kendall's W           (weight: 0.40)
     - Mean topology Jaccard (weight: 0.35)
     - Root placement rate   (weight: 0.25; total-rate version)
   ISS ∈ [0, 1]; high = robust to integration choice, low = heavily sensitive.

Input contract
--------------
All public metric functions accept a list of CoupledRunResult objects, one per
integration condition for a fixed (TI method, lineage task) pair.

Important distinction
---------------------
The sensitivity metrics in this file are TI-centred summaries across
integration methods. They are NOT themselves the formal Chapter 6 marginals
s̄_{t,ℓ}^{(TI)} and s̄_{m,ℓ}^{(int)} from the problem definition.

Those formal marginals average a per-run scalar quality score s_{m,t,ℓ}.
To support that, CoupledRunResult includes an optional field:
    run_quality_score : Optional[float]

If provided, aggregate_ti_marginals() and aggregate_integration_marginals()
compute the formal Chapter 6 marginals from the raw run records.

All metrics are exception-isolated: a failure in one metric never blocks others.
Every result uses the {"status", "value", "reason"} structure consistent with
the Chapter 4 metrics framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .utils import merge_json_shallow, write_json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoupledRunResult:
    """
    Holds the trajectory inference output for one (integration method, TI method,
    lineage task) experiment. One instance per integration condition.

    Parameters
    ----------
    integration_method : str
        Name of the integration method used upstream (e.g. "harmony", "scvi").
    ti_method : str
        Name of the trajectory inference method (e.g. "tscan", "slingshot").
    task_name : str
        Lineage task identifier (e.g. "task1_monocyte_macrophage_tam").
    pseudotime : pd.Series or None
        Cell pseudotime indexed by cell_id. None if the run failed or produced
        an invalid trajectory.
    edge_list : pd.DataFrame or None
        Topology edges with columns ["source", "target"] at minimum.
        None if the method does not produce topology or the run failed.
    root_cell_id : str or None
        The cell_id selected as the root for this run.
    root_group : str
        The biologically designated root cell-type label.
    group_key : str
        The obs column that maps cells to cell-type labels.
    cell_obs : pd.DataFrame
        The obs DataFrame for this run's cells. Must include group_key as a column.
    status : str
        "ok" if trajectory inference succeeded, "error" otherwise.
    run_quality_score : float or None
        Optional scalar per-run biological quality score s_{m,t,ℓ}.
        This is used ONLY for the formal Chapter 6 marginal aggregations,
        not for the TI-centred sensitivity metrics.
    """
    integration_method: str
    ti_method: str
    task_name: str

    pseudotime: Optional[pd.Series] = None
    edge_list: Optional[pd.DataFrame] = None

    root_cell_id: Optional[str] = None
    root_group: str = ""
    group_key: str = ""
    cell_obs: Optional[pd.DataFrame] = None

    status: str = "ok"
    error_msg: Optional[str] = None

    # Formal Chapter 6 marginal support
    run_quality_score: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

class _MetricResult:
    def __init__(
        self,
        status: str,
        value: Optional[Any] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.status = status
        self.value = value
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "value": self.value, "reason": self.reason}


def _ok(value: Any) -> Dict[str, Any]:
    return _MetricResult("ok", value).to_dict()


def _skip(reason: str) -> Dict[str, Any]:
    return _MetricResult("skipped", reason=reason).to_dict()


def _err(reason: str) -> Dict[str, Any]:
    return _MetricResult("error", reason=reason).to_dict()


def _safe(fn, *args, label: str = "", **kwargs) -> Dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return _err(f"{label}: {type(e).__name__}: {e}")


def _validate_same_context(runs: List[CoupledRunResult]) -> None:
    if not runs:
        raise ValueError("runs list is empty")

    ti_methods = {r.ti_method for r in runs}
    task_names = {r.task_name for r in runs}
    if len(ti_methods) != 1:
        raise ValueError(
            f"evaluate_coupled_metrics expects runs from one TI method only; got {sorted(ti_methods)}"
        )
    if len(task_names) != 1:
        raise ValueError(
            f"evaluate_coupled_metrics expects runs from one task only; got {sorted(task_names)}"
        )

    integration_methods = [r.integration_method for r in runs]
    dupes = pd.Series(integration_methods).duplicated()
    if bool(dupes.any()):
        duped_methods = sorted(pd.Series(integration_methods)[dupes].unique().tolist())
        raise ValueError(
            f"evaluate_coupled_metrics expects at most one run per integration method; "
            f"duplicates found for {duped_methods}"
        )


def _finite_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return x[np.isfinite(x.values)]


def _finite_index(s: pd.Series) -> pd.Index:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    return x.index[np.isfinite(x.values)]


def _edge_set_and_nodes(
    edge_list: Optional[pd.DataFrame],
) -> Tuple[frozenset, frozenset]:
    """Return (edge_frozenset, node_frozenset) for an undirected edge list."""
    if edge_list is None or not isinstance(edge_list, pd.DataFrame) or edge_list.empty:
        return frozenset(), frozenset()
    if not {"source", "target"}.issubset(edge_list.columns):
        return frozenset(), frozenset()

    edges = []
    nodes: set = set()
    for a, b in zip(
        edge_list["source"].astype(str).values,
        edge_list["target"].astype(str).values,
    ):
        if a == b or a in ("", "nan") or b in ("", "nan"):
            continue
        edges.append(tuple(sorted((a, b))))
        nodes.add(a)
        nodes.add(b)
    return frozenset(edges), frozenset(nodes)


def _has_valid_topology(run: CoupledRunResult) -> bool:
    if run.status != "ok":
        return False
    E, N = _edge_set_and_nodes(run.edge_list)
    return len(E) > 0 and len(N) >= 2


def _valid_topology_runs(runs: List[CoupledRunResult]) -> List[CoupledRunResult]:
    return [r for r in runs if _has_valid_topology(r)]


def _jaccard(A: frozenset, B: frozenset) -> float:
    union = len(A | B)
    if union == 0:
        return 1.0
    return float(len(A & B) / union)


def _leaf_and_branch_counts(edge_list: Optional[pd.DataFrame]) -> Tuple[int, int]:
    """Return (n_leaves, n_branchpoints) from an undirected edge list."""
    E, nodes = _edge_set_and_nodes(edge_list)
    if not nodes or not E:
        return 0, 0

    deg: Dict[str, int] = {}
    for a, b in E:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1

    leaves = sum(1 for d in deg.values() if d == 1)
    branchpoints = sum(1 for d in deg.values() if d >= 3)
    return int(leaves), int(branchpoints)


def _mean_std(vals: Iterable[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    a = np.asarray(vals, dtype=float)
    return {
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std()), 6),
        "n": int(len(a)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Kendall's W (coefficient of concordance)
# ─────────────────────────────────────────────────────────────────────────────

def metric_kendalls_w(
    runs: List[CoupledRunResult],
) -> Dict[str, Any]:
    """
    Compute Kendall's W (coefficient of concordance) across all integration
    conditions for a fixed TI method.

    Only cells present with finite pseudotime in ALL successful runs are used.
    This avoids undefined ranks from NaN/Inf values.

    Returns
    -------
    dict with status, value dict containing:
        W               : Kendall's W ∈ [0, 1]
        n_rankers       : number of integration methods contributing
        n_cells_common  : number of cells in the common finite intersection
        chi2            : chi-squared approximation test statistic
        p_value         : p-value for H0: rankings are independent
    """
    ok_runs = [r for r in runs if r.status == "ok" and r.pseudotime is not None]
    if len(ok_runs) < 2:
        return _skip(
            f"fewer than 2 successful runs for Kendall's W (n_ok={len(ok_runs)})"
        )

    common_cells = _finite_index(ok_runs[0].pseudotime)  # type: ignore[arg-type]
    for r in ok_runs[1:]:
        common_cells = common_cells.intersection(_finite_index(r.pseudotime))  # type: ignore[arg-type]

    n_common = int(len(common_cells))
    if n_common < 10:
        return _skip(
            f"too few common finite cells across integration conditions "
            f"(n_common={n_common}, minimum=10)"
        )

    k = int(len(ok_runs))  # number of rankers
    n = int(n_common)      # number of jointly ranked cells

    rank_matrix = np.zeros((k, n), dtype=np.float64)
    for i, r in enumerate(ok_runs):
        pt = pd.to_numeric(
            r.pseudotime.reindex(common_cells), errors="coerce"  # type: ignore[union-attr]
        ).astype(float)
        if not np.isfinite(pt.values).all():
            return _err(
                "Kendall's W encountered non-finite pseudotime values after "
                "common-cell filtering; input runs are inconsistent."
            )
        rank_matrix[i, :] = stats.rankdata(pt.values, method="average")

    R = rank_matrix.sum(axis=0)
    R_bar = float(R.mean())
    S = float(np.sum((R - R_bar) ** 2))

    denom = (k ** 2) * (n ** 3 - n)
    if denom == 0:
        return _err("Kendall's W denominator is zero (degenerate input)")

    W = float(12.0 * S / denom)
    W = float(np.clip(W, 0.0, 1.0))

    chi2 = float(k * (n - 1) * W)
    df = int(n - 1)
    p_value = float(1.0 - stats.chi2.cdf(chi2, df=df)) if df > 0 else float("nan")

    return _ok({
        "W": round(W, 6),
        "n_rankers": int(k),
        "n_cells_common": int(n),
        "S": round(S, 4),
        "chi2": round(chi2, 4),
        "df": int(df),
        "p_value": round(p_value, 6),
        "integration_methods": [r.integration_method for r in ok_runs],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2a — Pairwise topology Jaccard across integration conditions
# ─────────────────────────────────────────────────────────────────────────────

def metric_topology_jaccard(
    runs: List[CoupledRunResult],
) -> Dict[str, Any]:
    """
    Compute pairwise topology Jaccard similarity across integration conditions
    for a fixed TI method, using the common-node restriction from Chapter 4.

    Only runs with a valid topology edge list are included.
    """
    topo_runs = _valid_topology_runs(runs)
    if len(topo_runs) < 2:
        return _skip(
            f"fewer than 2 runs with valid topology for topology Jaccard "
            f"(n_topology_ok={len(topo_runs)})"
        )

    edge_sets = []
    node_sets = []
    for r in topo_runs:
        E, N = _edge_set_and_nodes(r.edge_list)
        edge_sets.append(E)
        node_sets.append(N)

    values: List[float] = []
    labels: List[Tuple[str, str]] = []

    for i in range(len(topo_runs)):
        for j in range(i + 1, len(topo_runs)):
            common_nodes = node_sets[i] & node_sets[j]
            if not common_nodes:
                continue

            Ei = frozenset(
                e for e in edge_sets[i]
                if e[0] in common_nodes and e[1] in common_nodes
            )
            Ej = frozenset(
                e for e in edge_sets[j]
                if e[0] in common_nodes and e[1] in common_nodes
            )

            if not Ei and not Ej:
                values.append(1.0)
            else:
                values.append(_jaccard(Ei, Ej))

            labels.append((
                topo_runs[i].integration_method,
                topo_runs[j].integration_method,
            ))

    if not values:
        return _skip("no valid node-overlapping topology pairs found for topology Jaccard")

    arr = np.asarray(values, dtype=float)
    return _ok({
        "jaccard_mean": round(float(arr.mean()), 6),
        "jaccard_std": round(float(arr.std()), 6),
        "jaccard_min": round(float(arr.min()), 6),
        "jaccard_max": round(float(arr.max()), 6),
        "n_pairs": int(len(values)),
        "n_topology_ok": int(len(topo_runs)),
        "pairwise_values": [round(v, 6) for v in values],
        "pair_labels": [list(lbl) for lbl in labels],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2b — Branch count variance across integration conditions
# ─────────────────────────────────────────────────────────────────────────────

def metric_branch_count_variance(
    runs: List[CoupledRunResult],
) -> Dict[str, Any]:
    """
    Quantify how much the number of recovered branches changes across
    integration conditions for a fixed TI method.

    Only runs with a valid topology edge list are included.
    """
    topo_runs = _valid_topology_runs(runs)
    if len(topo_runs) < 2:
        return _skip(
            f"fewer than 2 runs with valid topology for branch count variance "
            f"(n_topology_ok={len(topo_runs)})"
        )

    leaf_counts: List[int] = []
    branch_counts: List[int] = []
    per_method: List[Dict[str, Any]] = []

    for r in topo_runs:
        n_leaves, n_branchpoints = _leaf_and_branch_counts(r.edge_list)
        leaf_counts.append(n_leaves)
        branch_counts.append(n_branchpoints)
        per_method.append({
            "integration_method": r.integration_method,
            "n_leaves": int(n_leaves),
            "n_branchpoints": int(n_branchpoints),
        })

    def _summary(vals: List[int]) -> Dict[str, Any]:
        a = np.asarray(vals, dtype=float)
        mean = float(a.mean())
        std = float(a.std())
        cv = float(std / mean) if mean > 0 else float("nan")
        return {
            "values": [int(v) for v in vals],
            "mean": round(mean, 4),
            "std": round(std, 4),
            "min": int(a.min()),
            "max": int(a.max()),
            "cv": round(cv, 4) if np.isfinite(cv) else None,
        }

    return _ok({
        "leaves": _summary(leaf_counts),
        "branchpoints": _summary(branch_counts),
        "n_topology_ok": int(len(topo_runs)),
        "per_method": per_method,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — Root placement consistency
# ─────────────────────────────────────────────────────────────────────────────

def metric_root_placement_consistency(
    runs: List[CoupledRunResult],
) -> Dict[str, Any]:
    """
    Across all integration conditions, what fraction of runs correctly anchor
    the trajectory root in the biologically designated root population?

    Important design choice:
    - consistency_rate_total uses ALL runs as denominator, so failures and
      non-evaluable runs are penalised.
    - consistency_rate_evaluable uses only evaluable runs as denominator.
    """
    if len(runs) == 0:
        return _skip("runs list is empty")

    n_total = int(len(runs))
    n_failed = int(sum(1 for r in runs if r.status == "error"))
    n_correct = 0
    n_evaluable = 0

    per_method: List[Dict[str, Any]] = []

    for r in runs:
        record = {
            "integration_method": r.integration_method,
            "root_cell_id": r.root_cell_id,
            "root_assigned_group": None,
            "root_group_expected": r.root_group if r.root_group != "" else None,
            "correct": None,
            "counted_as_incorrect_in_total": True,
        }

        evaluable = (
            r.status == "ok"
            and r.root_cell_id is not None
            and r.root_group != ""
            and r.group_key != ""
            and r.cell_obs is not None
        )

        if not evaluable:
            record["note"] = "run not evaluable for root placement"
            per_method.append(record)
            continue

        assert r.cell_obs is not None
        assert r.root_cell_id is not None

        if r.root_cell_id not in r.cell_obs.index:
            record["note"] = "root_cell_id not found in cell_obs.index"
            per_method.append(record)
            continue

        if r.group_key not in r.cell_obs.columns:
            record["note"] = f"group_key '{r.group_key}' not in cell_obs.columns"
            per_method.append(record)
            continue

        assigned_group = str(r.cell_obs.loc[r.root_cell_id, r.group_key])
        correct = assigned_group == str(r.root_group)

        n_evaluable += 1
        if correct:
            n_correct += 1

        record["root_assigned_group"] = assigned_group
        record["correct"] = bool(correct)
        record["counted_as_incorrect_in_total"] = not bool(correct)
        per_method.append(record)

    consistency_rate_total = float(n_correct / n_total)
    consistency_rate_evaluable = (
        float(n_correct / n_evaluable) if n_evaluable > 0 else None
    )

    return _ok({
        "consistency_rate_total": round(consistency_rate_total, 6),
        "consistency_rate_evaluable": (
            round(consistency_rate_evaluable, 6)
            if consistency_rate_evaluable is not None else None
        ),
        "n_correct": int(n_correct),
        "n_evaluable": int(n_evaluable),
        "n_total_runs": int(n_total),
        "n_failed_runs": int(n_failed),
        "root_group_expected": runs[0].root_group if runs else None,
        "group_key": runs[0].group_key if runs else None,
        "per_method": per_method,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Metric 4 — Integration Sensitivity Score (composite)
# ─────────────────────────────────────────────────────────────────────────────

ISS_WEIGHTS: Dict[str, float] = {
    "kendalls_w": 0.40,
    "topology_jaccard": 0.35,
    "root_consistency": 0.25,
}


def metric_integration_sensitivity_score(
    kendalls_w_result: Dict[str, Any],
    topology_jaccard_result: Dict[str, Any],
    root_consistency_result: Dict[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Integration Sensitivity Score (ISS): a composite per-(TI method, task) scalar
    summarising robustness to upstream integration choice.

    Uses root consistency over ALL runs, so failures / non-evaluable runs
    are penalised in the composite.
    """
    if weights is None:
        weights = ISS_WEIGHTS.copy()

    required_keys = {"kendalls_w", "topology_jaccard", "root_consistency"}
    if set(weights.keys()) != required_keys:
        return _err(
            f"ISS weights must have exactly keys {sorted(required_keys)}, got {sorted(weights.keys())}"
        )

    def _extract(result: Dict[str, Any], key: str) -> Optional[float]:
        if result.get("status") != "ok":
            return None
        v = result.get("value")
        if not isinstance(v, dict):
            return None
        val = v.get(key)
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None

    w_val = _extract(kendalls_w_result, "W")
    j_val = _extract(topology_jaccard_result, "jaccard_mean")
    rc_val = _extract(root_consistency_result, "consistency_rate_total")

    scores: Dict[str, Optional[float]] = {
        "kendalls_w": w_val,
        "topology_jaccard": j_val,
        "root_consistency": rc_val,
    }
    available: Dict[str, bool] = {k: (v is not None) for k, v in scores.items()}

    active_weight_sum = sum(weights[k] for k in weights if available[k])
    if active_weight_sum == 0.0:
        return _skip(
            "no component of ISS is available "
            "(Kendall's W, topology Jaccard, and root consistency all unavailable)"
        )

    renormalised = not all(available.values())
    active_weights: Dict[str, float] = {
        k: (weights[k] / active_weight_sum if available[k] else 0.0)
        for k in weights
    }

    ISS = sum(
        active_weights[k] * scores[k]
        for k in weights
        if available[k] and scores[k] is not None
    )
    ISS = float(np.clip(float(ISS), 0.0, 1.0))

    return _ok({
        "ISS": round(ISS, 6),
        "components_used": [k for k, avail in available.items() if avail],
        "components_available": available,
        "component_scores": {
            k: round(float(v), 6) if v is not None else None
            for k, v in scores.items()
        },
        "weights_used": {
            k: round(float(v), 6) for k, v in active_weights.items()
        },
        "weights_renormalised": bool(renormalised),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_coupled_metrics(
    runs: List[CoupledRunResult],
    *,
    iss_weights: Optional[Dict[str, float]] = None,
    out_dir: Optional[str] = None,
    run_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate all coupled benchmark sensitivity metrics for one
    (TI method, lineage task) across all integration conditions.
    """
    _validate_same_context(runs)

    ti_method = runs[0].ti_method
    task_name = runs[0].task_name
    n_ok = int(sum(1 for r in runs if r.status == "ok"))
    n_total = int(len(runs))
    methods = [r.integration_method for r in runs]

    results: Dict[str, Any] = {
        "meta": {
            "ti_method": ti_method,
            "task_name": task_name,
            "n_integration_methods": n_total,
            "n_successful_runs": n_ok,
            "n_failed_runs": n_total - n_ok,
            "integration_methods": methods,
        }
    }

    results["kendalls_w"] = _safe(
        metric_kendalls_w, runs,
        label="kendalls_w",
    )

    results["topology_jaccard"] = _safe(
        metric_topology_jaccard, runs,
        label="topology_jaccard",
    )

    results["branch_count_variance"] = _safe(
        metric_branch_count_variance, runs,
        label="branch_count_variance",
    )

    results["root_placement_consistency"] = _safe(
        metric_root_placement_consistency, runs,
        label="root_placement_consistency",
    )

    results["integration_sensitivity_score"] = _safe(
        metric_integration_sensitivity_score,
        results["kendalls_w"],
        results["topology_jaccard"],
        results["root_placement_consistency"],
        weights=iss_weights,
        label="integration_sensitivity_score",
    )

    if out_dir is not None:
        od = Path(out_dir)
        (od / "tables").mkdir(parents=True, exist_ok=True)
        write_json(
            od / "tables" / "metrics_coupled_summary.json",
            results,
            indent=2,
            sort_keys=False,
            atomic=True,
        )

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"coupled_metrics": results})

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Formal Chapter 6 marginals from per-run scalar quality score
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_ti_marginals(
    runs: List[CoupledRunResult],
) -> Dict[str, Any]:
    """
    Compute the formal TI-marginal scores s̄_{t,ℓ}^{(TI)} from raw run records.

    Definition
    ----------
    For each fixed (ti_method, task_name), average the per-run scalar
    quality score s_{m,t,ℓ} across integration methods m.

    Requirement
    -----------
    CoupledRunResult.run_quality_score must be populated.
    """
    if not runs:
        return {"marginals": [], "n_total_runs": 0}

    rows = []
    for r in runs:
        if r.status != "ok":
            continue
        if r.run_quality_score is None:
            continue
        try:
            q = float(r.run_quality_score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(q):
            continue
        rows.append({
            "ti_method": r.ti_method,
            "task_name": r.task_name,
            "integration_method": r.integration_method,
            "run_quality_score": q,
        })

    if not rows:
        return {"marginals": [], "n_total_runs": len(runs), "reason": "no finite run_quality_score values"}

    df = pd.DataFrame(rows)
    out = []
    for (ti_method, task_name), sub in df.groupby(["ti_method", "task_name"], sort=True):
        vals = sub["run_quality_score"].to_numpy(dtype=float)
        out.append({
            "ti_method": str(ti_method),
            "task_name": str(task_name),
            "s_bar_ti": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
            "n_integration_methods": int(len(vals)),
            "integration_methods": sorted(sub["integration_method"].astype(str).unique().tolist()),
        })

    return {
        "marginals": out,
        "n_total_runs": int(len(runs)),
        "n_used_runs": int(len(df)),
    }


def aggregate_integration_marginals(
    runs: List[CoupledRunResult],
) -> Dict[str, Any]:
    """
    Compute the formal integration-marginal scores s̄_{m,ℓ}^{(int)} from raw run records.

    Definition
    ----------
    For each fixed (integration_method, task_name), average the per-run scalar
    quality score s_{m,t,ℓ} across TI methods t.

    Requirement
    -----------
    CoupledRunResult.run_quality_score must be populated.
    """
    if not runs:
        return {"marginals": [], "n_total_runs": 0}

    rows = []
    for r in runs:
        if r.status != "ok":
            continue
        if r.run_quality_score is None:
            continue
        try:
            q = float(r.run_quality_score)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(q):
            continue
        rows.append({
            "integration_method": r.integration_method,
            "task_name": r.task_name,
            "ti_method": r.ti_method,
            "run_quality_score": q,
        })

    if not rows:
        return {"marginals": [], "n_total_runs": len(runs), "reason": "no finite run_quality_score values"}

    df = pd.DataFrame(rows)
    out = []
    for (integration_method, task_name), sub in df.groupby(["integration_method", "task_name"], sort=True):
        vals = sub["run_quality_score"].to_numpy(dtype=float)
        out.append({
            "integration_method": str(integration_method),
            "task_name": str(task_name),
            "s_bar_int": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
            "n_ti_methods": int(len(vals)),
            "ti_methods": sorted(sub["ti_method"].astype(str).unique().tolist()),
        })

    return {
        "marginals": out,
        "n_total_runs": int(len(runs)),
        "n_used_runs": int(len(df)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# cross-task summary of the TI-centred sensitivity metrics
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_ti_sensitivity_across_tasks(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Summarise the new TI-centred sensitivity metrics across tasks for one TI method.

    This is NOT the formal Chapter 6 TI-marginal. It is an optional
    cross-task descriptive summary of:
      - Kendall's W
      - topology Jaccard mean
      - root consistency total
      - ISS
    """
    def _extract_scalar(res: Dict[str, Any], block: str, key: str) -> Optional[float]:
        v = res.get(block, {}).get("value")
        if not isinstance(v, dict):
            return None
        val = v.get(key)
        if val is None:
            return None
        try:
            f = float(val)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None

    ti_method = all_results[0]["meta"]["ti_method"] if all_results else "unknown"

    agg: Dict[str, Any] = {
        "ti_method": ti_method,
        "n_tasks": int(len(all_results)),
        "kendalls_w": _mean_std(_extract_scalar(r, "kendalls_w", "W") for r in all_results),
        "topology_jaccard_mean": _mean_std(_extract_scalar(r, "topology_jaccard", "jaccard_mean") for r in all_results),
        "root_consistency_total": _mean_std(_extract_scalar(r, "root_placement_consistency", "consistency_rate_total") for r in all_results),
        "ISS": _mean_std(_extract_scalar(r, "integration_sensitivity_score", "ISS") for r in all_results),
    }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "CoupledRunResult",
    "ISS_WEIGHTS",
    "metric_kendalls_w",
    "metric_topology_jaccard",
    "metric_branch_count_variance",
    "metric_root_placement_consistency",
    "metric_integration_sensitivity_score",
    "evaluate_coupled_metrics",
    "aggregate_ti_marginals",
    "aggregate_integration_marginals",
    "aggregate_ti_sensitivity_across_tasks",
]