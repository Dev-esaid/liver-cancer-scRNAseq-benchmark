"""
Trajectory Inference Benchmarking: Evaluation Metrics (metrics.py)

Reference-free / weak-prior evaluation suite for TI benchmarking.

This module is intentionally *strict* about pseudotime validity and *defensive*
about missing inputs, because TI outputs are heterogeneous across methods.

Implemented metrics (publication-oriented)
------------------------------------------
1) Pseudotime sanity / validity gate
   - numeric, finite for all cells, non-constant (+ summary stats).

2) Marker program concordance (Spearman)
   - Spearman correlation of pseudotime vs gene-set score per marker program.

3) Marker program monotonicity (binned violations)
   - Bin pseudotime into quantiles and measure monotonicity of binned program
     scores with respect to expected direction.

4) kNN smoothness
   - Mean absolute deviation of pseudotime from neighbor-averaged pseudotime
     using a kNN connectivity graph.

5) Root purity score (RPS)
   - Enrichment of the expected root label in the earliest α-quantile of
     pseudotime.

6) Topology–Pseudotime Consistency (cluster-level)
   - Spearman correlation between (a) cluster median pseudotime and
     (b) shortest-path distance from root in the inferred cluster-level graph.

Design guarantees
-----------------
- All downstream metrics are skipped if pseudotime sanity fails.
- Each metric call is exception-isolated so one metric cannot crash evaluation.
- JSON output written atomically via utils.write_json.

Notes
-----
- These metrics do not require ground-truth lineage labels.
- "Weak priors" here means using minimal biological expectations:
  a handful of marker gene programs and/or an expected root group label.

"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("metrics.py requires anndata to be installed.") from e

from .shared_types import MetricStatus, MarkerProgram, TaskPriors, TIOutput
from .utils import merge_json_shallow, write_json


class MetricResult:
    """Standard wrapper so every metric returns {status, value, reason}."""

    def __init__(self, status: MetricStatus, value: Optional[Any] = None, reason: Optional[str] = None):
        self.status = status
        self.value = value
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "value": self.value, "reason": self.reason}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _is_sparse(x: Any) -> bool:
    return sp is not None and sp.issparse(x)  # type: ignore[attr-defined]


def _safe_metric(fn, *args, **kwargs) -> Dict[str, Any]:
    """Run a metric and return a serialisable {status,value,reason} dict."""
    try:
        r = fn(*args, **kwargs)
        if isinstance(r, MetricResult):
            return r.to_dict()
        return MetricResult(status="ok", value=r).to_dict()
    except Exception as e:
        return MetricResult(status="error", reason=f"{type(e).__name__}: {e}").to_dict()


def _align_series_to_obs(
    s: pd.Series,
    obs_names: pd.Index,
    *,
    name: str,
    allow_missing: bool,
) -> pd.Series:
    """
    Align a Series to adata.obs_names.

    Contract:
      - Requires a pandas Series with unique index.
      - Reindexes to obs_names (strict by default).
    """
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if s.index.has_duplicates:
        raise ValueError(f"{name} index contains duplicates.")

    # Keep only overlapping cells, then check missingness.
    s = s.loc[s.index.intersection(obs_names)]
    missing = obs_names.difference(s.index)
    if len(missing) > 0 and not allow_missing:
        raise ValueError(f"{name} missing {len(missing)} cells (example: {list(missing[:5])}).")

    return s.reindex(obs_names)


def rescale_0_1(pt: pd.Series) -> pd.Series:
    """Rescale pseudotime to [0, 1]. Raises if constant or has no finite values."""
    pt_num = pd.to_numeric(pt, errors="coerce").astype(float)
    x = pt_num.values
    m = np.isfinite(x)
    if int(m.sum()) == 0:
        raise ValueError("pseudotime has no finite values")

    lo = float(np.nanmin(x[m]))
    hi = float(np.nanmax(x[m]))
    if np.isclose(hi, lo):
        raise ValueError("pseudotime is constant (min == max)")

    y = (pt_num - lo) / (hi - lo)
    return pd.Series(np.asarray(y.values, dtype=float), index=pt.index, name=pt.name or "pseudotime")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """
    Spearman correlation with finite-mask handling.

    Returns (rho, p, n_used). If n_used < 3, returns (nan, nan, n_used).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[m], y[m])
    return float(rho), float(p), n


# -----------------------------------------------------------------------------
# Gene set scoring
# -----------------------------------------------------------------------------
def gene_set_score(
    adata: "ad.AnnData",
    genes: Sequence[str],
    *,
    layer: Optional[str] = None,
    zscore_per_gene: bool = True,
    allow_partial: bool = True,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Compute a per-cell gene-set score (mean z-scored expression over genes).

    This deliberately avoids Scanpy scoring utilities to keep the benchmark
    self-contained and reproducible across environments.

    Parameters
    ----------
    adata:
      AnnData with genes in adata.var_names
    genes:
      Iterable of gene identifiers (strings)
    layer:
      Expression layer to use; if None, uses adata.X
    zscore_per_gene:
      If True, z-score each gene across cells before averaging.
    allow_partial:
      If True, ignore missing genes; otherwise error if any gene missing.

    Returns
    -------
    score: pd.Series indexed by adata.obs_names
    report: dict with gene presence/missing counts
    """
    genes = [str(g) for g in genes]
    present = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]

    if len(present) == 0:
        raise ValueError("No requested genes are present in adata.var_names.")
    if (not allow_partial) and len(missing) > 0:
        raise ValueError(f"Missing {len(missing)} genes (example: {missing[:10]}).")

    X = adata.layers[layer] if layer is not None else adata.X
    idx = [int(adata.var_names.get_loc(g)) for g in present]
    Xsub = X[:, idx]

    # Convert to dense (gene sets should be small).
    if _is_sparse(Xsub):
        Xd = np.asarray(Xsub.toarray(), dtype=float)
    else:
        Xd = np.asarray(Xsub, dtype=float)

    report = {
        "n_requested": int(len(genes)),
        "n_present": int(len(present)),
        "n_missing": int(len(missing)),
        "missing_example": missing[:10],
        "layer": layer,
        "zscore_per_gene": bool(zscore_per_gene),
    }

    if not zscore_per_gene:
        return pd.Series(Xd.mean(axis=1), index=adata.obs_names), report

    mu = Xd.mean(axis=0)
    sd = Xd.std(axis=0, ddof=0)
    # Guard against zero-variance genes.
    sd = np.where(sd == 0, 1.0, sd)

    z = (Xd - mu[None, :]) / sd[None, :]
    score = z.mean(axis=1)
    return pd.Series(score, index=adata.obs_names), report


# -----------------------------------------------------------------------------
# Metric I: Pseudotime sanity gate
# -----------------------------------------------------------------------------
def metric_pseudotime_sanity(pt: pd.Series) -> MetricResult:
    """
    Strict pseudotime validity check.

    Requirements:
      - convertible to float
      - finite for all cells (no NaN/Inf)
      - not constant (>= 2 unique finite values)

    Returns summary statistics if ok.
    """
    pt_num = pd.to_numeric(pt, errors="coerce").astype(float)
    x = pt_num.values
    finite = np.isfinite(x)

    n = int(len(x))
    n_finite = int(finite.sum())
    frac = float(n_finite / max(1, n))

    if n_finite == 0:
        return MetricResult(status="error", reason="pseudotime has no finite values")

    if n_finite != n:
        return MetricResult(status="error", reason=f"pseudotime contains non-finite values (finite_fraction={frac:.3f})")

    # Count unique after finiteness gate.
    n_unique = int(pd.Series(x).nunique(dropna=True))
    if n_unique < 2:
        return MetricResult(status="error", reason="pseudotime is constant (n_unique < 2)")

    # A very large fraction of ties can be a red flag for some TI methods.
    tie_frac = float(1.0 - (n_unique / max(1, n)))

    return MetricResult(
        status="ok",
        value={
            "n_cells": n,
            "n_finite": n_finite,
            "finite_fraction": frac,
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "n_unique": n_unique,
            "tie_fraction": tie_frac,
        },
    )


# -----------------------------------------------------------------------------
# Metric IV: Marker program concordance & monotonicity
# -----------------------------------------------------------------------------
def _expected_direction(prog: MarkerProgram) -> str:
    d = str(getattr(prog, "expected_direction", "either")).lower()
    if d not in ("positive", "negative", "either"):
        d = "either"
    return d


def metric_marker_concordance(
    adata: "ad.AnnData",
    pt: pd.Series,
    programs: Sequence[MarkerProgram],
    *,
    layer: Optional[str] = None,
) -> MetricResult:
    """
    Spearman correlation between pseudotime and marker program score.

    Output includes:
      - rho_raw: raw correlation
      - rho_aligned: aligned so "higher is better" given expected_direction
        (positive: rho, negative: -rho, either: abs(rho))
    """
    if not programs:
        return MetricResult(status="skipped", reason="no marker_programs provided")

    rows: List[Dict[str, Any]] = []
    for prog in programs:
        pname = str(getattr(prog, "name", "program"))
        try:
            score, rep = gene_set_score(adata, getattr(prog, "genes", []), layer=layer, zscore_per_gene=True, allow_partial=True)
            rho, p, n = safe_spearman(pt.values, score.values)

            d = _expected_direction(prog)
            if d == "positive":
                rho_aligned = rho
            elif d == "negative":
                rho_aligned = -rho
            else:
                rho_aligned = float(abs(rho)) if np.isfinite(rho) else np.nan

            rows.append(
                {
                    "program": pname,
                    "expected_direction": d,
                    "used_for_root": bool(getattr(prog, "used_for_root", False)),
                    "spearman_rho_raw": rho,
                    "spearman_rho_aligned": float(rho_aligned) if np.isfinite(rho_aligned) else np.nan,
                    "spearman_p": p,
                    "n_used": n,
                    "genes_requested": int(rep["n_requested"]),
                    "genes_present": int(rep["n_present"]),
                    "genes_missing": int(rep["n_missing"]),
                }
            )
        except Exception as e:
            rows.append({"program": pname, "status": "error", "error": f"{type(e).__name__}: {e}"})

    return MetricResult(status="ok", value=rows)


def _binned_means(values: np.ndarray, pt: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute mean(values) within pseudotime quantile bins.

    Returns an array of bin means in increasing pseudotime-bin order.
    """
    v = np.asarray(values, dtype=float)
    t = np.asarray(pt, dtype=float)
    m = np.isfinite(v) & np.isfinite(t)
    if int(m.sum()) < 3:
        return np.array([])

    # Use qcut on the filtered pseudotime values.
    try:
        bins = pd.qcut(pd.Series(t[m]), q=int(n_bins), labels=False, duplicates="drop")
    except Exception:
        return np.array([])

    if bins.isna().all():
        return np.array([])

    df = pd.DataFrame({"bin": bins.to_numpy(dtype="float"), "val": v[m]})
    df = df.dropna()
    if df.empty:
        return np.array([])

    means = df.groupby("bin", sort=True)["val"].mean().to_numpy()
    return np.asarray(means, dtype=float)


def _monotonicity_score(bin_means: np.ndarray, direction: str, tol: float) -> float:
    """
    Monotonicity score based on adjacent-bin violation counting.

    Score in [0,1] where 1 means no violations.

    direction:
      - positive: bin_means should be non-decreasing
      - negative: non-increasing
      - either: best of positive/negative (direction-free monotonicity)
    """
    if bin_means.size < 2:
        return np.nan

    diffs = np.diff(bin_means)
    denom = max(1, diffs.size)

    if direction == "positive":
        violations = int(np.sum(diffs < -float(tol)))
    elif direction == "negative":
        violations = int(np.sum(diffs > float(tol)))
    else:
        # direction-free: whichever direction gives fewer violations.
        v_pos = int(np.sum(diffs < -float(tol)))
        v_neg = int(np.sum(diffs > float(tol)))
        violations = min(v_pos, v_neg)

    return float(1.0 - (violations / denom))


def metric_marker_monotonicity(
    adata: "ad.AnnData",
    pt: pd.Series,
    programs: Sequence[MarkerProgram],
    *,
    layer: Optional[str] = None,
    n_bins: int = 20,
    tol: float = 0.0,
) -> MetricResult:
    """
    Quantile-binned marker monotonicity.

    For each marker program:
      - score cells by gene set
      - bin pseudotime into quantiles
      - compute mean score per bin
      - compute monotonicity score based on violations (see _monotonicity_score)
    """
    if not programs:
        return MetricResult(status="skipped", reason="no marker_programs provided")

    out: List[Dict[str, Any]] = []
    for prog in programs:
        pname = str(getattr(prog, "name", "program"))
        try:
            score, rep = gene_set_score(adata, getattr(prog, "genes", []), layer=layer, zscore_per_gene=True, allow_partial=True)
            means = _binned_means(score.values, pt.values, n_bins=int(n_bins))
            d = _expected_direction(prog)
            mono = _monotonicity_score(means, d, float(tol))

            out.append(
                {
                    "program": pname,
                    "expected_direction": d,
                    "used_for_root": bool(getattr(prog, "used_for_root", False)),
                    "binned_monotonicity": float(mono) if np.isfinite(mono) else np.nan,
                    "n_bins_requested": int(n_bins),
                    "n_bins_used": int(len(means)),
                    "tol": float(tol),
                    "genes_present": int(rep["n_present"]),
                }
            )
        except Exception as e:
            out.append({"program": pname, "status": "error", "error": f"{type(e).__name__}: {e}"})

    return MetricResult(status="ok", value=out)


# -----------------------------------------------------------------------------
# Metric III-ish: kNN smoothness on connectivity graph
# -----------------------------------------------------------------------------
def metric_knn_smoothness(pt: pd.Series, connectivities: Any) -> MetricResult:
    """
    Pseudotime smoothness over a kNN graph.

    Given a sparse connectivities matrix C (n_cells x n_cells), compute:
      neigh_mean[i] = sum_j C[i,j] * pt[j] / sum_j C[i,j]
      MAD = mean_i |pt[i] - neigh_mean[i]|

    Notes:
      - The diagonal is explicitly zeroed to avoid self-smoothing artefacts.
      - Cells with zero degree contribute |pt - 0| unless handled; we instead
        treat zero-degree rows as having neigh_mean == pt (so contribution 0)
        by setting row_sum to 1 and dot to pt (implemented below).
    """
    if connectivities is None:
        return MetricResult(status="skipped", reason="connectivities not provided")
    if not _is_sparse(connectivities):
        return MetricResult(status="skipped", reason="connectivities must be a sparse matrix")

    x = pd.to_numeric(pt, errors="coerce").astype(float).values
    if not np.isfinite(x).all():
        return MetricResult(status="error", reason="pseudotime contains non-finite values (expected pre-gated pt)")

    C = connectivities.tocsr(copy=True)

    if C.shape[0] != x.shape[0] or C.shape[1] != x.shape[0]:
        return MetricResult(status="error", reason=f"connectivities shape {C.shape} incompatible with n_cells={x.shape[0]}")

    # Remove diagonal self-connections (common in some kNN graphs).
    C.setdiag(0)
    C.eliminate_zeros()

    rs = np.asarray(C.sum(axis=1)).ravel().astype(float)
    zero_deg = int(np.sum(rs == 0))

    # For zero-degree nodes, define neigh_mean == pt (perfectly smooth).
    # Implemented by temporarily setting row_sum to 1 and dot(x) to x.
    rs_safe = rs.copy()
    rs_safe[rs_safe == 0] = 1.0
    neigh_sum = np.asarray(C.dot(x)).ravel().astype(float)
    neigh_mean = neigh_sum / rs_safe
    neigh_mean[rs == 0] = x[rs == 0]

    mad = float(np.mean(np.abs(x - neigh_mean)))

    return MetricResult(
        status="ok",
        value={
            "mean_abs_deviation_to_neighbor_mean": mad,
            "n_zero_degree": int(zero_deg),
            "mean_row_sum": float(np.mean(rs)),
        },
    )


# -----------------------------------------------------------------------------
# Metric IV.3: Root purity
# -----------------------------------------------------------------------------
def metric_root_purity(
    adata: "ad.AnnData",
    pt: pd.Series,
    priors: TaskPriors,
    *,
    alpha: float = 0.1,
    group_key: Optional[str] = None,
) -> MetricResult:
    """
    Root purity score (RPS): enrichment of expected root label in earliest α tail.

    Requires:
      - expected root label: priors.root_group
      - label key in adata.obs: group_key (or priors.root_group_key / priors.group_key)

    Output:
      - root_purity: fraction of tail cells with root label
      - baseline_fraction: global fraction of root label
      - fold_enrichment: root_purity / baseline_fraction
    """
    root_label = getattr(priors, "root_group", None)
    if root_label is None:
        return MetricResult(status="skipped", reason="priors.root_group not provided")

    if group_key is None:
        group_key = getattr(priors, "root_group_key", None) or getattr(priors, "group_key", None)
    if group_key is None:
        return MetricResult(status="skipped", reason="no group_key/root_group_key available for root purity")

    if group_key not in adata.obs.columns:
        return MetricResult(status="skipped", reason=f"group_key '{group_key}' not in adata.obs")

    labels = adata.obs[group_key].astype(str)
    root_label = str(root_label)

    if not (0.0 < float(alpha) < 1.0):
        return MetricResult(status="error", reason="alpha must be in (0,1)")

    thr = float(np.nanquantile(pt.values.astype(float), float(alpha)))
    tail = pt.values.astype(float) <= thr
    n_tail = int(np.sum(tail))
    if n_tail < 10:
        return MetricResult(status="skipped", reason=f"too few cells in alpha tail (n_tail={n_tail})")

    y = (labels.values == root_label).astype(int)
    root_purity = float(y[tail].mean()) if n_tail > 0 else np.nan
    baseline = float(y.mean()) if y.size > 0 else np.nan
    fold = float(root_purity / baseline) if (np.isfinite(baseline) and baseline > 0) else np.nan

    return MetricResult(
        status="ok",
        value={
            "group_key": str(group_key),
            "root_label": root_label,
            "alpha": float(alpha),
            "n_tail": int(n_tail),
            "root_purity": root_purity,
            "baseline_fraction": baseline,
            "fold_enrichment": fold,
        },
    )


# -----------------------------------------------------------------------------
# Topology–pseudotime consistency (cluster-level)
# -----------------------------------------------------------------------------
def _build_graph_from_edges(
    edges: pd.DataFrame,
    *,
    distance_transform: str = "inverse_weight",
    force_undirected: bool = True,
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Any]]:
    """
    Build an adjacency list from a topology edge table.

    Expected columns:
      - source (required)
      - target (required)
      - weight (optional; treated as similarity by default)
      - directed (optional bool)

    distance_transform:
      - 'inverse_weight': dist = 1/weight if weight>0 else 1
      - 'unit': dist = 1 for all edges
      - 'neglog_weight': dist = -log(weight) for 0<weight<=1 else 1 (common for similarities)
    """
    if edges is None or edges.empty:
        return {}, {"status": "skipped", "reason": "empty topology edge list"}

    if not {"source", "target"}.issubset(edges.columns):
        return {}, {"status": "error", "reason": "topology edge list missing required columns {'source','target'}"}

    adj: Dict[str, List[Tuple[str, float]]] = {}

    n_rows = int(edges.shape[0])
    n_used = 0
    n_skipped = 0

    has_weight = "weight" in edges.columns
    has_directed = "directed" in edges.columns

    for _, row in edges.iterrows():
        a = str(row["source"])
        b = str(row["target"])
        if a in ("", "nan") or b in ("", "nan") or a == b:
            n_skipped += 1
            continue

        # Determine if this edge should be treated as directed.
        directed = bool(row["directed"]) if has_directed else False
        if force_undirected:
            directed = False

        w = None
        if has_weight:
            try:
                w = float(row["weight"])
            except Exception:
                w = None

        if distance_transform == "unit":
            dist = 1.0
        elif distance_transform == "neglog_weight":
            if w is not None and np.isfinite(w) and w > 0:
                dist = float(-np.log(w))
            else:
                dist = 1.0
        else:  # inverse_weight (default)
            if w is not None and np.isfinite(w) and w > 0:
                dist = float(1.0 / w)
            else:
                dist = 1.0

        adj.setdefault(a, []).append((b, dist))
        if not directed:
            adj.setdefault(b, []).append((a, dist))
        n_used += 1

    report = {
        "status": "ok",
        "n_rows": n_rows,
        "n_edges_used": int(n_used),
        "n_edges_skipped": int(n_skipped),
        "distance_transform": str(distance_transform),
        "force_undirected": bool(force_undirected),
        "has_weight": bool(has_weight),
        "has_directed": bool(has_directed),
        "n_nodes": int(len(adj)),
    }
    if n_used == 0:
        report["status"] = "skipped"
        report["reason"] = "no valid edges after filtering"
    return adj, report


def _dijkstra_distances(adj: Dict[str, List[Tuple[str, float]]], root: str) -> Dict[str, float]:
    """Single-source shortest path distances using Dijkstra (non-negative weights)."""
    dist: Dict[str, float] = {root: 0.0}
    heap: List[Tuple[float, str]] = [(0.0, root)]

    while heap:
        d_u, u = heapq.heappop(heap)
        if d_u != dist.get(u, np.inf):
            continue
        for v, w in adj.get(u, []):
            nd = d_u + float(w)
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def metric_topology_pseudotime_consistency(
    adata: "ad.AnnData",
    pt: pd.Series,
    topology_edges: Optional[pd.DataFrame],
    priors: TaskPriors,
    *,
    group_key: Optional[str] = None,
    distance_transform: str = "inverse_weight",
    force_undirected: bool = True,
) -> MetricResult:
    """
    Cluster-level topology–pseudotime consistency.

    Steps:
      1) Build graph from topology_edges.
      2) Choose root node using priors.root_group.
      3) Compute shortest-path distances from root to each node.
      4) Compute median pseudotime per cluster from adata.obs[group_key].
      5) Spearman correlation between cluster median pseudotime and graph distance.

    Interpretation:
      - High positive rho: pseudotime increases as you move away from the root in the topology.
      - Negative rho: pseudotime direction is inconsistent with the chosen root (or topology is wrong).
    """
    if topology_edges is None or topology_edges.empty:
        return MetricResult(status="skipped", reason="topology_edges not provided/empty")

    root_group = getattr(priors, "root_group", None)
    if root_group is None:
        return MetricResult(status="skipped", reason="priors.root_group not provided (needed for topology distances)")

    if group_key is None:
        group_key = getattr(priors, "group_key", None) or getattr(priors, "root_group_key", None)
    if group_key is None:
        return MetricResult(status="skipped", reason="no group_key available for cluster median pseudotime")

    if group_key not in adata.obs.columns:
        return MetricResult(status="skipped", reason=f"group_key '{group_key}' not in adata.obs")

    adj, rep = _build_graph_from_edges(
        topology_edges,
        distance_transform=str(distance_transform),
        force_undirected=bool(force_undirected),
    )
    if rep.get("status") != "ok":
        return MetricResult(status="skipped", reason=str(rep.get("reason", "unable to build graph")))

    root = str(root_group)
    if root not in adj:
        return MetricResult(status="skipped", reason=f"root_group '{root}' not present in topology graph nodes")

    dist = _dijkstra_distances(adj, root)

    # Cluster median pseudotime
    labels = adata.obs[group_key].astype(str)
    df = pd.DataFrame({"cluster": labels.values, "pt": pt.values.astype(float)})
    med = df.groupby("cluster", sort=False)["pt"].median()

    # Align clusters in both.
    common = med.index.intersection(pd.Index(dist.keys()).astype(str))
    if len(common) < 3:
        return MetricResult(
            status="skipped",
            reason=f"too few clusters for correlation after alignment (n_common={len(common)})",
        )

    x = np.asarray([dist[str(c)] for c in common], dtype=float)
    y = med.loc[common].to_numpy(dtype=float)

    rho, p, n = safe_spearman(x, y)
    return MetricResult(
        status="ok",
        value={
            "group_key": str(group_key),
            "root_group": root,
            "distance_transform": str(distance_transform),
            "force_undirected": bool(force_undirected),
            "spearman_rho": rho,
            "spearman_rho_abs": float(abs(rho)) if np.isfinite(rho) else np.nan,
            "spearman_p": p,
            "n_clusters_used": int(n),
            "n_nodes_graph": int(rep.get("n_nodes", 0)),
            "n_nodes_reachable": int(len(dist)),
            "graph_build_report": rep,
        },
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def evaluate_all_metrics(
    adata: "ad.AnnData",
    ti: TIOutput,
    priors: TaskPriors,
    *,
    expression_layer: Optional[str] = None,
    connectivities: Any = None,
    out_dir: Optional[str] = None,
    run_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate all implemented metrics for a single TI output.

    Parameters
    ----------
    adata:
      AnnData for the task (cells used for TI evaluation)
    ti:
      TIOutput with fields pseudotime and edge_list/topology edges
    priors:
      TaskPriors (marker programs, group_key, root_group, ...)
    expression_layer:
      Expression layer to use for gene-set scoring (None => adata.X)
    connectivities:
      Sparse kNN connectivities matrix for kNN smoothness
    out_dir:
      If provided, writes a metrics_summary.json into out_dir/tables/
    run_config_path:
      If provided, merges the results into an existing run_config json.

    Returns
    -------
    results dict (JSON-serialisable)
    """
    results: Dict[str, Any] = {
        "meta": {
            "dataset_name": getattr(ti, "dataset_name", None),
            "task_name": getattr(ti, "task_name", None),
            "method_name": getattr(ti, "method_name", None),
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
        }
    }

    def _skip_all(reason: str) -> Dict[str, Any]:
        return {"status": "skipped", "value": None, "reason": reason}

    # 1) Align + validate pseudotime
    try:
        pt_raw = _align_series_to_obs(getattr(ti, "pseudotime"), adata.obs_names, name="pseudotime", allow_missing=False)
        pt_raw = pd.to_numeric(pt_raw, errors="coerce").astype(float)
    except Exception as e:
        err = f"invalid pseudotime alignment: {type(e).__name__}: {e}"
        results["pseudotime_sanity_raw"] = MetricResult(status="error", reason=err).to_dict()

        # Skip everything else.
        results["marker_concordance"] = _skip_all(err)
        results["marker_monotonicity"] = _skip_all(err)
        results["knn_smoothness"] = _skip_all(err)
        results["root_purity"] = _skip_all(err)
        results["topology_pseudotime_consistency"] = _skip_all(err)
        return _finalise(results, out_dir=out_dir, run_config_path=run_config_path)

    sanity = metric_pseudotime_sanity(pt_raw)
    results["pseudotime_sanity_raw"] = sanity.to_dict()
    if sanity.status != "ok":
        err = sanity.reason or "invalid pseudotime"
        results["marker_concordance"] = _skip_all(err)
        results["marker_monotonicity"] = _skip_all(err)
        results["knn_smoothness"] = _skip_all(err)
        results["root_purity"] = _skip_all(err)
        results["topology_pseudotime_consistency"] = _skip_all(err)
        return _finalise(results, out_dir=out_dir, run_config_path=run_config_path)

    # 2) Rescale to [0,1] for numerical stability (rank metrics are unaffected).
    pt = rescale_0_1(pt_raw)

    # 3) Run metrics (exception-isolated)
    results["marker_concordance"] = _safe_metric(
        metric_marker_concordance, adata, pt, getattr(priors, "marker_programs", []), layer=expression_layer
    )
    results["marker_monotonicity"] = _safe_metric(
        metric_marker_monotonicity, adata, pt, getattr(priors, "marker_programs", []), layer=expression_layer
    )
    results["knn_smoothness"] = _safe_metric(metric_knn_smoothness, pt, connectivities)

    results["root_purity"] = _safe_metric(metric_root_purity, adata, pt, priors)

    # Topology–pseudotime consistency uses the inferred topology edges.
    topo_edges = getattr(ti, "topology_edges", None)
    if topo_edges is None:
        topo_edges = getattr(ti, "edge_list", None)
    results["topology_pseudotime_consistency"] = _safe_metric(
        metric_topology_pseudotime_consistency, adata, pt, topo_edges, priors
    )

    return _finalise(results, out_dir=out_dir, run_config_path=run_config_path)


def _finalise(results: Dict[str, Any], *, out_dir: Optional[str], run_config_path: Optional[str]) -> Dict[str, Any]:
    """Write JSON outputs and merge into run_config if requested."""
    if out_dir is not None:
        od = Path(out_dir)
        (od / "tables").mkdir(parents=True, exist_ok=True)
        write_json(od / "tables" / "metrics_summary.json", results, indent=2, sort_keys=False, atomic=True)

    if run_config_path is not None:
        merge_json_shallow(run_config_path, {"metrics": results})

    return results


__all__ = [
    "MetricResult",
    "rescale_0_1",
    "safe_spearman",
    "gene_set_score",
    "metric_pseudotime_sanity",
    "metric_marker_concordance",
    "metric_marker_monotonicity",
    "metric_knn_smoothness",
    "metric_root_purity",
    "metric_topology_pseudotime_consistency",
    "evaluate_all_metrics",
]
