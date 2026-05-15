"""
Trajectory Inference Benchmarking: Dataset Priors Loader (priors.py)

Loads and validates curated per-dataset, per-task priors from JSON/YAML config
files and returns canonical TaskPriors objects defined in shared_types.py.

Fixes applied
-------------
- All data types (TaskPriors, MarkerProgram, OrdinalCovariate, RootPrior) are
  imported from shared_types.py — no local re-definitions (fixes C2, H3, H4).
- All JSON writes use utils.merge_json_shallow for atomic writes (fixes H6, L2).
- priors.py now imports utils.py (previously isolated from it) (fixes L2).
- TaskPriors.root_group is accessible via property defined in shared_types.py,
  so method_runner.priors.root_group works correctly on all load paths (fixes C3).

Design
------
- Priors are stored as JSON (or YAML, if PyYAML is installed) under a structured
  directory: <priors_root>/<dataset>/<task>.json
- The loader validates required fields and returns immutable TaskPriors objects.
- All collection fields are converted to tuples for immutability/hashability.
- A manifest of loaded priors can be written atomically to run_config.json.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# All shared types from the single canonical source
from .shared_types import (
    MarkerProgram,
    OrdinalCovariate,
    RootPrior,
    TaskPriors,
)
from .utils import (
    OpResult,
    ensure_dir,
    merge_json_shallow,
    ok,
    skipped,
    error,
    read_json,
)


# ---------------------------------------------------------------------------
# Optional YAML support
# ---------------------------------------------------------------------------

try:
    import yaml as _yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_marker_program(raw: Dict[str, Any]) -> MarkerProgram:
    name = str(raw.get("name", "unnamed"))
    genes_raw = raw.get("genes", [])
    if not isinstance(genes_raw, (list, tuple)):
        raise ValueError(f"MarkerProgram '{name}': 'genes' must be a list, got {type(genes_raw).__name__}")
    genes: Tuple[str, ...] = tuple(str(g) for g in genes_raw)
    direction = str(raw.get("expected_direction", "unknown")).lower()
    if direction not in ("positive", "negative", "unknown"):
        warnings.warn(
            f"MarkerProgram '{name}': expected_direction '{direction}' is not in "
            f"('positive','negative','unknown'). Defaulting to 'unknown'.",
            stacklevel=3,
        )
        direction = "unknown"
    used_for_root = bool(raw.get("used_for_root", False))
    return MarkerProgram(
        name=name,
        genes=genes,
        expected_direction=direction,  # type: ignore[arg-type]
        used_for_root=used_for_root,
    )


def _parse_ordinal_covariate(raw: Dict[str, Any]) -> OrdinalCovariate:
    name = str(raw.get("name", "unnamed"))
    obs_key = str(raw.get("obs_key", ""))
    if not obs_key:
        raise ValueError(f"OrdinalCovariate '{name}': 'obs_key' is required.")
    rank_map_raw = raw.get("rank_map", {})
    if not isinstance(rank_map_raw, dict):
        raise ValueError(f"OrdinalCovariate '{name}': 'rank_map' must be a dict.")
    rank_map = {str(k): float(v) for k, v in rank_map_raw.items()}
    return OrdinalCovariate(name=name, obs_key=obs_key, rank_map=rank_map)


def _parse_root_prior(raw: Optional[Dict[str, Any]]) -> Optional[RootPrior]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"'root' must be a dict, got {type(raw).__name__}")
    early_raw = raw.get("early_marker_programs", [])
    if not isinstance(early_raw, (list, tuple)):
        early_raw = []
    return RootPrior(
        root_cell_id=raw.get("root_cell_id") or None,
        root_group=raw.get("root_group") or None,
        root_group_key=raw.get("root_group_key") or None,
        early_marker_programs=tuple(str(p) for p in early_raw),
        k_nearest_centroid=int(raw["k_nearest_centroid"])
        if raw.get("k_nearest_centroid") is not None else None,
    )


def _dict_to_task_priors(
    raw: Dict[str, Any],
    *,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
) -> TaskPriors:
    """
    Parse a raw dict (from JSON/YAML) into a canonical TaskPriors object.
    dataset/task override whatever is in the dict (allows path-inferred values).
    """
    ds = dataset or raw.get("dataset") or None
    tk = task or raw.get("task") or None

    marker_programs: Tuple[MarkerProgram, ...] = tuple(
        _parse_marker_program(p) for p in raw.get("marker_programs", [])
    )
    ordinal_covariates: Tuple[OrdinalCovariate, ...] = tuple(
        _parse_ordinal_covariate(c) for c in raw.get("ordinal_covariates", [])
    )
    root = _parse_root_prior(raw.get("root"))
    terminal_labels_raw = raw.get("terminal_labels")
    terminal_labels: Optional[Tuple[str, ...]] = (
        tuple(str(x) for x in terminal_labels_raw)
        if terminal_labels_raw is not None else None
    )

    return TaskPriors(
        dataset=ds,
        task=tk,
        group_key=raw.get("group_key") or None,
        marker_programs=marker_programs,
        ordinal_covariates=ordinal_covariates,
        terminal_label_key=raw.get("terminal_label_key") or None,
        terminal_labels=terminal_labels,
        root=root,
        notes=raw.get("notes") or None,
    )


# ---------------------------------------------------------------------------
# Public API: file-based loading
# ---------------------------------------------------------------------------

def load_task_priors(
    path: Union[str, Path],
    *,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
) -> Tuple[TaskPriors, OpResult]:
    """
    Load a TaskPriors object from a JSON or YAML file.

    Returns (priors, OpResult). On failure, returns a minimal empty TaskPriors
    and an error OpResult so callers can handle gracefully.

    The returned TaskPriors always has a working .root_group property (defined
    in shared_types.py). This resolves the C3 AttributeError.
    """
    p = Path(path)
    if not p.exists():
        return TaskPriors(dataset=dataset, task=task), error(f"priors file not found: {p}")

    suffix = p.suffix.lower()
    raw: Optional[Dict[str, Any]] = None

    if suffix in (".json",):
        raw, res = read_json(p, strict=True)
        if raw is None:
            return TaskPriors(dataset=dataset, task=task), res
    elif suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            return TaskPriors(dataset=dataset, task=task), error(
                "PyYAML is required to load .yaml priors files. Install with: pip install pyyaml"
            )
        try:
            import yaml
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return TaskPriors(dataset=dataset, task=task), error(
                    f"YAML file must decode to a mapping: {p}"
                )
        except Exception as e:
            return TaskPriors(dataset=dataset, task=task), error(f"failed to parse YAML: {e}")
    else:
        return TaskPriors(dataset=dataset, task=task), error(
            f"unsupported priors file format '{suffix}'. Use .json or .yaml"
        )

    try:
        priors = _dict_to_task_priors(raw, dataset=dataset, task=task)
        return priors, ok({"path": str(p), "n_programs": len(priors.marker_programs),
                           "n_covariates": len(priors.ordinal_covariates)})
    except Exception as e:
        return TaskPriors(dataset=dataset, task=task), error(f"failed to parse priors dict: {e}")


def load_task_priors_from_registry(
    priors_root: Union[str, Path],
    dataset: str,
    task: str,
    *,
    extensions: Sequence[str] = (".json", ".yaml", ".yml"),
) -> Tuple[TaskPriors, OpResult]:
    """
    Load priors from a structured registry directory:
      <priors_root>/<dataset>/<task>.<ext>

    Tries extensions in order. Returns first match.
    """
    root = Path(priors_root)
    for ext in extensions:
        candidate = root / dataset / f"{task}{ext}"
        if candidate.exists():
            return load_task_priors(candidate, dataset=dataset, task=task)
    return TaskPriors(dataset=dataset, task=task), error(
        f"no priors file found for dataset='{dataset}', task='{task}' "
        f"under {root} (tried extensions: {list(extensions)})"
    )


def load_priors_from_dict(
    raw: Dict[str, Any],
    *,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
) -> Tuple[TaskPriors, OpResult]:
    """
    Construct a TaskPriors directly from a raw dictionary (e.g. from a larger
    config file). Useful when priors are embedded in run configs.
    """
    try:
        priors = _dict_to_task_priors(raw, dataset=dataset, task=task)
        return priors, ok({"source": "dict", "n_programs": len(priors.marker_programs)})
    except Exception as e:
        return TaskPriors(dataset=dataset, task=task), error(f"failed to parse priors dict: {e}")


# ---------------------------------------------------------------------------
# Registry introspection
# ---------------------------------------------------------------------------

def list_available_priors(
    priors_root: Union[str, Path],
    *,
    extensions: Sequence[str] = (".json", ".yaml", ".yml"),
) -> List[Dict[str, str]]:
    """
    Scan the priors registry and return a list of dicts with dataset/task/path.
    Useful for enumerating all available priors configs.
    """
    root = Path(priors_root)
    if not root.exists():
        return []
    found: List[Dict[str, str]] = []
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for f in sorted(dataset_dir.iterdir()):
            if f.suffix.lower() in extensions:
                found.append({
                    "dataset": dataset_dir.name,
                    "task": f.stem,
                    "path": str(f),
                })
    return found


def validate_priors_against_adata(
    priors: TaskPriors,
    adata: Any,
    *,
    warn_only: bool = True,
) -> OpResult:
    """
    Validate that priors fields reference obs columns and gene names that
    actually exist in the provided AnnData. Returns ok/skipped/error.

    warn_only=True: issues are emitted as warnings but do not raise.
    warn_only=False: first issue raises ValueError.
    """
    issues: List[str] = []

    # group_key
    if priors.group_key is not None and priors.group_key not in adata.obs.columns:
        issues.append(f"group_key '{priors.group_key}' not in adata.obs")

    # terminal_label_key
    if (priors.terminal_label_key is not None
            and priors.terminal_label_key not in adata.obs.columns):
        issues.append(f"terminal_label_key '{priors.terminal_label_key}' not in adata.obs")

    # ordinal covariate obs_keys
    for cov in priors.ordinal_covariates:
        if cov.obs_key not in adata.obs.columns:
            issues.append(f"ordinal covariate obs_key '{cov.obs_key}' not in adata.obs")

    # marker gene presence (count only; partial is ok)
    for prog in priors.marker_programs:
        var_names = set(adata.var_names)
        present = [g for g in prog.genes if g in var_names]
        frac = len(present) / max(1, len(prog.genes))
        if frac == 0.0:
            issues.append(
                f"marker program '{prog.name}': none of {len(prog.genes)} genes present in adata.var_names"
            )
        elif frac < 0.5:
            issues.append(
                f"marker program '{prog.name}': only {len(present)}/{len(prog.genes)} genes present "
                f"({100*frac:.0f}%)"
            )

    if not issues:
        return ok({"validated": True, "priors_dataset": priors.dataset, "priors_task": priors.task})

    msg = f"{len(issues)} validation issue(s): " + "; ".join(issues)
    if warn_only:
        warnings.warn(f"validate_priors_against_adata: {msg}", stacklevel=2)
        return skipped(reason=msg, details={"issues": issues})

    raise ValueError(f"validate_priors_against_adata: {msg}")


# ---------------------------------------------------------------------------
# Logging priors metadata to run_config  (atomic via utils)
# ---------------------------------------------------------------------------

def log_priors_to_run_config(
    priors: TaskPriors,
    run_config_path: Union[str, Path],
) -> OpResult:
    """
    Merge a summary of loaded priors into logs/run_config.json.
    Uses utils.merge_json_shallow for atomic writes (tempfile + os.replace).
    """
    payload: Dict[str, Any] = {
        "priors": {
            "dataset": priors.dataset,
            "task": priors.task,
            "group_key": priors.group_key,
            "n_marker_programs": len(priors.marker_programs),
            "marker_programs": [p.name for p in priors.marker_programs],
            "n_ordinal_covariates": len(priors.ordinal_covariates),
            "terminal_label_key": priors.terminal_label_key,
            "root_group": priors.root_group,
            "root_group_key": priors.root_group_key,
            "has_root_cell_id": priors.root_cell_id is not None,
        }
    }
    return merge_json_shallow(run_config_path, payload)


# ---------------------------------------------------------------------------
# Convenience: build minimal TaskPriors from CLI / inline arguments
# ---------------------------------------------------------------------------

def build_minimal_priors(
    *,
    dataset: Optional[str] = None,
    task: Optional[str] = None,
    group_key: Optional[str] = None,
    root_group: Optional[str] = None,
    root_group_key: Optional[str] = None,
    root_cell_id: Optional[str] = None,
    terminal_label_key: Optional[str] = None,
    terminal_labels: Optional[Sequence[str]] = None,
) -> TaskPriors:
    """
    Build a minimal TaskPriors from scalar arguments (e.g. CLI flags).
    Useful as the fallback when no priors file is available.
    """
    root: Optional[RootPrior] = None
    if any(x is not None for x in (root_group, root_group_key, root_cell_id)):
        root = RootPrior(
            root_group=root_group,
            root_group_key=root_group_key,
            root_cell_id=root_cell_id,
        )
    return TaskPriors(
        dataset=dataset,
        task=task,
        group_key=group_key,
        root=root,
        terminal_label_key=terminal_label_key,
        terminal_labels=tuple(terminal_labels) if terminal_labels is not None else None,
    )


__all__ = [
    "load_task_priors",
    "load_task_priors_from_registry",
    "load_priors_from_dict",
    "list_available_priors",
    "validate_priors_against_adata",
    "log_priors_to_run_config",
    "build_minimal_priors",
]