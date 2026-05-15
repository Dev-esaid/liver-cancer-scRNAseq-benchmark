"""
Trajectory Inference Benchmarking: Shared Data Contracts (shared_types.py)

Single authoritative source for ALL cross-module data types used in the TI
benchmarking framework.

Update (2026-02-25)
------------------
- Added TIOutput.extras (Optional[Dict[str, Any]]) to preserve method-specific,
  JSON-safe provenance (e.g., R log paths, parameter values).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import pandas as pd


Status = Literal["ok", "skipped", "error"]
WriteStatus = Literal["ok", "skipped", "error"]
MetricStatus = Literal["ok", "skipped", "error"]

ExpectedDirection = Literal["positive", "negative", "unknown"]


@dataclass(frozen=True)
class MarkerProgram:
    """
    Curated marker gene program for a dataset x task.

    expected_direction:
    - "positive": expected to increase with pseudotime (terminal-like)
    - "negative": expected to decrease with pseudotime (early-like)
    - "unknown" : no sign expectation

    used_for_root:
    Provenance flag — True if this program was used (directly or indirectly)
    in root cell selection. Enables circularity-aware metric reporting.
    """
    name: str
    genes: Tuple[str, ...]
    expected_direction: ExpectedDirection = "unknown"
    used_for_root: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["genes"] = list(self.genes)
        return d


@dataclass(frozen=True)
class OrdinalCovariate:
    """
    Curated ordinal covariate mapping (e.g. compartment ordering, severity).

    rank_map maps category string → numeric rank (lower = earlier in trajectory).
    """
    name: str
    obs_key: str
    rank_map: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "obs_key": self.obs_key,
            "rank_map": {str(k): float(v) for k, v in self.rank_map.items()},
        }


@dataclass(frozen=True)
class RootPrior:
    """
    Optional root selection provenance hints.

    Stored alongside a run for full reproducibility. The actual root selection
    algorithm lives in root_selection.py; this object only carries the hints.
    """
    root_cell_id: Optional[str] = None
    root_group: Optional[str] = None
    root_group_key: Optional[str] = None
    early_marker_programs: Tuple[str, ...] = field(default_factory=tuple)
    k_nearest_centroid: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cell_id": self.root_cell_id,
            "root_group": self.root_group,
            "root_group_key": self.root_group_key,
            "early_marker_programs": list(self.early_marker_programs),
            "k_nearest_centroid": self.k_nearest_centroid,
        }


@dataclass(frozen=True)
class TaskPriors:
    """
    Curated priors for one dataset x task run.

    This is the SINGLE canonical definition used by every module.

    Key design decisions
    --------------------
    - root_group is a top-level convenience property derived from self.root,
      so code can always access priors.root_group safely.
    - All collection fields are tuples (immutable/hashable).
    """
    dataset: Optional[str] = None
    task: Optional[str] = None

    group_key: Optional[str] = None

    marker_programs: Tuple[MarkerProgram, ...] = field(default_factory=tuple)
    ordinal_covariates: Tuple[OrdinalCovariate, ...] = field(default_factory=tuple)

    terminal_label_key: Optional[str] = None
    terminal_labels: Optional[Tuple[str, ...]] = None

    root: Optional[RootPrior] = None
    notes: Optional[str] = None

    @property
    def root_group(self) -> Optional[str]:
        if self.root is not None:
            return self.root.root_group
        return None

    @property
    def root_group_key(self) -> Optional[str]:
        if self.root is not None:
            return self.root.root_group_key
        return None

    @property
    def root_cell_id(self) -> Optional[str]:
        if self.root is not None:
            return self.root.root_cell_id
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "task": self.task,
            "group_key": self.group_key,
            "marker_programs": [p.to_dict() for p in self.marker_programs],
            "ordinal_covariates": [c.to_dict() for c in self.ordinal_covariates],
            "terminal_label_key": self.terminal_label_key,
            "terminal_labels": list(self.terminal_labels) if self.terminal_labels is not None else None,
            "root": self.root.to_dict() if self.root is not None else None,
            "notes": self.notes,
        }


@dataclass
class TIOutput:
    """
    Unified standardized TI method output.

    Required
    --------
    pseudotime : pd.Series
        Index = cell_id

    Optional
    --------
    topology_matrix : pd.DataFrame
    edge_list : pd.DataFrame  (columns: source, target, weight?, directed?)
    branch_labels : pd.Series
    terminal_probabilities : pd.DataFrame (index=cells, columns=terminal states)

    Metadata
    --------
    method_name, dataset_name, task_name : Optional[str]

    extras
    ------
    JSON-safe, method-specific provenance/parameters. Keep it SMALL and serializable.
    (No DataFrames here; put large matrices into the canonical output tables instead.)
    """
    pseudotime: pd.Series

    topology_matrix: Optional[pd.DataFrame] = None
    edge_list: Optional[pd.DataFrame] = None
    branch_labels: Optional[pd.Series] = None
    terminal_probabilities: Optional[pd.DataFrame] = None

    method_name: Optional[str] = None
    dataset_name: Optional[str] = None
    task_name: Optional[str] = None

    extras: Optional[Dict[str, Any]] = None


__all__ = [
    "Status",
    "WriteStatus",
    "MetricStatus",
    "ExpectedDirection",
    "MarkerProgram",
    "OrdinalCovariate",
    "RootPrior",
    "TaskPriors",
    "TIOutput",
]