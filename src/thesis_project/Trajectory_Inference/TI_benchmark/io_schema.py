"""
Trajectory Inference Benchmarking: I/O Schema (io_schema.py)

Update (2026-02-25)
------------------
- Added require_any_finite_pseudotime flags to allow writing placeholder
  pseudotimes (all NaN) on failed runs while still preserving schema.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("io_schema.py requires anndata to be installed.") from e

from .shared_types import WriteStatus
from .utils import merge_json_shallow, write_csv, write_json


IO_SCHEMA_VERSION = "1.0.0"

TABLE_CELL_PSEUDOTIME = "tables/cell_pseudotime.csv"
TABLE_GROUP_SUMMARY = "tables/group_summary_pseudotime.csv"
TABLE_TOPOLOGY_EDGES = "tables/topology_edges.csv"
TABLE_BRANCH_LABELS = "tables/cell_branch_labels.csv"
TABLE_TERMINAL_PROBS = "tables/terminal_probabilities.csv"
LOG_SCHEMA_MANIFEST = "logs/io_schema_manifest.json"
LOG_RUN_CONFIG = "logs/run_config.json"


@dataclass
class WriteArtifact:
    path: str
    status: WriteStatus
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    reason: Optional[str] = None
    notes: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchemaManifest:
    schema_version: str
    artifacts: Dict[str, WriteArtifact]
    global_notes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifacts": {k: v.to_dict() for k, v in self.artifacts.items()},
            "global_notes": self.global_notes,
        }


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    tables_dir: Path
    figures_dir: Path
    logs_dir: Path
    adata_dir: Path

    @staticmethod
    def from_run_dir(run_dir: Union[str, Path]) -> "RunPaths":
        rd = Path(run_dir)
        return RunPaths(
            run_dir=rd,
            tables_dir=rd / "tables",
            figures_dir=rd / "figures",
            logs_dir=rd / "logs",
            adata_dir=rd / "adata",
        )

    def mkdirs(self) -> "RunPaths":
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.adata_dir.mkdir(parents=True, exist_ok=True)
        return self


def _is_sparse(x: Any) -> bool:
    return sp is not None and sp.issparse(x)  # type: ignore[attr-defined]


def align_series_to_obs(
    s: pd.Series,
    obs_names: pd.Index,
    *,
    name: str,
    allow_missing: bool,
) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    if s.index.has_duplicates:
        raise ValueError(f"{name} index contains duplicates.")
    s2 = s.loc[s.index.intersection(obs_names)]
    missing = obs_names.difference(s2.index)
    if len(missing) > 0 and not allow_missing:
        raise ValueError(f"{name} missing {len(missing)} cells (example: {list(missing[:5])}).")
    return s2.reindex(obs_names)


def coerce_numeric_series(
    s: pd.Series,
    *,
    name: str,
    require_any_finite: bool = True,
    require_all_finite: bool = False,
) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    n_finite = int(np.isfinite(out.values).sum())
    if require_any_finite and n_finite == 0:
        raise ValueError(f"{name} has no finite numeric values after coercion.")
    if require_all_finite and n_finite != int(len(out)):
        raise ValueError(f"{name} contains non-finite values (finite={n_finite}/{len(out)}).")
    return out


def validate_edge_list(
    edge_list: pd.DataFrame,
    *,
    require_columns: Sequence[str] = ("source", "target"),
    allow_self_loops: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if edge_list is None:
        raise TypeError("edge_list is None; pass a DataFrame or handle None upstream.")
    if not isinstance(edge_list, pd.DataFrame):
        raise TypeError("edge_list must be a pandas DataFrame.")
    missing_cols = [c for c in require_columns if c not in edge_list.columns]
    if missing_cols:
        raise ValueError(f"edge_list missing required columns: {missing_cols}")

    df = edge_list.copy()
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)

    before = int(df.shape[0])
    df = df.replace({"source": {"nan": np.nan}, "target": {"nan": np.nan}})
    df = df.dropna(subset=["source", "target"])
    df = df[(df["source"].str.len() > 0) & (df["target"].str.len() > 0)]

    if not allow_self_loops:
        df = df[df["source"] != df["target"]]

    if "directed" in df.columns:
        df["directed"] = df["directed"].fillna(False).astype(bool)
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").astype(float)

    df = df.drop_duplicates()

    report = {
        "n_rows_in": before,
        "n_rows_out": int(df.shape[0]),
        "n_dropped": int(before - df.shape[0]),
        "columns": list(df.columns),
        "allow_self_loops": bool(allow_self_loops),
    }
    return df, report


def sanitize_obs_keys(adata: ad.AnnData, keys: Sequence[str]) -> Tuple[list, list]:
    present, missing = [], []
    for k in keys:
        if k in adata.obs.columns:
            present.append(k)
        else:
            missing.append(k)
    return present, missing


def _write_df_csv(df: pd.DataFrame, path: Path) -> WriteArtifact:
    res = write_csv(path, df, index=False, atomic=True)
    if res.status == "ok":
        return WriteArtifact(path=str(path), status="ok", n_rows=int(df.shape[0]), n_cols=int(df.shape[1]))
    return WriteArtifact(
        path=str(path),
        status="error",
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        reason=res.reason,
        notes=res.details,
    )


def _write_placeholder(path: Path, columns: Sequence[str]) -> None:
    df = pd.DataFrame(columns=list(columns))
    write_csv(path, df, index=False, atomic=True)


def write_cell_pseudotime_table(
    adata: ad.AnnData,
    pseudotime: pd.Series,
    out_path: Union[str, Path],
    *,
    group_key: Optional[str] = None,
    include_obs_keys: Optional[Sequence[str]] = None,
    allow_missing_pseudotime: bool = False,
    require_any_finite_pseudotime: bool = True,
    require_all_finite_pseudotime: bool = True,
) -> Tuple[WriteArtifact, Dict[str, Any]]:
    notes: Dict[str, Any] = {}
    pt = align_series_to_obs(pseudotime, adata.obs_names, name="pseudotime", allow_missing=allow_missing_pseudotime)
    pt = coerce_numeric_series(
        pt,
        name="pseudotime",
        require_any_finite=bool(require_any_finite_pseudotime),
        require_all_finite=bool(require_all_finite_pseudotime),
    )

    df = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "pseudotime": pt.values})

    if group_key is not None:
        if group_key in adata.obs.columns:
            df[group_key] = adata.obs[group_key].astype(str).values
        else:
            notes["group_key_missing"] = group_key

    if include_obs_keys is not None:
        present, missing = sanitize_obs_keys(adata, include_obs_keys)
        for k in present:
            df[k] = adata.obs[k].values
        if missing:
            notes["missing_obs_keys"] = missing

    artifact = _write_df_csv(df, Path(out_path))
    return artifact, notes


def write_group_summary_pseudotime(
    adata: ad.AnnData,
    pseudotime: pd.Series,
    out_path: Union[str, Path],
    *,
    group_key: Optional[str],
    allow_missing_pseudotime: bool = False,
    require_any_finite_pseudotime: bool = True,
) -> Tuple[WriteArtifact, Dict[str, Any]]:
    notes: Dict[str, Any] = {}
    path = Path(out_path)
    cols = ["group", "n_cells", "n_nan_pseudotime", "pseudotime_mean", "pseudotime_median", "pseudotime_min", "pseudotime_max"]

    if group_key is None:
        notes["skipped_reason"] = "group_key not provided"
        _write_placeholder(path, cols)
        return WriteArtifact(path=str(path), status="skipped", n_rows=0, n_cols=len(cols), reason=notes["skipped_reason"]), notes

    if group_key not in adata.obs.columns:
        notes["skipped_reason"] = f"group_key '{group_key}' not found in adata.obs"
        _write_placeholder(path, cols)
        return WriteArtifact(path=str(path), status="skipped", n_rows=0, n_cols=len(cols), reason=notes["skipped_reason"]), notes

    pt = align_series_to_obs(pseudotime, adata.obs_names, name="pseudotime", allow_missing=allow_missing_pseudotime)
    pt = coerce_numeric_series(pt, name="pseudotime", require_any_finite=bool(require_any_finite_pseudotime), require_all_finite=False)

    groups = adata.obs[group_key].astype(str)
    df = pd.DataFrame({"group": groups.values, "pseudotime": pt.values})
    g = df.groupby("group", sort=True)["pseudotime"]

    out = pd.DataFrame({
        "group": g.size().index.astype(str),
        "n_cells": g.size().values.astype(int),
        "n_nan_pseudotime": df.groupby("group", sort=True)["pseudotime"].apply(lambda x: int(pd.isna(x).sum())).values.astype(int),
        "pseudotime_mean": g.mean().values.astype(float),
        "pseudotime_median": g.median().values.astype(float),
        "pseudotime_min": g.min().values.astype(float),
        "pseudotime_max": g.max().values.astype(float),
    })

    artifact = _write_df_csv(out, path)
    return artifact, notes


def write_topology_edges(
    edge_list: Optional[pd.DataFrame],
    out_path: Union[str, Path],
    *,
    allow_self_loops: bool = False,
) -> Tuple[WriteArtifact, Dict[str, Any]]:
    notes: Dict[str, Any] = {}
    path = Path(out_path)
    contract_cols = ["source", "target", "weight", "directed"]

    if edge_list is None or (isinstance(edge_list, pd.DataFrame) and edge_list.empty):
        notes["skipped_reason"] = "edge_list not provided (None or empty)"
        _write_placeholder(path, contract_cols)
        return WriteArtifact(path=str(path), status="skipped", n_rows=0, n_cols=len(contract_cols), reason=notes["skipped_reason"]), notes

    df, rep = validate_edge_list(edge_list, allow_self_loops=allow_self_loops)
    notes["validation"] = rep

    for c in contract_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[contract_cols]

    artifact = _write_df_csv(df, path)
    return artifact, notes


def write_branch_labels(
    adata: ad.AnnData,
    branch_labels: Optional[pd.Series],
    out_path: Union[str, Path],
) -> Tuple[WriteArtifact, Dict[str, Any]]:
    notes: Dict[str, Any] = {}
    path = Path(out_path)
    cols = ["cell_id", "branch_label"]

    if branch_labels is None:
        notes["skipped_reason"] = "branch_labels not provided"
        _write_placeholder(path, cols)
        return WriteArtifact(path=str(path), status="skipped", n_rows=0, n_cols=len(cols), reason=notes["skipped_reason"]), notes

    bl = align_series_to_obs(branch_labels, adata.obs_names, name="branch_labels", allow_missing=True)
    df = pd.DataFrame({"cell_id": adata.obs_names.astype(str), "branch_label": bl.astype("string").values})
    artifact = _write_df_csv(df, path)
    return artifact, notes


def write_terminal_probabilities(
    adata: ad.AnnData,
    terminal_probabilities: Optional[pd.DataFrame],
    out_path: Union[str, Path],
) -> Tuple[WriteArtifact, Dict[str, Any]]:
    notes: Dict[str, Any] = {}
    path = Path(out_path)

    if terminal_probabilities is None:
        notes["skipped_reason"] = "terminal_probabilities not provided"
        _write_placeholder(path, ["cell_id"])
        return WriteArtifact(path=str(path), status="skipped", n_rows=0, n_cols=1, reason=notes["skipped_reason"]), notes

    if not isinstance(terminal_probabilities, pd.DataFrame):
        raise TypeError("terminal_probabilities must be a pandas DataFrame (or None).")
    if terminal_probabilities.index.has_duplicates:
        raise ValueError("terminal_probabilities index contains duplicates.")

    dfp = terminal_probabilities.loc[terminal_probabilities.index.intersection(adata.obs_names)].copy()
    dfp = dfp.reindex(adata.obs_names)
    for c in dfp.columns:
        dfp[c] = pd.to_numeric(dfp[c], errors="coerce").astype(float)

    out = pd.DataFrame({"cell_id": adata.obs_names.astype(str)})
    for c in dfp.columns.astype(str):
        out[str(c)] = dfp[c].values

    artifact = _write_df_csv(out, path)
    return artifact, notes


def write_all_canonical_outputs(
    adata: ad.AnnData,
    *,
    pseudotime: pd.Series,
    run_dir: Union[str, Path],
    group_key: Optional[str] = None,
    include_obs_keys: Optional[Sequence[str]] = None,
    edge_list: Optional[pd.DataFrame] = None,
    branch_labels: Optional[pd.Series] = None,
    terminal_probabilities: Optional[pd.DataFrame] = None,
    allow_missing_pseudotime: bool = False,
    require_any_finite_pseudotime: bool = True,
    require_all_finite_pseudotime: bool = True,
    allow_self_loops: bool = False,
    merge_into_run_config: Optional[Dict[str, Any]] = None,
) -> SchemaManifest:
    paths = RunPaths.from_run_dir(run_dir).mkdirs()
    artifacts: Dict[str, WriteArtifact] = {}
    global_notes: Dict[str, Any] = {}

    # 1) cell pseudotime
    cell_cols = ["cell_id", "pseudotime"] + ([group_key] if group_key else [])
    try:
        art, notes = write_cell_pseudotime_table(
            adata,
            pseudotime,
            paths.run_dir / TABLE_CELL_PSEUDOTIME,
            group_key=group_key,
            include_obs_keys=include_obs_keys,
            allow_missing_pseudotime=allow_missing_pseudotime,
            require_any_finite_pseudotime=require_any_finite_pseudotime,
            require_all_finite_pseudotime=require_all_finite_pseudotime,
        )
        artifacts["cell_pseudotime"] = art
        if notes:
            global_notes["cell_pseudotime_notes"] = notes
    except Exception as e:
        _write_placeholder(paths.run_dir / TABLE_CELL_PSEUDOTIME, cell_cols)
        artifacts["cell_pseudotime"] = WriteArtifact(
            path=str(paths.run_dir / TABLE_CELL_PSEUDOTIME),
            status="error",
            n_rows=0,
            n_cols=len(cell_cols),
            reason=f"{type(e).__name__}: {e}",
        )

    # 2) group summary
    group_cols = ["group", "n_cells", "n_nan_pseudotime", "pseudotime_mean", "pseudotime_median", "pseudotime_min", "pseudotime_max"]
    try:
        art, notes = write_group_summary_pseudotime(
            adata,
            pseudotime,
            paths.run_dir / TABLE_GROUP_SUMMARY,
            group_key=group_key,
            allow_missing_pseudotime=allow_missing_pseudotime,
            require_any_finite_pseudotime=require_any_finite_pseudotime,
        )
        artifacts["group_summary_pseudotime"] = art
        if notes:
            global_notes["group_summary_notes"] = notes
    except Exception as e:
        _write_placeholder(paths.run_dir / TABLE_GROUP_SUMMARY, group_cols)
        artifacts["group_summary_pseudotime"] = WriteArtifact(
            path=str(paths.run_dir / TABLE_GROUP_SUMMARY),
            status="error",
            n_rows=0,
            n_cols=len(group_cols),
            reason=f"{type(e).__name__}: {e}",
        )

    # 3) topology edges
    edge_cols = ["source", "target", "weight", "directed"]
    try:
        art, notes = write_topology_edges(edge_list, paths.run_dir / TABLE_TOPOLOGY_EDGES, allow_self_loops=allow_self_loops)
        artifacts["topology_edges"] = art
        if notes:
            global_notes["topology_edges_notes"] = notes
    except Exception as e:
        _write_placeholder(paths.run_dir / TABLE_TOPOLOGY_EDGES, edge_cols)
        artifacts["topology_edges"] = WriteArtifact(
            path=str(paths.run_dir / TABLE_TOPOLOGY_EDGES),
            status="error",
            n_rows=0,
            n_cols=len(edge_cols),
            reason=f"{type(e).__name__}: {e}",
        )

    # 4) branch labels
    bl_cols = ["cell_id", "branch_label"]
    try:
        art, notes = write_branch_labels(adata, branch_labels, paths.run_dir / TABLE_BRANCH_LABELS)
        artifacts["cell_branch_labels"] = art
        if notes:
            global_notes["branch_labels_notes"] = notes
    except Exception as e:
        _write_placeholder(paths.run_dir / TABLE_BRANCH_LABELS, bl_cols)
        artifacts["cell_branch_labels"] = WriteArtifact(
            path=str(paths.run_dir / TABLE_BRANCH_LABELS),
            status="error",
            n_rows=0,
            n_cols=len(bl_cols),
            reason=f"{type(e).__name__}: {e}",
        )

    # 5) terminal probabilities
    try:
        art, notes = write_terminal_probabilities(adata, terminal_probabilities, paths.run_dir / TABLE_TERMINAL_PROBS)
        artifacts["terminal_probabilities"] = art
        if notes:
            global_notes["terminal_probabilities_notes"] = notes
    except Exception as e:
        _write_placeholder(paths.run_dir / TABLE_TERMINAL_PROBS, ["cell_id"])
        artifacts["terminal_probabilities"] = WriteArtifact(
            path=str(paths.run_dir / TABLE_TERMINAL_PROBS),
            status="error",
            n_rows=0,
            n_cols=1,
            reason=f"{type(e).__name__}: {e}",
        )

    if merge_into_run_config is not None:
        res = merge_json_shallow(paths.run_dir / LOG_RUN_CONFIG, merge_into_run_config)
        global_notes["run_config_merge"] = {"status": res.status, "keys": list(merge_into_run_config.keys())}

    manifest = SchemaManifest(schema_version=IO_SCHEMA_VERSION, artifacts=artifacts, global_notes=global_notes)

    manifest_path = paths.run_dir / LOG_SCHEMA_MANIFEST
    write_json(manifest_path, manifest.to_dict(), indent=2, sort_keys=False, atomic=True)

    return manifest


__all__ = [
    "IO_SCHEMA_VERSION",
    "RunPaths",
    "SchemaManifest",
    "WriteArtifact",
    "align_series_to_obs",
    "coerce_numeric_series",
    "validate_edge_list",
    "sanitize_obs_keys",
    "write_cell_pseudotime_table",
    "write_group_summary_pseudotime",
    "write_topology_edges",
    "write_branch_labels",
    "write_terminal_probabilities",
    "write_all_canonical_outputs",
    "TABLE_CELL_PSEUDOTIME",
    "TABLE_GROUP_SUMMARY",
    "TABLE_TOPOLOGY_EDGES",
    "TABLE_BRANCH_LABELS",
    "TABLE_TERMINAL_PROBS",
    "LOG_SCHEMA_MANIFEST",
    "LOG_RUN_CONFIG",
]