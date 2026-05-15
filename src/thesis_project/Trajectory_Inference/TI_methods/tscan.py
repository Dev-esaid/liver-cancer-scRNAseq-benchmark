#!/usr/bin/env python3
"""
TSCAN adapter (R) for TI benchmarking (TIOutput contract).

Hardening update (2026-04-06)
-----------------------------
This version fixes the most likely post-plotting-update failure mode:
pseudotime.csv is produced by R, but Python reindexing turns it into all-NaN
because of cell-ID mismatch, stale temp outputs, or silent CSV carry-over.

Main fixes:
1) Fresh temp directory per run (main + each bootstrap) via cleanup/recreate.
2) Strong cell-ID sanitation on both Python and R handoff sides:
   - cast to str
   - strip whitespace
   - detect duplicates
3) Detailed overlap diagnostics before reindexing pseudotime to adata.obs_names.
4) Better R failure messages with tail of tscan_R.log included in exception.
5) Optional diagnostic JSON/CSV artifacts written to tmp_dir for debugging.
6) Safe AnnData export support for nullable strings.

Requires:
- group_key/cluster_key in adata.obs
- embedding in adata.obsm[rep_key] (default: X_pca)
- R script writes:
    out_dir/pseudotime.csv  columns: cell_id, pseudotime
    out_dir/edges.csv       columns: source, target, weight, directed
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad

try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
TI_ROOT = THIS_DIR.parent
sys.path.insert(0, str(TI_ROOT))

from TI_benchmark.shared_types import TIOutput
from TI_benchmark.method_runner import run_benchmark, build_arg_parser, args_to_run_spec

logger = logging.getLogger(__name__)
METHOD_NAME = "tscan"


# ---------------------------------------------------------------------
# CLI (method-specific)
# ---------------------------------------------------------------------
def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("TSCAN options")
    g.add_argument("--rscript", default="Rscript", help="Path to Rscript binary")
    g.add_argument(
        "--tscan-n-dims",
        type=int,
        default=30,
        help="Number of embedding dims passed to TSCAN",
    )
    g.add_argument(
        "--pt-normalize-01",
        action="store_true",
        help="Normalize pseudotime to [0, 1] if finite.",
    )


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def sh_quote(s: str) -> str:
    if any(c in s for c in (" ", "\t", "\n", '"', "'", "(", ")", "&", "|", ";")):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _safe_tail(path: Path, n_lines: int = 80) -> str:
    try:
        if not path.exists():
            return f"[missing log: {path}]"
        txt = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = txt[-n_lines:]
        return "\n".join(tail)
    except Exception as e:
        return f"[could not read log tail: {e}]"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _sanitize_index_like(x: pd.Index | pd.Series | np.ndarray | list) -> pd.Index:
    vals = pd.Index(pd.Series(list(x), dtype="object").astype(str).str.strip().tolist())
    return vals


def _sanitize_series_index(s: pd.Series) -> pd.Series:
    out = s.copy()
    out.index = _sanitize_index_like(out.index)
    return out


def _normalize_01(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce").astype(float)
    m = np.isfinite(vals.values)
    if m.sum() == 0:
        return vals
    lo = float(np.nanmin(vals.values[m]))
    hi = float(np.nanmax(vals.values[m]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vals
    return (vals - lo) / (hi - lo)


def _validate_pseudotime(pt: pd.Series) -> Tuple[bool, str]:
    if pt is None or len(pt) == 0:
        return False, "pseudotime is empty"
    vals = pd.to_numeric(pt, errors="coerce").astype(float).values
    n_finite = int(np.isfinite(vals).sum())
    if n_finite == 0:
        return False, "pseudotime has no finite values"
    return True, f"ok (finite={n_finite}/{len(vals)})"


def _ensure_edge_list_schema(edge_list: Optional[pd.DataFrame]) -> pd.DataFrame:
    if edge_list is None or edge_list.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    df = edge_list.copy()

    if "from" in df.columns and "source" not in df.columns:
        df = df.rename(columns={"from": "source"})
    if "to" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"to": "target"})

    required = {"source", "target"}
    if not required.issubset(df.columns):
        logger.warning("edges.csv missing required columns %s. Found=%s", required, list(df.columns))
        return pd.DataFrame(columns=["source", "target", "weight", "directed"])

    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "directed" not in df.columns:
        df["directed"] = False

    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(float)

    if df["directed"].dtype != bool:
        df["directed"] = (
            df["directed"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "t", "yes", "y"])
        )

    df = df[
        (df["source"] != df["target"]) &
        (df["source"] != "") &
        (df["target"] != "") &
        (df["source"].ne("nan")) &
        (df["target"].ne("nan"))
    ].reset_index(drop=True)

    return df


def _build_topology_matrix(edge_list: pd.DataFrame) -> Optional[pd.DataFrame]:
    if edge_list is None or edge_list.empty:
        return None

    nodes = sorted(set(edge_list["source"]) | set(edge_list["target"]))
    if len(nodes) == 0:
        return None

    idx = {n: i for i, n in enumerate(nodes)}
    mat = np.zeros((len(nodes), len(nodes)), dtype=float)

    for s, t, w in zip(edge_list["source"], edge_list["target"], edge_list["weight"]):
        if s == t:
            continue
        i, j = idx.get(s), idx.get(t)
        if i is None or j is None:
            continue
        ww = float(w) if np.isfinite(w) else 1.0
        mat[i, j] = max(mat[i, j], ww)
        mat[j, i] = max(mat[j, i], ww)

    return pd.DataFrame(mat, index=nodes, columns=nodes)


def _cast_nullable_strings_inplace(adata: ad.AnnData) -> None:
    def _fix_df(df: pd.DataFrame, name: str) -> None:
        for col in df.columns:
            s = df[col]
            try:
                if pd.api.types.is_string_dtype(s.dtype):
                    df[col] = s.astype("object")
            except Exception as e:
                logger.warning("Could not cast %s[%s] away from string dtype: %s", name, col, e)

    _fix_df(adata.obs, "obs")
    _fix_df(adata.var, "var")


def _check_duplicate_ids(ids: pd.Index, label: str) -> None:
    dup = ids[ids.duplicated()].tolist()
    if dup:
        preview = dup[:10]
        raise RuntimeError(
            f"{label} contains duplicated IDs (n_dup={len(dup)}). "
            f"Examples: {preview}"
        )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not write diagnostics JSON %s: %s", path, e)


def _diagnose_pseudotime_alignment(
    adata_ids: pd.Index,
    pt_ids: pd.Index,
    tmp_dir: Path,
) -> Dict[str, Any]:
    adata_set = set(adata_ids)
    pt_set = set(pt_ids)

    common = adata_set & pt_set
    only_adata = [x for x in adata_ids if x not in pt_set][:20]
    only_pt = [x for x in pt_ids if x not in adata_set][:20]

    payload = {
        "n_adata_ids": int(len(adata_ids)),
        "n_pt_ids": int(len(pt_ids)),
        "n_common": int(len(common)),
        "frac_common_vs_adata": float(len(common) / max(1, len(adata_ids))),
        "frac_common_vs_pt": float(len(common) / max(1, len(pt_ids))),
        "adata_only_examples": only_adata,
        "pt_only_examples": only_pt,
        "adata_head": list(map(str, adata_ids[:10])),
        "pt_head": list(map(str, pt_ids[:10])),
    }

    _write_json(tmp_dir / "alignment_diagnostics.json", payload)
    return payload


# ---------------------------------------------------------------------
# R runner
# ---------------------------------------------------------------------
def _run_rscript(
    *,
    rscript: str,
    script_path: Path,
    args: Dict[str, Any],
    log_path: Path,
) -> None:
    if not script_path.exists():
        raise FileNotFoundError(f"R script not found: {script_path}")

    cmd = [str(rscript), str(script_path)]
    for k, v in args.items():
        if v is None:
            continue
        cmd += [f"--{k}", str(v)]

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Running Rscript: %s", " ".join([sh_quote(x) for x in cmd]))
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("# Command:\n")
        fh.write(" ".join([sh_quote(x) for x in cmd]) + "\n\n")
        fh.flush()

        p = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if p.returncode != 0:
        tail = _safe_tail(log_path, n_lines=120)
        raise RuntimeError(
            f"TSCAN R step failed (exit={p.returncode}). See log: {log_path}\n"
            f"--- R log tail ---\n{tail}"
        )


# ---------------------------------------------------------------------
# Core TI runner
# ---------------------------------------------------------------------
def ti_runner(
    adata: ad.AnnData,
    root_cell_id: str,
    seed: int,
    *,
    bootstrap_index: Optional[int] = None,
    rep_key: str = "X_pca",
    group_key: Optional[str] = None,
    cluster_key: Optional[str] = None,
    run_dir: Optional[str] = None,
    rscript: str = "Rscript",
    tscan_n_dims: int = 30,
    pt_normalize_01: bool = False,
    **kwargs: Any,
) -> TIOutput:
    _cast_nullable_strings_inplace(adata)

    gk = group_key or cluster_key
    if gk is None or gk not in adata.obs.columns:
        raise ValueError(
            "TSCAN requires group_key/cluster_key in adata.obs. "
            f"Got group_key={group_key}, cluster_key={cluster_key}."
        )
    if rep_key not in adata.obsm:
        raise KeyError(
            f"rep_key '{rep_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    if str(root_cell_id) not in set(adata.obs_names.astype(str)):
        raise ValueError(f"root_cell_id '{root_cell_id}' not found in adata.obs_names.")

    # ---- sanitize IDs before any handoff ------------------------------------
    adata_ids = _sanitize_index_like(adata.obs_names.astype(str))
    _check_duplicate_ids(adata_ids, "adata.obs_names")

    group_vals = adata.obs[gk].astype(str).str.strip()
    root_id_s = str(root_cell_id).strip()

    try:
        root_pos = list(adata.obs_names.astype(str)).index(str(root_cell_id))
    except ValueError:
        root_pos = None
    if root_pos is None:
        raise RuntimeError(f"Could not locate root_cell_id '{root_cell_id}' in adata.obs_names.")

    root_cluster = str(group_vals.iloc[root_pos]).strip()
    if root_cluster == "":
        raise RuntimeError(f"Root cluster label for root_cell_id '{root_cell_id}' is empty.")

    base = Path(run_dir) if run_dir is not None else Path.cwd()
    tag = "main" if bootstrap_index is None else f"bootstrap_{int(bootstrap_index)}"
    tmp_dir = base / "logs" / "tmp" / "tscan" / tag

    # Critical fix: remove stale artifacts from previous runs.
    _reset_dir(tmp_dir)
    r_out = tmp_dir / "r_out"
    r_out.mkdir(parents=True, exist_ok=True)

    X = np.asarray(adata.obsm[rep_key])
    if X.ndim != 2 or X.shape[0] != adata.n_obs:
        raise RuntimeError(f"Invalid embedding at obsm['{rep_key}']: shape={getattr(X, 'shape', None)}")

    n_dims = int(min(int(tscan_n_dims), int(X.shape[1])))
    if n_dims < 2:
        raise RuntimeError(f"Need at least 2 dims for TSCAN; got n_dims={n_dims} from embedding shape={X.shape}")

    # ---- write fresh inputs --------------------------------------------------
    emb = pd.DataFrame(
        X[:, :n_dims],
        index=adata_ids,
    )
    emb.index.name = "cell_id"
    emb_path = tmp_dir / "embedding.csv"
    emb.to_csv(emb_path)

    meta = pd.DataFrame(
        {
            "cell_id": adata_ids,
            gk: group_vals.astype(str).str.strip().values,
        }
    )
    meta_path = tmp_dir / "meta.csv"
    meta.to_csv(meta_path, index=False)

    # diagnostics before R
    _write_json(
        tmp_dir / "python_input_diagnostics.json",
        {
            "method": METHOD_NAME,
            "tag": tag,
            "n_cells": int(adata.n_obs),
            "n_dims_written": int(n_dims),
            "rep_key": str(rep_key),
            "group_key": str(gk),
            "root_cell_id": root_id_s,
            "root_cluster": root_cluster,
            "embedding_csv": str(emb_path),
            "meta_csv": str(meta_path),
            "embedding_index_head": list(map(str, emb.index[:10])),
            "meta_cell_id_head": meta["cell_id"].head(10).astype(str).tolist(),
            "group_head": meta[gk].head(10).astype(str).tolist(),
        },
    )

    r_script = THIS_DIR / "R" / "tscan_run.R"
    log_path = tmp_dir / "tscan_R.log"

    _run_rscript(
        rscript=str(rscript),
        script_path=r_script,
        args={
            "embedding": emb_path,
            "meta": meta_path,
            "cluster_key": gk,
            "root_cluster": root_cluster,
            "seed": int(seed),
            "out_dir": r_out,
        },
        log_path=log_path,
    )

    # ---- load pseudotime ----------------------------------------------------
    pt_path = r_out / "pseudotime.csv"
    if not pt_path.exists():
        tail = _safe_tail(log_path, n_lines=120)
        raise RuntimeError(
            f"R completed but pseudotime.csv not found at: {pt_path}\n"
            f"--- R log tail ---\n{tail}"
        )

    pt_df = pd.read_csv(pt_path)
    if not {"cell_id", "pseudotime"}.issubset(pt_df.columns):
        raise RuntimeError(
            f"pseudotime.csv missing required columns. Found: {list(pt_df.columns)}"
        )

    pt_df["cell_id"] = pt_df["cell_id"].astype(str).str.strip()
    pt_df["pseudotime"] = pd.to_numeric(pt_df["pseudotime"], errors="coerce")

    if pt_df["cell_id"].duplicated().any():
        dup = pt_df.loc[pt_df["cell_id"].duplicated(), "cell_id"].tolist()[:10]
        raise RuntimeError(
            f"pseudotime.csv contains duplicated cell_id values. Examples: {dup}"
        )

    pt_ids = pd.Index(pt_df["cell_id"].tolist())
    _check_duplicate_ids(pt_ids, "pseudotime.csv cell_id")

    align_diag = _diagnose_pseudotime_alignment(adata_ids, pt_ids, tmp_dir)
    n_common = int(align_diag["n_common"])

    if n_common == 0:
        tail = _safe_tail(log_path, n_lines=120)
        raise RuntimeError(
            "TSCAN pseudotime alignment failed: zero overlap between "
            "adata.obs_names and pseudotime.csv cell_id.\n"
            f"Diagnostics: {tmp_dir / 'alignment_diagnostics.json'}\n"
            f"--- R log tail ---\n{tail}"
        )

    pt = pd.Series(
        pt_df["pseudotime"].values,
        index=pt_ids,
        name="pseudotime",
    )

    # Reindex to adata order using sanitized IDs
    pt = pt.reindex(adata_ids)
    pt.index = adata.obs_names

    if pt_normalize_01:
        pt = _normalize_01(pt)
        pt.name = "pseudotime"

    ok, reason = _validate_pseudotime(pt)
    if not ok:
        tail = _safe_tail(log_path, n_lines=120)
        raise RuntimeError(
            f"{reason}\n"
            f"Alignment diagnostics: {tmp_dir / 'alignment_diagnostics.json'}\n"
            f"R log: {log_path}\n"
            f"--- R log tail ---\n{tail}"
        )

    # ---- load edges ---------------------------------------------------------
    edges_path = r_out / "edges.csv"
    if edges_path.exists():
        edge_list = pd.read_csv(edges_path)
    else:
        edge_list = pd.DataFrame(columns=["source", "target", "weight", "directed"])

    edge_list = _ensure_edge_list_schema(edge_list)
    topology_matrix = _build_topology_matrix(edge_list)

    # final diagnostics
    _write_json(
        tmp_dir / "python_output_diagnostics.json",
        {
            "n_pseudotime_rows_raw": int(len(pt_df)),
            "n_pseudotime_ids_unique": int(pt_ids.nunique()),
            "n_common_ids": int(n_common),
            "n_finite_pseudotime_after_reindex": int(np.isfinite(pd.to_numeric(pt, errors="coerce")).sum()),
            "n_edges": int(len(edge_list)),
            "n_topology_nodes": int(0 if topology_matrix is None else topology_matrix.shape[0]),
            "r_log": str(log_path),
            "pseudotime_csv": str(pt_path),
            "edges_csv": str(edges_path),
        },
    )

    return TIOutput(
        pseudotime=pt,
        edge_list=edge_list,
        topology_matrix=topology_matrix,
        method_name=METHOD_NAME,
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    p = build_arg_parser()
    add_method_args(p)
    a = p.parse_args()

    spec = args_to_run_spec(a)
    spec.method_name = METHOD_NAME

    runner_extra_kwargs = {
        "rscript": str(a.rscript),
        "tscan_n_dims": int(a.tscan_n_dims),
        "pt_normalize_01": bool(getattr(a, "pt_normalize_01", False)),
    }
    run_benchmark(spec, ti_runner, runner_extra_kwargs=runner_extra_kwargs)


if __name__ == "__main__":
    main()