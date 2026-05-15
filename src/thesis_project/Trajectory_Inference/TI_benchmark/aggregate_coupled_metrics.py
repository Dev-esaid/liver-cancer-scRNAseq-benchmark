from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise ImportError("aggregate_coupled_metrics.py requires anndata.") from e


PROJECT_SRC = Path(os.environ.get("PROJECT_SRC", "/data1/esraa/Thesis-Project/src"))
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from thesis_project.Trajectory_Inference.TI_benchmark.coupled_metrics import (  # noqa: E402
    CoupledRunResult,
    evaluate_coupled_metrics,
)
from thesis_project.Trajectory_Inference.TI_benchmark.utils import write_json  # noqa: E402


COUPLED_ROOT = Path(
    os.environ.get(
        "COUPLED_ROOT",
        "/data1/esraa/Thesis-Project/Results/coupled_benchmark",
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "OUTPUT_DIR",
        "/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics",
    )
)

TASK_DIRS = {
    "task_1": COUPLED_ROOT / "task_1",
    "task_2": COUPLED_ROOT / "task_2",
}

# Used only as fallback if run_config does not carry these fields.
TASK_DEFAULTS: Dict[str, Dict[str, str]] = {
    "task_1": {
        "task_name": "task1_Monocyte_macrophage_TAM",
        "root_group": "Tissue Monocyte",
        "group_key": "cell_subtype_L2",
    },
    "task_2": {
        "task_name": "task2_CD8_Tcell_differentiation",
        "root_group": "Naive T",
        "group_key": "cell_subtype_L2",
    },
}

IGNORE_INTEGRATION_DIRS = {
    "merged_results",
    "tables",
    "figures",
    "logs",
    "adata",
    "plots",
    "plots_pub",
}

PSEUDOTIME_CANDIDATES = [
    "tables/cell_pseudotime.csv",
]

EDGE_CANDIDATES = [
    "tables/topology_edges.csv",
]

ADATA_CANDIDATES = [
    "adata/adata_with_ti_outputs.h5ad",
    "adata/adata_post_geometry.h5ad",
    "adata/adata_post_preprocess.h5ad",
]


# helpers
def _print(msg: str) -> None:
    print(msg, flush=True)


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _find_first_existing(run_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for rel in candidates:
        p = run_dir / rel
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _standardize_task_label(task_dir_name: str) -> str:
    return str(task_dir_name).strip()


def _read_run_config(run_dir: Path) -> Optional[Dict[str, Any]]:
    rc = run_dir / "logs" / "run_config.json"
    if not rc.exists():
        return None
    try:
        return _load_json(rc)
    except Exception as e:
        _print(f"[WARN] Failed to read run_config.json in {run_dir}: {type(e).__name__}: {e}")
        return None


def _infer_status(run_config: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    if not isinstance(run_config, dict):
        return "error", "missing run_config.json"

    final = run_config.get("final", {})
    if isinstance(final, dict):
        success = final.get("success")
        reason = final.get("reason")
        if success is True:
            return "ok", None
        if success is False:
            return "error", str(reason) if reason is not None else "run marked unsuccessful"

    ti_block = run_config.get("ti_method", {})
    if isinstance(ti_block, dict):
        status = ti_block.get("status")
        err = ti_block.get("error_msg")
        if status == "ok":
            return "ok", None
        if status == "error":
            return "error", str(err) if err is not None else "ti_method.status=error"

    return "error", "could not infer final run status"


def _extract_root_cell_id(run_config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(run_config, dict):
        return None

    root_block = run_config.get("root_selection", {})
    if isinstance(root_block, dict):
        sel = root_block.get("selection", {})
        if isinstance(sel, dict) and sel.get("root_cell_id") is not None:
            return str(sel.get("root_cell_id"))
        if root_block.get("root_cell_id") is not None:
            return str(root_block.get("root_cell_id"))

    rs = run_config.get("run_spec", {})
    if isinstance(rs, dict) and rs.get("root_cell_id") is not None:
        return str(rs.get("root_cell_id"))

    return None


def _extract_root_group(run_config: Optional[Dict[str, Any]], task_dir_name: str) -> str:
    if isinstance(run_config, dict):
        priors = run_config.get("priors", {})
        if isinstance(priors, dict) and priors.get("root_group"):
            return str(priors.get("root_group"))

        rs = run_config.get("run_spec", {})
        if isinstance(rs, dict) and rs.get("root_group"):
            return str(rs.get("root_group"))

    return TASK_DEFAULTS.get(task_dir_name, {}).get("root_group", "")


def _extract_group_key(run_config: Optional[Dict[str, Any]], task_dir_name: str) -> str:
    if isinstance(run_config, dict):
        priors = run_config.get("priors", {})
        if isinstance(priors, dict) and priors.get("group_key"):
            return str(priors.get("group_key"))
        if isinstance(priors, dict) and priors.get("root_group_key"):
            return str(priors.get("root_group_key"))

        rs = run_config.get("run_spec", {})
        if isinstance(rs, dict) and rs.get("group_key"):
            return str(rs.get("group_key"))

    return TASK_DEFAULTS.get(task_dir_name, {}).get("group_key", "")


def _extract_task_name(run_config: Optional[Dict[str, Any]], task_dir_name: str) -> str:
    if isinstance(run_config, dict):
        run_meta = run_config.get("run_meta", {})
        if isinstance(run_meta, dict) and run_meta.get("task"):
            return str(run_meta.get("task"))
    return TASK_DEFAULTS.get(task_dir_name, {}).get("task_name", task_dir_name)


def _extract_ti_method(run_config: Optional[Dict[str, Any]], ti_dir_name: str) -> str:
    if isinstance(run_config, dict):
        run_meta = run_config.get("run_meta", {})
        if isinstance(run_meta, dict) and run_meta.get("method"):
            return str(run_meta.get("method"))
    return ti_dir_name


def _load_pseudotime(run_dir: Path) -> Optional[pd.Series]:
    pt_path = _find_first_existing(run_dir, PSEUDOTIME_CANDIDATES)
    if pt_path is None:
        return None

    df = pd.read_csv(pt_path)
    cols_lower = {c.lower(): c for c in df.columns}

    cell_col = cols_lower.get("cell_id") or cols_lower.get("cell") or cols_lower.get("obs_name")
    pt_col = cols_lower.get("pseudotime")
    if cell_col is None or pt_col is None:
        raise ValueError(
            f"Pseudotime file {pt_path} missing required columns. Found: {list(df.columns)}"
        )

    s = pd.Series(
        pd.to_numeric(df[pt_col], errors="coerce").astype(float).values,
        index=df[cell_col].astype(str).values,
        name="pseudotime",
    )
    return s


def _load_edge_list(run_dir: Path) -> Optional[pd.DataFrame]:
    edge_path = _find_first_existing(run_dir, EDGE_CANDIDATES)
    if edge_path is None:
        return None

    df = pd.read_csv(edge_path)
    if not {"source", "target"}.issubset(df.columns):
        return None

    out = df.copy()
    out["source"] = out["source"].astype(str)
    out["target"] = out["target"].astype(str)
    return out


def _load_cell_obs(run_dir: Path, group_key: str) -> Optional[pd.DataFrame]:
    if not group_key:
        return None

    adata_path = _find_first_existing(run_dir, ADATA_CANDIDATES)
    if adata_path is None:
        return None

    try:
        adata = ad.read_h5ad(str(adata_path))
    except Exception as e:
        _print(f"[WARN] Failed to open {adata_path}: {type(e).__name__}: {e}")
        return None

    try:
        if group_key not in adata.obs.columns:
            return None
        obs = adata.obs[[group_key]].copy()
        obs.index = obs.index.astype(str)
        return obs
    finally:
        try:
            del adata
        except Exception:
            pass


def _build_run_result(
    run_dir: Path,
    *,
    task_dir_name: str,
    ti_dir_name: str,
    integration_method: str,
) -> CoupledRunResult:
    run_config = _read_run_config(run_dir)
    status, error_msg = _infer_status(run_config)

    task_name = _extract_task_name(run_config, task_dir_name)
    ti_method = _extract_ti_method(run_config, ti_dir_name)
    root_group = _extract_root_group(run_config, task_dir_name)
    group_key = _extract_group_key(run_config, task_dir_name)
    root_cell_id = _extract_root_cell_id(run_config)

    pseudotime: Optional[pd.Series] = None
    edge_list: Optional[pd.DataFrame] = None
    cell_obs: Optional[pd.DataFrame] = None

    # Load artifacts best-effort even for failed runs; metrics module can decide.
    try:
        pseudotime = _load_pseudotime(run_dir)
    except Exception as e:
        _print(f"[WARN] Failed to load pseudotime for {run_dir}: {type(e).__name__}: {e}")
        if status == "ok":
            status = "error"
            error_msg = f"successful run missing/invalid pseudotime: {type(e).__name__}: {e}"

    try:
        edge_list = _load_edge_list(run_dir)
    except Exception as e:
        _print(f"[WARN] Failed to load edge list for {run_dir}: {type(e).__name__}: {e}")
        edge_list = None

    try:
        cell_obs = _load_cell_obs(run_dir, group_key)
    except Exception as e:
        _print(f"[WARN] Failed to load cell_obs for {run_dir}: {type(e).__name__}: {e}")
        cell_obs = None

    if status == "ok" and pseudotime is None:
        status = "error"
        if error_msg is None:
            error_msg = "successful run missing pseudotime artifact"

    return CoupledRunResult(
        integration_method=integration_method,
        ti_method=ti_method,
        task_name=task_name,
        pseudotime=pseudotime,
        edge_list=edge_list,
        root_cell_id=root_cell_id,
        root_group=root_group,
        group_key=group_key,
        cell_obs=cell_obs,
        status=status,
        error_msg=error_msg,
    )


def _collect_runs_for_ti_task(task_dir: Path, ti_dir: Path) -> List[CoupledRunResult]:
    runs: List[CoupledRunResult] = []
    for child in sorted(ti_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in IGNORE_INTEGRATION_DIRS:
            continue
        runs.append(
            _build_run_result(
                child,
                task_dir_name=task_dir.name,
                ti_dir_name=ti_dir.name,
                integration_method=child.name,
            )
        )
    return runs


def _scalar_from_result(res: Dict[str, Any], block: str, key: str) -> Optional[float]:
    block_obj = res.get(block, {})
    if not isinstance(block_obj, dict):
        return None
    if block_obj.get("status") != "ok":
        return None
    value = block_obj.get("value", {})
    if not isinstance(value, dict):
        return None
    x = value.get(key)
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if pd.notna(xf) else None


def _nested_scalar_from_result(
    res: Dict[str, Any],
    block: str,
    *nested_keys: str,
) -> Optional[float]:
    block_obj = res.get(block, {})
    if not isinstance(block_obj, dict):
        return None
    if block_obj.get("status") != "ok":
        return None

    cur: Any = block_obj.get("value", {})
    if not isinstance(cur, dict):
        return None

    for key in nested_keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None

    try:
        xf = float(cur)
    except (TypeError, ValueError):
        return None
    return xf if pd.notna(xf) else None


def _build_master_row(task_dir_name: str, ti_method: str, results: Dict[str, Any]) -> Dict[str, Any]:
    meta = results.get("meta", {})
    return {
        "task": task_dir_name,
        "task_name": meta.get("task_name"),
        "ti_method": meta.get("ti_method", ti_method),
        "n_integration_methods": meta.get("n_integration_methods"),
        "n_successful_runs": meta.get("n_successful_runs"),
        "n_failed_runs": meta.get("n_failed_runs"),
        "kendalls_w": _scalar_from_result(results, "kendalls_w", "W"),
        "topology_jaccard_mean": _scalar_from_result(results, "topology_jaccard", "jaccard_mean"),
        "branch_leaves_cv": _nested_scalar_from_result(
            results,
            "branch_count_variance",
            "leaves",
            "cv",
        ),
        "branch_points_cv": _nested_scalar_from_result(
            results,
            "branch_count_variance",
            "branchpoints",
            "cv",
        ),
        "root_consistency_total": _scalar_from_result(results, "root_placement_consistency", "consistency_rate_total"),
        "root_consistency_evaluable": _scalar_from_result(results, "root_placement_consistency", "consistency_rate_evaluable"),
        "integration_sensitivity_score": _scalar_from_result(results, "integration_sensitivity_score", "ISS"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_master_rows: List[Dict[str, Any]] = []

    for task_dir_name, task_dir in TASK_DIRS.items():
        if not task_dir.exists():
            _print(f"[WARN] Task directory not found, skipping: {task_dir}")
            continue

        _print(f"\n=== Processing {task_dir_name} ===")

        ti_dirs = [
            p for p in sorted(task_dir.iterdir())
            if p.is_dir() and p.name not in IGNORE_INTEGRATION_DIRS
        ]

        for ti_dir in ti_dirs:
            ti_method = ti_dir.name
            _print(f"[INFO] Collecting runs for {task_dir_name} / {ti_method}")

            runs = _collect_runs_for_ti_task(task_dir, ti_dir)
            if not runs:
                _print(f"[WARN] No integration-method runs found under {ti_dir}")
                continue

            n_total = len(runs)
            n_ok = sum(1 for r in runs if r.status == "ok")
            _print(f"[INFO]   found {n_total} runs ({n_ok} ok, {n_total - n_ok} error)")

            out_dir = OUTPUT_DIR / task_dir_name / ti_method
            out_dir.mkdir(parents=True, exist_ok=True)

            try:
                results = evaluate_coupled_metrics(
                    runs,
                    out_dir=str(out_dir),
                    run_config_path=None,
                )
            except Exception as e:
                _print(
                    f"[ERROR] Coupled metric evaluation failed for {task_dir_name}/{ti_method}: "
                    f"{type(e).__name__}: {e}"
                )
                continue

            all_master_rows.append(_build_master_row(task_dir_name, ti_method, results))
            _print(f"[OK]     wrote summary to {out_dir / 'tables' / 'metrics_coupled_summary.json'}")

    # Write master tables
    if not all_master_rows:
        _print("[WARN] No coupled metric summaries were generated.")
        return

    master_df = pd.DataFrame(all_master_rows)
    master_df = master_df.sort_values(["task", "ti_method"], kind="mergesort").reset_index(drop=True)

    master_csv = OUTPUT_DIR / "coupled_metrics_master.csv"
    master_json = OUTPUT_DIR / "coupled_metrics_master.json"

    master_df.to_csv(master_csv, index=False)
    write_json(master_json, master_df.to_dict(orient="records"), indent=2, sort_keys=False, atomic=True)

    _print("\n=== DONE ===")
    _print(f"[OK] Master CSV : {master_csv}")
    _print(f"[OK] Master JSON: {master_json}")


if __name__ == "__main__":
    main()