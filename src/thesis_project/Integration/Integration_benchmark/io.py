import os
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def _json_friendly(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def save_json(obj: Dict[str, Any], path: str):
    """Save dictionary as JSON file."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_friendly)


def save_run_artifacts(
    outdir: str,
    *,
    metrics: Dict[str, Any],
    config: Any,
    perf_df: Optional[pd.DataFrame],
    adata=None,
    save_h5ad: bool = False,
    h5ad_name: str = "adata_result.h5ad",
) -> Tuple[str, str, str]:
    """
    Standardized saving used across all methods:
      - metrics.json
      - config.json
      - perf_log.csv
      - optional h5ad

    Returns: (metrics_path, config_path, perf_path)
    """
    ensure_dir(outdir)

    metrics_path = os.path.join(outdir, "metrics.json")
    config_path = os.path.join(outdir, "config.json")
    perf_path = os.path.join(outdir, "perf_log.csv")

    # Save metrics JSON
    save_json(metrics, metrics_path)

    # Save config JSON
    cfg_dict = asdict(config) if is_dataclass(config) else dict(config)
    save_json(cfg_dict, config_path)

    # Save perf log CSV
    if perf_df is not None:
        perf_df.to_csv(perf_path, index=False)
    else:
        pd.DataFrame([]).to_csv(perf_path, index=False)

    # Optional h5ad
    if save_h5ad and adata is not None:
        try:
            adata.write_h5ad(os.path.join(outdir, h5ad_name))
        except Exception:
            # best-effort: don't break artifacts saving if h5ad fails
            pass

    return metrics_path, config_path, perf_path