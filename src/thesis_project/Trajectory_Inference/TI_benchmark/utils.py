"""
Trajectory Inference Benchmarking: Shared Utilities (utils.py)

Core principles
---------------
- Deterministic: stable ordering; explicit casting; reproducible randomness via explicit seeds.
- Transparent: operations return OpResult with status {ok, skipped, error} and explicit reasons.
- Non-assumptive: no dataset/task semantics assumed. Callers pass explicit inputs.
- Robust I/O: safe JSON read/write with ATOMIC writes (tempfile + os.replace).
- Robust CSV I/O: ATOMIC CSV writes (same HPC-preemption safety as JSON).
- Reproducibility: explicit global seeding helper + runtime environment logging.

All JSON write operations in this framework use merge_json_shallow() or write_json().
All CSV write operations in this framework SHOULD use write_csv() (atomic).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore

try:
    import anndata as ad  # type: ignore
except Exception:  # pragma: no cover
    ad = None  # type: ignore

from .shared_types import Status


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class OpResult:
    """
    Standard operation result for contract-style transparent logging.

    status  : ok | skipped | error
    reason  : explanation (required for skipped / error)
    details : optional structured payload
    """
    status: Status
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ok(details: Optional[Dict[str, Any]] = None) -> OpResult:
    return OpResult(status="ok", details={} if details is None else dict(details))


def skipped(reason: str, details: Optional[Dict[str, Any]] = None) -> OpResult:
    return OpResult(status="skipped", reason=str(reason), details={} if details is None else dict(details))


def error(reason: str, details: Optional[Dict[str, Any]] = None) -> OpResult:
    return OpResult(status="error", reason=str(reason), details={} if details is None else dict(details))


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_global_seeds(seed: int) -> OpResult:
    """
    Set global RNG seeds for libraries that still rely on global state.

    Notes
    -----
    - This does NOT retroactively make hash randomization deterministic
      (PYTHONHASHSEED must be set at process start).
    - This *does* help for any library using `random` or numpy's legacy RandomState.
    """
    try:
        s = int(seed)
        random.seed(s)
        np.random.seed(s)
        os.environ.setdefault("PYTHONHASHSEED", str(s))  # informative only unless set pre-launch
        return ok({"seed": s, "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")})
    except Exception as e:
        return error(f"failed to set global seeds: {type(e).__name__}: {e}")


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version  # py3.8+
        return version(name)
    except Exception:
        return None


def get_runtime_info(extra_packages: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Collect runtime/environment metadata suitable for run_config.json.
    """
    pkgs = [
        "numpy",
        "pandas",
        "scipy",
        "anndata",
        "scanpy",
        "umap-learn",
    ]
    if extra_packages:
        pkgs.extend(list(extra_packages))

    versions = {p: _pkg_version(p) for p in pkgs}

    return {
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": sys.platform,
        },
        "versions": versions,
    }


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists; returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """Ensure parent directory exists; returns Path(path)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# JSON / text I/O — ALL writes are ATOMIC (tempfile + os.replace)
# ---------------------------------------------------------------------------

class _NumpyPandasJSONEncoder(json.JSONEncoder):
    """Safely converts numpy/pandas scalars to JSON primitives."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def read_text(path: Union[str, Path], *, encoding: str = "utf-8") -> str:
    return Path(path).read_text(encoding=encoding)


def write_text(
    path: Union[str, Path],
    text: str,
    *,
    encoding: str = "utf-8",
    atomic: bool = True,
) -> OpResult:
    """Write text to disk. Uses atomic replace by default."""
    p = ensure_parent_dir(path)
    try:
        if not atomic:
            p.write_text(text, encoding=encoding)
            return ok({"path": str(p), "atomic": False})
        tmp_dir = str(p.parent)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, encoding=encoding) as tf:
            tf.write(text)
            tmp_name = tf.name
        os.replace(tmp_name, str(p))
        return ok({"path": str(p), "atomic": True})
    except Exception as e:
        return error(f"failed to write text: {type(e).__name__}: {e}", {"path": str(p)})


def read_json(
    path: Union[str, Path],
    *,
    default: Optional[Dict[str, Any]] = None,
    strict: bool = False,
) -> Tuple[Optional[Dict[str, Any]], OpResult]:
    """
    Read JSON file returning (obj, result).

    - Missing file  + strict=False → (default or {}, skipped)
    - Invalid JSON  + strict=False → (default or {}, error)
    - strict=True                  → (None, error) on any failure
    """
    p = Path(path)
    if not p.exists():
        if strict:
            return None, error(f"json not found: {p}")
        return ({} if default is None else dict(default)), skipped(f"json not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            if strict:
                return None, error(f"json must decode to a mapping: {p}")
            return ({} if default is None else dict(default)), error(f"json must decode to a mapping: {p}")
        return obj, ok({"path": str(p)})
    except Exception as e:
        if strict:
            return None, error(f"failed to parse json: {type(e).__name__}: {e}")
        return ({} if default is None else dict(default)), error(f"failed to parse json: {type(e).__name__}: {e}")


def write_json(
    path: Union[str, Path],
    obj: Any,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    atomic: bool = True,
) -> OpResult:
    """Write JSON deterministically with numpy/pandas-safe encoder. Atomic by default."""
    p = ensure_parent_dir(path)
    try:
        text = json.dumps(
            obj,
            indent=int(indent),
            sort_keys=bool(sort_keys),
            cls=_NumpyPandasJSONEncoder,
        )
        return write_text(p, text, encoding="utf-8", atomic=atomic)
    except Exception as e:
        return error(f"failed to write json: {type(e).__name__}: {e}", {"path": str(p)})


def merge_json_shallow(
    path: Union[str, Path],
    payload: Mapping[str, Any],
    *,
    atomic: bool = True,
) -> OpResult:
    """
    Shallow-merge payload into an existing JSON object at `path`.

    - Payload keys override existing keys.
    - Missing or invalid existing file is treated as {}.
    - Write is ATOMIC (tempfile + os.replace) to protect against HPC preemption.
    """
    p = Path(path)
    existing, res = read_json(p, default={}, strict=False)
    if existing is None:
        existing = {}
    merged = dict(existing)
    merged.update(dict(payload))
    out = write_json(p, merged, indent=2, sort_keys=False, atomic=atomic)
    out.details.update({"path": str(p), "merged_keys": list(payload.keys()), "read_status": res.status})
    if res.reason:
        out.details["read_reason"] = res.reason
    return out


def deep_update(dst: MutableMapping[str, Any], src: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Deep (recursive) update of a nested mapping."""
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), Mapping):
            deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def merge_json_deep(
    path: Union[str, Path],
    payload: Mapping[str, Any],
    *,
    atomic: bool = True,
) -> OpResult:
    """Deep-merge payload into an existing JSON object. Atomic write."""
    p = Path(path)
    existing, res = read_json(p, default={}, strict=False)
    if existing is None:
        existing = {}
    merged = deep_update(dict(existing), dict(payload))
    out = write_json(p, merged, indent=2, sort_keys=False, atomic=atomic)
    out.details.update({"path": str(p), "merged_keys": list(payload.keys()), "read_status": res.status})
    if res.reason:
        out.details["read_reason"] = res.reason
    return out


# ---------------------------------------------------------------------------
# CSV I/O — ATOMIC by default (preemption-safe)
# ---------------------------------------------------------------------------

def write_csv(
    path: Union[str, Path],
    df: pd.DataFrame,
    *,
    index: bool = False,
    encoding: str = "utf-8",
    atomic: bool = True,
    **to_csv_kwargs: Any,
) -> OpResult:
    """
    Write a CSV file. Atomic by default (tempfile + os.replace).
    """
    p = ensure_parent_dir(path)
    try:
        if not atomic:
            df.to_csv(p, index=index, encoding=encoding, **to_csv_kwargs)
            return ok({"path": str(p), "atomic": False, "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])})

        tmp_dir = str(p.parent)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=tmp_dir, encoding=encoding, newline="") as tf:
            df.to_csv(tf, index=index, encoding=encoding, **to_csv_kwargs)
            tmp_name = tf.name
        os.replace(tmp_name, str(p))
        return ok({"path": str(p), "atomic": True, "n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])})
    except Exception as e:
        return error(f"failed to write csv: {type(e).__name__}: {e}", {"path": str(p)})


# ---------------------------------------------------------------------------
# Hashing / provenance
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    return sha256_bytes(text.encode(encoding))


def sha256_file(path: Union[str, Path], *, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file deterministically."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(int(chunk_size))
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def json_canonical_dumps(obj: Any) -> str:
    """Canonical JSON string for hashing: sort_keys=True, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), cls=_NumpyPandasJSONEncoder)


def sha256_json(obj: Any) -> str:
    return sha256_text(json_canonical_dumps(obj))


# ---------------------------------------------------------------------------
# Array / pandas alignment helpers
# ---------------------------------------------------------------------------

def is_sparse(x: Any) -> bool:
    return sp is not None and sp.issparse(x)  # type: ignore[attr-defined]


def require_unique_index(index: Union[pd.Index, Sequence[Any]], *, name: str) -> None:
    idx = index if isinstance(index, pd.Index) else pd.Index(index)
    if idx.has_duplicates:
        raise ValueError(f"{name} contains duplicates (n={idx.size}, unique={idx.nunique()}).")


def align_series_to_index(
    s: pd.Series,
    index: pd.Index,
    *,
    name: str,
    allow_missing: bool,
) -> pd.Series:
    """
    Align a Series to an index deterministically:
    - Drop extras, reindex to match index order.
    - If allow_missing=False: raise if any index values are missing.
    """
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")
    require_unique_index(s.index, name=f"{name}.index")
    s2 = s.loc[s.index.intersection(index)]
    missing = index.difference(s2.index)
    if len(missing) > 0 and not allow_missing:
        raise ValueError(f"{name} missing {len(missing)} entries (example: {list(missing[:5])}).")
    return s2.reindex(index)


def align_df_to_index(
    df: pd.DataFrame,
    index: pd.Index,
    *,
    name: str,
    allow_missing: bool,
) -> pd.DataFrame:
    """Align a DataFrame to an index deterministically."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    require_unique_index(df.index, name=f"{name}.index")
    df2 = df.loc[df.index.intersection(index)]
    missing = index.difference(df2.index)
    if len(missing) > 0 and not allow_missing:
        raise ValueError(f"{name} missing {len(missing)} entries (example: {list(missing[:5])}).")
    return df2.reindex(index)


def coerce_numeric_series(
    s: pd.Series,
    *,
    name: str,
    require_any_finite: bool = True,
) -> pd.Series:
    """Coerce a Series to float."""
    out = pd.to_numeric(s, errors="coerce").astype(float)
    if require_any_finite and int(np.isfinite(out.values).sum()) == 0:
        raise ValueError(f"{name} has no finite numeric values after coercion.")
    return out


def stable_unique(seq: Sequence[Any]) -> List[Any]:
    """Deterministic unique list preserving first-occurrence order."""
    seen: set = set()
    out: List[Any] = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ---------------------------------------------------------------------------
# Reproducible randomness
# ---------------------------------------------------------------------------

def make_rng(seed: Optional[int]) -> np.random.Generator:
    """
    Create a numpy Generator from an explicit seed.
    Raises if seed is None (callers must choose seeds explicitly).
    """
    if seed is None:
        raise ValueError("seed must be provided explicitly (None is not allowed).")
    if isinstance(seed, bool):
        raise TypeError("seed must be an int, not bool.")
    return np.random.default_rng(int(seed))


def stratified_sample_indices(
    groups: Sequence[Any],
    *,
    frac: float,
    min_per_group: int,
    rng: np.random.Generator,
    replace: bool,
    strict_min: bool = True,
    sort_indices: bool = True,
) -> np.ndarray:
    """
    Stratified sampling over groups.

    IMPORTANT:
    - For AnnData stability analysis, returning duplicate indices is dangerous
      (it creates duplicated obs_names). Therefore, even when replace=True,
      this function returns UNIQUE indices (at most group size).
    """
    if not (0.0 < float(frac) <= 1.0):
        raise ValueError("frac must be in (0, 1].")
    if int(min_per_group) < 0:
        raise ValueError("min_per_group must be >= 0.")
    g = pd.Series(list(groups))
    n = int(g.shape[0])
    if n == 0:
        return np.array([], dtype=int)

    out_idx: List[int] = []
    for label, idx in g.groupby(g, sort=True).groups.items():
        idx = np.asarray(list(idx), dtype=int)
        m = int(idx.size)
        k = int(np.ceil(m * float(frac)))
        k = max(k, int(min_per_group))

        if not replace:
            if k > m:
                if strict_min:
                    raise ValueError(
                        f"group '{label}' size {m} < requested k={k} without replacement "
                        f"(min_per_group={min_per_group}, frac={frac})."
                    )
                k = m
            chosen = rng.choice(idx, size=k, replace=False)
        else:
            # Keep uniqueness to avoid duplicated obs_names.
            if k >= m:
                chosen = idx
            else:
                chosen = rng.choice(idx, size=k, replace=False)

        out_idx.extend(np.asarray(chosen, dtype=int).tolist())

    arr = np.asarray(out_idx, dtype=int)
    return np.sort(arr) if sort_indices else arr


# ---------------------------------------------------------------------------
# AnnData helpers
# ---------------------------------------------------------------------------

def require_anndata() -> None:
    if ad is None:
        raise ImportError("anndata is required but is not installed.")


def validate_anndata_basic(
    adata: Any,
    *,
    require_X: bool = True,
    require_unique_obs: bool = True,
    require_unique_var: bool = True,
) -> OpResult:
    """Basic structural validation for AnnData. No biological assumptions."""
    if ad is None:
        return error("anndata is not installed; cannot validate AnnData.")
    if not isinstance(adata, ad.AnnData):  # type: ignore[attr-defined]
        return error(f"expected AnnData; got {type(adata).__name__}.")
    details: Dict[str, Any] = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "has_X": adata.X is not None,
        "layers": list(adata.layers.keys()) if hasattr(adata, "layers") else [],
    }
    if require_X and adata.X is None:
        return error("adata.X is required but is None.", details)
    if require_unique_obs and adata.obs_names.has_duplicates:
        return error("adata.obs_names contains duplicates.", details)
    if require_unique_var and adata.var_names.has_duplicates:
        return error("adata.var_names contains duplicates.", details)
    return ok(details)


def get_matrix(adata: Any, *, layer: Optional[str] = None) -> Any:
    """Retrieve expression matrix from AnnData without assumptions."""
    require_anndata()
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Requested layer '{layer}' not found in adata.layers.")
    return adata.layers[layer]


__all__ = [
    "Status",
    "OpResult", "ok", "skipped", "error",
    "set_global_seeds", "get_runtime_info",
    "ensure_dir", "ensure_parent_dir",
    "read_text", "write_text",
    "read_json", "write_json", "merge_json_shallow", "deep_update", "merge_json_deep",
    "write_csv",
    "sha256_bytes", "sha256_text", "sha256_file", "json_canonical_dumps", "sha256_json",
    "is_sparse", "require_unique_index", "align_series_to_index", "align_df_to_index",
    "coerce_numeric_series", "stable_unique",
    "make_rng", "stratified_sample_indices",
    "validate_anndata_basic", "get_matrix",
    "_NumpyPandasJSONEncoder",
]