from __future__ import annotations

import contextlib
import copy
import importlib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


@dataclass
class MetricSubset:
    adata_int: Any
    adata_pre: Optional[Any]
    emb_key: Optional[str]
    conn_key: Optional[str]
    dist_key: Optional[str]
    neighbors_uns_key: Optional[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(fn, *args, label: str = "", **kwargs) -> Tuple[float, Optional[str]]:
    """Call fn(*args, **kwargs); return (scalar, error_string_or_None)."""
    try:
        result = fn(*args, **kwargs)
        if result is None:
            return float("nan"), f"{label}: returned None"
        val = float(np.nanmean(np.asarray(result, dtype=float).ravel()))
        return val, None
    except Exception as exc:
        return float("nan"), f"{label}: {type(exc).__name__}: {exc}"


def _import_scib():
    """Import scib and its metrics sub-module. Raises ImportError if absent."""
    scib = importlib.import_module("scib")
    me = importlib.import_module("scib.metrics")
    return scib, me


def _ensure_obs_categorical(adata, *keys: str):
    """Ensure selected adata.obs columns are pandas categorical dtype."""
    if adata is None:
        return None
    for key in keys:
        if key is None or key not in adata.obs:
            continue
        if str(adata.obs[key].dtype) != "category":
            adata.obs[key] = adata.obs[key].astype("category")
    return adata


def _resolve_existing_graph_keys(
    adata,
    conn_key: Optional[str],
    dist_key: Optional[str],
    neighbors_uns_key: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return the graph keys that actually exist on this AnnData object."""
    eff_conn = conn_key if conn_key and conn_key in adata.obsp else None
    if eff_conn is None and "connectivities" in adata.obsp:
        eff_conn = "connectivities"

    eff_dist = dist_key if dist_key and dist_key in adata.obsp else None
    if eff_dist is None and "distances" in adata.obsp:
        eff_dist = "distances"

    eff_uns = neighbors_uns_key if neighbors_uns_key and neighbors_uns_key in adata.uns else None
    if eff_uns is None and "neighbors" in adata.uns:
        eff_uns = "neighbors"

    if eff_uns is not None:
        uns_obj = adata.uns.get(eff_uns, {})
        if eff_conn is None:
            cand = uns_obj.get("connectivities_key")
            if isinstance(cand, str) and cand in adata.obsp:
                eff_conn = cand
        if eff_dist is None:
            cand = uns_obj.get("distances_key")
            if isinstance(cand, str) and cand in adata.obsp:
                eff_dist = cand

    # final fallback: if we have graph matrices but no explicit uns entry,
    # leave eff_uns as None and let _alias_graph fabricate a minimal one
    return eff_conn, eff_dist, eff_uns


def _make_minimal_neighbors_uns(
    adata,
    conn_key: str,
    dist_key: Optional[str],
    n_neighbors: int = 30,
) -> Dict[str, Any]:
    """Create a minimal neighbors-style metadata dict."""
    out = {
        "connectivities_key": conn_key,
        "params": {"n_neighbors": int(max(2, min(n_neighbors, max(2, adata.n_obs - 1))))},
    }
    if dist_key is not None and dist_key in adata.obsp:
        out["distances_key"] = dist_key
    return out


@contextlib.contextmanager
def _alias_graph(
    adata,
    conn_key: Optional[str],
    dist_key: Optional[str] = None,
    neighbors_uns_key: Optional[str] = None,
):
    """
    Temporarily install a method-specific kNN graph into the standard slots
    expected by scib/scanpy helper code.
    """
    if conn_key is None:
        yield
        return
    if conn_key not in adata.obsp:
        raise KeyError(
            f"conn_key '{conn_key}' not found in adata.obsp. "
            f"Available keys: {list(adata.obsp.keys())}"
        )

    orig_conn = adata.obsp.get("connectivities", None)
    orig_dist = adata.obsp.get("distances", None)
    orig_uns = adata.uns.get("neighbors", None)

    adata.obsp["connectivities"] = adata.obsp[conn_key]

    if dist_key is not None and dist_key in adata.obsp:
        adata.obsp["distances"] = adata.obsp[dist_key]

    if neighbors_uns_key is not None and neighbors_uns_key in adata.uns:
        adata.uns["neighbors"] = copy.deepcopy(adata.uns[neighbors_uns_key])
        if "params" not in adata.uns["neighbors"] or not isinstance(adata.uns["neighbors"]["params"], dict):
            adata.uns["neighbors"]["params"] = {}
        adata.uns["neighbors"]["connectivities_key"] = "connectivities"
        if dist_key is not None and dist_key in adata.obsp:
            adata.uns["neighbors"]["distances_key"] = "distances"
    else:
        # BBKNN / subset safety: fabricate a minimal neighbors dict if the
        # namespaced uns entry is absent but the graph matrices are present.
        n_neighbors = 30
        if orig_uns is not None and isinstance(orig_uns, dict):
            try:
                n_neighbors = int(orig_uns.get("params", {}).get("n_neighbors", 30))
            except Exception:
                n_neighbors = 30
        adata.uns["neighbors"] = _make_minimal_neighbors_uns(
            adata,
            conn_key="connectivities",
            dist_key="distances" if (dist_key is not None and dist_key in adata.obsp) else None,
            n_neighbors=n_neighbors,
        )

    try:
        yield
    finally:
        if orig_conn is not None:
            adata.obsp["connectivities"] = orig_conn
        elif "connectivities" in adata.obsp:
            del adata.obsp["connectivities"]

        if orig_dist is not None:
            adata.obsp["distances"] = orig_dist
        elif "distances" in adata.obsp:
            del adata.obsp["distances"]

        if orig_uns is not None:
            adata.uns["neighbors"] = orig_uns
        elif "neighbors" in adata.uns:
            del adata.uns["neighbors"]


def _infer_n_neighbors(
    adata,
    neighbors_uns_key: Optional[str] = None,
    default: int = 30,
) -> int:
    """Infer n_neighbors from an existing neighbors entry if possible."""
    candidate_uns = []
    if neighbors_uns_key is not None:
        candidate_uns.append(neighbors_uns_key)
    candidate_uns.append("neighbors")

    for key in candidate_uns:
        if key in adata.uns:
            params = adata.uns[key].get("params", {})
            try:
                n_neighbors = int(params.get("n_neighbors", default))
                return max(2, min(n_neighbors, max(2, adata.n_obs - 1)))
            except Exception:
                continue

    return max(2, min(default, max(2, adata.n_obs - 1)))


def _ensure_metric_embedding(
    adata,
    emb_key: Optional[str],
    output_type: str,
    n_comps: int = 50,
) -> Optional[str]:
    """
    Return a usable embedding key for embedding-based metrics.

    - embed methods: use the provided embedding key
    - full methods: use existing X_pca or compute PCA on the subset
    - knn methods: return None
    """
    if emb_key is not None and emb_key in adata.obsm:
        return emb_key

    if output_type == "full":
        if "X_pca" in adata.obsm:
            return "X_pca"

        max_obs_comps = adata.n_obs - 1
        max_var_comps = adata.n_vars - 1
        max_comps = min(int(n_comps), max_obs_comps, max_var_comps)
        if max_comps < 2:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.tl.pca(adata, n_comps=max_comps, svd_solver="arpack")
        return "X_pca"

    return None


def _allocate_group_sample_sizes(
    counts: np.ndarray,
    n_target: int,
    ensure_one_per_group: bool,
) -> np.ndarray:
    """Allocate integer sample sizes per stratum while respecting capacities."""
    counts = np.asarray(counts, dtype=int)
    total = int(counts.sum())
    if n_target >= total:
        return counts.copy()
    if n_target <= 0:
        return np.zeros_like(counts, dtype=int)

    quotas = counts.astype(float) / float(total) * float(n_target)
    take = np.floor(quotas).astype(int)

    if ensure_one_per_group and n_target >= len(counts):
        min_take = (counts > 0).astype(int)
        take = np.maximum(take, min_take)
        take = np.minimum(take, counts)
    else:
        min_take = np.zeros_like(counts, dtype=int)

    current = int(take.sum())
    if current > n_target:
        overflow = current - n_target
        removable = np.where(take > min_take)[0]
        frac = quotas - np.floor(quotas)
        order = removable[np.argsort(frac[removable])]
        for idx in order:
            if overflow <= 0:
                break
            room = int(take[idx] - min_take[idx])
            if room <= 0:
                continue
            delta = min(room, overflow)
            take[idx] -= delta
            overflow -= delta

    remaining = int(n_target - take.sum())
    if remaining > 0:
        frac = quotas - np.floor(quotas)
        capacity_left = counts - take
        while remaining > 0:
            candidates = np.where(capacity_left > 0)[0]
            if candidates.size == 0:
                break
            order = candidates[np.argsort(-frac[candidates])]
            progressed = False
            for idx in order:
                if remaining <= 0:
                    break
                if capacity_left[idx] <= 0:
                    continue
                take[idx] += 1
                capacity_left[idx] -= 1
                remaining -= 1
                progressed = True
            if not progressed:
                break

    return np.minimum(take, counts)


def _balanced_sample_obs_names(
    adata,
    n_target: Optional[int],
    strata_keys: Sequence[Optional[str]],
    random_state: int = 0,
):
    """
    Balanced subsampling that preserves batch/label composition as much as possible.
    """
    if n_target is None or int(n_target) >= adata.n_obs:
        return np.asarray(adata.obs_names, dtype=object)

    n_target = int(max(1, n_target))
    rng = np.random.default_rng(int(random_state))
    valid_keys = [k for k in strata_keys if k is not None and k in adata.obs]

    if not valid_keys:
        idx = np.sort(rng.choice(adata.n_obs, size=n_target, replace=False))
        return np.asarray(adata.obs_names[idx], dtype=object)

    strata_df = adata.obs.loc[:, valid_keys].copy()
    for key in valid_keys:
        strata_df[key] = strata_df[key].astype(str).fillna("__NA__")

    strata = strata_df.astype(str).agg("||".join, axis=1).to_numpy()
    _, inv = np.unique(strata, return_inverse=True)
    counts = np.bincount(inv)
    take = _allocate_group_sample_sizes(
        counts,
        n_target=n_target,
        ensure_one_per_group=(n_target >= len(counts)),
    )

    selected = []
    for group_id, group_take in enumerate(take):
        if group_take <= 0:
            continue
        members = np.flatnonzero(inv == group_id)
        if group_take >= members.size:
            chosen = members
        else:
            chosen = rng.choice(members, size=int(group_take), replace=False)
        selected.append(np.asarray(chosen, dtype=int))

    if not selected:
        idx = np.sort(rng.choice(adata.n_obs, size=n_target, replace=False))
    else:
        idx = np.sort(np.concatenate(selected))
        if idx.size > n_target:
            idx = np.sort(rng.choice(idx, size=n_target, replace=False))
        elif idx.size < n_target:
            remaining = np.setdiff1d(np.arange(adata.n_obs), idx, assume_unique=False)
            extra = rng.choice(remaining, size=n_target - idx.size, replace=False)
            idx = np.sort(np.concatenate([idx, extra]))

    return np.asarray(adata.obs_names[idx], dtype=object)


def _build_subset_graph(
    adata,
    use_rep: Optional[str],
    neighbors_uns_key: str,
    n_neighbors: int,
) -> Tuple[str, Optional[str], str]:
    """Build a fresh kNN graph on a subset AnnData."""
    n_neighbors = max(2, min(int(n_neighbors), max(2, adata.n_obs - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            key_added=neighbors_uns_key,
        )
    return (
        adata.uns[neighbors_uns_key]["connectivities_key"],
        adata.uns[neighbors_uns_key].get("distances_key"),
        neighbors_uns_key,
    )


def _prepare_metric_subset(
    adata_int,
    adata_pre,
    *,
    batch_key: str,
    label_key: str,
    cluster_key: str,
    emb_key: Optional[str],
    conn_key: Optional[str],
    dist_key: Optional[str],
    neighbors_uns_key: Optional[str],
    output_type: str,
    n_target: Optional[int],
    tag: str,
    random_state: int,
) -> MetricSubset:
    """
    Create a bounded-size working copy for metric computation.
    """
    obs_names = _balanced_sample_obs_names(
        adata_int,
        n_target=n_target,
        strata_keys=(batch_key, label_key),
        random_state=random_state,
    )

    if len(obs_names) == adata_int.n_obs:
        ad_int = adata_int.copy()
    else:
        ad_int = adata_int[obs_names].copy()

    ad_pre_sub = None
    if adata_pre is not None:
        missing = [name for name in obs_names if name not in adata_pre.obs_names]
        if missing:
            raise KeyError(
                f"{len(missing)} sampled cells are missing from adata_pre. "
                "adata_pre must contain the same cells as adata_int."
            )
        ad_pre_sub = adata_pre[obs_names].copy()

    _ensure_obs_categorical(ad_int, batch_key, label_key, cluster_key)
    _ensure_obs_categorical(ad_pre_sub, batch_key, label_key, cluster_key)

    metric_emb_key = _ensure_metric_embedding(ad_int, emb_key, output_type)
    eff_conn, eff_dist, eff_uns = _resolve_existing_graph_keys(
        ad_int,
        conn_key=conn_key,
        dist_key=dist_key,
        neighbors_uns_key=neighbors_uns_key,
    )

    subset_is_smaller = len(obs_names) < adata_int.n_obs

    if output_type == "knn":
        # For knn-native methods like BBKNN, we should preserve the existing
        # graph if subsetting kept it. If not, try to reconstruct minimal graph
        # metadata from available matrices.
        if eff_conn is not None and eff_uns is None:
            eff_uns = None  # let _alias_graph fabricate a minimal neighbors dict
    else:
        needs_graph = subset_is_smaller or eff_conn is None or eff_uns is None
        if needs_graph and ad_int.n_obs >= 3:
            use_rep = metric_emb_key if metric_emb_key is not None else None
            n_neighbors = _infer_n_neighbors(
                adata_int,
                neighbors_uns_key=neighbors_uns_key,
                default=30,
            )
            temp_uns = f"neighbors__scib_{tag}"
            eff_conn, eff_dist, eff_uns = _build_subset_graph(
                ad_int,
                use_rep=use_rep,
                neighbors_uns_key=temp_uns,
                n_neighbors=n_neighbors,
            )

    return MetricSubset(
        adata_int=ad_int,
        adata_pre=ad_pre_sub,
        emb_key=metric_emb_key,
        conn_key=eff_conn,
        dist_key=eff_dist,
        neighbors_uns_key=eff_uns,
    )


def _ensure_cluster(
    adata,
    cluster_key: str,
    label_key: str,
    *,
    neighbors_uns_key: Optional[str],
    emb_key: Optional[str],
    verbose: bool = False,
) -> None:
    """
    Ensure a clustering exists.

    Fast-path behavior:
    - if cluster_key already exists, keep it
    - otherwise run one Leiden clustering on the available graph
      (or compute a temporary graph if needed)
    """
    if label_key in adata.obs and str(adata.obs[label_key].dtype) != "category":
        adata.obs[label_key] = adata.obs[label_key].astype("category")

    if cluster_key in adata.obs:
        if str(adata.obs[cluster_key].dtype) != "category":
            adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
        return

    local_neighbors_key = neighbors_uns_key if neighbors_uns_key in adata.uns else None
    if local_neighbors_key is None and adata.n_obs >= 3:
        temp_uns = "neighbors__scib_cluster"
        n_neighbors = _infer_n_neighbors(adata, neighbors_uns_key=None, default=30)
        use_rep = emb_key if emb_key is not None else None
        _build_subset_graph(
            adata,
            use_rep=use_rep,
            neighbors_uns_key=temp_uns,
            n_neighbors=n_neighbors,
        )
        local_neighbors_key = temp_uns

    if local_neighbors_key is None:
        raise RuntimeError(
            f"Unable to create cluster_key '{cluster_key}': no usable graph and too few cells."
        )

    if verbose:
        print(f"[scib_compat] Running Leiden -> '{cluster_key}' on {adata.n_obs:,} cells")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.leiden(
            adata,
            neighbors_key=local_neighbors_key,
            key_added=cluster_key,
            resolution=1.0,
        )

    if str(adata.obs[cluster_key].dtype) != "category":
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")


def _resolve_caps(
    n_obs: int,
    metric_subsample_n: Optional[int],
    heavy_metric_subsample_n: Optional[int],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Resolve the 100k / 50k caps used by the working subsets.
    """
    full_cap = None if metric_subsample_n is None else int(metric_subsample_n)
    heavy_cap = None if heavy_metric_subsample_n is None else int(heavy_metric_subsample_n)

    if full_cap is not None:
        full_cap = max(1, min(full_cap, n_obs))
    if heavy_cap is not None:
        heavy_cap = max(1, min(heavy_cap, n_obs))

    if full_cap is not None and heavy_cap is not None:
        heavy_cap = min(heavy_cap, full_cap)

    return full_cap, heavy_cap


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_scib_metrics(
    adata_int,
    adata_pre,
    *,
    batch_key: str,
    label_key: str,
    cluster_key: str,
    emb_key: Optional[str],
    conn_key: Optional[str],
    dist_key: Optional[str] = None,
    neighbors_uns_key: Optional[str] = None,
    output_type: str = "embed",
    organism: str = "human",
    n_isolated: Optional[int] = None,
    subsample: Optional[int] = 100,
    compute_trajectory: bool = False,
    trajectory_adata_pre: Optional[Any] = None,
    pseudotime_key: str = "dpt_pseudotime",
    verbose: bool = False,
    metric_subsample_n: Optional[int] = 100_000,
    heavy_metric_subsample_n: Optional[int] = 50_000,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Compute all 14 scIB metrics with bounded-size working subsets.
    """
    try:
        _, me = _import_scib()
    except ImportError as exc:
        raise ImportError("scib is required. Install via: pip install scib") from exc

    out: Dict[str, Any] = {}

    full_cap, heavy_cap = _resolve_caps(
        adata_int.n_obs,
        metric_subsample_n=metric_subsample_n,
        heavy_metric_subsample_n=heavy_metric_subsample_n,
    )

    general = _prepare_metric_subset(
        adata_int,
        adata_pre,
        batch_key=batch_key,
        label_key=label_key,
        cluster_key=cluster_key,
        emb_key=emb_key,
        conn_key=conn_key,
        dist_key=dist_key,
        neighbors_uns_key=neighbors_uns_key,
        output_type=output_type,
        n_target=full_cap,
        tag="general",
        random_state=random_state,
    )

    heavy = _prepare_metric_subset(
        adata_int,
        adata_pre,
        batch_key=batch_key,
        label_key=label_key,
        cluster_key=cluster_key,
        emb_key=emb_key,
        conn_key=conn_key,
        dist_key=dist_key,
        neighbors_uns_key=neighbors_uns_key,
        output_type=output_type,
        n_target=heavy_cap,
        tag="heavy",
        random_state=random_state + 17,
    )

    lisi_pct = 100 if subsample is None else int(subsample)
    lisi_pct = max(1, min(100, lisi_pct))

    # -----------------------------------------------------------------------
    # Step 0 — ensure cluster assignments exist on the general subset
    # -----------------------------------------------------------------------
    try:
        _ensure_cluster(
            general.adata_int,
            cluster_key=cluster_key,
            label_key=label_key,
            neighbors_uns_key=general.neighbors_uns_key,
            emb_key=general.emb_key,
            verbose=verbose,
        )
    except Exception as exc:
        out["clustering_error"] = f"{type(exc).__name__}: {exc}"

    # -----------------------------------------------------------------------
    # 1. NMI
    # -----------------------------------------------------------------------
    if cluster_key in general.adata_int.obs:
        val, err = _safe(me.nmi, general.adata_int, cluster_key, label_key, label="NMI")
        out["NMI"] = val
        if err:
            out["NMI_error"] = err
    else:
        out["NMI"] = float("nan")
        out["NMI_error"] = f"cluster_key '{cluster_key}' missing from obs"

    # -----------------------------------------------------------------------
    # 2. ARI
    # -----------------------------------------------------------------------
    if cluster_key in general.adata_int.obs:
        val, err = _safe(me.ari, general.adata_int, cluster_key, label_key, label="ARI")
        out["ARI"] = val
        if err:
            out["ARI_error"] = err
    else:
        out["ARI"] = float("nan")
        out["ARI_error"] = f"cluster_key '{cluster_key}' missing from obs"

    # -----------------------------------------------------------------------
    # 3. Cell type ASW
    # -----------------------------------------------------------------------
    if heavy.emb_key is not None:
        val, err = _safe(
            me.silhouette,
            heavy.adata_int,
            label_key,
            heavy.emb_key,
            metric="euclidean",
            scale=True,
            label="cell_type_ASW",
        )
        out["cell_type_ASW"] = val
        if err:
            out["cell_type_ASW_error"] = err
    else:
        out["cell_type_ASW"] = float("nan")
        out["cell_type_ASW_note"] = "no embedding available for this output type"

    # -----------------------------------------------------------------------
    # 4. Isolated label F1
    # -----------------------------------------------------------------------
    if output_type == "knn" and heavy.conn_key is not None:
        try:
            with _alias_graph(
                heavy.adata_int,
                heavy.conn_key,
                dist_key=heavy.dist_key,
                neighbors_uns_key=heavy.neighbors_uns_key,
            ):
                val = me.isolated_labels_f1(
                    heavy.adata_int,
                    label_key=label_key,
                    batch_key=batch_key,
                    embed=None,
                    cluster_key="iso_label",
                    iso_threshold=n_isolated,
                    verbose=verbose,
                    force=False,
                )
            out["isolated_label_F1"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["isolated_label_F1"] = float("nan")
            out["isolated_label_F1_error"] = f"{type(exc).__name__}: {exc}"
    elif heavy.emb_key is not None:
        try:
            val = me.isolated_labels_f1(
                heavy.adata_int,
                label_key=label_key,
                batch_key=batch_key,
                embed=heavy.emb_key,
                cluster_key="iso_label",
                iso_threshold=n_isolated,
                verbose=verbose,
                force=False,
            )
            out["isolated_label_F1"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["isolated_label_F1"] = float("nan")
            out["isolated_label_F1_error"] = f"{type(exc).__name__}: {exc}"
    elif heavy.conn_key is not None:
        try:
            with _alias_graph(
                heavy.adata_int,
                heavy.conn_key,
                dist_key=heavy.dist_key,
                neighbors_uns_key=heavy.neighbors_uns_key,
            ):
                val = me.isolated_labels_f1(
                    heavy.adata_int,
                    label_key=label_key,
                    batch_key=batch_key,
                    embed=None,
                    cluster_key="iso_label",
                    iso_threshold=n_isolated,
                    verbose=verbose,
                    force=False,
                )
            out["isolated_label_F1"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["isolated_label_F1"] = float("nan")
            out["isolated_label_F1_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["isolated_label_F1"] = float("nan")
        out["isolated_label_F1_note"] = "no graph or embedding available"

    # -----------------------------------------------------------------------
    # 5. Isolated label ASW
    # -----------------------------------------------------------------------
    if heavy.emb_key is not None:
        try:
            val = me.isolated_labels_asw(
                heavy.adata_int,
                label_key=label_key,
                batch_key=batch_key,
                embed=heavy.emb_key,
                iso_threshold=n_isolated,
                scale=True,
                verbose=verbose,
            )
            out["isolated_label_ASW"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["isolated_label_ASW"] = float("nan")
            out["isolated_label_ASW_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["isolated_label_ASW"] = float("nan")
        out["isolated_label_ASW_note"] = "no embedding available for this output type"

    # -----------------------------------------------------------------------
    # 6. Cell cycle conservation
    # -----------------------------------------------------------------------
    if heavy.adata_pre is not None:
        try:
            val = me.cell_cycle(
                heavy.adata_pre,
                heavy.adata_int,
                batch_key=batch_key,
                embed=heavy.emb_key,
                organism=organism,
            )
            out["cell_cycle_conservation"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["cell_cycle_conservation"] = float("nan")
            out["cell_cycle_conservation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["cell_cycle_conservation"] = float("nan")
        out["cell_cycle_conservation_note"] = "adata_pre not provided"

    # -----------------------------------------------------------------------
    # 7. HVG conservation
    # -----------------------------------------------------------------------
    if general.adata_pre is not None:
        try:
            val = me.hvg_overlap(general.adata_pre, general.adata_int, batch_key)
            out["hvg_conservation"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["hvg_conservation"] = float("nan")
            out["hvg_conservation_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["hvg_conservation"] = float("nan")
        out["hvg_conservation_note"] = "adata_pre not provided"

    # -----------------------------------------------------------------------
    # 8. Trajectory conservation
    # -----------------------------------------------------------------------
    if compute_trajectory:
        if trajectory_adata_pre is not None:
            try:
                missing = [name for name in general.adata_int.obs_names if name not in trajectory_adata_pre.obs_names]
                if missing:
                    raise KeyError(
                        f"{len(missing)} trajectory cells are missing from trajectory_adata_pre"
                    )
                traj_pre = trajectory_adata_pre[general.adata_int.obs_names].copy()
                _ensure_obs_categorical(traj_pre, batch_key, label_key, cluster_key)
            except Exception as exc:
                traj_pre = None
                out["trajectory_conservation"] = float("nan")
                out["trajectory_conservation_error"] = f"{type(exc).__name__}: {exc}"
            else:
                try:
                    if pseudotime_key not in traj_pre.obs:
                        raise KeyError(
                            f"'{pseudotime_key}' not found in trajectory_adata_pre.obs"
                        )
                    with _alias_graph(
                        general.adata_int,
                        general.conn_key,
                        dist_key=general.dist_key,
                        neighbors_uns_key=general.neighbors_uns_key,
                    ):
                        val = me.trajectory_conservation(
                            traj_pre,
                            general.adata_int,
                            label_key=label_key,
                            pseudotime_key=pseudotime_key,
                            batch_key=batch_key,
                        )
                    out["trajectory_conservation"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
                except Exception as exc:
                    out["trajectory_conservation"] = float("nan")
                    out["trajectory_conservation_error"] = f"{type(exc).__name__}: {exc}"
        elif general.adata_pre is not None:
            try:
                if pseudotime_key not in general.adata_pre.obs:
                    raise KeyError(
                        f"'{pseudotime_key}' not found in trajectory_adata_pre/adata_pre.obs"
                    )
                with _alias_graph(
                    general.adata_int,
                    general.conn_key,
                    dist_key=general.dist_key,
                    neighbors_uns_key=general.neighbors_uns_key,
                ):
                    val = me.trajectory_conservation(
                        general.adata_pre,
                        general.adata_int,
                        label_key=label_key,
                        pseudotime_key=pseudotime_key,
                        batch_key=batch_key,
                    )
                out["trajectory_conservation"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
            except Exception as exc:
                out["trajectory_conservation"] = float("nan")
                out["trajectory_conservation_error"] = f"{type(exc).__name__}: {exc}"
        else:
            out["trajectory_conservation"] = float("nan")
            out["trajectory_conservation_note"] = "adata_pre not provided for trajectory"
    else:
        out["trajectory_conservation"] = float("nan")
        out["trajectory_conservation_note"] = "compute_trajectory=False"

    # -----------------------------------------------------------------------
    # 9. Batch ASW
    # -----------------------------------------------------------------------
    if heavy.emb_key is not None:
        val, err = _safe(
            me.silhouette_batch,
            heavy.adata_int,
            batch_key,
            label_key,
            heavy.emb_key,
            metric="euclidean",
            scale=True,
            label="batch_ASW",
        )
        out["batch_ASW"] = val
        if err:
            out["batch_ASW_error"] = err
    else:
        out["batch_ASW"] = float("nan")
        out["batch_ASW_note"] = "no embedding available for this output type"

    # -----------------------------------------------------------------------
    # 10. PCR comparison
    # -----------------------------------------------------------------------
    if general.adata_pre is not None:
        try:
            val = me.pcr_comparison(
                general.adata_pre,
                general.adata_int,
                covariate=batch_key,
                embed=general.emb_key,
                n_comps=50,
                scale=True,
                verbose=verbose,
            )
            out["pcr_comparison"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["pcr_comparison"] = float("nan")
            out["pcr_comparison_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["pcr_comparison"] = float("nan")
        out["pcr_comparison_note"] = "adata_pre not provided"

    # -----------------------------------------------------------------------
    # 11. kBET
    # -----------------------------------------------------------------------
    kbet_embed = heavy.emb_key if output_type != "knn" else None
    try:
        with _alias_graph(
            heavy.adata_int,
            heavy.conn_key if output_type == "knn" else None,
            dist_key=heavy.dist_key,
            neighbors_uns_key=heavy.neighbors_uns_key,
        ):
            val = me.kBET(
                heavy.adata_int,
                batch_key=batch_key,
                label_key=label_key,
                type_=output_type,
                embed=kbet_embed,
                scaled=True,
                return_df=False,
                verbose=verbose,
            )
        out["kBET"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
    except Exception as exc:
        out["kBET"] = float("nan")
        out["kBET_error"] = f"{type(exc).__name__}: {exc}"

    # -----------------------------------------------------------------------
    # 12. Graph connectivity
    # -----------------------------------------------------------------------
    if general.conn_key is not None:
        try:
            with _alias_graph(
                general.adata_int,
                general.conn_key,
                dist_key=general.dist_key,
                neighbors_uns_key=general.neighbors_uns_key,
            ):
                val = me.graph_connectivity(general.adata_int, label_key)
            out["graph_connectivity"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
        except Exception as exc:
            out["graph_connectivity"] = float("nan")
            out["graph_connectivity_error"] = f"{type(exc).__name__}: {exc}"
    else:
        out["graph_connectivity"] = float("nan")
        out["graph_connectivity_note"] = "no connectivities in obsp"

    # -----------------------------------------------------------------------
    # 13. Graph iLISI
    # -----------------------------------------------------------------------
    lisi_use_rep = heavy.emb_key if output_type == "embed" else None
    try:
        with _alias_graph(
            heavy.adata_int,
            heavy.conn_key if output_type == "knn" else None,
            dist_key=heavy.dist_key,
            neighbors_uns_key=heavy.neighbors_uns_key,
        ):
            val = me.ilisi_graph(
                heavy.adata_int,
                batch_key=batch_key,
                type_=output_type,
                use_rep=lisi_use_rep,
                k0=90,
                subsample=lisi_pct,
                scale=True,
                n_cores=1,
                verbose=verbose,
            )
        out["iLISI"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
    except Exception as exc:
        out["iLISI"] = float("nan")
        out["iLISI_error"] = f"{type(exc).__name__}: {exc}"

    # -----------------------------------------------------------------------
    # 14. Graph cLISI
    # -----------------------------------------------------------------------
    try:
        with _alias_graph(
            heavy.adata_int,
            heavy.conn_key if output_type == "knn" else None,
            dist_key=heavy.dist_key,
            neighbors_uns_key=heavy.neighbors_uns_key,
        ):
            val = me.clisi_graph(
                heavy.adata_int,
                label_key=label_key,
                type_=output_type,
                use_rep=lisi_use_rep,
                k0=90,
                subsample=lisi_pct,
                scale=True,
                n_cores=1,
                verbose=verbose,
            )
        out["cLISI"] = float(np.nanmean(np.asarray(val, dtype=float).ravel()))
    except Exception as exc:
        out["cLISI"] = float("nan")
        out["cLISI_error"] = f"{type(exc).__name__}: {exc}"

    return out