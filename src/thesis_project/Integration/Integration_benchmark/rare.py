from typing import List, Optional
import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
except Exception:
    sp = None

def rare_labels_from_obs(labels, rare_freq_threshold: float = 0.01) -> List[str]:
    freq = pd.Series(labels).value_counts(normalize=True)
    return freq[freq < rare_freq_threshold].index.astype(str).tolist()

def fast_rare_knn_purity(
    ad,
    label_key: str,
    conn,
    rare_types: List[str],
    k: int = 50,
    max_cells_per_type: int = 500,
    seed: int = 0,
) -> float:
    """Fast proxy used in BBKNN/Harmony/Scanorama scripts."""
    labels = ad.obs[label_key].astype(str).values
    rng = np.random.default_rng(seed)
    purities = []

    for rt in rare_types:
        idx = np.where(labels == rt)[0]
        if idx.size < 20:
            continue

        take = rng.choice(idx, size=min(max_cells_per_type, idx.size), replace=False)
        per_cell = []
        for i in take:
            row = conn[i]
            if sp is not None and sp.issparse(row):
                row = row.toarray().ravel()
            else:
                row = np.asarray(row).ravel()
            nn = np.argsort(row)[::-1]
            nn = nn[nn != i][:k]
            if nn.size:
                per_cell.append(float(np.mean(labels[nn] == rt)))
        if per_cell:
            purities.append(float(np.mean(per_cell)))

    return float(np.mean(purities)) if purities else float("nan")
