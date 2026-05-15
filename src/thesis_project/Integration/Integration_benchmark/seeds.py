# src/thesis_project/benchmark/seeds.py
import os
import random
import numpy as np

def set_global_seed(seed: int = 0, use_torch: bool = False):
    """
    Set a single global seed for full reproducibility.
    Default seed = 0 (used across all experiments).
    """

    # Python + NumPy
    random.seed(seed)
    np.random.seed(seed)

    # Scanpy / sklearn follow numpy RNG
    try:
        import scanpy as sc
        sc.settings.set_figure_params(dpi=160, frameon=False)
    except Exception:
        pass

    # PyTorch (for scVI / scANVI / scGen)
    if use_torch:
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # Performance-friendly reproducibility
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    return seed
