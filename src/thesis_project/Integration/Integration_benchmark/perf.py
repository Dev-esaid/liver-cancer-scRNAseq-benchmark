import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import psutil
import pandas as pd

@dataclass
class PerfRecord:
    step: str
    event: str           # "start" / "end"
    time_s: Optional[float]
    rss_gb: float
    gpu_mem_gb: Optional[float] = None

class PerfLogger:
    """Step-level runtime + memory (CPU + optional GPU)."""
    def __init__(self, track_gpu: bool = False):
        self.proc = psutil.Process(os.getpid())
        self.track_gpu = track_gpu
        self._t0: Optional[float] = None
        self.records: List[PerfRecord] = []

    def _rss_gb(self) -> float:
        return self.proc.memory_info().rss / (1024**3)

    def _gpu_mem_gb(self) -> Optional[float]:
        if not self.track_gpu:
            return None
        try:
            import torch
            if torch.cuda.is_available():
                return float(torch.cuda.memory_allocated() / (1024**3))
        except Exception:
            pass
        return None

    def start(self, step: str):
        self._t0 = time.time()
        self.records.append(PerfRecord(step=step, event="start", time_s=None, rss_gb=self._rss_gb(), gpu_mem_gb=self._gpu_mem_gb()))

    def end(self, step: str):
        dt = time.time() - self._t0 if self._t0 is not None else None
        self.records.append(PerfRecord(step=step, event="end", time_s=dt, rss_gb=self._rss_gb(), gpu_mem_gb=self._gpu_mem_gb()))
        self._t0 = None

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])

    def save_csv(self, path: str) -> pd.DataFrame:
        df = self.to_df()
        df.to_csv(path, index=False)
        return df
