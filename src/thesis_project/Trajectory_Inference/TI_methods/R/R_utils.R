#!/usr/bin/env python3
"""
Shared R-script execution utility for TI benchmarking framework.

Centralises the _run_rscript helper previously duplicated across
TSCAN, Slingshot, SCORPIUS, Monocle2, and Monocle3 adapters.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict


def run_rscript(
    rscript: str,
    script_path: Path,
    args: Dict[str, Any],
    log_path: Path,
) -> None:
    """
    Execute an R script via Rscript with key-value CLI arguments.

    Parameters
    ----------
    rscript:
        Path (or name on PATH) of the Rscript executable.
    script_path:
        Absolute path to the .R script to run.
    args:
        Mapping of argument names to values; each pair is passed as
        ``--<name> <value>`` on the command line.
    log_path:
        File to which stdout + stderr are written.  The parent
        directory is created automatically if it does not exist.

    Raises
    ------
    RuntimeError
        If Rscript exits with a non-zero return code.
    """
    # Ensure the log directory exists before opening the file (Bug 4 fix)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [rscript, str(script_path)]
    for k, v in args.items():
        cmd.extend([f"--{k}", str(v)])

    with log_path.open("w") as fh:
        proc = subprocess.run(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript failed with exit code {proc.returncode}. "
            f"See log for details: {log_path}"
        )