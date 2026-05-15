#!/usr/bin/env python3

import os
import json
import pandas as pd
from pathlib import Path

# ================================
# CONFIG
# ================================
BASE_DIR = Path("/data1/esraa/Thesis-Project/Results/coupled_benchmark")
OUTPUT_PATH = Path("/data1/esraa/Thesis-Project/Results/coupled_benchmark/Quantitive_metrics/coupled_ti_runtime.csv")

TASKS = ["task_1", "task_2"]

IGNORE_DIRS = {
    "merged_results",
    "tables",
    "figures",
    "logs",
    "adata",
    "plots",
    "plots_pub",
}

# ================================
# COLLECT RUNTIME
# ================================
records = []

for task in TASKS:
    task_dir = BASE_DIR / task

    if not task_dir.exists():
        print(f"⚠ Missing task dir: {task_dir}")
        continue

    for ti_method in sorted(os.listdir(task_dir)):
        ti_dir = task_dir / ti_method

        if not ti_dir.is_dir():
            continue

        for integration in sorted(os.listdir(ti_dir)):
            if integration in IGNORE_DIRS:
                continue

            run_dir = ti_dir / integration
            rc_path = run_dir / "logs" / "run_config.json"

            if not rc_path.exists():
                print(f"⚠ Missing run_config: {rc_path}")
                continue

            try:
                with open(rc_path) as f:
                    data = json.load(f)

                elapsed = data.get("ti_method", {}).get("elapsed_seconds", None)

                if elapsed is None:
                    print(f"⚠ No elapsed_seconds: {rc_path}")
                    continue

                records.append({
                    "task": task,
                    "ti_method": ti_method,
                    "integration": integration,
                    "elapsed_seconds": float(elapsed),
                })

            except Exception as e:
                print(f"❌ Failed reading {rc_path}: {e}")


# ================================
# BUILD DATAFRAME
# ================================
df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("No runtime data collected.")

# ================================
# AGGREGATION
# ================================
summary_rows = []

for ti_method, g in df.groupby("ti_method"):
    row = {"ti_method": ti_method}

    task_totals = {}
    task_means = {}

    for task in TASKS:
        g_task = g[g["task"] == task]

        total = g_task["elapsed_seconds"].sum()
        mean = g_task["elapsed_seconds"].mean()

        row[f"{task}_total"] = round(total, 2)
        row[f"{task}_mean"] = round(mean, 2)

        task_totals[task] = total

    # Overall
    overall_total = sum(task_totals.values())
    overall_mean = g["elapsed_seconds"].mean()

    row["overall_total"] = round(overall_total, 2)
    row["overall_mean"] = round(overall_mean, 2)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Sort by fastest overall runtime
summary_df = summary_df.sort_values("overall_total")

# ================================
# SAVE
# ================================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Saved runtime table:")
print(OUTPUT_PATH)
print(summary_df)