import os
import json
import pandas as pd

base_dir = "/data1/esraa/Thesis-Project/Results/Trajectory_Inference"
results = []

for root, dirs, files in os.walk(base_dir):
    if "run_config.json" in files:
        json_path = os.path.join(root, "run_config.json")
        # Extract dataset, method, task from path
        parts = json_path.split(os.sep)
        try:
            dataset = parts[-5]
            method = parts[-4]
            task = parts[-3]
        except IndexError:
            continue  # skip if path is malformed
        # Read elapsed_seconds
        with open(json_path) as f:
            data = json.load(f)
            elapsed = data.get("ti_method", {}).get("elapsed_seconds", None)
        results.append({
            "method": method,
            "task": task,
            "elapsed_seconds": elapsed
        })

# Create DataFrame
df = pd.DataFrame(results)
pivot = df.pivot_table(index="method", columns="task", values="elapsed_seconds")
pivot.to_csv("ti_methods_runtime_table.csv")
print("Saved: ti_methods_runtime_table.csv")