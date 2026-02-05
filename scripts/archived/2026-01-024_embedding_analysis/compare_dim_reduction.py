"""Compare embedding compression strategies: quantization and PCA dimension reduction.

Fetches runs from the mike-quant-experiment project and groups them by config:
- fp32: baseline float32 embeddings
- int8: int8 quantized embeddings
- int8_dim512: int8 + PCA to 512 dims
- int8_dim256: int8 + PCA to 256 dims

Outputs a comparison CSV with eval metrics and percent change vs fp32 baseline.

Output: dim_reduction_comparison.csv
"""

import pandas as pd
import wandb

WANDB_ENTITY = "eai-ai2"
PROJECT = "mike-quant-experiment"

# Map run name patterns to column names
CONFIG_MAP = {
    "_df_qt_dim256": "int8_dim256",
    "_df_qt_dim512": "int8_dim512",
    "_df_qt": "int8",
    "_df": "fp32",  # Must be last since others contain "_df"
}

api = wandb.Api()

print("Fetching run list...")
run_info = [(r.id, r.name) for r in api.runs(f"{WANDB_ENTITY}/{PROJECT}")]
print(f"Found {len(run_info)} total runs")

# Collect metrics per config
config_metrics = {name: {} for name in CONFIG_MAP.values()}

for run_id, run_name in run_info:
    # Determine which config this run belongs to
    config_name = None
    for pattern, name in CONFIG_MAP.items():
        if pattern in run_name:
            config_name = name
            break

    if config_name is None:
        print(f"Skipping unknown run: {run_name}")
        continue

    print(f"Processing {run_name} -> {config_name}")

    # Fetch run and get metrics
    try:
        run = api.run(f"{WANDB_ENTITY}/{PROJECT}/{run_id}")
        for key, value in run.summary.items():
            if not key.startswith("eval/") or key.startswith("eval/test/"):
                continue
            if not isinstance(value, int | float):
                continue
            task = key.replace("eval/", "")
            config_metrics[config_name][task] = value
    except Exception as e:
        print(f"  Error: {e}")

# Build comparison dataframe
all_tasks = sorted(set().union(*[m.keys() for m in config_metrics.values()]))

rows = []
for task in all_tasks:
    row = {"task": task}
    fp32_val = config_metrics["fp32"].get(task)
    for config in ["fp32", "int8", "int8_dim512", "int8_dim256"]:
        val = config_metrics[config].get(task)
        row[config] = val
        # Add pct change vs fp32
        if config != "fp32" and val is not None and fp32_val is not None:
            pct = (val - fp32_val) / fp32_val * 100
            row[f"{config}_pct"] = pct
    rows.append(row)

df = pd.DataFrame(rows)

# Reorder columns
cols = [
    "task",
    "fp32",
    "int8",
    "int8_pct",
    "int8_dim512",
    "int8_dim512_pct",
    "int8_dim256",
    "int8_dim256_pct",
]
df = df[[c for c in cols if c in df.columns]]

print("\n" + df.to_string(index=False))

# Save
df.to_csv("mike-do-not-commit/dim_reduction_comparison.csv", index=False)
print("\nSaved to mike-do-not-commit/dim_reduction_comparison.csv")

# Print averages
print("\nAverage pct change vs fp32:")
for config in ["int8", "int8_dim512", "int8_dim256"]:
    col = f"{config}_pct"
    if col in df.columns:
        avg = df[col].mean()
        print(f"  {config}: {avg:.2f}%")
