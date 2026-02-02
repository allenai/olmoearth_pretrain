"""Compare fp32 vs int8 quantized embedding eval metrics from a wandb sweep.

Fetches runs from the mike-quant-experiment-int8-sweep project, separates them
by quantization type (based on "_qt" suffix in run name), takes the max metric
across runs for each task, and outputs a comparison CSV showing the difference
and percent change between fp32 and int8.

Output: quant_comparison.csv
"""

import pandas as pd
import wandb

WANDB_ENTITY = "eai-ai2"
PROJECT = "mike-quant-experiment-int8-sweep"

api = wandb.Api()

# First get all run IDs and names
print("Fetching run list...")
run_info = [(r.id, r.name) for r in api.runs(f"{WANDB_ENTITY}/{PROJECT}")]
print(f"Found {len(run_info)} total runs")

# Separate runs by quantization
qt_metrics = {}
fp_metrics = {}
qt_count = 0
fp_count = 0
skipped = 0

for run_id, run_name in run_info:
    is_qt = "_qt" in run_name
    target = qt_metrics if is_qt else fp_metrics

    # Re-fetch the run individually (avoids lazy loading issues)
    try:
        run = api.run(f"{WANDB_ENTITY}/{PROJECT}/{run_id}")
        summary_items = list(run.summary.items())
    except Exception as e:
        print(f"Skipping {run_name}: {type(e).__name__}: {e}")
        skipped += 1
        continue

    if is_qt:
        qt_count += 1
    else:
        fp_count += 1

    for key, value in summary_items:
        if not key.startswith("eval/") or key.startswith("eval/test/"):
            continue
        if not isinstance(value, int | float):
            continue
        # Keep max across sweep
        target[key] = max(target.get(key, float("-inf")), value)

print(f"Processed: {fp_count} fp32 runs, {qt_count} int8 runs, {skipped} skipped")
print(f"FP32 metrics found: {len(fp_metrics)}")
print(f"INT8 metrics found: {len(qt_metrics)}")

# Build comparison dataframe
tasks = sorted(set(qt_metrics.keys()) | set(fp_metrics.keys()))
rows = []
for task in tasks:
    task_name = task.replace("eval/", "")
    fp_val = fp_metrics.get(task)
    qt_val = qt_metrics.get(task)
    diff = (qt_val - fp_val) if (fp_val and qt_val) else None
    pct = (diff / fp_val * 100) if (fp_val and diff) else None
    rows.append(
        {
            "task": task_name,
            "fp32": fp_val,
            "int8": qt_val,
            "diff": diff,
            "pct_change": pct,
        }
    )

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("quant_comparison.csv", index=False)
print("\nSaved to quant_comparison.csv")
