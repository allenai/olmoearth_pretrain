"""Compare baseline vs percentile-quantized embedding eval metrics from a wandb sweep.

Fetches runs from the percentile-quant-sweep project, separates them by quantization
bit level (based on run config or name), takes the max metric across runs for each
task, and outputs a comparison CSV showing the difference and percent change vs baseline.

Output: percentile_quant_comparison.csv
"""

import pandas as pd
import wandb

WANDB_ENTITY = "eai-ai2"
PROJECTS = [
    "percentile-quant-sweep-fix8b",
    "percentile-quant-sweep",
]

# Map to detect bit levels from run config or name
BIT_LEVELS = ["baseline", "8bit", "4bit", "2bit", "1bit"]

api = wandb.Api()

# First get all run IDs and names
print("Fetching run list...")
run_info = [
    (project, r.id, r.name)
    for project in PROJECTS
    for r in api.runs(f"{WANDB_ENTITY}/{project}")
]
print(f"Found {len(run_info)} total runs")

# Collect metrics per bit level
config_metrics: dict[str, dict[str, float]] = {level: {} for level in BIT_LEVELS}
config_counts: dict[str, int] = {level: 0 for level in BIT_LEVELS}
skipped = 0


def get_bit_level(run: wandb.apis.public.Run) -> str | None:
    """Determine the bit level from run config or name."""
    # Check config first
    config = run.config
    if config.get("quantize_embeddings"):
        bits = config.get("quantize_bits")
        if bits is not None:
            return f"{bits}bit"
    elif not config.get("quantize_embeddings", True):
        # Explicitly disabled quantization = baseline
        return "baseline"

    # Fall back to name patterns
    name = run.name.lower()
    if "_qt1" in name or "_1bit" in name:
        return "1bit"
    if "_qt2" in name or "_2bit" in name:
        return "2bit"
    if "_qt4" in name or "_4bit" in name:
        return "4bit"
    if "_qt8" in name or "_8bit" in name:
        return "8bit"
    if "_qt" not in name and "bit" not in name:
        return "baseline"

    return None


for project, run_id, run_name in run_info:
    try:
        run = api.run(f"{WANDB_ENTITY}/{project}/{run_id}")
        bit_level = get_bit_level(run)

        if bit_level is None:
            print(f"Skipping {run_name}: couldn't determine bit level")
            skipped += 1
            continue

        config_counts[bit_level] += 1
        target = config_metrics[bit_level]

        for key, value in run.summary.items():
            if not key.startswith("eval/") or key.startswith("eval/test/"):
                continue
            if not isinstance(value, int | float):
                continue
            # Keep max across sweep
            target[key] = max(target.get(key, float("-inf")), value)

    except Exception as e:
        print(f"Skipping {run_name}: {type(e).__name__}: {e}")
        skipped += 1
        continue

print("\nRun counts per config:")
for level in BIT_LEVELS:
    print(
        f"  {level}: {config_counts[level]} runs, {len(config_metrics[level])} metrics"
    )
print(f"  skipped: {skipped}")

# Build comparison dataframe
all_tasks = sorted(set().union(*[set(m.keys()) for m in config_metrics.values()]))

rows = []
for task in all_tasks:
    task_name = task.replace("eval/", "")
    row: dict[str, str | float | None] = {"task": task_name}

    for level in BIT_LEVELS:
        row[level] = config_metrics[level].get(task)

    rows.append(row)

df = pd.DataFrame(rows)

# Add average row at the bottom
avg_row: dict[str, str | float | None] = {"task": "AVERAGE"}
for level in BIT_LEVELS:
    avg_row[level] = df[level].mean()
df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

# Reorder columns
cols = ["task"] + BIT_LEVELS
df = df[[c for c in cols if c in df.columns]]

print("\n" + df.to_string(index=False))

output_path = "percentile_quant_comparison.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
