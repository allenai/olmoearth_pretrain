"""Count, per eval dataset, which modality stack each sample actually gets.

The year-aligned model.yamls declare S2/S1 as required and Landsat as optional,
every one of them with load_all_layers over twelve monthly layers. rslearn
treats load_all_layers as all-or-nothing (is_data_input_available), so a window
missing a single Landsat month loses the whole Landsat year rather than eleven
twelfths of it -- and because a required input missing means the window is
dropped during resolution, "S2 only" is not a state a surviving sample can be
in. This script measures that claim per dataset instead of assuming it.

Reads completion markers directly rather than building the datasets: rslearn's
own resolution is the ground truth, but building it eight times over ~160k
windows is far slower than a threaded stat sweep, and the marker is exactly
what is_layer_completed checks.

Usage:
    python scripts/tools/audit_modality_coverage.py [--datasets a,b] [--workers 64]
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

import yaml

EVAL_ROOT = "/weka/dfive-default/olmoearth/eval_datasets"
CONFIG_ROOT = "data/rslearn_dataset_configs"

DATASETS = [
    "ethiopia_crops_year_aligned",
    "africa_crop_mask_year_aligned",
    "canada_crops_fine_year_aligned",
    "canada_crops_coarse_year_aligned",
    "descals_year_aligned",
    "lcmap_lu_year_aligned",
    "glance_year_aligned",
    "us_trees_year_aligned",
]

# Mask inputs resolve to the modality they mask and are not part of the stack.
MASK_PREFIXES = ("sentinel2_scl", "landsat_qa")


def load_inputs(dataset: str) -> dict[str, dict]:
    """Per-input {layers, load_all_layers, required} from the repo model.yaml."""
    path = os.path.join(CONFIG_ROOT, dataset, "model.yaml")
    config = yaml.safe_load(open(path))
    inputs = config["data"]["init_args"]["inputs"]
    out = {}
    for name, cfg in inputs.items():
        if cfg.get("is_target"):
            continue
        layers = cfg.get("layers") or []
        if not layers or layers[0].startswith(MASK_PREFIXES):
            continue
        out[name] = {
            "layers": layers,
            "all": bool(cfg.get("load_all_layers")),
            "required": cfg.get("required", True),
        }
    return out


def input_complete(window_dir: str, spec: dict) -> bool:
    """Mirror rslearn's is_data_input_available for a single window."""
    completed = 0
    for layer in spec["layers"]:
        if os.path.exists(os.path.join(window_dir, "layers", layer, "completed")):
            completed += 1
            if not spec["all"]:
                return True
    return completed == len(spec["layers"]) if spec["all"] else completed > 0


def classify(window_dir: str, inputs: dict[str, dict]) -> tuple[str, str]:
    """Return (split, bucket) for one window."""
    try:
        with open(os.path.join(window_dir, "metadata.json")) as f:
            split = json.load(f).get("options", {}).get("eval_split") or "?"
    except OSError:
        split = "?"

    for name, spec in inputs.items():
        if spec["required"] and not input_complete(window_dir, spec):
            return split, f"dropped ({name} incomplete)"

    landsat = inputs.get("landsat")
    if landsat is None:
        return split, "no landsat input in config"
    if input_complete(window_dir, landsat):
        return split, "S2+S1+Landsat"
    return split, "S2+S1 (landsat dropped whole)"


def audit(dataset: str, workers: int) -> None:
    """Print the stack breakdown for one dataset."""
    inputs = load_inputs(dataset)
    root = os.path.join(EVAL_ROOT, dataset, "windows")
    window_dirs = [
        entry.path
        for group in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, group))
        for entry in os.scandir(os.path.join(root, group))
        if entry.is_dir()
    ]

    counts: dict[tuple[str, str], int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for split, bucket in pool.map(
            lambda w: classify(w, inputs), window_dirs, chunksize=64
        ):
            counts[(split, bucket)] = counts.get((split, bucket), 0) + 1

    required = sorted(n for n, s in inputs.items() if s["required"])
    print(f"\n=== {dataset}  ({len(window_dirs)} windows on disk)")
    print(f"    required inputs: {required}")
    by_bucket: dict[str, int] = {}
    for (split, bucket), n in counts.items():
        by_bucket[bucket] = by_bucket.get(bucket, 0) + n
    total = sum(by_bucket.values())
    for bucket, n in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        per_split = {s: c for (s, b), c in sorted(counts.items()) if b == bucket}
        print(f"    {bucket:34s} {n:6d} ({100 * n / total:6.2f}%)  {per_split}")
    print(flush=True)


def main() -> None:
    """Audit every dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    names = args.datasets.split(",") if args.datasets else DATASETS
    for dataset in names:
        audit(dataset, args.workers)


if __name__ == "__main__":
    main()
