"""Verify the srtm_terrain normalization values against real data.

srtm_terrain (slope + aspect_sin + aspect_cos) is a derived modality computed at load
time from the raw SRTM elevation band, so it is not covered by the machine-generated
computed.json. Its committed norm values were seeded from theoretical bounds:
  - predefined.json: slope in [0, pi/2], aspect_sin/cos in [-1, 1]
  - computed.json:   slope mean=pi/4 std=pi/8, aspect_sin/cos mean=0 std=0.5
This script measures the *actual* per-band statistics over the dataset so you can check
whether those seeds are reasonable and, if not, replace them.

It reports, per band (slope, aspect_sin, aspect_cos):
  - streaming mean / std over all valid (non-missing) pixels
  - min / max and percentiles (p0.1, p1, p50, p99, p99.9) from a random pixel reservoir
  - the fraction of pixels that were missing (the slope/aspect missing-halo)
then prints the committed config values side by side and emits paste-ready JSON blocks
for both predefined.json and computed.json.

Example usage:
    python3 scripts/tools/20260720_verify_srtm_terrain_norm.py \
        --h5py_dir /path/to/h5py_data \
        --estimate_from 500 \
        --output_path srtm_terrain_norm_report.json
"""

import argparse
import json
import logging
import random
from importlib.resources import files
from typing import Any

import numpy as np
from olmo_core.utils import prepare_cli_environment
from tqdm import tqdm

from olmoearth_pretrain.data.constants import (
    IMAGE_TILE_SIZE,
    MISSING_VALUE,
    Modality,
)
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthDatasetConfig,
)
from olmoearth_pretrain.data.utils import update_streaming_stats

logger = logging.getLogger(__name__)

MODALITY = Modality.SRTM_TERRAIN
# Cap on the number of pixel values kept per band for percentile/min/max estimation.
# 4M float64 values ~= 32 MB per band, which is plenty for stable percentiles.
RESERVOIR_CAP = 4_000_000
PERCENTILES = [0.1, 1.0, 50.0, 99.0, 99.9]


def _load_committed(config_name: str) -> dict[str, Any]:
    """Load a committed norm config from package resources."""
    with (files("olmoearth_pretrain.data.norm_configs") / config_name).open() as f:
        return json.load(f)


def measure(
    dataset: OlmoEarthDataset,
    estimate_from: int | None,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure per-band statistics for srtm_terrain over the dataset.

    Missing pixels are masked out per band (rather than skipping whole samples) because
    the derived slope/aspect always has a missing halo wherever the elevation had voids
    or the tile edge, so whole-sample skipping would bias toward void-free tiles.
    """
    rng = np.random.default_rng(seed)
    bands = MODALITY.band_order

    dataset_len = len(dataset)
    if estimate_from is not None and estimate_from < dataset_len:
        indices = random.sample(range(dataset_len), k=estimate_from)
    else:
        indices = list(range(dataset_len))

    stats = {b: {"count": 0, "mean": 0.0, "var": 0.0} for b in bands}
    reservoir: dict[str, list[np.ndarray]] = {b: [] for b in bands}
    reservoir_n = {b: 0 for b in bands}
    total_pixels = {b: 0 for b in bands}
    missing_pixels = {b: 0 for b in bands}
    samples_with_data = 0

    for i in tqdm(indices):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]
        data = sample.as_dict().get(MODALITY.name)
        if data is None:
            continue
        samples_with_data += 1
        for idx, band in enumerate(bands):
            band_data = np.asarray(data[..., idx]).reshape(-1)
            missing = band_data == MISSING_VALUE
            total_pixels[band] += band_data.size
            missing_pixels[band] += int(missing.sum())
            valid = band_data[~missing].astype(np.float64)
            if valid.size == 0:
                continue
            # Exact streaming mean / variance over all valid pixels.
            new_count, new_mean, new_var = update_streaming_stats(
                stats[band]["count"], stats[band]["mean"], stats[band]["var"], valid
            )
            stats[band]["count"] = int(new_count)
            stats[band]["mean"] = float(new_mean)
            stats[band]["var"] = float(new_var)
            # Reservoir sample for percentiles / min / max.
            reservoir_n[band] += valid.size
            if reservoir_n[band] <= RESERVOIR_CAP:
                reservoir[band].append(valid)
            else:
                # Keep a random subset proportional to the reservoir cap.
                keep = rng.random(valid.size) < (RESERVOIR_CAP / reservoir_n[band])
                if keep.any():
                    reservoir[band].append(valid[keep])

    result: dict[str, Any] = {"bands": {}}
    for band in bands:
        count = stats[band]["count"]
        std = (stats[band]["var"] / count) ** 0.5 if count > 0 else float("nan")
        pooled = (
            np.concatenate(reservoir[band])
            if reservoir[band]
            else np.array([], dtype=np.float64)
        )
        pcts = (
            {f"p{p}": float(np.percentile(pooled, p)) for p in PERCENTILES}
            if pooled.size
            else {}
        )
        result["bands"][band] = {
            "count": count,
            "mean": stats[band]["mean"],
            "std": std,
            "var": stats[band]["var"],  # M2 accumulator, matches compute_h5_norm.py
            "min": float(pooled.min()) if pooled.size else float("nan"),
            "max": float(pooled.max()) if pooled.size else float("nan"),
            "percentiles": pcts,
            "missing_fraction": (
                missing_pixels[band] / total_pixels[band]
                if total_pixels[band]
                else float("nan")
            ),
        }
    result["samples_sampled"] = len(indices)
    result["samples_with_srtm_terrain"] = samples_with_data
    result["dataset_len"] = dataset_len
    result["h5py_dir"] = str(dataset.h5py_dir)
    return result


def report(result: dict[str, Any]) -> None:
    """Print a human-readable comparison of measured vs committed values."""
    predefined = _load_committed("predefined.json").get(MODALITY.name, {})
    computed = _load_committed("computed.json").get(MODALITY.name, {})

    print("\n" + "=" * 78)
    print(
        f"srtm_terrain norm verification  "
        f"({result['samples_with_srtm_terrain']}/{result['samples_sampled']} "
        f"sampled tiles had data)"
    )
    print("=" * 78)
    for band, m in result["bands"].items():
        pc = m["percentiles"]
        print(f"\n[{band}]  missing_fraction={m['missing_fraction']:.4f}")
        print(
            f"  measured : mean={m['mean']:.5f}  std={m['std']:.5f}  "
            f"min={m['min']:.4f}  max={m['max']:.4f}"
        )
        if pc:
            print("  quantiles: " + "  ".join(f"{k}={v:.4f}" for k, v in pc.items()))
        # Committed computed (mean/std) — this is what actually drives normalization.
        c = computed.get(band)
        if c:
            print(
                f"  committed computed.json : mean={c['mean']:.5f}  std={c['std']:.5f}"
            )
        # Committed predefined (min/max) — the fallback strategy.
        p = predefined.get(band)
        if p:
            print(f"  committed predefined.json: min={p['min']}  max={p['max']}")

    # Paste-ready blocks. computed.json is tried first at runtime, so it matters most.
    computed_block = {
        band: {
            "count": m["count"],
            "mean": m["mean"],
            "std": m["std"],
            "var": m["var"],
        }
        for band, m in result["bands"].items()
    }
    # For predefined min/max, use robust percentile bounds rather than raw min/max so a
    # handful of extreme pixels don't blow out the window (mirrors the ~2 sigma spirit
    # of the computed strategy).
    predefined_block = {
        band: {
            "min": round(m["percentiles"].get("p0.1", m["min"]), 6),
            "max": round(m["percentiles"].get("p99.9", m["max"]), 6),
        }
        for band, m in result["bands"].items()
    }
    print("\n" + "-" * 78)
    print("Paste-ready computed.json block (drives normalization; tried first):")
    print(json.dumps({MODALITY.name: computed_block}, indent=2))
    print("\nPaste-ready predefined.json block (fallback; p0.1/p99.9 bounds):")
    print(json.dumps({MODALITY.name: predefined_block}, indent=2))
    print("-" * 78)


if __name__ == "__main__":
    prepare_cli_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5py_dir", type=str, required=True)
    parser.add_argument(
        "--estimate_from",
        type=int,
        default=None,
        help="Number of random tiles to sample (default: all).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to write the full measured stats as JSON.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # srtm_terrain is derived in read_h5_file whenever it is a training modality and the
    # raw srtm band is present in the h5. normalize=False so we measure raw values.
    dataset_config = OlmoEarthDatasetConfig(
        h5py_dir=args.h5py_dir,
        training_modalities=[MODALITY.name],
        normalize=False,
        seed=args.seed,
    )
    dataset = dataset_config.build()
    dataset.prepare()

    result = measure(dataset, estimate_from=args.estimate_from, seed=args.seed)
    report(result)

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote full stats to {args.output_path}")
