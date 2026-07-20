"""Verify the srtm normalization values against real data.

srtm is a single terrain modality [elevation, slope, aspect_sin, aspect_cos]; only the
elevation band is stored on disk, the rest are derived from it at load time
(compute_srtm_bands). The slope/aspect stats are therefore not produced by the standard
machine-generated compute_h5_norm run, so this script measures the *actual* per-band
statistics over the dataset to check whether the committed norm values are reasonable
and, if not, replace them. Elevation is measured too, as a sanity check against the
existing committed values.

It reads the stored ``srtm`` elevation band straight out of the per-sample h5 files and
runs the same ``compute_srtm_bands`` used at train time, rather than going through the
full sample pipeline. That is both faster (one band read, no subsetting) and avoids the
missing-modality fill step, which cannot determine a tile's H/W when srtm is the only
modality requested and a given sample happens to lack it.

It reports, per band (elevation, slope, aspect_sin, aspect_cos):
  - streaming mean / std over all valid (non-missing) pixels
  - min / max and percentiles (p0.1, p1, p50, p99, p99.9) from a random pixel reservoir
  - the fraction of pixels that were missing (the slope/aspect missing-halo)
then prints the committed config values side by side and emits paste-ready JSON blocks
for both predefined.json and computed.json.

Example usage:
    python3 scripts/tools/20260720_verify_srtm_terrain_norm.py \
        --h5py_dir /path/to/h5py_data \
        --estimate_from 500 \
        --output_path srtm_norm_report.json
"""

import argparse
import json
import logging
import random
from importlib.resources import files
from typing import Any

import h5py

# hdf5 plugin is needed to decompress the data for certain compression types (e.g. zstd)
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd
from olmo_core.utils import prepare_cli_environment
from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import compute_srtm_bands
from olmoearth_pretrain.data.utils import update_streaming_stats
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py

logger = logging.getLogger(__name__)

MODALITY = Modality.SRTM
# Cap on the number of pixel values kept per band for percentile/min/max estimation.
# 4M float64 values ~= 32 MB per band, which is plenty for stable percentiles.
RESERVOIR_CAP = 4_000_000
PERCENTILES = [0.1, 1.0, 50.0, 99.0, 99.9]


def _load_committed(config_name: str) -> dict[str, Any]:
    """Load a committed norm config from package resources."""
    with (files("olmoearth_pretrain.data.norm_configs") / config_name).open() as f:
        return json.load(f)


def _srtm_present_indices(h5py_dir: UPath) -> list[int]:
    """Row indices (== h5 filename indices) of samples whose raw srtm band is present.

    File names are ``sample_{index}.h5`` where index is the row position in the metadata
    CSV (see OlmoEarthDataset.prepare, which sets sample_indices = arange(num_samples)).
    The CSV has a per-modality presence column, so we select rows where srtm is nonzero
    to avoid opening files we know have no elevation to derive from.
    """
    meta_path = h5py_dir / ConvertToH5py.sample_metadata_fname
    metadata_df = pd.read_csv(str(meta_path))
    if Modality.SRTM.name not in metadata_df.columns:
        raise ValueError(
            f"metadata CSV at {meta_path} has no '{Modality.SRTM.name}' column; "
            f"columns are {list(metadata_df.columns)}"
        )
    present = pd.to_numeric(metadata_df[Modality.SRTM.name], errors="coerce").fillna(0)
    return metadata_df.index[present > 0].tolist()


def _read_srtm(h5py_dir: UPath, index: int) -> np.ndarray | None:
    """Read the raw srtm band for one sample, or None if it is absent in the file."""
    fpath = h5py_dir / ConvertToH5py.sample_file_pattern.format(index=index)
    with fpath.open("rb") as f:
        with h5py.File(f, "r") as h5file:
            if Modality.SRTM.name not in h5file:
                return None
            return h5file[Modality.SRTM.name][()]


def measure(
    h5py_dir: UPath,
    estimate_from: int | None,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure per-band statistics for the srtm modality over the dataset.

    Missing pixels are masked out per band (rather than skipping whole samples) because
    the derived slope/aspect always has a missing halo wherever the elevation had voids
    or the tile edge, so whole-sample skipping would bias toward void-free tiles.
    """
    rng = np.random.default_rng(seed)
    bands = MODALITY.band_order

    candidates = _srtm_present_indices(h5py_dir)
    logger.info(f"{len(candidates)} samples have a raw srtm band")
    if estimate_from is not None and estimate_from < len(candidates):
        random.Random(seed).shuffle(candidates)
        indices = candidates[:estimate_from]
    else:
        indices = candidates

    stats = {b: {"count": 0, "mean": 0.0, "var": 0.0} for b in bands}
    reservoir: dict[str, list[np.ndarray]] = {b: [] for b in bands}
    reservoir_n = {b: 0 for b in bands}
    total_pixels = {b: 0 for b in bands}
    missing_pixels = {b: 0 for b in bands}
    samples_with_data = 0
    samples_missing_srtm = 0

    for i in tqdm(indices):
        raw = _read_srtm(h5py_dir, i)
        if raw is None:
            samples_missing_srtm += 1
            continue
        srtm_bands = compute_srtm_bands(raw, np.dtype(np.float32))  # [H, W, T, 4]
        samples_with_data += 1
        for idx, band in enumerate(bands):
            band_data = np.asarray(srtm_bands[..., idx]).reshape(-1)
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
    result["samples_with_srtm"] = samples_with_data
    result["samples_missing_srtm"] = samples_missing_srtm
    result["h5py_dir"] = str(h5py_dir)
    return result


def report(result: dict[str, Any]) -> None:
    """Print a human-readable comparison of measured vs committed values."""
    predefined = _load_committed("predefined.json").get(MODALITY.name, {})
    computed = _load_committed("computed.json").get(MODALITY.name, {})

    print("\n" + "=" * 78)
    print(f"srtm norm verification  ({result['samples_with_srtm']} tiles measured)")
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
        c = computed.get(band)
        if c:
            print(
                f"  committed computed.json : mean={c['mean']:.5f}  std={c['std']:.5f}"
            )
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
        help="Number of random srtm-bearing tiles to sample (default: all).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to write the full measured stats as JSON.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    h5py_dir = UPath(args.h5py_dir)
    if not h5py_dir.exists():
        raise FileNotFoundError(f"H5PY directory does not exist: {h5py_dir}")

    result = measure(h5py_dir, estimate_from=args.estimate_from, seed=args.seed)
    report(result)

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote full stats to {args.output_path}")
