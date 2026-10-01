"""Per-band mean/std of allcap modalities, in the norm_configs/computed.json format.

Streams raw h5 values over a random subset of samples, excluding fill pixels
(0 for the uint16 reflectance products) and non-finite values, and writes
``{modality: {band: {count, mean, std, var}}}``, plus the proposed
``landsat_l2`` entry for ``computed.json`` under the key ``landsat_l2_entry``.

The allcap corpus is not cloud filtered, so its optical stats are cloud-inflated
(sentinel2_l2a visible-band means ~3x the existing computed.json values, SWIR
within ~10-20%, i.e. same units). The existing sentinel2_l2a / sentinel1 stats
are kept (evals normalize with them too). For ``landsat_l2`` (no existing
entry), B1-B7 are converted from the wavelength-matched sentinel2_l2a
surface-reflectance stats so both optical sensors share one normalized scale
(Landsat C2 L2: reflectance = DN * 2.75e-5 - 0.2; S2 L2A: reflectance = DN /
10000); thermal B10 has no S2 counterpart and uses the measured all-sky stats.

Usage:
    python scripts/vnext/allcap_fact9k/compute_norm_stats.py \
        --h5py_dir <allcap h5 dir> --modalities landsat_l2 sentinel2_l2a sentinel1 \
        --num_samples 300 --output /tmp/allcap_norm.json
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (zstd filter)
import numpy as np

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import load_computed_config

# Fill value per modality: uint16 reflectance products use 0 for no data.
FILL_VALUE = {"landsat_l2": 0, "sentinel2_l2a": 0}

LANDSAT_L2_SCALE, LANDSAT_L2_OFFSET = 2.75e-5, -0.2
# Landsat 8/9 OLI band -> wavelength-matched Sentinel-2 MSI band.
LANDSAT_TO_S2_BAND = {
    "B1": "B01",
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B5": "B8A",
    "B6": "B11",
    "B7": "B12",
}


def landsat_l2_entry(s2_stats: dict, measured_b10: dict) -> dict:
    """computed.json entry for landsat_l2 (see module docstring)."""
    entry = {}
    for band, s2_band in LANDSAT_TO_S2_BAND.items():
        reflectance_mean = s2_stats[s2_band]["mean"] / 10000.0
        reflectance_std = s2_stats[s2_band]["std"] / 10000.0
        std = reflectance_std / LANDSAT_L2_SCALE
        entry[band] = {
            "mean": (reflectance_mean - LANDSAT_L2_OFFSET) / LANDSAT_L2_SCALE,
            "std": std,
            "var": std * std,
        }
    entry["B10"] = measured_b10
    return dict(sorted(entry.items()))


def main() -> None:
    """Stream per-band sums over a random sample of h5 files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5py_dir", required=True)
    parser.add_argument("--modalities", nargs="+", required=True)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    h5py_dir = Path(args.h5py_dir)
    num_files = int(h5py_dir.name)
    indices = random.Random(args.seed).sample(range(num_files), args.num_samples)
    sums = {
        m: np.zeros((3, len(Modality.get(m).band_order)), dtype=np.float64)
        for m in args.modalities
    }
    for n, idx in enumerate(indices):
        with h5py.File(h5py_dir / f"sample_{idx}.h5", "r") as f:
            for m in args.modalities:
                if m not in f:
                    continue
                data = f[m][()].astype(np.float64)  # (H, W, T, C)
                flat = data.reshape(-1, data.shape[-1])
                valid = np.isfinite(flat)
                if m in FILL_VALUE:
                    valid &= flat != FILL_VALUE[m]
                x = np.where(valid, flat, 0.0)
                sums[m][0] += valid.sum(axis=0)
                sums[m][1] += x.sum(axis=0)
                sums[m][2] += (x * x).sum(axis=0)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(indices)} files", flush=True)

    out: dict = {}
    for m, (count, s1, s2) in sums.items():
        mean = s1 / count
        var = s2 / count - mean**2
        out[m] = {
            band: {
                "count": int(count[i]),
                "mean": float(mean[i]),
                "std": float(np.sqrt(var[i])),
                "var": float(var[i]),
            }
            for i, band in enumerate(Modality.get(m).band_order)
        }
    if "landsat_l2" in out:
        out["landsat_l2_entry"] = landsat_l2_entry(
            load_computed_config()["sentinel2_l2a"], out["landsat_l2"]["B10"]
        )
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
