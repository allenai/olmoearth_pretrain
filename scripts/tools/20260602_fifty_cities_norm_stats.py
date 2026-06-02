"""Compute per-band normalization stats for the 50Cities eval dataset.

Writes ``norm_stats.json`` into the dataset dir, which FiftyCitiesDataset loads
when ``norm_stats_from_pretrained=False``. Stats are computed over *all* tiles
(split-mode independent), in the stored eval-band order:

    S2: EVAL_S2_L2A_BAND_NAMES  (raw uint16 reflectance, B10/cirrus dropped)
    S1: EVAL_S1_BAND_NAMES      (dB; the -60 dB nodata floor is excluded)

Run on a machine with the dataset mounted (e.g. weka):

    python scripts/tools/20260602_fifty_cities_norm_stats.py \
        --data_dir /weka/dfive-default/presto_eval_sets/fifty_cities
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets import paths
from olmoearth_pretrain.evals.datasets.constants import EVAL_S2_L2A_BAND_NAMES
from olmoearth_pretrain.evals.datasets.fifty_cities_dataset import S1_STORED_BAND_NAMES
from olmoearth_pretrain.evals.datasets.fifty_cities_processor import S1_LINEAR_FLOOR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fifty_cities_norm_stats")

# S1 pixels at (or below) the dB nodata floor are excluded from S1 statistics.
S1_FLOOR_DB = 10.0 * np.log10(S1_LINEAR_FLOOR) + 1e-3


class BandAccumulator:
    """Streaming per-band sum / sumsq / min / max accumulator."""

    def __init__(self, n_bands: int):
        """Allocate accumulators for ``n_bands`` bands."""
        self.n = n_bands
        self.sum = np.zeros(n_bands, dtype=np.float64)
        self.sumsq = np.zeros(n_bands, dtype=np.float64)
        self.count = np.zeros(n_bands, dtype=np.float64)
        self.min = np.full(n_bands, np.inf, dtype=np.float64)
        self.max = np.full(n_bands, -np.inf, dtype=np.float64)

    def update(self, bands_pixels: np.ndarray, mask: np.ndarray | None = None) -> None:
        """Accumulate a (n_bands, n_pixels) array, optionally masking nodata."""
        for b in range(self.n):
            vals = bands_pixels[b]
            if mask is not None:
                vals = vals[mask[b]]
            if vals.size == 0:
                continue
            self.sum[b] += vals.sum()
            self.sumsq[b] += np.square(vals, dtype=np.float64).sum()
            self.count[b] += vals.size
            self.min[b] = min(self.min[b], float(vals.min()))
            self.max[b] = max(self.max[b], float(vals.max()))

    def merge(self, other: "BandAccumulator") -> None:
        """Fold another accumulator's totals into this one."""
        self.sum += other.sum
        self.sumsq += other.sumsq
        self.count += other.count
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)

    def finalize(self, band_names: list[str]) -> dict[str, dict[str, float]]:
        """Return per-band mean/std/min/max/count keyed by ``band_names``."""
        mean = self.sum / self.count
        var = np.clip(self.sumsq / self.count - mean**2, 0.0, None)
        std = np.sqrt(var)
        return {
            name: {
                "mean": float(mean[i]),
                "std": float(std[i]),
                "min": float(self.min[i]),
                "max": float(self.max[i]),
                "count": int(self.count[i]),
            }
            for i, name in enumerate(band_names)
        }


def _accumulate_city(
    city_dir: Path, n_tiles: int
) -> tuple[BandAccumulator, BandAccumulator]:
    """Accumulate S2 and S1 stats over every tile of one city."""
    s2_acc = BandAccumulator(len(EVAL_S2_L2A_BAND_NAMES))
    s1_acc = BandAccumulator(len(S1_STORED_BAND_NAMES))
    for idx in range(n_tiles):
        tile = torch.load(city_dir / f"{idx}.pt")
        s2 = (
            tile["s2"]
            .numpy()
            .reshape(len(EVAL_S2_L2A_BAND_NAMES), -1)
            .astype(np.float64)
        )
        s1 = (
            tile["s1"].numpy().reshape(len(S1_STORED_BAND_NAMES), -1).astype(np.float64)
        )
        s2_acc.update(s2)
        s1_acc.update(s1, mask=s1 > S1_FLOOR_DB)
    return s2_acc, s1_acc


def main() -> None:
    """Compute and write norm_stats.json."""
    parser = argparse.ArgumentParser(description="Compute 50Cities norm stats.")
    parser.add_argument(
        "--data_dir",
        default=str(paths.FIFTY_CITIES_DIR),
        help="Processed 50Cities dir (holds tiles/ and manifest.json).",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel city readers.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        counts = json.load(f)["counts"]
    logger.info(
        "Computing stats over %d cities, %d tiles", len(counts), sum(counts.values())
    )

    s2_total = BandAccumulator(len(EVAL_S2_L2A_BAND_NAMES))
    s1_total = BandAccumulator(len(S1_STORED_BAND_NAMES))

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_accumulate_city, data_dir / "tiles" / city, n): city
            for city, n in counts.items()
        }
        for fut in tqdm(futures, total=len(futures), desc="cities"):
            s2_acc, s1_acc = fut.result()
            s2_total.merge(s2_acc)
            s1_total.merge(s1_acc)

    stats = {
        Modality.SENTINEL2_L2A.name: s2_total.finalize(EVAL_S2_L2A_BAND_NAMES),
        Modality.SENTINEL1.name: s1_total.finalize(S1_STORED_BAND_NAMES),
    }

    out_path = data_dir / "norm_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote %s", out_path)

    for mod, band_stats in stats.items():
        print(f"\n{mod}:")
        for name, s in band_stats.items():
            print(
                f"  {name:28s} mean={s['mean']:10.3f} std={s['std']:10.3f} "
                f"min={s['min']:10.3f} max={s['max']:10.3f}"
            )


if __name__ == "__main__":
    main()
