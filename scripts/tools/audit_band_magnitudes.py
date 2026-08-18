"""Post-normalization per-band magnitude audit for the AEF year-aligned evals.

Motivated by the 2026-08-18 eurosat finding: eurosat is L1C imagery fed through
L2A-derived normalization, and its B12 band comes out with magnitudes large
enough to dominate the embedding -- zeroing the band moves KNN accuracy from an
unstable ~0.92 to a stable ~0.95. That bug was found by accident; nothing has
ever screened the release-blocking datasets for the same class of defect. This
script is that screen.

For each dataset it builds the SAME dataset object the embedding evals consume
(``from_registry_entry``: same registry entry, same model.yaml, same
normalization path, ws16 center-pixel windows) and streams a deterministic
subsample of windows per split, accumulating per-(dataset, modality, band)
histograms of the post-normalization |value| the model would actually see.

Reported per band: p50 / p99 / p999 / max of |x|, plus the fraction of exactly-
zero values (absent-band zeroing and ragged-slot padding land at exactly 0.0,
so a large frac_zero is coverage information, not a magnitude problem) and the
fractions beyond |4| and |8| (values a z-scored band should essentially never
reach; eurosat's B12 is the calibration point for "band that breaks a task").

CPU-only by design: no model, no GPU -- runs as a 0-GPU Beaker job with weka
mounted. Writes a CSV plus a flagged-rows summary to --out_dir.
"""

import argparse
import csv
import logging
import os

import numpy as np

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.rslearn_dataset import from_registry_entry
from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry
from olmoearth_pretrain.internal.all_evals import AEF_SUPPLEMENTAL_YEAR_ALIGNED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Log-spaced |x| histogram: resolves 1e-6 .. 1e6, which comfortably brackets
# both a healthy z-scored band (p999 ~ 3-4) and a eurosat-B12-class outlier.
BIN_EDGES = np.logspace(-6, 6, 241)
# |x| levels a z-scored band should essentially never reach.
FLAG_LEVELS = (4.0, 8.0)
# A band whose p999 exceeds this is flagged in the summary. eurosat's healthy
# bands sit ~3; its broken B12 is far beyond.
P999_FLAG = 8.0


class BandStats:
    """Streaming |x| histogram + exact-zero count for one (modality, band)."""

    def __init__(self) -> None:
        """Zeroed accumulators; one instance per (modality, band)."""
        self.hist = np.zeros(len(BIN_EDGES) + 1, dtype=np.int64)
        self.n_zero = 0
        self.n_total = 0
        self.abs_max = 0.0
        self.n_nonfinite = 0

    def update(self, values: np.ndarray) -> None:
        """Accumulate a flat float array of one band's post-norm values."""
        v = np.abs(values.ravel())
        finite = np.isfinite(v)
        self.n_nonfinite += int((~finite).sum())
        v = v[finite]
        self.n_total += v.size
        self.n_zero += int((v == 0).sum())
        if v.size:
            self.abs_max = max(self.abs_max, float(v.max()))
            # searchsorted rather than np.histogram so underflow/overflow land
            # in the first/last bucket instead of being dropped.
            idx = np.searchsorted(BIN_EDGES, v, side="right")
            self.hist += np.bincount(idx, minlength=len(BIN_EDGES) + 1)

    def quantile(self, q: float) -> float:
        """Approximate |x| quantile from the histogram (upper bin edge)."""
        if self.n_total == 0:
            return float("nan")
        target = q * self.n_total
        cum = np.cumsum(self.hist)
        idx = int(np.searchsorted(cum, target, side="left"))
        if idx == 0:
            return 0.0
        return float(BIN_EDGES[min(idx - 1, len(BIN_EDGES) - 1)])

    def frac_above(self, level: float) -> float:
        """Fraction of values with |x| > level."""
        if self.n_total == 0:
            return float("nan")
        idx = int(np.searchsorted(BIN_EDGES, level, side="right"))
        return float(self.hist[idx:].sum() / self.n_total)


def audit_dataset(
    name: str, splits: list[str], cap_per_split: int
) -> dict[tuple[str, str], BandStats]:
    """Stream one dataset's splits into per-(modality, band) stats."""
    stats: dict[tuple[str, str], BandStats] = {}
    entry = get_dataset_entry(name)
    # Imagery only: the entries also carry precomputed embedding products
    # (gse / tessera*), which are exempt from imagery normalization and are the
    # largest arrays in each window -- loading them costs most of the wall
    # clock while auditing nothing.
    imagery = [
        m
        for m in entry.modalities
        if m.lower() in ("sentinel2_l2a", "sentinel2", "sentinel1", "landsat")
    ]
    for split in splits:
        ds = from_registry_entry(
            entry,
            split=split,
            input_modalities_override=imagery,
            window_size=16,
            label_at_center_pixel=True,
        )
        n = len(ds)
        stride = max(1, n // cap_per_split)
        indices = list(range(0, n, stride))[:cap_per_split]
        logger.info("%s/%s: %d windows, auditing %d", name, split, n, len(indices))
        for i in indices:
            masked_sample, _ = ds[i]
            for modality, tensor in masked_sample.as_dict().items():
                if modality == "timestamps" or modality.endswith("_mask"):
                    continue
                if tensor is None:
                    continue
                spec = Modality.get(modality)
                bands = spec.band_order
                arr = np.asarray(tensor, dtype=np.float32)
                if arr.shape[-1] != len(bands):
                    logger.warning(
                        "%s/%s %s: last dim %d != %d bands, skipping",
                        name,
                        split,
                        modality,
                        arr.shape[-1],
                        len(bands),
                    )
                    continue
                for b, band in enumerate(bands):
                    stats.setdefault((modality, band), BandStats()).update(arr[..., b])
    return stats


def main() -> None:
    """Run the audit and write the CSV + flag summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(AEF_SUPPLEMENTAL_YEAR_ALIGNED),
        help="Comma-separated registry dataset names.",
    )
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--cap_per_split", type=int, default=200)
    parser.add_argument("--out_dir", default="/outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "band_magnitude_audit.csv")
    rows = []
    flags = []
    for name in args.datasets.split(","):
        try:
            stats = audit_dataset(name, args.splits.split(","), args.cap_per_split)
        except Exception:
            logger.exception("dataset %s failed, continuing", name)
            continue
        for (modality, band), s in sorted(stats.items()):
            row = {
                "dataset": name,
                "modality": modality,
                "band": band,
                "n": s.n_total,
                "n_nonfinite": s.n_nonfinite,
                "frac_zero": round(s.n_zero / s.n_total, 5) if s.n_total else None,
                "p50": round(s.quantile(0.50), 4),
                "p99": round(s.quantile(0.99), 4),
                "p999": round(s.quantile(0.999), 4),
                "abs_max": round(s.abs_max, 3),
                "frac_gt4": round(s.frac_above(4.0), 6),
                "frac_gt8": round(s.frac_above(8.0), 6),
            }
            rows.append(row)
            if (row["p999"] and row["p999"] > P999_FLAG) or s.n_nonfinite:
                flags.append(row)
            logger.info(
                "%s %s %s: p50=%.3f p99=%.3f p999=%.3f max=%.2f "
                "zero=%.3f >4=%.5f >8=%.6f nonfinite=%d",
                name,
                modality,
                band,
                row["p50"],
                row["p99"],
                row["p999"],
                s.abs_max,
                row["frac_zero"] or 0,
                row["frac_gt4"],
                row["frac_gt8"],
                s.n_nonfinite,
            )

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %d rows to %s", len(rows), out_csv)

    print(f"\n===== FLAGGED BANDS (p999 > {P999_FLAG:.1f} or non-finite values) =====")
    if not flags:
        print("none -- no eurosat-class magnitude outliers in the audited datasets")
    for row in flags:
        print(
            f"  {row['dataset']} {row['modality']} {row['band']}: "
            f"p999={row['p999']} max={row['abs_max']} "
            f">8={row['frac_gt8']} nonfinite={row['n_nonfinite']}"
        )


if __name__ == "__main__":
    main()
