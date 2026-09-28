"""Process the 50Cities dataset into 64x64 tiles for OlmoEarth eval.

The raw dataset is one directory per city, each containing:

    S1.tif      (2 bands, float32, *linear* power -- VV, VH)
    S2.tif      (12 bands, uint16, L2A reflectance, B10/cirrus dropped)
    label.png   (RGBA, land-cover classes encoded as an RGB colormap)

Only the cities that ship a ``label.png`` are usable for a supervised
segmentation eval; the rest are dropped.

``process()`` tiles every labeled city into non-overlapping 64x64 patches and
writes ONE combined file per tile under ``tiles/<City>/<idx>.pt`` holding a dict
with that tile's S2, S1 and label tensors together (plus its row/col). Per-tile
files keep ``__getitem__`` memory-light (the full set is ~18GB), and
``manifest.json`` records the per-city tile counts. This is the only expensive
step and only needs to run once.

* S2 is stored as raw uint16 reflectance (the pretrained normalizer expects raw
  L2A reflectance). NOTE: because the source S2.tif is *unsigned* (uint16), its
  per-band min is structurally 0 -- unlike float L2A products (e.g. PASTIS),
  which can hold small *negative* reflectances from atmospheric overcorrection
  of dark targets. This only affects the stored min/max (used by the
  ``min_max_int`` norm method); the default 2-std normalization uses mean/std.
* S1 is converted to dB (``10*log10``) so it matches the units the pretrained S1
  normalizer was computed on; NaN/non-positive nodata is floored.
* Labels: the RGB colormap is auto-discovered across all cities (anti-aliasing
  noise at class boundaries is snapped to the nearest real palette color), so
  class indices are consistent across cities. The two nodata/background colors
  (black, white) are mapped to ``SEGMENTATION_IGNORE_LABEL``; the remaining 13
  colors become contiguous semantic classes 0..12. The mapping is written to
  ``colormap.json``.

Train/valid/test splits are NOT written here -- they are defined in code in
``fifty_cities_dataset.py`` (random / by_city / by_continent) and computed
deterministically from ``manifest.json`` at load time.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import torch
from PIL import Image

from olmoearth_pretrain.evals.metrics import SEGMENTATION_IGNORE_LABEL

logger = logging.getLogger(__name__)

# Colors that are nodata / background rather than a semantic class. They are
# mapped to SEGMENTATION_IGNORE_LABEL; the remaining colors become the 13
# contiguous semantic classes.
IGNORE_COLORS = [(0, 0, 0), (255, 255, 255)]

# Order of the 12 S2 bands as they appear in S2.tif. Assumed to be standard L2A
# with B10 (cirrus) dropped; update here if the source ordering differs.
S2_TIF_BAND_NAMES = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
# S1.tif channel order is VH, VV (determined empirically: channel-0 backscatter
# is ~6 dB weaker than channel-1, and matches the pretrained vh/vv means). Tiles
# are stored in this raw tif order; FiftyCitiesDataset maps it to the model's
# band order at load time.
S1_TIF_BAND_NAMES = ["vh", "vv"]

TILE_SIZE = 64

# Floor (in linear power) applied before the dB conversion so that nodata zeros
# and NaNs don't blow up to -inf. 1e-6 linear -> -60 dB.
S1_LINEAR_FLOOR = 1e-6

# Colors rarer than this fraction of total labeled pixels are treated as
# anti-aliasing artifacts and snapped to the nearest "real" palette color
# rather than being assigned their own class.
PALETTE_MIN_FRACTION = 1e-4


def _pack_rgb(rgb: np.ndarray) -> np.ndarray:
    """Pack an (..., 3) uint8 RGB array into a single int32 per pixel."""
    rgb = rgb.astype(np.int32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def _unpack_rgb(packed: int) -> tuple[int, int, int]:
    """Inverse of :func:`_pack_rgb` for a single packed value."""
    return ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)


class FiftyCitiesProcessor:
    """Tile the 50Cities dataset into 64x64 patches and build splits."""

    def __init__(self, data_dir: str, output_dir: str):
        """Init the processor.

        Args:
            data_dir: Path to the raw 50Cities dataset (one dir per city).
            output_dir: Where to write tiles/, colormap.json and splits/.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.tiles_dir = self.output_dir / "tiles"
        for d in (self.output_dir, self.tiles_dir):
            d.mkdir(parents=True, exist_ok=True)

    def labeled_cities(self) -> list[str]:
        """Return the sorted list of cities that have a label.png."""
        cities = [
            p.name
            for p in self.data_dir.iterdir()
            if p.is_dir() and (p / "label.png").exists()
        ]
        return sorted(cities)

    # ------------------------------------------------------------------ #
    # Palette discovery
    # ------------------------------------------------------------------ #
    def _read_label_rgb(self, city: str) -> np.ndarray:
        """Read the RGB (dropping alpha) label image for a city as (H, W, 3)."""
        arr = np.array(Image.open(self.data_dir / city / "label.png"))
        return arr[..., :3]

    def discover_palette(self, cities: list[str]) -> np.ndarray:
        """Discover a global RGB palette across all labeled cities.

        Returns an (num_classes, 3) uint8 array, sorted deterministically by
        color, after dropping rare (anti-aliasing) colors.
        """
        counts: dict[int, int] = defaultdict(int)
        total = 0
        for city in cities:
            packed = _pack_rgb(self._read_label_rgb(city)).reshape(-1)
            vals, cnts = np.unique(packed, return_counts=True)
            total += packed.size
            for v, c in zip(vals.tolist(), cnts.tolist()):
                counts[v] += c

        threshold = total * PALETTE_MIN_FRACTION
        kept = sorted(v for v, c in counts.items() if c >= threshold)
        palette = np.array([_unpack_rgb(v) for v in kept], dtype=np.uint8)
        logger.info(
            "Discovered %d palette colors (from %d raw colors, threshold=%d px)",
            len(palette),
            len(counts),
            int(threshold),
        )
        return palette

    @staticmethod
    def _build_label_map(palette: np.ndarray) -> np.ndarray:
        """Map each palette row to its final class label.

        Nodata/background colors (see ``IGNORE_COLORS``) map to
        ``SEGMENTATION_IGNORE_LABEL``; the remaining colors are assigned
        contiguous semantic indices 0, 1, 2, ... in palette order.
        """
        ignore_set = {tuple(c) for c in IGNORE_COLORS}
        palette_to_label = np.empty(len(palette), dtype=np.int64)
        sem_idx = 0
        for i, rgb in enumerate(palette):
            if tuple(int(x) for x in rgb) in ignore_set:
                palette_to_label[i] = SEGMENTATION_IGNORE_LABEL
            else:
                palette_to_label[i] = sem_idx
                sem_idx += 1
        return palette_to_label

    def _save_colormap(
        self,
        palette: np.ndarray,
        palette_to_label: np.ndarray,
        per_class: np.ndarray,
    ) -> None:
        """Persist the semantic class -> RGB mapping (+ pixel counts) to JSON."""
        packed_palette = _pack_rgb(palette)
        classes = []
        ignored = []
        for i in range(len(palette)):
            entry = {
                "rgb": [int(x) for x in palette[i]],
                "packed": int(packed_palette[i]),
            }
            label = int(palette_to_label[i])
            if label == SEGMENTATION_IGNORE_LABEL:
                ignored.append(entry)
            else:
                entry["index"] = label
                entry["pixel_count"] = int(per_class[label])
                classes.append(entry)
        # classes are appended in increasing palette order, which is also
        # increasing semantic-index order, so no explicit sort is needed.
        payload = {
            "num_classes": len(classes),
            "ignore_label": SEGMENTATION_IGNORE_LABEL,
            "note": (
                "Auto-discovered palette. Semantic class index == palette order; "
                "nodata/background colors are mapped to ignore_label. Rename "
                "classes once semantics are known."
            ),
            "classes": classes,
            "ignored_colors": ignored,
        }
        with open(self.output_dir / "colormap.json", "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "Wrote colormap.json: %d semantic classes, %d ignored colors",
            len(classes),
            len(ignored),
        )

    @staticmethod
    def _rgb_to_class(
        rgb: np.ndarray, palette: np.ndarray, palette_to_label: np.ndarray
    ) -> np.ndarray:
        """Map an (H, W, 3) RGB image to (H, W) class labels.

        Every pixel is snapped to the nearest palette color (Euclidean in RGB),
        which both decodes the colormap and snaps anti-aliasing noise to a real
        class, then mapped to its final label via ``palette_to_label``. Done over
        unique colors only, so it's cheap.
        """
        packed = _pack_rgb(rgb)
        uniq, inverse = np.unique(packed.reshape(-1), return_inverse=True)
        uniq_rgb = np.array([_unpack_rgb(int(v)) for v in uniq], dtype=np.int32)
        # (num_unique, num_palette) squared distances
        dists = (
            (uniq_rgb[:, None, :] - palette[None, :, :].astype(np.int32)) ** 2
        ).sum(axis=2)
        uniq_to_label = palette_to_label[dists.argmin(axis=1)].astype(np.int16)
        return uniq_to_label[inverse].reshape(rgb.shape[:2])

    # ------------------------------------------------------------------ #
    # Tiling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_tif(path: Path) -> np.ndarray:
        """Read a GeoTIFF into a (bands, H, W) numpy array."""
        with rasterio.open(path) as src:
            return src.read()

    def _tile_city(
        self,
        city: str,
        palette: np.ndarray,
        palette_to_label: np.ndarray,
        num_classes: int,
        drop_empty: bool,
    ) -> tuple[int, np.ndarray]:
        """Tile a city into 64x64 patches, one combined file per tile.

        Returns the number of tiles written and the per-semantic-class histogram.
        """
        city_dir = self.tiles_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)

        s2 = self._read_tif(self.data_dir / city / "S2.tif")  # (12, H, W) uint16
        s1 = self._read_tif(self.data_dir / city / "S1.tif")  # (2, H, W) float32
        label_rgb = self._read_label_rgb(city)  # (H, W, 3)

        assert s2.shape[0] == len(S2_TIF_BAND_NAMES), (
            f"{city}: expected {len(S2_TIF_BAND_NAMES)} S2 bands, got {s2.shape[0]}"
        )
        assert s1.shape[0] == len(S1_TIF_BAND_NAMES), (
            f"{city}: expected {len(S1_TIF_BAND_NAMES)} S1 bands, got {s1.shape[0]}"
        )
        h, w = s2.shape[1:]
        assert s1.shape[1:] == (h, w), f"{city}: S1/S2 shape mismatch"
        assert label_rgb.shape[:2] == (h, w), f"{city}: label/S2 shape mismatch"

        # Linear power -> dB, flooring nodata (0 / NaN / negatives).
        s1 = np.nan_to_num(s1, nan=S1_LINEAR_FLOOR, posinf=None, neginf=S1_LINEAR_FLOOR)
        s1 = 10.0 * np.log10(np.clip(s1, S1_LINEAR_FLOOR, None))

        labels = self._rgb_to_class(label_rgb, palette, palette_to_label)  # (H, W)

        n_rows, n_cols = h // TILE_SIZE, w // TILE_SIZE
        per_class = np.zeros(num_classes, dtype=np.int64)
        count = 0
        for r in range(n_rows):
            for c in range(n_cols):
                ys, xs = r * TILE_SIZE, c * TILE_SIZE
                s2_t = s2[:, ys : ys + TILE_SIZE, xs : xs + TILE_SIZE]
                if drop_empty and not s2_t.any():
                    continue  # off-scene / black tile
                lbl_t = labels[ys : ys + TILE_SIZE, xs : xs + TILE_SIZE]
                semantic = lbl_t[lbl_t >= 0]  # exclude ignore (-1)
                per_class += np.bincount(semantic.reshape(-1), minlength=num_classes)
                tile = {
                    # int32 (not int16): raw uint16 reflectance can exceed 32767.
                    # Source is unsigned, so values are >= 0 (no negative
                    # reflectance, unlike float L2A products); see module docstring.
                    "s2": torch.from_numpy(s2_t.astype(np.int32)),  # (12, 64, 64)
                    "s1": torch.from_numpy(
                        s1[:, ys : ys + TILE_SIZE, xs : xs + TILE_SIZE].astype(
                            np.float32
                        )
                    ),  # (2, 64, 64) dB
                    "label": torch.from_numpy(lbl_t.astype(np.int16)),  # (64, 64)
                    "row": r,
                    "col": c,
                }
                torch.save(tile, city_dir / f"{count}.pt")
                count += 1

        return count, per_class

    def process(self, drop_empty: bool = True) -> None:
        """Run stage 1: discover palette and tile every labeled city."""
        cities = self.labeled_cities()
        logger.info("Found %d labeled cities: %s", len(cities), cities)

        palette = self.discover_palette(cities)
        palette_to_label = self._build_label_map(palette)
        num_classes = int((palette_to_label >= 0).sum())
        np.save(self.output_dir / "palette.npy", palette)

        counts: dict[str, int] = {}
        per_class = np.zeros(num_classes, dtype=np.int64)
        for city in cities:
            n, hist = self._tile_city(
                city, palette, palette_to_label, num_classes, drop_empty
            )
            counts[city] = n
            per_class += hist
            logger.info("%s: %d tiles", city, n)

        self._save_colormap(palette, palette_to_label, per_class)
        manifest = {
            "tile_size": TILE_SIZE,
            "drop_empty": drop_empty,
            "num_classes": num_classes,
            "counts": counts,
        }
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Wrote manifest.json: %d tiles total", sum(counts.values()))


def main() -> None:
    """CLI entry point.

    Stage 1 only: tile the cities and write tiles/, manifest.json and
    colormap.json. Train/valid/test splits are defined in code in
    ``fifty_cities_dataset.py`` (random / by_city / by_continent), so no split
    files are written here.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Process the 50Cities eval dataset.")
    parser.add_argument("--data_dir", required=True, help="Raw 50Cities directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--keep_empty",
        action="store_true",
        help="Keep all-zero (off-scene) tiles instead of dropping them.",
    )
    args = parser.parse_args()

    proc = FiftyCitiesProcessor(args.data_dir, args.output_dir)
    proc.process(drop_empty=not args.keep_empty)


if __name__ == "__main__":
    main()
