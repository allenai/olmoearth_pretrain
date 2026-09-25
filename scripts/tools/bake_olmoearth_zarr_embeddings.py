r"""Bake OlmoEarth's published large-scale embeddings into a PASTIS eval dataset.

The rslearn_projects large-scale embedding pipeline publishes a geozarr store:
one group per UTM zone (``utm01`` .. ``utm60``) holding a
``(year, 128, y, x)`` int8 ``embeddings`` array on a 10 m grid, quantized with
AlphaEarth's power scheme and -128 as nodata
(rslp/large_scale_embeddings/README.md). Every PASTIS window already sits on
its patch's native 10 m UTM grid (pastis_processor.patch_grid_from_geometry),
so each window's embedding is a plain array slice of its zone's group: no
reprojection or resampling.

This builds ``pastis_year_aligned_oe13zarr``, a sibling of
``pastis_year_aligned`` rather than a new layer in it, because adding a layer
changes that dataset's config.json, whose hash every eval job on it verifies.
The sibling holds:

- the same windows (projection, bounds, time range, options -- so the
  eval_split tags) restricted to those carrying the ``gse`` layer, which
  pastis_year_aligned's model.yaml requires, so the window set matches
  exactly what AEF, Tessera and the forward-pass OlmoEarth runs were scored
  on;
- the ``label`` layer, copied file-for-file;
- the ``olmoearth_emb`` layer: the zarr slab for the window's time-range
  midpoint year, dequantized to float32 the way the README specifies
  (``sign(x) * |x|^2`` with ``x = v / 127.5``); nodata pixels are written as
  -1.0 in every band, the value the AEF layer uses (the dequantized range is
  (-0.993, 0.993), so -1.0 is unambiguous).

config.json is copied byte-for-byte from
data/rslearn_dataset_configs/pastis_year_aligned_oe13zarr/, so its registry
hash is known before the bake runs. A manifest in the embedding materializer's
shape (plus nodata counts) is written next to the dataset.

Run on a Weka-mounted machine (zarr is not a project dependency)::

    uv run --with 'zarr>=3' python scripts/tools/bake_olmoearth_zarr_embeddings.py \
        --zarr_url https://storage.googleapis.com/ai2-olmoearth-embeddings-us-central1/large_scale_embeddings/geozarr_global_v1.3/s2_s1_landsat_distilled_ps1_ws16_overlap4/embeddings.zarr
"""

import argparse
import json
import logging
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rslearn.dataset import Dataset, Window
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
    get_target_year,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_JSON = (
    REPO_ROOT / "data/rslearn_dataset_configs/pastis_year_aligned_oe13zarr/config.json"
)
DEFAULT_SRC = "/weka/dfive-default/olmoearth/eval_datasets/pastis_year_aligned"
DEFAULT_DST = "/weka/dfive-default/olmoearth/eval_datasets/pastis_year_aligned_oe13zarr"

MODALITY = Modality.OLMOEARTH_EMB
# The layer every model on pastis_year_aligned requires (see module docstring).
REQUIRED_SRC_LAYER = Modality.GSE.name
LABEL_LAYER = "label"
ZARR_NODATA = -128
NODATA_VALUE = -1.0
RESOLUTION = 10


def dequantize(v: np.ndarray) -> np.ndarray:
    """Invert the AlphaEarth power quantizer (rslp large_scale_embeddings README)."""
    x = v.astype(np.float32) / 127.5
    return np.sign(x) * np.abs(x) ** 2.0


class ZarrReader:
    """Reads a window's embedding slab out of the per-UTM-zone geozarr store."""

    def __init__(self, url: str) -> None:
        """Open the store and cache each zone's grid on first use."""
        import zarr

        self.root = zarr.open_group(url, mode="r")
        self._zones: dict[str, tuple[Any, list[int], float, float]] = {}
        self._lock = threading.Lock()

    def _zone(self, epsg: int) -> tuple[Any, list[int], float, float]:
        """(embeddings array, years, x origin, y origin) for a UTM EPSG code."""
        if not (32601 <= epsg <= 32660):
            raise ValueError(f"EPSG:{epsg} is not a northern UTM zone")
        name = f"utm{epsg - 32600:02d}"
        with self._lock:
            if name not in self._zones:
                group = self.root[name]
                a, b, x0, d, e, y0 = group.attrs["spatial:transform"]
                if (a, b, d, e) != (RESOLUTION, 0, 0, -RESOLUTION):
                    raise ValueError(f"{name}: unexpected transform {(a, b, d, e)}")
                years = [int(y) for y in group["time"][:]]
                self._zones[name] = (group["embeddings"], years, x0, y0)
        return self._zones[name]

    def read(self, window: Window, year: int) -> np.ndarray:
        """The (128, H, W) int8 slab covering a window's bounds."""
        proj = window.projection
        if (proj.x_resolution, proj.y_resolution) != (RESOLUTION, -RESOLUTION):
            raise ValueError(
                f"{window.name}: window grid is not 10 m "
                f"({proj.x_resolution}, {proj.y_resolution})"
            )
        embeddings, years, x0, y0 = self._zone(proj.crs.to_epsg())
        if year not in years:
            raise ValueError(f"{window.name}: year {year} not in store years {years}")
        # rslearn pixel (col, row) has geographic (col * 10, -row * 10).
        left, top, right, bottom = window.bounds
        col = left - round(x0 / RESOLUTION)
        row = top + round(y0 / RESOLUTION)
        return embeddings[
            years.index(year), :, row : row + bottom - top, col : col + right - left
        ]


def copy_window(src: Window, dst_dataset: Dataset) -> Window:
    """Recreate a window in the destination dataset and copy its label layer."""
    dst = Window(
        storage=dst_dataset.storage,
        group=src.group,
        name=src.name,
        projection=src.projection,
        bounds=src.bounds,
        time_range=src.time_range,
        options=src.options,
    )
    dst.save()
    src_dir = src.get_layer_dir(LABEL_LAYER)
    dst_dir = dst.get_layer_dir(LABEL_LAYER)
    if dst_dir.exists():
        shutil.rmtree(str(dst_dir))
    shutil.copytree(str(src_dir), str(dst_dir))
    dst.mark_layer_completed(LABEL_LAYER)
    return dst


def main() -> None:
    """Build the sibling dataset and write its manifest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zarr_url", required=True)
    parser.add_argument("--src_ds_path", default=DEFAULT_SRC)
    parser.add_argument("--dst_ds_path", default=DEFAULT_DST)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    dst_path = UPath(args.dst_ds_path)
    dst_path.mkdir(parents=True, exist_ok=True)
    with (dst_path / "config.json").open("wb") as f:
        f.write(CONFIG_JSON.read_bytes())
    dst_dataset = Dataset(dst_path)
    provider = RslearnWindowProvider(dst_path)

    src_windows = Dataset(UPath(args.src_ds_path)).storage.get_windows()
    windows = [
        w
        for w in src_windows
        if w.is_layer_completed(REQUIRED_SRC_LAYER)
        and w.is_layer_completed(LABEL_LAYER)
    ]
    kept = {(w.group, w.name) for w in windows}
    excluded = sorted(
        f"{w.group}/{w.name}" for w in src_windows if (w.group, w.name) not in kept
    )
    logger.info(
        f"{len(windows)}/{len(src_windows)} source windows carry "
        f"'{REQUIRED_SRC_LAYER}' and '{LABEL_LAYER}'"
    )

    reader = ZarrReader(args.zarr_url)
    results: dict[str, dict[str, Any]] = {}
    failed: list[str] = []

    def handle(src: Window) -> None:
        window_id = f"{src.group}/{src.name}"
        try:
            year = get_target_year(src)
            slab = reader.read(src, year)
            nodata = (slab == ZARR_NODATA).all(axis=0)
            array = dequantize(slab)
            array[:, nodata] = NODATA_VALUE
            dst = copy_window(src, dst_dataset)
            provider.write_embedding(dst, MODALITY, array, NODATA_VALUE)
            valid = array[:, ~nodata]
            results[window_id] = {
                "year": year,
                "split": src.options.get("eval_split", src.options.get("split")),
                "nodata_pixels": int(nodata.sum()),
                "mean_norm": float(np.linalg.norm(valid, axis=0).mean())
                if valid.size
                else None,
            }
        except Exception:
            logger.exception(f"Window {window_id} failed")
            failed.append(window_id)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for done, _ in enumerate(executor.map(handle, windows), start=1):
            if done % 100 == 0 or done == len(windows):
                logger.info(f"Baked {done}/{len(windows)} windows")

    nodata_windows = sorted(k for k, v in results.items() if v["nodata_pixels"])
    splits: dict[str, int] = {}
    for v in results.values():
        splits[str(v["split"])] = splits.get(str(v["split"]), 0) + 1
    norms = [v["mean_norm"] for v in results.values() if v["mean_norm"] is not None]
    manifest = {
        "product": MODALITY.name,
        "product_version": args.zarr_url,
        "modality": MODALITY.name,
        "year_policy": "window_time_range_midpoint",
        "source_dataset": args.src_ds_path,
        "num_windows_written": len(results),
        "num_windows_skipped_existing": 0,
        "num_coverage_gaps": 0,
        "num_windows_without_year": 0,
        "num_windows_failed": len(failed),
        "windows_failed": sorted(failed),
        "num_source_windows_excluded": len(excluded),
        "source_windows_excluded": excluded,
        "split_counts": splits,
        "years": sorted({v["year"] for v in results.values()}),
        "num_windows_with_nodata": len(nodata_windows),
        "total_nodata_pixels": sum(v["nodata_pixels"] for v in results.values()),
        "windows_with_nodata": nodata_windows,
        "mean_embedding_norm": float(np.mean(norms)) if norms else None,
        "cli_args": vars(args),
    }
    manifest_path = dst_path / f"embedding_materializer_manifest_{MODALITY.name}.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"Wrote {manifest_path}: "
        + json.dumps({k: v for k, v in manifest.items() if not isinstance(v, list)})
    )
    if failed:
        raise SystemExit(f"{len(failed)} windows failed; re-run to retry them")


if __name__ == "__main__":
    main()
