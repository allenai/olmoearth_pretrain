"""Measure what a Tessera v2 export would actually cost to fetch.

Phase 0 recon, part B: hits the Planetary Computer STAC API, no dataset writes.

Tessera v2 embeds one calendar year of *every* acquisition per pixel, so its
fetch is dominated by the number of scenes intersecting each window-year --
roughly 20x the OlmoEarth monthly-S1 export over the same windows. That
multiple is the whole scoping decision (all 160k windows vs a subsample), and
it is worth measuring rather than guessing.

This counts, for a sample of windows, how many Sentinel-2 L2A and Sentinel-1
RTC (ascending / descending) items intersect the window's target year, then
extrapolates to the dataset's full window count. The STAC filters and the
``max_matches`` caps are read out of the ``*_all`` layers of
``data/rslearn_dataset_configs/config_pastis_rslearn.json`` -- the same layers
the real fetch would use -- so the probe cannot drift from the fetch config.

Two things to read off the output:

* the projected scene-read and file counts, which decide whether v2 runs on
  the full corpus or a stratified subsample;
* ``>CAP`` flags, meaning that window-year has more acquisitions than the
  layer's ``max_matches`` allows, so v2 would silently see a truncated time
  series there. Raise the cap in the fetch config or accept it knowingly.

Zero ascending *or* descending S1 over a window is normal (orbit geometry);
``build_dpixel_inputs`` handles a missing direction. Zero *S2* is not -- that
would fail v2 inference for that window.

Requires ``pystac_client`` and ``planetary_computer``.

Example:
    python scripts/extract_tesserav2/fetch_cost_probe.py \
        --datasets ethiopia_crops,descals --sample 15
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import STGeometry
from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.registry import get_dataset_entry
from olmoearth_pretrain.internal.all_evals import AEF_SUPPLEMENTAL_DATASETS

logger = logging.getLogger(__name__)

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Repo-checkout location of the config carrying the *_all fetch layers,
# resolved relative to this file so the script works from any directory.
DEFAULT_FETCH_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "data/rslearn_dataset_configs/config_pastis_rslearn.json"
)

# The layers a Tessera v2 fetch materializes, in report order.
ALL_SCENE_LAYERS = (
    "sentinel2_l2a_all",
    "sentinel1_ascending_all",
    "sentinel1_descending_all",
)

# rslearn data source class -> Planetary Computer collection. Mirrors the
# COLLECTION_NAME on each class; kept here so the probe does not need the
# planetary_computer import chain that those classes pull in.
CLASS_TO_COLLECTION = {
    "Sentinel1": "sentinel-1-rtc",
    "Sentinel2": "sentinel-2-l2a",
}


class LayerProbe:
    """One ``*_all`` layer's STAC query, cap, and file-count multiplier."""

    def __init__(self, name: str, layer_config: dict[str, Any]) -> None:
        """Build a probe from an rslearn raster layer config.

        Args:
            name: the layer name (e.g. "sentinel1_ascending_all").
            layer_config: the parsed ``layers[<name>]`` object.

        Raises:
            ValueError: if the layer's data source class is not one this probe
                knows how to map to a STAC collection.
        """
        self.name = name
        data_source = layer_config["data_source"]
        class_name = data_source["class_path"].rsplit(".", 1)[-1]
        if class_name not in CLASS_TO_COLLECTION:
            raise ValueError(
                f"layer '{name}' uses unmapped data source '{class_name}'; "
                f"add it to CLASS_TO_COLLECTION"
            )
        self.collection = CLASS_TO_COLLECTION[class_name]
        self.query = data_source.get("init_args", {}).get("query") or None
        self.max_matches = data_source.get("query_config", {}).get("max_matches")
        # Each band set materializes its own GeoTIFF per item group, so this is
        # the file multiplier per matched scene.
        self.files_per_scene = len(layer_config.get("band_sets", []))


def load_layer_probes(config_path: Path) -> list[LayerProbe]:
    """Read the ``*_all`` fetch layers out of a dataset config.

    Args:
        config_path: path to the rslearn dataset config JSON.

    Returns:
        one LayerProbe per layer in ALL_SCENE_LAYERS.

    Raises:
        ValueError: if the config is missing one of those layers.
    """
    config = json.loads(config_path.read_text())
    layers = config.get("layers", {})
    probes = []
    for name in ALL_SCENE_LAYERS:
        if name not in layers:
            raise ValueError(f"{config_path} has no '{name}' layer")
        probes.append(LayerProbe(name, layers[name]))
    return probes


def target_year(window: Window, year_from: str) -> int | None:
    """Return the calendar year a Tessera v2 product would be fetched for.

    Args:
        window: the window to derive a year for.
        year_from: "midpoint" (matches the embedding materializer's convention)
            or "start".

    Returns:
        the year, or None if the window carries no time range.
    """
    if window.time_range is None:
        return None
    start, end = window.time_range
    if year_from == "start":
        return start.year
    return (start + (end - start) / 2).year


def window_bbox(window: Window) -> tuple[float, float, float, float]:
    """Return the window's WGS84 (min_lon, min_lat, max_lon, max_lat) bounds.

    Args:
        window: the window to reproject.

    Returns:
        the lon/lat bounding box.
    """
    geometry = STGeometry(window.projection, shapely.box(*window.bounds), None)
    return geometry.to_projection(WGS84_PROJECTION).shp.bounds


def format_means(means: dict[str, float]) -> str:
    """Format the per-layer mean scene counts for printing.

    Args:
        means: layer name -> mean scenes per window-year.

    Returns:
        a compact one-line rendering.
    """
    return "  ".join(f"{name}={value:.1f}" for name, value in means.items())


def probe_dataset(
    name: str,
    catalog: Any,
    probes: list[LayerProbe],
    sample: int,
    seed: int,
    year_from: str,
) -> dict[str, Any]:
    """Count intersecting scenes for a sample of one dataset's windows.

    Args:
        name: registry dataset name.
        catalog: an open pystac_client Client.
        probes: the ``*_all`` layer probes to count against.
        sample: number of windows to probe.
        seed: RNG seed, so re-runs probe the same windows.
        year_from: how to derive each window's target year.

    Returns:
        a JSON-serializable summary including per-window counts and the
        full-corpus projection.

    Raises:
        ValueError: if the registry entry has no weka_path.
    """
    entry = get_dataset_entry(name)
    if not entry.weka_path:
        raise ValueError(f"Registry entry '{name}' has no weka_path.")
    total_windows = sum(split["count"] for split in entry.split_stats.values())

    print(f"\n{'=' * 78}\n{name}  ({total_windows} windows)\n{'=' * 78}")

    windows = Dataset(UPath(entry.weka_path)).load_windows()
    chosen = (
        list(windows)
        if len(windows) <= sample
        else random.Random(seed).sample(windows, sample)
    )

    header = "  ".join(f"{p.name.replace('sentinel', 's'):>22s}" for p in probes)
    print(f"{'window':28s} {'year':>5s}  {header}")

    rows = []
    sums = {probe.name: 0 for probe in probes}
    probed = 0
    for window in chosen:
        year = target_year(window, year_from)
        if year is None:
            logger.warning(f"{window.name}: no time range, skipping")
            continue
        bbox = window_bbox(window)
        counts = {}
        flags = []
        for probe in probes:
            matched = catalog.search(
                collections=[probe.collection],
                bbox=bbox,
                datetime=f"{year}-01-01/{year}-12-31",
                query=probe.query,
            ).matched()
            counts[probe.name] = matched
            sums[probe.name] += matched
            if probe.max_matches is not None and matched > probe.max_matches:
                flags.append(f"{probe.name}>CAP({probe.max_matches})")
        probed += 1
        cells = "  ".join(f"{counts[p.name]:>22d}" for p in probes)
        print(f"{window.name[:28]:28s} {year:>5d}  {cells}  {' '.join(flags)}")
        rows.append({"window": window.name, "year": year, "counts": counts})

    if probed == 0:
        print("  no probeable windows")
        return {"dataset": name, "probed": 0}

    means = {name_: total / probed for name_, total in sums.items()}
    reads_per_window = sum(means.values())
    files_per_window = sum(
        means[probe.name] * probe.files_per_scene for probe in probes
    )
    print(f"\n  mean scenes/window-year: {format_means(means)}")
    print(f"  -> {reads_per_window:.0f} scene reads and ~{files_per_window:.0f} files")
    print(
        f"  -> full corpus ({total_windows} windows): "
        f"{reads_per_window * total_windows / 1e6:.1f}M reads, "
        f"{files_per_window * total_windows / 1e6:.1f}M files"
    )

    return {
        "dataset": name,
        "probed": probed,
        "total_windows": total_windows,
        "mean_scenes_per_window": means,
        "reads_per_window": reads_per_window,
        "files_per_window": files_per_window,
        "projected_reads": reads_per_window * total_windows,
        "projected_files": files_per_window * total_windows,
        "rows": rows,
    }


def main() -> None:
    """Probe the requested datasets and print the full-corpus projection."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(AEF_SUPPLEMENTAL_DATASETS),
        help="Comma-separated registry dataset names to probe.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Windows to probe per dataset (one STAC search per layer each).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed, so re-runs match."
    )
    parser.add_argument(
        "--year_from",
        choices=["midpoint", "start"],
        default="midpoint",
        help=(
            "How to derive each window's target year. 'midpoint' matches the "
            "embedding materializer's convention."
        ),
    )
    parser.add_argument(
        "--fetch_config",
        type=str,
        default=str(DEFAULT_FETCH_CONFIG),
        help="Config carrying the *_all layers whose queries/caps are probed.",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to write the machine-readable summary.",
    )
    args = parser.parse_args()

    try:
        import planetary_computer
        import pystac_client
    except ImportError as e:
        raise SystemExit(
            "fetch_cost_probe needs pystac_client and planetary_computer: "
            "pip install pystac-client planetary-computer"
        ) from e

    probes = load_layer_probes(Path(args.fetch_config))
    logger.info(
        "probing layers: "
        + ", ".join(
            f"{p.name} -> {p.collection} (cap {p.max_matches}, "
            f"{p.files_per_scene} file(s)/scene)"
            for p in probes
        )
    )
    catalog = pystac_client.Client.open(
        PC_STAC_URL, modifier=planetary_computer.sign_inplace
    )

    summaries = []
    for name in args.datasets.split(","):
        summaries.append(
            probe_dataset(
                name.strip(), catalog, probes, args.sample, args.seed, args.year_from
            )
        )

    measured = [s for s in summaries if s.get("probed")]
    if measured:
        print(f"\n{'=' * 78}\nPROJECTION\n{'=' * 78}")
        for summary in measured:
            print(
                f"{summary['dataset']:22s} "
                f"{summary['reads_per_window']:6.0f} reads/window  "
                f"{summary['projected_reads'] / 1e6:7.1f}M reads  "
                f"{summary['projected_files'] / 1e6:7.1f}M files"
            )
        print(
            f"{'TOTAL':22s} {'':6s}          "
            f"{sum(s['projected_reads'] for s in measured) / 1e6:7.1f}M reads  "
            f"{sum(s['projected_files'] for s in measured) / 1e6:7.1f}M files"
        )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summaries, f, indent=2)
        logger.info(f"Wrote summary to {args.json_out}")


if __name__ == "__main__":
    main()
