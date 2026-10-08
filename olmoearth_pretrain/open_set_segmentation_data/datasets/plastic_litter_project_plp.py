"""Plastic Litter Project (PLP2021) -> open-set-segmentation polygon labels.

Source: "Plastic Litter Project 2021 dataset" (Marine Remote Sensing Group, University of
the Aegean; ESA Discovery). Zenodo record 7085112. The campaign deployed artificial
floating-plastic targets in the Gulf of Gera (Lesvos, Greece) and imaged them with
Sentinel-2 on 22 acquisition dates from June-October 2021, alongside UAS RGB/hyperspectral
data and georeferenced UAS orthophoto maps. The archive ships one ~5-10 GB zip per date
(Sentinel-2 L1C + ACOLITE product + orthophoto + UAS image); it does NOT ship a ready-made
vector label of the target footprints.

We only need the LABELS (pretraining supplies its own imagery), so we do NOT bulk-download
the ~170 GB of imagery. Instead we derive the target footprints once from a single
georeferenced UAS orthophoto (20210716_ortho.tif, extracted from 20210716.zip via HTTP
range-read + inflate) by segmenting the two bright HDPE-mesh targets against the dark
water, and cache them as raw/{slug}/plp_targets.geojson. The deployment mooring is fixed
across all 2021 acquisitions, so the same two footprints are reused for every date; the
per-date Sentinel-2 acquisition timestamp and the target surface state come from the .SAFE
folder name and ancillary_data_log.pdf (cached in raw/).

label_type is polygons: for each S2-observable acquisition date we rasterize the two target
footprints (class 1 = plastic target) into a 32x32 UTM 10 m tile with a water background
(class 0). Dates on which the targets were submerged / mostly submerged (not detectable by
Sentinel-2) are excluded. This is a specific-date label (the target is a transient surface
object at a specific S2 acquisition), so time_range is a ~1-hour window around the S2
acquisition time (spec 5).

Run:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.plastic_litter_project_plp
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import shapely.geometry
from rslearn.const import WGS84_PROJECTION

from .. import io, manifest
from ..rasterize import geom_to_pixels, rasterize_shapes

SLUG = "plastic_litter_project_plp"
NAME = "Plastic Litter Project (PLP)"
ZENODO_RECORD = "7085112"
TILE = 32  # 320 m tile: targets are offshore, coast is ~270 m south -> all-water background.
RESOLUTION = 10

# Class scheme. Manifest lists ["plastic target", "water"]; water is the natural
# background, so it is class id 0 and plastic target is id 1.
CLASSES = [
    ("water", "Open sea-surface water background (Gulf of Gera, Lesvos)."),
    (
        "plastic target",
        "Artificial floating plastic target (HDPE mesh / structured mesh raft) deployed "
        "by the Plastic Litter Project and designed to be detectable by Sentinel-2.",
    ),
]
PLASTIC_ID = 1
WATER_ID = 0

# Target surface states (ancillary_data_log.pdf) that are still visible at the sea surface
# and therefore detectable by Sentinel-2. Fully/mostly submerged dates are excluded.
OBSERVABLE_STATES = {"floating", "part sub", "mix floating", "mix part sub"}

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    f"dataset_summaries/{SLUG}.md"
)

# One orthophoto is enough to fix the (mooring-fixed) target footprints.
ORTHO_ZIP_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/20210716.zip/content"
)
ORTHO_INNER = "20210716/20210716_ortho/20210716_ortho.tif"


def _derive_targets_from_ortho(raw) -> None:
    """Derive the two target footprint polygons from the 20210716 UAS orthophoto.

    Only runs if raw/plp_targets.geojson is missing. Extracts the orthophoto from the
    Zenodo zip via a single HTTP range-read of the (deflate-compressed) member, inflates
    it, segments the two bright mesh targets against dark water, and writes their convex
    hulls (WGS84) to raw/plp_targets.geojson. This is a one-time ~515 MB pull; the imagery
    itself is not retained.
    """
    import struct
    import zipfile
    import zlib

    import rasterio
    from rasterio.transform import xy
    from scipy import ndimage

    from ..download import HttpRangeFile

    rf = HttpRangeFile(ORTHO_ZIP_URL)
    zf = zipfile.ZipFile(rf)
    info = next(i for i in zf.infolist() if i.filename == ORTHO_INNER)
    hdr = rf._range(info.header_offset, info.header_offset + 29)
    n = struct.unpack("<H", hdr[26:28])[0]
    e = struct.unpack("<H", hdr[28:30])[0]
    data_start = info.header_offset + 30 + n + e
    comp = rf._range(data_start, data_start + info.compress_size - 1)
    raw_bytes = zlib.decompress(comp, -15)
    tmp_tif = raw / "20210716_ortho.tif"
    with tmp_tif.open("wb") as f:
        f.write(raw_bytes)

    with rasterio.open(tmp_tif.path) as ds:
        # ROI around the deployment (fraction of the scene) to avoid the coastline.
        x0, x1 = int(0.50 * ds.width), int(0.69 * ds.width)
        y0, y1 = int(0.23 * ds.height), int(0.50 * ds.height)
        from rasterio.windows import Window

        arr = ds.read(window=Window(x0, y0, x1 - x0, y1 - y0))
        transform = ds.transform
    R, G, B = arr[0].astype(np.int32), arr[1].astype(np.int32), arr[2].astype(np.int32)
    alpha = arr[3] if arr.shape[0] >= 4 else np.full(R.shape, 255)
    valid = alpha > 10
    bright = (R + G + B) / 3
    mask = valid & (R > B + 8) & (bright > 95)
    mask = ndimage.binary_closing(mask, iterations=3)
    lab, ncomp = ndimage.label(mask)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, ncomp + 1))
    big = [i + 1 for i, s in enumerate(sizes) if s > 20000]
    feats = []
    for k, cid in enumerate(big):
        comp = ndimage.binary_fill_holes(lab == cid)
        ys, xs = np.where(comp)
        hull = shapely.geometry.MultiPoint(
            [(int(px), int(py)) for px, py in zip(xs, ys)]
        ).convex_hull
        coords = [
            list(xy(transform, y0 + int(py), x0 + int(px)))
            for px, py in hull.exterior.coords
        ]
        feats.append(
            {
                "type": "Feature",
                "properties": {"class": "plastic target", "target_index": k},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    with (raw / "plp_targets.geojson").open("w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    # Do not retain the 515 MB orthophoto.
    tmp_tif.unlink()


def _load_targets(raw) -> list[shapely.geometry.Polygon]:
    with (raw / "plp_targets.geojson").open() as f:
        fc = json.load(f)
    return [shapely.geometry.shape(feat["geometry"]) for feat in fc["features"]]


def _one_hour_window(s2_ts: str) -> tuple[datetime, datetime]:
    """~1-hour window centered on the Sentinel-2 acquisition instant (YYYYMMDDThhmmss)."""
    t = datetime.strptime(s2_ts, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    return (t - timedelta(minutes=30), t + timedelta(minutes=30))


def main() -> None:
    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "plp_targets.geojson").exists():
        print("deriving target footprints from orthophoto (one-time ~515 MB pull) ...")
        _derive_targets_from_ortho(raw)

    with (raw / "acquisition_dates.json").open() as f:
        acq: dict[str, dict[str, str]] = json.load(f)

    targets = _load_targets(raw)
    print(f"loaded {len(targets)} target footprint polygon(s)")

    # Fixed UTM tile geometry (mooring fixed across all dates). Center on the union
    # centroid of the target footprints.
    union = shapely.geometry.MultiPolygon(targets) if len(targets) > 1 else targets[0]
    clon, clat = union.centroid.x, union.centroid.y
    proj = io.utm_projection_for_lonlat(clon, clat)

    # Reproject target polygons into the projection's pixel space.
    pix_targets = [geom_to_pixels(t, WGS84_PROJECTION, proj) for t in targets]
    union_pix = (
        shapely.geometry.MultiPolygon(pix_targets)
        if len(pix_targets) > 1
        else pix_targets[0]
    )
    col = int(round(union_pix.centroid.x))
    row = int(round(union_pix.centroid.y))
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Rasterize the target footprints (class 1) over a water (class 0) background.
    # all_touched so the few-pixel targets register.
    shapes = [(pt, PLASTIC_ID) for pt in pix_targets]
    label = rasterize_shapes(
        shapes, bounds, fill=WATER_ID, dtype="uint8", all_touched=True
    )[0]
    n_target_px = int((label == PLASTIC_ID).sum())
    print(
        f"tile {TILE}x{TILE} at {proj.crs}, bounds={bounds}, "
        f"plastic-target pixels={n_target_px}, water pixels={int((label == WATER_ID).sum())}"
    )
    if n_target_px == 0:
        raise RuntimeError("no plastic-target pixels rasterized; check footprints")

    observable = [d for d, m in sorted(acq.items()) if m["state"] in OBSERVABLE_STATES]
    excluded = [
        d for d, m in sorted(acq.items()) if m["state"] not in OBSERVABLE_STATES
    ]
    print(f"observable dates: {len(observable)}  excluded (submerged): {len(excluded)}")

    n = 0
    for date in observable:
        sample_id = f"{n:06d}"
        s2 = acq[date]["s2"]
        tr = _one_hour_window(s2)
        io.write_label_geotiff(
            SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            tr,
            source_id=f"PLP2021/{date}/S2_{s2}/{acq[date]['state']}",
            classes_present=[WATER_ID, PLASTIC_ID],
        )
        n += 1
    print(f"wrote {n} label patches")

    metadata: dict[str, Any] = {
        "dataset": SLUG,
        "name": NAME,
        "task_type": "classification",
        "source": "Zenodo",
        "license": "open (Zenodo) / CC-BY",
        "provenance": {
            "url": "https://zenodo.org/records/7085112",
            "have_locally": False,
            "annotation_method": (
                "controlled field-deployed plastic targets; footprints derived by "
                "segmenting the georeferenced UAS orthophoto"
            ),
        },
        "sensors_relevant": ["sentinel2"],
        "classes": [
            {"id": i, "name": name, "description": desc}
            for i, (name, desc) in enumerate(CLASSES)
        ],
        "nodata_value": io.CLASS_NODATA,
        "num_samples": n,
        "notes": (
            "Two artificial floating-plastic target footprints (oval ~24x32 m, square "
            "~33x34 m) rasterized into a 32x32 UTM (EPSG:32635) 10 m tile: class 1 = "
            "plastic target, class 0 = water background. Footprints derived once from the "
            "20210716 UAS orthophoto and reused for all dates (fixed mooring). One tile per "
            "S2-observable acquisition date (16 of 22 dates; 6 submerged/mostly-submerged "
            "dates excluded as not S2-detectable). time_range = ~1-hour window around the "
            "Sentinel-2 acquisition instant (specific-date surface label). Single "
            "deployment location, so all tiles share the same geometry; imagery not "
            "retained (labels only)."
        ),
    }
    io.write_dataset_metadata(SLUG, metadata)

    _write_summary(n, observable, excluded, acq, n_target_px)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n
    )
    print(f"STATUS: completed classification num_samples={n}")


def _write_summary(
    n: int,
    observable: list[str],
    excluded: list[str],
    acq: dict[str, dict[str, str]],
    n_target_px: int,
) -> None:
    lines = [
        f"# Plastic Litter Project (PLP) — {SLUG}",
        "",
        "- **Status**: completed",
        "- **Task type**: classification (polygons rasterized to a mask)",
        f"- **Samples**: {n} label patches (32x32, UTM EPSG:32635, 10 m, uint8, nodata=255)",
        "- **Source**: Zenodo record 7085112 — *Plastic Litter Project 2021 dataset* "
        "(Marine Remote Sensing Group, University of the Aegean; ESA Discovery). Open access.",
        "- **URL**: https://zenodo.org/records/7085112",
        "- **Access**: public Zenodo download, no credentials. Only the georeferenced "
        "orthophoto label signal is used — imagery is not retained.",
        "",
        "## What PLP2021 is",
        "",
        "A controlled field campaign that deployed artificial floating-plastic targets in "
        "the Gulf of Gera (Lesvos, Greece) and imaged them with Sentinel-2 on 22 dates "
        "(Jun-Oct 2021), together with UAS RGB/hyperspectral data and georeferenced UAS "
        "orthophoto maps. The targets are large HDPE-mesh rafts designed to be detectable "
        "at Sentinel-2's 10 m resolution. The archive ships one ~5-10 GB zip per date "
        "(S2 L1C + ACOLITE product + orthophoto + UAS image) and **no** ready-made vector "
        "label of the target footprints.",
        "",
        "## Why we did not bulk-download",
        "",
        "The full archive is ~170 GB and only the *labels* are needed (pretraining supplies "
        "its own imagery). We therefore extracted a single georeferenced UAS orthophoto "
        "(`20210716_ortho.tif`, 525 MB) from `20210716.zip` via an HTTP range-read of the "
        "deflate-compressed member + inflate (no full-zip download), segmented the two "
        "bright mesh targets against the dark water, and cached their convex-hull footprints "
        "to `raw/plp_targets.geojson`. The orthophoto is not retained.",
        "",
        "## Labels & processing",
        "",
        "- **Two target footprints** derived from the 20210716 orthophoto: an oval mesh "
        "target (~24 x 32 m) and a square structured-mesh target (~33 x 34 m), both "
        "offshore in open water ~270 m north of the coastline. The deployment mooring is "
        "fixed across all 2021 acquisitions, so the same footprints are reused for every "
        "date.",
        "- **Rasterization**: footprints rasterized (`all_touched=True`) into a 32x32 UTM "
        f"10 m tile centered on the targets — class 1 = plastic target ({n_target_px} "
        "pixels), class 0 = water background. 320 m tile keeps the whole context on water.",
        "- **Class scheme**: manifest classes `[plastic target, water]` remapped so water "
        "(the natural background) = id 0 and plastic target = id 1.",
        "- **Time range**: each sample is a specific Sentinel-2 acquisition (a transient "
        "surface object), so `time_range` is a ~1-hour window centered on the S2 "
        "acquisition instant parsed from the L1C `.SAFE` product name (well under 1 year).",
        "- **Observability filter**: target surface state per date comes from "
        "`ancillary_data_log.pdf`. Dates where the target was *submerged* or *mostly "
        f"submerged* (not detectable by Sentinel-2) were excluded ({len(excluded)} dates: "
        f"{', '.join(excluded)}). Kept {len(observable)} S2-observable dates "
        "(floating / part-sub / mix-floating / mix-part-sub).",
        "",
        "## Caveats",
        "",
        "- **Single deployment location**: all tiles share the same geometry and footprint; "
        "diversity is temporal only (one site, one target pair, 16 dates). This is a small, "
        "high-precision controlled positive-signal dataset for marine plastic.",
        "- Footprints were derived from one date's orthophoto and reused; per-date target "
        "configuration/exact position may vary slightly, but at 10 m the fixed footprint is "
        "a faithful approximation and the mooring is fixed.",
        "- Older PLP campaigns (2018/2019) are separate Zenodo records and are not included "
        "here (this record is PLP2021).",
        "",
        "## Verification",
        "",
        "Output tifs are single-band uint8, UTM EPSG:32635 at 10 m, 32x32, values in "
        "{0 water, 1 plastic target} (nodata 255 declared but unused); each tif has a "
        "matching JSON with a ~1-hour `time_range` around the S2 acquisition instant.",
        "",
        "## Reproduce",
        "",
        "```",
        f"python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.{SLUG}",
        "```",
        "(Re-derives `raw/plp_targets.geojson` from the orthophoto only if it is missing.)",
        "",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
