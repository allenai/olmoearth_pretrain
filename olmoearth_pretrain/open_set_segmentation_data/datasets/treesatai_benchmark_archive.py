"""Process the TreeSatAI Benchmark Archive into open-set-segmentation label patches.

Source: Zenodo record 6598390 (TreeSatAI Benchmark Archive for Deep Learning in Forest
Applications; Ahlswede et al. 2023, ESSD). Multi-sensor benchmark over Lower Saxony,
Germany, pairing 60 m aerial + Sentinel-1 + Sentinel-2 patches with tree genus labels
derived from the state forest inventory (field reference). We use the **Sentinel-2**
component (native UTM 10 m), preferring the 200 m patches (20x20 px) which give more
spatial context than the 60 m ones.

Georeferencing: every S2 patch is a real GeoTIFF carrying EPSG:326xx (UTM 32N) + a 10 m
geotransform over a Lower-Saxony forest-inventory stand, so labels place directly on the
S2 grid (no coordinate recovery needed).

Labels: ``labels/TreeSatBA_v9_60m_multi_labels.json`` maps each patch filename to a
multi-label list ``[[genus, area_fraction], ...]`` (15 genus classes incl. "Cleared").
Each 200 m patch is cut around a single inventoried stand, so where one genus dominates
the patch (>= DOMINANCE) we treat the patch as a coherent single-genus land-cover tile
(spec 4 "scene-level" -> uniform-class tile) and emit a uniform label patch filled with
that genus's class id, georeferenced to the S2 patch's exact CRS/geotransform.

Class scheme: the 15 TreeSatBA genera, ids 0-14 by descending dominant-patch frequency
(uint8, well under the 254 cap). Time range: forest-stand genus is persistent, so we
anchor a 1-year window on the inventory YEAR clamped into the Sentinel era (>= 2016).

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.treesatai_benchmark_archive
"""

import argparse
import json
import multiprocessing
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "treesatai_benchmark_archive"
NAME = "TreeSatAI Benchmark Archive"
ZENODO_RECORD = "6598390"
DOMINANCE = 0.7  # patch must be >= this fraction one genus to label it uniformly
PER_CLASS = 1000
RESOLUTION = io.RESOLUTION

# 15 genus classes, ids 0-14 by descending dominant-patch frequency (computed once from
# the multi-label file). Descriptions from the archive's BT_ENG common names.
CLASSES = [
    ("Pinus", "Pine (Scots pine, black pine, Weymouth/eastern white pine)."),
    ("Quercus", "Oak (pedunculate/English oak, sessile oak, red oak)."),
    ("Fagus", "European beech (Fagus sylvatica)."),
    ("Picea", "Spruce (Norway spruce, Picea abies)."),
    ("Cleared", "Cleared / harvested forest area with no standing tree species."),
    ("Larix", "Larch (European larch, Japanese larch)."),
    ("Pseudotsuga", "Douglas fir (Pseudotsuga menziesii)."),
    ("Acer", "Maple (sycamore maple, Acer pseudoplatanus)."),
    ("Fraxinus", "Ash (Fraxinus excelsior)."),
    ("Betula", "Birch (Betula spp.)."),
    ("Alnus", "Alder (Alnus spp.)."),
    ("Abies", "Silver fir (Abies alba)."),
    ("Populus", "Poplar (Populus spp.)."),
    ("Prunus", "Cherry (Prunus spp.)."),
    ("Tilia", "Linden / lime (Tilia spp.)."),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    "dataset_summaries/treesatai_benchmark_archive.md"
)


def _s2_dir() -> Path:
    return Path(io.raw_dir(SLUG).path) / "s2" / "200m"


def _labels() -> dict[str, list]:
    lp = Path(io.raw_dir(SLUG).path) / "labels" / "TreeSatBA_v9_60m_multi_labels.json"
    with open(lp) as f:
        return json.load(f)


def _years() -> dict[str, int]:
    """Map IMG_ID -> inventory YEAR from p.GeoJSON."""
    gp = Path(io.raw_dir(SLUG).path) / "geojson" / "p.GeoJSON"
    with open(gp) as f:
        gj = json.load(f)
    out: dict[str, int] = {}
    for feat in gj["features"]:
        p = feat["properties"]
        out[p["IMG_ID"]] = int(p["YEAR"])
    return out


def _write_one(sample_id: str, rec: dict) -> tuple[str, int]:
    """Emit a uniform single-genus label tile matching the S2 patch's georeferencing."""
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, rec["class_id"]

    with rasterio.open(rec["tif_path"]) as ds:
        h, w = ds.height, ds.width
        crs = ds.crs
        transform = ds.transform
    h = min(h, io.MAX_TILE)
    w = min(w, io.MAX_TILE)

    out = np.full((h, w), rec["class_id"], dtype=np.uint8)

    proj = Projection(CRS.from_string(crs.to_string()), RESOLUTION, -RESOLUTION)
    x_ul = transform.c  # world x of upper-left corner
    y_ul = transform.f  # world y of upper-left corner
    x_min = int(round(x_ul / RESOLUTION))
    y_min = int(round(-y_ul / RESOLUTION))
    bounds = (x_min, y_min, x_min + w, y_min + h)

    io.write_label_geotiff(SLUG, sample_id, out, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["img_id"],
        classes_present=[rec["class_id"]],
    )
    return sample_id, rec["class_id"]


def build_records() -> list[dict[str, Any]]:
    labels = _labels()
    years = _years()
    s2dir = _s2_dir()
    recs: list[dict[str, Any]] = []
    n_missing_tif = 0
    n_below = 0
    for fname, multi in labels.items():
        if not multi:
            continue
        genus, frac = max(multi, key=lambda x: x[1])
        if genus not in NAME_TO_ID:
            continue
        if frac < DOMINANCE:
            n_below += 1
            continue
        tif_path = s2dir / fname
        if not tif_path.exists():
            n_missing_tif += 1
            continue
        img_id = fname[:-4] if fname.endswith(".tif") else fname
        year = max(years.get(img_id, 2016), 2016)
        recs.append(
            {
                "img_id": img_id,
                "tif_path": str(tif_path),
                "genus": genus,
                "class_id": NAME_TO_ID[genus],
                "year": year,
            }
        )
    print(
        f"candidates: {len(recs)} patches (>= {DOMINANCE} dominant); "
        f"dropped {n_below} mixed, {n_missing_tif} missing-tif"
    )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    # Ensure S2 200 m patches are extracted (idempotent).
    if not _s2_dir().exists() or len(list(_s2_dir().glob("*.tif"))) < 50000:
        s2zip = raw / "s2.zip"
        if s2zip.exists():
            print("extracting s2/200m patches ...")
            with zipfile.ZipFile(s2zip.path) as z:
                members = [
                    m
                    for m in z.namelist()
                    if m.startswith("s2/200m/") and m.endswith(".tif")
                ]
                z.extractall(raw.path, members=members)

    recs = build_records()

    # Class-balanced selection, <=1000/class, subject to 25k total cap.
    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "genus", per_class=PER_CLASS)
    print(f"selected {len(selected)} patches (<= {PER_CLASS}/class, 25k cap)")

    tasks = [{"sample_id": f"{i:06d}", "rec": rec} for i, rec in enumerate(selected)]
    written = 0
    with multiprocessing.Pool(args.workers) as pool:
        for _sid, _cid in star_imap_unordered(pool, _write_one, tasks):
            written += 1
            if written % 2000 == 0:
                print(f"  wrote {written}/{len(tasks)}")
                io.check_disk()
    print(f"wrote {written} label patches")

    counts = Counter(r["genus"] for r in selected)
    metadata = {
        "dataset": SLUG,
        "name": NAME,
        "task_type": "classification",
        "source": "Zenodo record 6598390",
        "license": "CC-BY-4.0",
        "provenance": {
            "url": "https://doi.org/10.5281/zenodo.6598390",
            "have_locally": False,
            "annotation_method": "forest inventory (field reference), Lower Saxony (NLF/BI)",
        },
        "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
        "classes": [
            {"id": i, "name": name, "description": desc}
            for i, (name, desc) in enumerate(CLASSES)
        ],
        "nodata_value": io.CLASS_NODATA,
        "num_samples": written,
        "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
        "notes": (
            "Sentinel-2 200 m patches (20x20 px, native UTM 32N 10 m). Each patch is cut "
            f"around one inventoried forest stand; patches with >= {DOMINANCE} dominant "
            "genus are emitted as uniform single-genus land-cover tiles (scene-level). "
            "Class ids 0-14 by descending dominant-patch frequency. Time range: 1-year "
            "window anchored on the inventory YEAR clamped to >= 2016 (stand genus is "
            "persistent). All source train/test splits used."
        ),
    }
    io.write_dataset_metadata(SLUG, metadata)
    _write_summary(written, counts)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=written
    )
    print("done")


def _write_summary(n_samples: int, counts: Counter) -> None:
    lines = [
        f"# TreeSatAI Benchmark Archive — {n_samples} label patches (classification)",
        "",
        f"- **Slug**: `{SLUG}`",
        f"- **Source**: Zenodo record {ZENODO_RECORD} "
        "(https://doi.org/10.5281/zenodo.6598390); Ahlswede et al. 2023, ESSD.",
        "- **Region / annotation**: Lower Saxony, Germany; state forest inventory "
        "(field reference, NLF/BI).",
        "- **License**: CC-BY-4.0. Public, no credentials.",
        "- **Task**: classification (tree genus), 15 classes, uint8, nodata 255.",
        "",
        "## What it is",
        "Multi-sensor benchmark pairing 60 m aerial + Sentinel-1 + Sentinel-2 patches with "
        "tree genus labels from the German forest inventory. We use the **Sentinel-2 200 m** "
        "patches (20x20 px, native EPSG:326xx UTM 32N at 10 m) — real GeoTIFFs with embedded "
        "CRS + geotransform, so labels drop straight onto the S2 grid.",
        "",
        "## Labels & class scheme",
        "`labels/TreeSatBA_v9_60m_multi_labels.json` gives each patch a multi-label list "
        "`[[genus, area_fraction], ...]` over 15 genera (14 tree genera + `Cleared`). Each "
        "200 m patch is cut around a single inventoried stand; where one genus covers "
        f">= {DOMINANCE} of the patch we emit a **uniform single-genus tile** (spec 4 "
        "scene-level coherent land-cover patch). Class ids 0-14 assigned by descending "
        "dominant-patch frequency:",
        "",
        "| id | genus | selected patches |",
        "|----|-------|------------------|",
    ]
    for i, (name, _d) in enumerate(CLASSES):
        lines.append(f"| {i} | {name} | {counts.get(name, 0)} |")
    lines += [
        "",
        "## Georeferencing / tiles",
        "Reused each S2 patch's exact CRS + geotransform (origin snapped to the integer "
        "10 m pixel grid, sub-metre shift). Tiles are single-band uint8, 20x20, UTM 32N at "
        "10 m; every pixel = the dominant genus's class id.",
        "",
        "## Time range",
        "Forest-stand genus is persistent, so each sample gets a 1-year window anchored on "
        "the inventory `YEAR` clamped to the Sentinel era (>= 2016). Inventory years span "
        "2011-2020; pre-2016 stands are anchored at 2016. `change_time` is null (state, not "
        "change). Caveat: a few `Cleared` patches with pre-2016 inventory years may have "
        "regrown by the anchored window.",
        "",
        "## Sampling",
        f"Class-balanced, up to {PER_CLASS}/class subject to the 25k cap "
        f"(`balance_by_class`). Total written: {n_samples}. Rare genera (Tilia, Prunus, "
        "Populus, Abies) contribute all available patches.",
        "",
        "## Reproduce",
        "```",
        "python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.treesatai_benchmark_archive",
        "```",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
