"""CEMS Wildfire Dataset -> open-set-segmentation classification (burn severity).

Source: MatteoM95/CEMS-Wildfire-Dataset, hosted on HuggingFace as
``links-ads/wildfires-cems``. 500+ Copernicus EMS rapid-mapping wildfire activations
(Jun 2017 - Apr 2023, mostly Europe). Each sample directory
``EMSR{n}/AOI{a}/EMSR{n}_AOI{a}_{seq}/`` holds a post-fire Sentinel-2 L2A GeoTIFF plus
paired georeferenced label rasters:
  * ``*_DEL.tif`` - burned-area delineation, binary uint8 (0 unburned, 1 burned). Present
    for every sample.
  * ``*_GRA.tif`` - burn-severity grading, uint8 0-4 (0 no visible damage, 1 negligible-
    to-slight, 2 moderate, 3 high, 4 destroyed). Present for a subset of samples ("when
    available"). Its non-zero footprint exactly matches DEL, so GRA subsumes DEL.

All rasters are georeferenced in EPSG:4326 at ~10 m/pixel (0.0000905 deg lat, ~0.000146
deg lon at ~52N). We reproject each label to local UTM at 10 m (nearest resampling -
categorical) and tile into non-overlapping 64x64 UTM patches.

Unified class scheme (per spec sec.5 multi-target combine: delineation + severity in ONE
scheme). Where GRA exists we use the 0-4 severity grade directly; for DEL-only samples the
burned pixels get a dedicated "burned_ungraded" class (5), so the delineation product stays
represented. Unburned (0) is a genuine observed background class here (the CEMS product
delineates the whole AOI), so we keep it - this dataset has a real negative class within
each tile and is NOT positive-only.

Burn scars are change/event labels: change_time = the post-fire S2 acquisition date;
time_range = a 360-day window centred on it (burn scars persist for months and are
detectable in post-fire imagery across that window).

Sampling: tiles-per-class balanced (rarest severity first), <=1000 tiles/class, 25k cap.

Run:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cems_wildfire_dataset
"""

import glob
import json
import multiprocessing
import os
import subprocess
import tarfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from .. import io, manifest
from ..sampling import select_tiles_per_class

SLUG = "cems_wildfire_dataset"
NAME = "CEMS Wildfire Dataset"
HF_REPO = "links-ads/wildfires-cems"
TILE = 64
RES = 10
PER_CLASS = 1000
HALF_WINDOW_DAYS = 180  # +/-180 d = 360 d change window (<= 1 year)

# id -> (name, description). ids 0-4 are the CEMS grading grades; id 5 is the
# delineation-only burned class (samples lacking a severity grading product).
CLASSES = [
    (
        "no_visible_damage",
        "No visible fire damage / unburned land within the CEMS activation AOI (grading grade 0).",
    ),
    (
        "negligible_to_slight",
        "Negligible-to-slight fire damage (Copernicus EMS grading grade 1).",
    ),
    (
        "moderately_damaged",
        "Moderately / possibly damaged by fire (Copernicus EMS grading grade 2).",
    ),
    ("highly_damaged", "Highly damaged burned area (Copernicus EMS grading grade 3)."),
    (
        "destroyed",
        "Completely destroyed / total destruction burned area (Copernicus EMS grading grade 4).",
    ),
    (
        "burned_ungraded",
        "Burned area from the binary delineation product for activations that lack a severity "
        "grading product (burn extent known, severity ungraded).",
    ),
]

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    f"dataset_summaries/{SLUG}.md"
)

SPLIT_PARTS = {
    "train": [f"train.tar.{i:04d}.gz.part" for i in range(11)],
    "test": [f"test.tar.{i:04d}.gz.part" for i in range(4)],
    "val": ["val.tar.00.gz.part", "val.tar.01.gz.part", "val.tar.02.gz.part"],
}


# --------------------------------------------------------------------------- extraction
def _ensure_extracted() -> str:
    """Concatenate the split .gz.part files per split, untar, return the extracted root."""
    raw = io.raw_dir(SLUG)
    root = raw / "extracted"
    root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    for split, parts in SPLIT_PARTS.items():
        marker = root / f".{split}_done"
        if marker.exists():
            continue
        # Ensure parts present (idempotent; skips existing).
        for p in parts:
            dst = raw / "data" / split / p
            if not dst.exists():
                hf_hub_download(
                    HF_REPO,
                    f"data/{split}/{p}",
                    repo_type="dataset",
                    local_dir=raw.path,
                )
        tgz = raw / "data" / split / f"{split}.tar.gz"
        part_paths = [str(raw / "data" / split / p) for p in parts]
        with open(tgz.path, "wb") as out:
            subprocess.run(["cat", *part_paths], stdout=out, check=True)
        print(f"[{split}] untarring {tgz} ...", flush=True)
        with tarfile.open(tgz.path, "r:gz") as t:
            t.extractall(root.path)
        os.remove(tgz.path)  # concatenated archive no longer needed
        marker.touch()
    return root.path


# --------------------------------------------------------------------------- helpers
def _utm_crs(lon: float, lat: float) -> CRS:
    return get_utm_ups_projection(lon, lat, RES, -RES).crs


def _parse_date(sample_dir: str, base: str) -> datetime | None:
    """Post-fire S2 acquisition date from the *_S2L2A.json sidecar."""
    jp = os.path.join(sample_dir, base + "_S2L2A.json")
    try:
        with open(jp) as f:
            j = json.load(f)
    except FileNotFoundError:
        return None
    payload = j.get("payload", {})
    ad = payload.get("acquisition_date")
    if ad:
        try:
            return datetime.strptime(ad[0], "%Y/%m/%d_%H:%M:%S").replace(tzinfo=UTC)
        except (ValueError, IndexError):
            pass
    try:
        frm = payload["input"]["data"][0]["dataFilter"]["timeRange"]["from"]
        return datetime.fromisoformat(frm.replace("Z", "+00:00"))
    except (KeyError, ValueError, IndexError):
        return None


def _reproject_label(src_path: str):
    """Reproject a categorical label raster to local UTM 10 m (nearest). Returns
    (dst_array uint8, dst_crs, dst_transform).
    """
    with rasterio.open(src_path) as src:
        arr = src.read(1)
        lon = (src.bounds.left + src.bounds.right) / 2
        lat = (src.bounds.top + src.bounds.bottom) / 2
        dst_crs = _utm_crs(lon, lat)
        transform, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=RES
        )
        dst = np.full((h, w), io.CLASS_NODATA, dtype=np.uint8)
        reproject(
            source=arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            dst_nodata=io.CLASS_NODATA,
        )
    return dst, dst_crs, transform


def _label_for_sample(sample_dir: str):
    """Return (reprojected label array, dst_crs, transform, use_gra) for a sample, with
    ids already remapped to the unified scheme, or None if no usable label.
    """
    base = os.path.basename(sample_dir)
    gra = os.path.join(sample_dir, base + "_GRA.tif")
    dele = os.path.join(sample_dir, base + "_DEL.tif")
    use_gra = os.path.exists(gra)
    src = gra if use_gra else dele
    if not os.path.exists(src):
        return None
    dst, dst_crs, transform = _reproject_label(src)
    if use_gra:
        # GRA already 0-4 == class ids 0-4.
        valid = {0, 1, 2, 3, 4}
    else:
        # DEL binary: burned (1) -> class 5, unburned (0) stays 0. DEL also carries a
        # native 255 nodata; leave it as nodata.
        dst = np.where(dst == 1, np.uint8(5), dst)
        valid = {0, 5}
    # Defensive: force any value outside the scheme (DEL 255 nodata, reprojection-border
    # 255, and rare warp artefacts such as a stray 254) to CLASS_NODATA so it is treated
    # as ignore and its tiles are dropped by the scan's nodata check.
    bad = ~np.isin(dst, list(valid))
    if bad.any():
        dst[bad] = io.CLASS_NODATA
    return dst, dst_crs, transform, use_gra


# --------------------------------------------------------------------------- scan
def _scan_one(sample_dir: str):
    base = os.path.basename(sample_dir)
    res = _label_for_sample(sample_dir)
    if res is None:
        return []
    dst, _crs, _tr, _use_gra = res
    date = _parse_date(sample_dir, base)
    if date is None:
        return []
    h, w = dst.shape
    recs = []
    for r0 in range(0, h - TILE + 1, TILE):
        for c0 in range(0, w - TILE + 1, TILE):
            crop = dst[r0 : r0 + TILE, c0 : c0 + TILE]
            if (crop == io.CLASS_NODATA).any():
                continue  # skip tiles straddling the reprojection border
            present = sorted(int(x) for x in np.unique(crop))
            if not (set(present) - {0}):
                continue  # keep only tiles containing burned pixels
            recs.append(
                {
                    "sample_dir": sample_dir,
                    "r0": r0,
                    "c0": c0,
                    "classes_present": present,
                    "date": date.isoformat(),
                    "source_id": f"{base}_r{r0}_c{c0}",
                }
            )
    return recs


# --------------------------------------------------------------------------- write
def _write_sample_tiles(sample_dir, tiles):
    """Write all selected tiles for one sample (one reprojection per sample).

    ``tiles`` is a list of (sample_id, rec). Invoked via star_imap_unordered, which unpacks
    each task dict as kwargs, so params must match the task dict keys.
    """
    # Idempotency: skip if every tile already written.
    if all((io.locations_dir(SLUG) / f"{sid}.tif").exists() for sid, _ in tiles):
        return len(tiles), []
    res = _label_for_sample(sample_dir)
    dst, dst_crs, transform, _ = res
    proj = Projection(dst_crs, RES, -RES)
    ox, oy = transform.c, transform.f  # UTM coords of dst[0,0] upper-left
    out_info = []
    for sid, rec in tiles:
        tif = io.locations_dir(SLUG) / f"{sid}.tif"
        r0, c0 = rec["r0"], rec["c0"]
        crop = dst[r0 : r0 + TILE, c0 : c0 + TILE].astype(np.uint8)
        x_ul = ox + c0 * RES
        y_ul = oy - r0 * RES
        x_min = int(round(x_ul / RES))
        y_min = int(round(-y_ul / RES))
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        date = datetime.fromisoformat(rec["date"])
        tr = (
            date - timedelta(days=HALF_WINDOW_DAYS),
            date + timedelta(days=HALF_WINDOW_DAYS),
        )
        if not tif.exists():
            io.write_label_geotiff(
                SLUG, sid, crop, proj, bounds, nodata=io.CLASS_NODATA
            )
            io.write_sample_json(
                SLUG,
                sid,
                proj,
                bounds,
                tr,
                change_time=date,
                source_id=rec["source_id"],
                classes_present=rec["classes_present"],
            )
        out_info.append((sid, rec["classes_present"]))
    return len(tiles), out_info


# --------------------------------------------------------------------------- main
def main() -> None:
    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    root = _ensure_extracted()
    sample_dirs = sorted(
        d
        for d in glob.glob(os.path.join(root, "*", "EMSR*", "AOI*", "EMSR*_AOI*_*"))
        if os.path.isdir(d)
    )
    print(f"total sample dirs: {len(sample_dirs)}", flush=True)

    print("scanning (reproject -> tile) ...", flush=True)
    records: list[dict] = []
    with multiprocessing.Pool(64) as pool:
        for recs in pool.imap_unordered(_scan_one, sample_dirs, chunksize=4):
            records.extend(recs)
    print(f"candidate burned 64x64 tiles: {len(records)}", flush=True)

    cand_per_class: Counter = Counter()
    for r in records:
        for c in r["classes_present"]:
            cand_per_class[c] += 1
    print(
        "candidate tiles-per-class:", dict(sorted(cand_per_class.items())), flush=True
    )

    selected = select_tiles_per_class(
        records, "classes_present", per_class=PER_CLASS, total_cap=25000
    )
    selected.sort(key=lambda r: (r["sample_dir"], r["r0"], r["c0"]))
    print(f"selected tiles: {len(selected)}", flush=True)

    # Assign sample ids and group by sample dir for one reprojection per sample.
    by_sample: dict[str, list] = {}
    for i, rec in enumerate(selected):
        sid = f"{i:06d}"
        by_sample.setdefault(rec["sample_dir"], []).append((sid, rec))

    tasks = [{"sample_dir": sd, "tiles": tl} for sd, tl in by_sample.items()]
    print(f"writing tiles from {len(tasks)} samples ...", flush=True)
    written = 0
    sel_per_class: Counter = Counter()
    with multiprocessing.Pool(64) as pool:
        for n, info in star_imap_unordered(pool, _write_sample_tiles, tasks):
            written += n
            for _sid, classes in info:
                for c in classes:
                    sel_per_class[c] += 1
            if written % 1000 < n:
                io.check_disk()
    print(f"wrote {written} tiles", flush=True)

    metadata = {
        "dataset": SLUG,
        "name": NAME,
        "task_type": "classification",
        "source": "HuggingFace links-ads/wildfires-cems (GitHub MatteoM95/CEMS-Wildfire-Dataset)",
        "license": "CC-BY-4.0",
        "provenance": {
            "url": "https://github.com/MatteoM95/CEMS-Wildfire-Dataset",
            "have_locally": False,
            "annotation_method": "Copernicus EMS rapid-mapping burn delineation + severity grading",
        },
        "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
        "classes": [
            {"id": i, "name": n, "description": d} for i, (n, d) in enumerate(CLASSES)
        ],
        "nodata_value": io.CLASS_NODATA,
        "num_samples": len(selected),
        "notes": (
            "Post-fire Sentinel-2 activations (Copernicus EMS). Labels reprojected from "
            "EPSG:4326 ~10 m to local UTM 10 m (nearest) and tiled into 64x64. Unified "
            "scheme combines burn-severity grading (ids 0-4 from *_GRA.tif) with the binary "
            "delineation product (ids 0/5 from *_DEL.tif for activations lacking grading). "
            "Class 0 (no visible damage) is a genuine observed background within each AOI. "
            "Change labels: change_time = post-fire S2 acquisition date; time_range = +/-180 d "
            "(360 d) around it. Tiles-per-class balanced, <=1000/class, 25k cap."
        ),
    }
    io.write_dataset_metadata(SLUG, metadata)

    _write_summary(cand_per_class, sel_per_class, len(selected), len(sample_dirs))
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"STATUS: completed classification num_samples={len(selected)}", flush=True)
    print("selected tiles-per-class:", dict(sorted(sel_per_class.items())), flush=True)


def _write_summary(cand: Counter, sel: Counter, n_samples: int, n_dirs: int) -> None:
    lines = [
        f"# CEMS Wildfire Dataset — {SLUG}",
        "",
        "- **Status**: completed",
        "- **Task type**: classification (dense_raster; burn-severity segmentation)",
        f"- **Samples**: {n_samples} label patches (64x64, UTM 10 m, uint8, nodata=255)",
        "- **Source**: HuggingFace `links-ads/wildfires-cems` "
        "(repo GitHub `MatteoM95/CEMS-Wildfire-Dataset`); CC-BY-4.0",
        "- **URL**: https://github.com/MatteoM95/CEMS-Wildfire-Dataset",
        "- **Access**: public HF download of split `*.tar.NNNN.gz.part` files "
        "(concatenated per split, then untarred). No credentials.",
        f"- **Source sample dirs scanned**: {n_dirs} (train+val+test all used)",
        "",
        "## What the dataset is",
        "",
        "500+ Copernicus EMS rapid-mapping wildfire activations (Jun 2017 - Apr 2023, "
        "mostly Europe). Each sample directory carries a post-fire Sentinel-2 L2A GeoTIFF "
        "plus georeferenced label rasters: `*_DEL.tif` (binary burned-area delineation, "
        "present for all samples) and `*_GRA.tif` (burn-severity grading 0-4, present for a "
        "subset). Rasters are EPSG:4326 at ~10 m; GRA non-zero footprint exactly matches "
        "DEL (verified), so GRA subsumes DEL where present.",
        "",
        "## Processing",
        "",
        "- Reprojected each categorical label from EPSG:4326 (~10 m) to local UTM at 10 m "
        "with **nearest** resampling (categorical), via `calculate_default_transform` + "
        "`rasterio.warp.reproject`.",
        "- Tiled each reprojected label into non-overlapping 64x64 UTM patches; dropped "
        "tiles touching the reprojection border (nodata) and tiles with no burned pixels.",
        "- **Unified class scheme** (spec sec.5 multi-target combine): severity grades map "
        "directly to ids 0-4; for delineation-only activations (no GRA) burned pixels get "
        "id 5 (`burned_ungraded`). Unburned (id 0) is a real observed background class "
        "(CEMS delineates the whole AOI), so this dataset is NOT positive-only.",
        "- **Time**: burn scars are change/event labels. `change_time` = post-fire S2 "
        "acquisition date (from `*_S2L2A.json`); `time_range` = +/-180 d (360 d) around it. "
        "Burn scars persist for months, so a yearly window is well-posed (not flagged).",
        "- **Sampling**: tiles-per-class balanced, rarest-severity-first, <=1000 tiles per "
        "class, 25k total cap (`sampling.select_tiles_per_class`).",
        "- Did NOT apply the cloud mask (`*_CM.tif`): the CEMS burn labels are authoritative "
        "vector rapid-mapping products independent of clouds in the particular S2 mosaic.",
        "",
        "## Classes (id: name — candidate tiles / selected tiles)",
        "",
    ]
    for i, (n, _d) in enumerate(CLASSES):
        lines.append(f"- {i}: {n} — {cand.get(i, 0)} / {sel.get(i, 0)}")
    lines += [
        "",
        "Class 1 (negligible_to_slight) is the least common severity grade but has enough "
        "candidate tiles to reach the ~1000/class target. Class 0 (background) is co-present "
        "in nearly every burned tile so it exceeds the 1000 guideline; this is inherent "
        "(cannot drop background without dropping burn signal). All severity classes reach "
        "~1000-1400 selected tiles; downstream assembly drops any class below its minimum.",
        "",
        "## Verification",
        "",
        "Output tifs are single-band uint8, UTM CRS at 10 m, 64x64, values in {0..5} plus "
        "255 nodata; each tif has a matching JSON with a 360-day `time_range` and a "
        "`change_time`. Georeferencing derived from the reprojection transform.",
        "",
        "## Reproduce",
        "",
        "```",
        f"python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.{SLUG}",
        "```",
        "",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
