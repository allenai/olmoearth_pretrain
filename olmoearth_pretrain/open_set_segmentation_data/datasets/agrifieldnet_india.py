"""Process AgriFieldNet India Competition into open-set-segmentation label patches.

Source: "AgriFieldNet Competition Dataset" (Radiant Earth Foundation & IDinsight, 2022),
originally distributed via Radiant MLHub (now retired) and mirrored openly on Source
Cooperative at ``radiantearth/agrifieldnet-competition`` (S3 via the
``https://data.source.coop`` unsigned proxy; bucket ``radiantearth``). Licensed
CC-BY-4.0. Ground-surveyed smallholder crop-type field labels across four northern Indian
states (Uttar Pradesh, Rajasthan, Odisha, Bihar), collected in-situ by IDinsight's Data
on Demand team and curated/QC'd against Sentinel-2 by Radiant Earth. Labeled season is the
2021-22 rabi (winter) crop cycle; we anchor a 1-year window on 2022 (the manifest year).

Unlike CV4A this mirror IS georeferenced: each 256x256 chip is a proper 10 m UTM COG.
Chips span multiple UTM zones (43N/44N/45N), so each chip's own CRS/transform is used.

Layout on the mirror:
  train_labels/ref_agrifieldnet_competition_v1_labels_train_{chip}.tif           -> crop-code raster
  train_labels/ref_agrifieldnet_competition_v1_labels_train_{chip}_field_ids.tif -> field-id raster
  test_labels/ ... _field_ids.tif   (test chips have NO crop labels -> unused here)
Only the train_labels chips carry crop codes, so we process those (1165 chips). Sentinel-2
imagery (source/) is NOT downloaded -- pretraining supplies its own imagery.

Task: per-pixel **classification** (crop type). EuroCrops/CV4A-style: one label patch per
labeled field -- a <=64x64 UTM 10 m tile centered on the field footprint, with the crop
class id burned at every labeled pixel in the window (neighboring labeled fields included)
and 255 (nodata/ignore) everywhere the field-id raster is 0 (unsurveyed land). We only have
a ground-truth crop label inside surveyed fields, so unlabeled land is ignore, not a
background class (spec 5 positive-only handling). "No crop/Fallow" (code 4) IS a real
labeled class, not background.

Crop codes (Documentation.pdf p.2; NON-contiguous) -> class ids 0..12 (ascending code):
    code  1 Wheat            -> 0
    code  2 Mustard          -> 1
    code  3 Lentil           -> 2
    code  4 No crop/Fallow   -> 3
    code  5 Green pea        -> 4
    code  6 Sugarcane        -> 5
    code  8 Garlic           -> 6
    code  9 Maize            -> 7
    code 13 Gram             -> 8
    code 14 Coriander        -> 9
    code 15 Potato           -> 10
    code 16 Bersem (berseem) -> 11
    code 36 Rice             -> 12

Sampling: tiles-per-class balanced on each field's majority crop, up to 1000 fields/class
(25k cap; per-class limit stays 1000 since 13 classes). Time range: 1-year window on 2022.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.agrifieldnet_india
"""

import argparse
import multiprocessing
import warnings
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "agrifieldnet_india"
NAME = "AgriFieldNet India"

S3_ENDPOINT = "https://data.source.coop"
S3_BUCKET = "radiantearth"
TRAIN_PREFIX = "agrifieldnet-competition/train_labels/"

YEAR = 2022
MAX_TILE = io.MAX_TILE  # 64
PER_CLASS = 1000

# crop code -> (class_id, name). Codes are non-contiguous (Documentation.pdf p.2).
CODE_TO_CLASS = {
    1: (0, "Wheat"),
    2: (1, "Mustard"),
    3: (2, "Lentil"),
    4: (3, "No crop/Fallow"),
    5: (4, "Green pea"),
    6: (5, "Sugarcane"),
    8: (6, "Garlic"),
    9: (7, "Maize"),
    13: (8, "Gram"),
    14: (9, "Coriander"),
    15: (10, "Potato"),
    16: (11, "Bersem"),
    36: (12, "Rice"),
}
# uint16-indexed remap table: crop code -> class id (0..12) or 255 (nodata) for code 0/other.
_REMAP = np.full(64, io.CLASS_NODATA, dtype=np.uint8)
for _code, (_cid, _name) in CODE_TO_CLASS.items():
    _REMAP[_code] = _cid


def list_chip_ids() -> list[str]:
    """List all train chip ids (chips that have a crop-label raster)."""
    import boto3
    import botocore

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )
    paginator = s3.get_paginator("list_objects_v2")
    chips = []
    prefix = TRAIN_PREFIX + "ref_agrifieldnet_competition_v1_labels_train_"
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=TRAIN_PREFIX):
        for o in page.get("Contents", []):
            k = o["Key"]
            if k.endswith(".tif") and not k.endswith("_field_ids.tif"):
                chips.append(k[len(prefix) : -len(".tif")])
    return sorted(chips)


def _download_chip(chip: str) -> str:
    """Download a chip's crop-label + field-id rasters into raw_dir (idempotent)."""
    raw = io.raw_dir(SLUG) / "train_labels"
    base = f"ref_agrifieldnet_competition_v1_labels_train_{chip}"
    for name in (f"{base}.tif", f"{base}_field_ids.tif"):
        download.download_s3_unsigned(
            S3_BUCKET, TRAIN_PREFIX + name, raw / name, endpoint_url=S3_ENDPOINT
        )
    return chip


def _scan_chip(chip: str) -> list[dict[str, Any]]:
    """Extract one per-field record per labeled field in a chip.

    Returns dicts with: class_id (field majority crop), bounds (px in chip CRS),
    crs (str), label (<=64x64 uint8 window, remapped, 255=nodata), source_id.
    """
    raw = io.raw_dir(SLUG) / "train_labels"
    base = f"ref_agrifieldnet_competition_v1_labels_train_{chip}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open((raw / f"{base}.tif").path) as ds:
            L = ds.read(1)
            crs = ds.crs.to_string()
            left, top = ds.bounds.left, ds.bounds.top
        with rasterio.open((raw / f"{base}_field_ids.tif").path) as ds:
            F = ds.read(1).astype(np.int64)

    H, W = L.shape
    ox = int(round(left)) // io.RESOLUTION
    oy = -int(round(top)) // io.RESOLUTION

    flat_f = F.ravel()
    flat_l = L.ravel()
    order = np.argsort(flat_f, kind="stable")
    sf = flat_f[order]
    sl = flat_l[order]
    fields = np.unique(sf)
    fields = fields[fields > 0]
    starts = np.searchsorted(sf, fields)
    starts = np.append(starts, len(sf))
    rows_all, cols_all = np.divmod(order, W)

    out: list[dict[str, Any]] = []
    for i, fld in enumerate(fields):
        s, e = starts[i], starts[i + 1]
        seg_l = sl[s:e]
        labeled = seg_l > 0
        if not labeled.any():
            continue  # field id present but no crop label (test-only field)
        vals, cnts = np.unique(seg_l[labeled], return_counts=True)
        code = int(vals[np.argmax(cnts)])
        if code not in CODE_TO_CLASS:
            continue  # unexpected code; skip
        cid = CODE_TO_CLASS[code][0]
        rr = rows_all[s:e]
        cc = cols_all[s:e]
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        bw = min(MAX_TILE, max(1, c1 - c0))
        bh = min(MAX_TILE, max(1, r1 - r0))
        cx, cy = (c0 + c1) // 2, (r0 + r1) // 2
        wc0 = min(max(0, cx - bw // 2), W - bw)
        wr0 = min(max(0, cy - bh // 2), H - bh)
        wc1, wr1 = wc0 + bw, wr0 + bh
        arr = _REMAP[np.clip(L[wr0:wr1, wc0:wc1], 0, len(_REMAP) - 1)]
        if not (arr != io.CLASS_NODATA).any():
            continue
        out.append(
            {
                "class_id": cid,
                "crs": crs,
                "bounds": (ox + wc0, oy + wr0, ox + wc1, oy + wr1),
                "label": arr,
                "source_id": f"{chip}/field_{int(fld)}",
            }
        )
    return out


def _write_tile(rec: dict[str, Any]) -> tuple[str, str, int]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip", rec["class_id"]
    try:
        arr = rec["label"]
        proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
        bounds = rec["bounds"]
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(YEAR),
            source_id=rec["source_id"],
            classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
        )
        return sample_id, "ok", rec["class_id"]
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error", rec["class_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    chips = list_chip_ids()
    print(f"train chips: {len(chips)}")

    # ---- download crop-label + field-id rasters (parallel, idempotent) --------------
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _download_chip, [dict(chip=c) for c in chips]),
            total=len(chips),
            desc="download",
        ):
            pass
    io.check_disk()

    # ---- scan chips -> per-field records (parallel) ---------------------------------
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_chip, [dict(chip=c) for c in chips]),
            total=len(chips),
            desc="scan",
        ):
            records.extend(recs)
    print(f"labeled fields: {len(records)}")
    raw_dist = Counter(r["class_id"] for r in records)
    print(
        "raw field class distribution:",
        {CODE_TO_CLASS_BY_ID[k]: raw_dist[k] for k in sorted(raw_dist)},
    )

    # ---- balance per class (<=1000 fields/class, 25k cap) ---------------------------
    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} fields after balancing")

    # ---- write in parallel ----------------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res, cid in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                written_by_class[cid] += 1
    print("write results:", dict(results))
    io.check_disk()

    # ---- metadata -------------------------------------------------------------------
    classes = [
        {
            "id": cid,
            "name": name,
            "description": (
                f"AgriFieldNet crop code {code} ({name}); ground-surveyed smallholder "
                "field, northern India rabi season."
            ),
        }
        for code, (cid, name) in sorted(CODE_TO_CLASS.items(), key=lambda kv: kv[1][0])
    ]
    class_counts = {
        name: int(written_by_class.get(cid, 0))
        for code, (cid, name) in sorted(CODE_TO_CLASS.items(), key=lambda kv: kv[1][0])
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Source Cooperative (radiantearth/agrifieldnet-competition)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://source.coop/radiantearth/agrifieldnet-competition",
                "stac_id": "ref_agrifieldnet_competition_v1",
                "have_locally": False,
                "annotation_method": (
                    "in-situ ground survey (IDinsight Data on Demand), curated/QC'd vs "
                    "Sentinel-2 by Radiant Earth Foundation"
                ),
                "doi": "10.34911/rdnt.wu92p1",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "Per-field crop-type label patches (<=64x64, native UTM 10 m COGs from the "
                "georeferenced Source Cooperative mirror), one per labeled train field, "
                "sized to the field footprint. Crop class id burned at labeled pixels "
                "(neighboring fields included); 255 (nodata/ignore) where the field-id "
                "raster is 0 (unsurveyed land) -- no synthetic background. 'No crop/Fallow' "
                "(code 4) is a real class. Crop codes 1..36 are non-contiguous; mapped to "
                "class ids 0..12 by ascending code (Documentation.pdf p.2). Test chips carry "
                "only field ids (no crop labels) and are excluded. Tiles-per-class balanced, "
                "up to 1000 fields/class. Time range = 2022 (rabi season) 1-year window. "
                "Chips span UTM zones 43N/44N/45N; each tile uses its chip's native CRS."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {len(CODE_TO_CLASS)} classes")


CODE_TO_CLASS_BY_ID = {cid: name for code, (cid, name) in CODE_TO_CLASS.items()}


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
