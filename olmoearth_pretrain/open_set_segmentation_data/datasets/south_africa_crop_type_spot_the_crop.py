"""Process South Africa Crop Type (Spot the Crop) into open-set-segmentation labels.

Source: "Crop Type Classification Dataset for Western Cape, South Africa" (Western Cape
Department of Agriculture + Radiant Earth Foundation, 2021), produced for the Radiant
Earth **Spot the Crop** Challenge. Mirrored on Source Cooperative at
``radiantearth/south-africa-crops-competition`` (data proxy
``https://data.source.coop``). Licensed CC-BY-4.0. Manual government field surveys over
the Western Cape, paired with 2017 Sentinel-1/2 time series.

The label layer is delivered as **georeferenced** 256x256 chips (NOT coordinate-free):
``train/labels/{tile}.tif`` is a per-pixel crop-code raster and
``train/labels/{tile}_field_ids.tif`` a per-pixel field-id raster. Both are EPSG:32634
(UTM 34, negative northings for the southern hemisphere), 10 m/pixel, north-up. There are
2650 train tiles. The **test** split ships field-id rasters but the crop labels are
withheld (competition holdout), so only the train split carries usable labels; test is
skipped.

GEOREFERENCING (verified): tile 1000 center reprojects to lon/lat ~ (18.51, -32.24),
Western Cape. We keep the source CRS EPSG:32634 and pixel grid verbatim (the spec allows
reusing a source window's CRS when it is already UTM at 10 m) -- lossless, no resampling.
pyproj transforms the negative-northing 32634 coordinates to the correct S-hemisphere
lon/lat, and downstream pairing projects via the CRS, so this is safe.

Task: per-pixel **classification** (crop type). One label patch per surveyed field
(EuroCrops / CV4A-Kenya style): a <=64x64 UTM 10 m tile sized to the field footprint and
centered on it, crop class id burned at every labeled pixel in the window (neighboring
labeled fields included) and 255 (nodata/ignore) everywhere unlabeled -- we only have a
ground-truth crop label inside surveyed fields, so unlabeled land is ignore, not a
background class.

Classes (from the dataset's authoritative ``labels.json``; code 0 = "No Data" -> nodata).
The manifest's class list is slightly off (it lists "barley"; the real legend has
"Weeds"), so we follow labels.json. Crop codes 1-9 -> class ids 0-8:
    0 Lucerne/Medics            (code 1)
    1 Planted pastures (perennial) (code 2)
    2 Fallow                    (code 3)
    3 Wine grapes               (code 4)
    4 Weeds                     (code 5)
    5 Small grain grazing       (code 6)
    6 Wheat                     (code 7)
    7 Canola                    (code 8)
    8 Rooibos                   (code 9)
Each field is painted with a single uniform crop code in the raster (verified against
field_info_train.csv), so the per-field class is the raster mode over the field pixels.

Sampling: tiles-per-class balanced, up to 1000 fields/class (25k cap; only 9 classes so
the cap is not binding). Time range: 1-year window on 2017 (growing season).

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.south_africa_crop_type_spot_the_crop
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

SLUG = "south_africa_crop_type_spot_the_crop"
NAME = "South Africa Crop Type (Spot the Crop)"
BUCKET = "radiantearth"
ENDPOINT = "https://data.source.coop"
KEY_PREFIX = "south-africa-crops-competition"
URL = "https://source.coop/radiantearth/south-africa-crops-competition"

EPSG = 32634
YEAR = 2017
NUM_TILES = 2650
TILE_PX = 256
MAX_TILE = io.MAX_TILE  # 64
PER_CLASS = 1000

# crop code -> (class_id, name), from labels.json. Code 0 ("No Data") -> nodata (255).
CODE_TO_CLASS = {
    1: (0, "Lucerne/Medics"),
    2: (1, "Planted pastures (perennial)"),
    3: (2, "Fallow"),
    4: (3, "Wine grapes"),
    5: (4, "Weeds"),
    6: (5, "Small grain grazing"),
    7: (6, "Wheat"),
    8: (7, "Canola"),
    9: (8, "Rooibos"),
}
_REMAP = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _code, (_cid, _name) in CODE_TO_CLASS.items():
    _REMAP[_code] = _cid

_PROJ = Projection(CRS.from_epsg(EPSG), io.RESOLUTION, -io.RESOLUTION)


def _label_key(tile: int) -> str:
    return f"{KEY_PREFIX}/train/labels/{tile}.tif"


def _fieldid_key(tile: int) -> str:
    return f"{KEY_PREFIX}/train/labels/{tile}_field_ids.tif"


def _download_tile(tile: int) -> tuple[int, str]:
    """Download a train tile's crop-label and field-id rasters to raw/."""
    raw = io.raw_dir(SLUG) / "train" / "labels"
    try:
        for key in (_label_key(tile), _fieldid_key(tile)):
            dst = raw / key.split("/")[-1]
            download.download_s3_unsigned(BUCKET, key, dst, endpoint_url=ENDPOINT)
        return tile, "ok"
    except Exception as e:  # noqa: BLE001
        print(f"download error tile {tile}: {e}")
        return tile, "error"


def _tile_origin(ds: "rasterio.io.DatasetReader") -> tuple[int, int]:
    """Return (proj_col0, proj_row0): pixel coords of the tile top-left under _PROJ.

    Source transform origin is (E0, N0) with N0 negative (S hemisphere). Under
    Projection(crs, 10, -10) a world point (x, y) has pixel (x/10, y/-10), so the tile
    top-left maps to (E0/10, -N0/10).
    """
    t = ds.transform
    return int(round(t.c / io.RESOLUTION)), int(round(-t.f / io.RESOLUTION))


def _extract_fields(tile: int) -> list[dict[str, Any]]:
    """Read one tile; return per-field records (majority crop code + bbox in tile px)."""
    raw = io.raw_dir(SLUG) / "train" / "labels"
    lab_path = (raw / f"{tile}.tif").path
    fid_path = (raw / f"{tile}_field_ids.tif").path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lab_ds = rasterio.open(lab_path)
        lab = lab_ds.read(1)
        fid = rasterio.open(fid_path).read(1)
    ox, oy = _tile_origin(lab_ds)
    H, W = lab.shape
    flat_f = fid.ravel()
    flat_l = lab.ravel()
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
        labeled = seg_l > 0  # code 0 = No Data
        if not labeled.any():
            continue
        vals, cnts = np.unique(seg_l[labeled], return_counts=True)
        code = int(vals[np.argmax(cnts)])
        if code not in CODE_TO_CLASS:
            continue
        rr = rows_all[s:e]
        cc = cols_all[s:e]
        out.append(
            {
                "tile": tile,
                "field_id": int(fld),
                "class_id": CODE_TO_CLASS[code][0],
                "bbox": (
                    int(cc.min()),
                    int(rr.min()),
                    int(cc.max()) + 1,
                    int(rr.max()) + 1,
                ),
                "ox": ox,
                "oy": oy,
                "H": H,
                "W": W,
            }
        )
    return out


def _write_tile_fields(
    tile: int, recs: list[dict[str, Any]]
) -> list[tuple[str, str, int]]:
    """Write all selected fields for one tile (read the tile label raster once)."""
    raw = io.raw_dir(SLUG) / "train" / "labels"
    results: list[tuple[str, str, int]] = []
    lab = None
    for rec in recs:
        sample_id = rec["sample_id"]
        cid = rec["class_id"]
        tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
        if tif.exists():
            results.append((sample_id, "skip", cid))
            continue
        try:
            if lab is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lab = rasterio.open((raw / f"{tile}.tif").path).read(1)
            c0, r0, c1, r1 = rec["bbox"]
            H, W = rec["H"], rec["W"]
            bw = min(MAX_TILE, max(1, c1 - c0))
            bh = min(MAX_TILE, max(1, r1 - r0))
            cx = (c0 + c1) // 2
            cy = (r0 + r1) // 2
            wc0 = min(max(0, cx - bw // 2), W - bw)
            wr0 = min(max(0, cy - bh // 2), H - bh)
            wc1, wr1 = wc0 + bw, wr0 + bh
            arr = _REMAP[lab[wr0:wr1, wc0:wc1]]
            if not (arr != io.CLASS_NODATA).any():
                results.append((sample_id, "empty", cid))
                continue
            ox, oy = rec["ox"], rec["oy"]
            bounds = (ox + wc0, oy + wr0, ox + wc1, oy + wr1)
            io.write_label_geotiff(
                SLUG, sample_id, arr, _PROJ, bounds, nodata=io.CLASS_NODATA
            )
            io.write_sample_json(
                SLUG,
                sample_id,
                _PROJ,
                bounds,
                io.year_range(YEAR),
                source_id=f"tile{tile}_field{rec['field_id']}",
                classes_present=sorted(
                    set(np.unique(arr).tolist()) - {io.CLASS_NODATA}
                ),
            )
            results.append((sample_id, "ok", cid))
        except Exception as e:  # noqa: BLE001
            print(f"error on {sample_id}: {e}")
            results.append((sample_id, "error", cid))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    tiles = list(range(1, NUM_TILES + 1))
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "South Africa Crop Type (Spot the Crop), CC-BY-4.0.\n"
            f"{URL}\nSource Cooperative data proxy: {ENDPOINT}/{BUCKET}/{KEY_PREFIX}/\n"
            "Downloaded: train/labels/{tile}.tif (crop code) + {tile}_field_ids.tif for "
            f"tiles 1..{NUM_TILES}. Test crop labels are withheld (skipped).\n"
        )

    # ---- Download all train label rasters (parallel) --------------------------------
    with multiprocessing.Pool(args.workers) as p:
        dl: Counter = Counter()
        for _tile, res in tqdm.tqdm(
            star_imap_unordered(p, _download_tile, [dict(tile=t) for t in tiles]),
            total=len(tiles),
            desc="download",
        ):
            dl[res] += 1
    print("download results:", dict(dl))
    io.check_disk()

    # ---- Pass 1: extract per-field records (parallel over tiles) --------------------
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _extract_fields, [dict(tile=t) for t in tiles]),
            total=len(tiles),
            desc="extract",
        ):
            records.extend(recs)
    print(f"total fields: {len(records)}")
    print("field class distribution:", dict(Counter(r["class_id"] for r in records)))

    # ---- Balance per class (<=1000 fields/class, 25k cap) ---------------------------
    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    print(f"selected {len(selected)} fields after balancing")

    for i, r in enumerate(sorted(selected, key=lambda r: (r["tile"], r["field_id"]))):
        r["sample_id"] = f"{i:06d}"
    by_tile: dict[int, list[dict[str, Any]]] = {}
    for r in selected:
        by_tile.setdefault(r["tile"], []).append(r)

    # ---- Pass 2: write windows (parallel over tiles) --------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    args_list = [dict(tile=t, recs=recs) for t, recs in by_tile.items()]
    with multiprocessing.Pool(args.workers) as p:
        for res_list in tqdm.tqdm(
            star_imap_unordered(p, _write_tile_fields, args_list),
            total=len(args_list),
            desc="write",
        ):
            for _sid, res, cid in res_list:
                results[res] += 1
                if res in ("ok", "skip"):
                    written_by_class[cid] += 1
    print("write results:", dict(results))
    io.check_disk()

    # ---- Metadata -------------------------------------------------------------------
    classes = [
        {
            "id": cid,
            "name": name,
            "description": (
                f"Western Cape crop-type legend code {code} ({name}); "
                "government field survey (Spot the Crop challenge)."
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
            "source": "Source Cooperative (radiantearth/south-africa-crops-competition)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "data_proxy": f"{ENDPOINT}/{BUCKET}/{KEY_PREFIX}/",
                "doi": "10.34911/rdnt.j0co8q",
                "have_locally": False,
                "annotation_method": "manual government field survey (Western Cape Dept. of Agriculture)",
                "georeferencing": (
                    "Source labels are georeferenced 256x256 chips, EPSG:32634 (UTM 34, "
                    "negative northings for S hemisphere), 10 m north-up. CRS + pixel grid "
                    "kept verbatim (no resampling); verified tile1000 center -> lon/lat "
                    "(18.51, -32.24), Western Cape."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "Per-field crop-type label patches (<=64x64 UTM 10 m), one per surveyed "
                "field, sized to the field footprint. Crop class burned at labeled pixels "
                "(neighboring fields included), 255 (nodata/ignore) on unlabeled land. "
                "Derived from train/labels/{tile}.tif (crop code) + {tile}_field_ids.tif; "
                "per-field class = raster mode (fields are uniform, verified vs "
                "field_info_train.csv). Codes 1-9 -> ids 0-8 per labels.json; code 0 "
                "(No Data) -> nodata. Test split crop labels are withheld -> train only. "
                "Tiles-per-class balanced, up to 1000 fields/class. Time range = 2017 "
                "growing-season 1-year window."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {len(CODE_TO_CLASS)} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
