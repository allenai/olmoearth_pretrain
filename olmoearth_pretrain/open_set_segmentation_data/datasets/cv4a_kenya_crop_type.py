"""Process CV4A Kenya Crop Type into open-set-segmentation label patches.

Source: "CV4A Kenya Crop Type Competition" / PlantVillage-Radiant Earth Kenya crop-type
training data (ICLR 2020 CV4A workshop challenge), mirrored on Source Cooperative at
``radiantearth/african-crops-kenya-02`` (STAC id ``ref_african_crops_kenya_02``).
Licensed CC-BY-SA-4.0. Smallholder crop-type field labels from western Kenya (Bungoma
area) collected in-situ via the PlantVillage app, quality-controlled against Sentinel-2
by Radiant Earth. Growing season 2019.

The source provides four 2016x3035 tiles (data/{0,1,2,3}/), each with a per-pixel
``{t}_label.tif`` (crop code 1-7, 0 = unlabeled/withheld-test) and ``{t}_field_id.tif``
(integer field id), plus the Sentinel-2 time series.

GEOREFERENCING RECOVERY (critical). The Source Cooperative mirror strips all
georeferencing: every tif is a plain ``tifffile.py`` array with an identity transform and
no CRS. We reconstruct it as follows and VALIDATE it against real Sentinel-2:
  * Edge cross-correlation of adjacent tile borders gives an unambiguous 2x2 mosaic:
        [tile1 | tile3]     (top row)
        [tile0 | tile2]     (bottom row)
    i.e. mosaic (6070 rows x 4032 cols) at 10 m = 40.32 km x 60.70 km.
  * The dataset's WGS84 bounding box (NASA CMR collection C2781412688-MLHUB:
    W 34.02206853 E 34.38442998 N 0.71604663 S 0.16702187) reprojected to UTM 36N
    (EPSG:32636) matches those dimensions to ~1 px; the mosaic top-left snaps to
    E=613740, N=79160.
  * Validation: the reconstructed mosaic B08 (2019-06-06) cross-correlated against the
    real Sentinel-2 scene S2B_36NXF_20190606 (same MGRS tile 36NXF, same date, from the
    open AWS sentinel-cogs bucket) peaks at correlation 0.9999999 at pixel offset (0, 0).
    The recovered grid is pixel-exact and aligned to the native S2 grid
    (S2 transform origin 600000/100020; our origin = 600000+1374*10 / 100020-2086*10).

Task: per-pixel **classification** (crop type). One label patch per labeled field
(EuroCrops-style): a <=64x64 UTM 10 m tile sized to the field footprint and centered on
it, with the crop class id burned at every labeled pixel in the window (neighboring
labeled fields included) and 255 (nodata/ignore) everywhere unlabeled -- we only have a
ground-truth crop label inside surveyed fields, so unlabeled land is ignore, not a
background class. Withheld test fields (label 0) are not used.

Classes (from the dataset Documentation.pdf Appendix D, the authoritative legend --
note the manifest's class list is INCORRECT for this dataset). Crop codes 1-7 -> class
ids 0-6:
    0 Maize
    1 Cassava
    2 Common Bean
    3 Maize & Common Bean (intercropping)
    4 Maize & Cassava   (intercropping)
    5 Maize & Soybean   (intercropping)
    6 Cassava & Common Bean (intercropping)

Sampling: tiles-per-class balanced, up to 1000 fields/class (25k cap; only Maize, with
1462 fields, is truncated). Time range: 1-year window on 2019 (growing season).

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cv4a_kenya_crop_type
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "cv4a_kenya_crop_type"
NAME = "CV4A Kenya Crop Type"

# Recovered mosaic georeferencing (validated pixel-exact against S2B_36NXF_20190606).
EPSG = 32636
MOSAIC_ORIGIN_E = 613740  # top-left easting
MOSAIC_ORIGIN_N = 79160  # top-left northing
TILE_W, TILE_H = 2016, 3035  # per source tile
# 2x2 layout: (row, col) -> source tile index.
LAYOUT = {(0, 0): 1, (0, 1): 3, (1, 0): 0, (1, 1): 2}

YEAR = 2019
MAX_TILE = io.MAX_TILE  # 64
PER_CLASS = 1000

# crop code -> (class_id, name). Codes are 1..7 in the source; class ids 0..6.
CODE_TO_CLASS = {
    1: (0, "Maize"),
    2: (1, "Cassava"),
    3: (2, "Common Bean"),
    4: (3, "Maize & Common Bean (intercropping)"),
    5: (4, "Maize & Cassava (intercropping)"),
    6: (5, "Maize & Soybean (intercropping)"),
    7: (6, "Cassava & Common Bean (intercropping)"),
}
# uint8 remap table: code (0..7) -> class id (0..6) or nodata (255) for code 0.
_REMAP = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _code, (_cid, _name) in CODE_TO_CLASS.items():
    _REMAP[_code] = _cid

_PROJ = Projection(CRS.from_epsg(EPSG), io.RESOLUTION, -io.RESOLUTION)


def build_mosaics() -> tuple[np.ndarray, np.ndarray]:
    """Assemble the 2x2 label and field-id mosaics (row 0 = north)."""
    raw = io.raw_dir(SLUG)
    lab: dict[int, np.ndarray] = {}
    fid: dict[int, np.ndarray] = {}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(4):
            lab[t] = rasterio.open((raw / f"{t}_label.tif").path).read(1)
            fid[t] = rasterio.open((raw / f"{t}_field_id.tif").path).read(1)

    def mos(d: dict[int, np.ndarray]) -> np.ndarray:
        top = np.hstack([d[LAYOUT[(0, 0)]], d[LAYOUT[(0, 1)]]])
        bot = np.hstack([d[LAYOUT[(1, 0)]], d[LAYOUT[(1, 1)]]])
        return np.vstack([top, bot])

    return mos(lab), mos(fid).astype(np.int64)


def mosaic_pixel_to_proj_bounds(
    c0: int, r0: int, c1: int, r1: int
) -> tuple[int, int, int, int]:
    """Mosaic pixel window [c0:c1, r0:r1] -> integer pixel bounds under _PROJ.

    A mosaic pixel (c, r) sits at easting E=MOSAIC_ORIGIN_E + c*10, northing
    N=MOSAIC_ORIGIN_N - r*10. Under Projection(crs, 10, -10) the pixel coordinate is
    (E/10, -N/10) = (E0/10 + c, -N0/10 + r).
    """
    ox = MOSAIC_ORIGIN_E // io.RESOLUTION
    oy = -MOSAIC_ORIGIN_N // io.RESOLUTION
    return (ox + c0, oy + r0, ox + c1, oy + r1)


def _write_tile(rec: dict[str, Any]) -> tuple[str, str, int]:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip", rec["class_id"]
    try:
        arr = rec["label"]  # (H, W) uint8, already remapped (255 = nodata)
        if not (arr != io.CLASS_NODATA).any():
            return sample_id, "empty", rec["class_id"]
        bounds = rec["bounds"]
        io.write_label_geotiff(
            SLUG, sample_id, arr, _PROJ, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            _PROJ,
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

    L, F = build_mosaics()
    H, W = L.shape
    print(f"mosaic {L.shape}; labeled px {int((L > 0).sum())}")

    # ---- field -> (class code, pixel bbox) via one sorted pass ----------------------
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

    records: list[dict[str, Any]] = []
    for i, fld in enumerate(fields):
        s, e = starts[i], starts[i + 1]
        seg_l = sl[s:e]
        labeled = seg_l > 0
        if not labeled.any():
            continue  # withheld test field
        vals, cnts = np.unique(seg_l[labeled], return_counts=True)
        code = int(vals[np.argmax(cnts)])
        cid = CODE_TO_CLASS[code][0]
        idxs = order[s:e]
        rr = rows_all[s:e]
        cc = cols_all[s:e]
        r0, r1 = int(rr.min()), int(rr.max()) + 1
        c0, c1 = int(cc.min()), int(cc.max()) + 1
        records.append(
            {
                "field_id": int(fld),
                "code": code,
                "class_id": cid,
                "bbox": (c0, r0, c1, r1),
            }
        )
    print(f"labeled fields: {len(records)}")

    # ---- balance per class (<=1000/field-class, 25k cap) ----------------------------
    selected = balance_by_class(
        records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    print(f"selected {len(selected)} fields after balancing")

    # ---- build <=64x64 windows centered on each field (label from mosaic) -----------
    tile_recs: list[dict[str, Any]] = []
    for r in selected:
        c0, r0, c1, r1 = r["bbox"]
        bw = min(MAX_TILE, max(1, c1 - c0))
        bh = min(MAX_TILE, max(1, r1 - r0))
        cx = (c0 + c1) // 2
        cy = (r0 + r1) // 2
        wc0 = min(max(0, cx - bw // 2), W - bw)
        wr0 = min(max(0, cy - bh // 2), H - bh)
        wc1, wr1 = wc0 + bw, wr0 + bh
        arr = _REMAP[L[wr0:wr1, wc0:wc1]]
        tile_recs.append(
            {
                "label": arr,
                "bounds": mosaic_pixel_to_proj_bounds(wc0, wr0, wc1, wr1),
                "class_id": r["class_id"],
                "source_id": f"field_{r['field_id']}",
            }
        )
    for i, r in enumerate(tile_recs):
        r["sample_id"] = f"{i:06d}"

    # ---- write in parallel ----------------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res, cid in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in tile_recs]),
            total=len(tile_recs),
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
            "description": f"CV4A crop code {code} ({name}); FAO crop list, PlantVillage field survey.",
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
            "source": "Source Cooperative (radiantearth/african-crops-kenya-02)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://source.coop/radiantearth/african-crops-kenya-02",
                "stac_id": "ref_african_crops_kenya_02",
                "have_locally": False,
                "annotation_method": "in-situ PlantVillage field survey, QC'd vs Sentinel-2",
                "georeferencing": (
                    "Recovered: source tifs are un-georeferenced arrays. 2x2 mosaic "
                    "[tile1|tile3]/[tile0|tile2] (edge cross-correlation), UTM 36N "
                    "(EPSG:32636), top-left E=613740 N=79160, 10 m. Validated pixel-exact "
                    "(corr 0.9999999, offset 0,0) vs S2B_36NXF_20190606."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "Per-field crop-type label patches (<=64x64 UTM 10 m), one per labeled "
                "field, sized to the field footprint. Crop class burned at labeled pixels "
                "(neighboring fields included), 255 (nodata/ignore) on unlabeled land. "
                "Withheld test fields (label 0) excluded. Class ids 0-6 map crop codes 1-7 "
                "per the dataset Documentation.pdf Appendix D (the manifest class list is "
                "wrong for this dataset). Tiles-per-class balanced, up to 1000 fields/class "
                "(only Maize truncated). Time range = 2019 growing-season 1-year window."
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
