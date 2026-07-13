"""Process MMFlood into open-set-segmentation label patches.

Source: MMFlood — "MMFlood: A Multimodal Dataset for Flood Delineation from
Satellite Imagery" (Montello, Arnaudo, Rossi; IEEE Access 2022). Data on Zenodo
record 6534637 (single ``mmflood.zip``, 11.3 GB, CC-BY-4.0). It covers 95
Copernicus EMS flood activations (EMSR codes) over 42 countries, each split into
one or more AOIs (``mmflood/EMSR{code}-{aoi}/``) holding four co-registered
modalities: ``s1_raw`` (Sentinel-1 SAR), ``DEM``, ``hydro`` (permanent
hydrography), and ``mask`` (the manually-delineated flood extent). Only the
``mask`` rasters are the label signal we need — pretraining supplies its own
imagery — so we pull ONLY the ~1748 mask GeoTIFFs (28 MB compressed) plus
``activations.json``, via HTTP range reads into the zip, never the 13 GB of SAR.

The zip is compressed with Deflate64 (method 9), which Python's stdlib ``zipfile``
cannot decode; we use the ``zipfile-deflate64`` shim.

Mask rasters are single-band float32 in EPSG:4326 (geographic) at ~10-14 m,
valued 0 = not-flooded, 1 = flooded (no nodata). ``activations.json`` gives each
activation's title/country, event start/end datetimes, lon/lat, and train/val/test
subset.

Class scheme (dense per-pixel CLASSIFICATION), matching the manifest's binary
flooded / not-flooded:
    id 0 = not-flooded  (mapped AOI land/water NOT flagged as flood inundation)
    id 1 = flooded      (Copernicus EMS flood-inundation delineation)
    255  = nodata/ignore (pixels outside the AOI footprint after reprojection)
We keep the manifest's binary scheme (we do NOT split permanent water out via the
``hydro`` layer, unlike worldfloods_v2/sen1floods11 — the manifest defines only
flooded/not-flooded here).

Processing (label_type = dense_raster, reprojected): each mask is geographic, so
we reproject it to the local UTM zone at 10 m/pixel (nearest resampling —
categorical) onto a grid snapped to the 10 m S2 grid, marking pixels outside the
source footprint as 255. We then tile the reprojected mask into 64x64 patches
(same approach as worldfloods_v2, plus reprojection). Tiles >50% nodata are
dropped; a tile counts toward a class only with >= MIN_CLASS_PX px of it.
Selection is tiles-per-class balanced (spec 5) via ``sampling.select_tiles_per_class``
(<= 1000 tiles/class, 25k cap) — the rare ``flooded`` class is filled first. All
three source subsets (train/val/test) are used (spec 5).

Time range: the flood is a dated EVENT (change label, spec 5). Copernicus EMS
activation ``start`` dates the flood onset to within days (median event span 3
days), well inside the ~1-2 month timing-precision requirement. We set
``change_time`` = activation start and make ``time_range`` a 360-day window
centered on it. Pretraining then only uses a sample when the sampled input window
spans the flood. Events whose start is before 2016 (9 of 95 activations, from
2014-2015) fall outside the Sentinel era and are dropped (spec 8); the remaining
86 activations are processed.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mmflood
"""

import argparse
import math
import multiprocessing
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import zipfile_deflate64 as z64
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "mmflood"
NAME = "MMFlood"
ZENODO_URL = "https://zenodo.org/api/records/6534637/files/mmflood.zip/content"

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata
HALF_WINDOW_DAYS = 180  # +/- around change_time => 360-day (<=1yr) window
MIN_YEAR = 2016  # Sentinel era; drop pre-2016 activations (spec 8)

NOTFLOOD, FLOOD = 0, 1
CLASSES = [
    (
        "not-flooded",
        "Mapped area within a Copernicus EMS flood-activation AOI that was NOT "
        "delineated as flood inundation (dry land and pre-existing water).",
    ),
    (
        "flooded",
        "Flood inundation extent manually delineated by Copernicus Emergency "
        "Management Service (photointerpretation), the MMFlood label.",
    ),
]


def raw_root():
    return io.raw_dir(SLUG)


# ---------------------------------------------------------------------------
# Download: pull the single Zenodo zip (11.3 GB, one sequential connection —
# Zenodo 429-rate-limits parallel range reads), then extract ONLY the mask tifs
# + activations.json locally (the Deflate64 zip needs the zipfile-deflate64 shim).
# Idempotent: skips the download and any already-extracted files.
# ---------------------------------------------------------------------------


def _zip_path():
    return raw_root() / "mmflood.zip"


def _download_zip(dst) -> None:
    """Resumable single-stream download of the 11.3 GB zip via HTTP range reads.

    Zenodo throttles aggregate bandwidth per IP (~6-7 MB/s) and 429-rate-limits by
    request COUNT, so we pull the whole file in a few large (64 MB) sequential range
    requests on one connection (fast, and far below the request-rate limit) rather
    than thousands of tiny per-file range reads. Resumes from a partial .tmp.
    """
    import os

    import requests

    url = "https://zenodo.org/records/6534637/files/mmflood.zip?download=1"
    tmp = dst.parent / (dst.name + ".tmp")
    pos = tmp.stat().st_size if tmp.exists() else 0
    s = requests.Session()
    h = s.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    total = int(h.headers["Content-Range"].split("/")[-1])
    with open(tmp.path, "ab") as f:
        while pos < total:
            end = min(pos + 64 * 1024 * 1024, total) - 1
            r = s.get(
                url, headers={"Range": f"bytes={pos}-{end}"}, timeout=300, stream=True
            )
            r.raise_for_status()
            for chunk in r.iter_content(8 * 1024 * 1024):
                f.write(chunk)
                pos += len(chunk)
    os.rename(tmp.path, dst.path)


def _extract_members(zip_path: str, members: list[str]) -> int:
    """Worker: extract a batch of zip members from the LOCAL zip.

    Idempotent and crash-safe: a member is re-extracted unless the on-disk file
    already matches the archive's uncompressed size (guards against truncated files
    left by an interrupted, non-atomic extract). Writes to a .part then renames.
    """
    import os

    root = raw_root()
    with z64.ZipFile(zip_path) as zf:
        infos = {i.filename: i for i in zf.infolist()}
        n = 0
        for m in members:
            info = infos[m]
            dst = os.path.join(root.path, *m.split("/"))
            if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            with zf.open(info) as src, open(tmp, "wb") as f:
                while True:
                    buf = src.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            os.rename(tmp, dst)
            n += 1
    return n


def download_raw(workers: int = 16) -> dict:
    """Download the zip (if absent) and extract activations.json + all mask tifs.

    Label-only: only mask GeoTIFFs and the metadata json are extracted from the
    archive; the SAR/DEM/hydro modalities are left inside the zip.
    """
    import json

    root = raw_root()
    root.mkdir(parents=True, exist_ok=True)
    io.check_disk()

    zip_path = _zip_path()
    if not zip_path.exists():
        print("  downloading mmflood.zip (11.3 GB, single connection)...")
        _download_zip(zip_path)
    print(f"  zip present ({zip_path.stat().st_size / 1e9:.1f} GB); listing members...")

    with z64.ZipFile(str(zip_path)) as zf:
        zf.extract("activations.json", path=root.path)
        masks = sorted(
            i.filename
            for i in zf.infolist()
            if "/mask/" in i.filename and i.filename.endswith(".tif")
        )

    with (root / "activations.json").open() as f:
        acts = json.load(f)

    print(f"  {len(masks)} mask tifs in zip; extracting missing ones...")
    n = max(1, len(masks) // workers)
    chunks = [masks[i : i + n] for i in range(0, len(masks), n)]
    total = 0
    with multiprocessing.Pool(workers) as p:
        for got in star_imap_unordered(
            p,
            _extract_members,
            [dict(zip_path=str(zip_path), members=c) for c in chunks],
        ):
            total += got
    print(f"  extracted {total} new mask tifs ({len(masks) - total} already present)")
    return acts


# ---------------------------------------------------------------------------
# Reprojection + tiling
# ---------------------------------------------------------------------------


def _mask_paths() -> list[str]:
    root = raw_root() / "mmflood"
    out = []
    for act_dir in sorted(root.iterdir()):
        mdir = act_dir / "mask"
        if mdir.exists():
            for t in sorted(mdir.iterdir()):
                if t.name.endswith(".tif"):
                    out.append(str(t))
    return out


def _reproject_mask(path: str):
    """Reproject a geographic mask to local UTM 10 m; return (uint8 array, proj, col0, row0).

    Pixels outside the source footprint are 255 (nodata). Grid is snapped to the
    global 10 m grid so rslearn pixel bounds are integers.
    """
    with rasterio.open(path) as src:
        a = src.read(1)
        scrs = src.crs
        st = src.transform
        sb = src.bounds
    cx = (sb.left + sb.right) / 2.0
    cy = (sb.bottom + sb.top) / 2.0
    proj = io.utm_projection_for_lonlat(cx, cy)
    dcrs = proj.crs
    left, bottom, right, top = transform_bounds(scrs, dcrs, *sb)
    x0 = 10 * math.floor(left / 10)
    y0 = 10 * math.ceil(top / 10)
    w = int(math.ceil((right - x0) / 10))
    h = int(math.ceil((y0 - bottom) / 10))
    dt = Affine(10, 0, x0, 0, -10, y0)
    data = np.zeros((h, w), np.uint8)
    val = np.zeros((h, w), np.uint8)
    src_u8 = (a > 0.5).astype(np.uint8)  # binarise (source is float 0.0/1.0)
    reproject(
        src_u8,
        data,
        src_transform=st,
        src_crs=scrs,
        dst_transform=dt,
        dst_crs=dcrs,
        resampling=Resampling.nearest,
    )
    reproject(
        np.ones_like(a, np.uint8),
        val,
        src_transform=st,
        src_crs=scrs,
        dst_transform=dt,
        dst_crs=dcrs,
        resampling=Resampling.nearest,
    )
    out = np.where(val == 1, data, io.CLASS_NODATA).astype(np.uint8)
    col0 = int(round(dt.c / 10))
    row0 = int(round(dt.f / -10))
    return out, Projection(dcrs, 10, -10), col0, row0


def _act_code(path: str) -> str:
    """Mask path .../mmflood/EMSR107-5/mask/EMSR107-5-0.tif -> activation code EMSR107."""
    import os

    folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return folder.rsplit("-", 1)[0]


def _scan_mask(path: str) -> list[dict[str, Any]]:
    """Return one candidate record per non-mostly-nodata 64x64 tile of a mask."""
    arr, _proj, _c0, _r0 = _reproject_mask(path)
    nty, ntx = arr.shape[0] // TILE, arr.shape[1] // TILE
    recs: list[dict[str, Any]] = []
    total_px = TILE * TILE
    for ti in range(nty):
        for tj in range(ntx):
            sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
            u, c = np.unique(sub, return_counts=True)
            counts = {int(k): int(v) for k, v in zip(u, c)}
            if counts.get(io.CLASS_NODATA, 0) > MAX_NODATA_FRAC * total_px:
                continue
            present = [
                cid
                for cid, _ in enumerate(CLASSES)
                if counts.get(cid, 0) >= MIN_CLASS_PX
            ]
            if not present:
                continue
            recs.append({"path": path, "ti": ti, "tj": tj, "classes_present": present})
    return recs


def _write_mask(path: str, tiles: list[dict[str, Any]], change_time_iso: str) -> None:
    """Reproject a mask and write all its selected tiles + sidecars."""
    arr, proj, col0, row0 = _reproject_mask(path)
    change_time = datetime.fromisoformat(change_time_iso)
    tr = (
        change_time - timedelta(days=HALF_WINDOW_DAYS),
        change_time + timedelta(days=HALF_WINDOW_DAYS),
    )
    for t in tiles:
        sample_id = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
            continue
        ti, tj = t["ti"], t["tj"]
        sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE].copy()
        x_min = col0 + tj * TILE
        y_min = row0 + ti * TILE
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        io.write_label_geotiff(
            SLUG, sample_id, sub, proj, bounds, nodata=io.CLASS_NODATA
        )
        present = sorted(int(x) for x in np.unique(sub) if x != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            tr,
            change_time=change_time,
            source_id=t["source_id"],
            classes_present=present,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Downloading MMFlood mask rasters (labels only) from Zenodo...")
    acts = download_raw(workers=min(args.workers, 16))
    io.check_disk()

    # change_time (activation start) per activation code; drop pre-2016 events.
    change_by_code: dict[str, str] = {}
    dropped_codes = []
    for code, v in acts.items():
        st = datetime.fromisoformat(v["start"]).replace(tzinfo=UTC)
        if st.year < MIN_YEAR:
            dropped_codes.append(code)
            continue
        change_by_code[code] = st.isoformat()
    print(
        f"  {len(change_by_code)} activations kept, {len(dropped_codes)} pre-{MIN_YEAR} dropped"
    )

    all_paths = _mask_paths()
    # keep only masks whose activation is post-2016 and present in metadata
    paths = [p for p in all_paths if _act_code(p) in change_by_code]
    print(
        f"  {len(paths)} mask tifs to process ({len(all_paths) - len(paths)} dropped by date/metadata)"
    )

    print("Scanning masks into 64x64 tiles (reproject to UTM 10 m)...")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in star_imap_unordered(p, _scan_mask, [dict(path=x) for x in paths]):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: (r["path"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        code = _act_code(r["path"])
        import os

        base = os.path.basename(r["path"])[:-4]
        r["sample_id"] = f"{i:06d}"
        r["source_id"] = f"{base}_r{r['ti']}_c{r['tj']}"
        r["_code"] = code
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    # group by source mask for the write phase
    by_path: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_path.setdefault(r["path"], []).append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_path)} masks...")
    write_args = [
        dict(path=pth, tiles=ts, change_time_iso=change_by_code[_act_code(pth)])
        for pth, ts in by_path.items()
    ]
    with multiprocessing.Pool(args.workers) as p:
        for _ in star_imap_unordered(p, _write_mask, write_args):
            pass

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo record 6534637 (Montello, Arnaudo, Rossi, IEEE Access 2022)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/6534637",
                "have_locally": False,
                "annotation_method": "Copernicus EMS flood delineation (manual photointerpretation)",
                "citation": "Montello, Arnaudo, Rossi, 'MMFlood: A Multimodal Dataset for Flood Delineation from Satellite Imagery', IEEE Access 2022; DOI 10.1109/ACCESS.2022.3205419",
                "subsets_used": ["train", "val", "test"],
            },
            "sensors_relevant": ["sentinel1", "sentinel2"],
            "classes": [
                {"id": i, "name": nm, "description": desc}
                for i, (nm, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "95 Copernicus EMS flood activations over 42 countries; only the manual "
                "flood-delineation mask rasters were used (label-only; the 13 GB of Sentinel-1 "
                "SAR and DEM/hydro were not downloaded). Masks are float32 binary (0 not-flooded, "
                "1 flooded) in EPSG:4326 at ~10-14 m; reprojected to local UTM at 10 m (nearest "
                "resampling) and tiled into 64x64 patches, pixels outside the AOI footprint set to "
                "255. Binary manifest scheme kept (permanent water NOT split out). Flood is a dated "
                "EVENT: change_time = Copernicus EMS activation start (event onset, known to within "
                "days; median activation span 3 days), time_range a 360-day window centered on it. "
                f"{len(dropped_codes)} of 95 activations with start before {MIN_YEAR} (2014-2015) were "
                "dropped as pre-Sentinel-era. Tiles-per-class balanced (<=1000/class); the rare "
                "'flooded' class is filled first. Deflate64-compressed zip extracted label-only via "
                "HTTP range reads."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
