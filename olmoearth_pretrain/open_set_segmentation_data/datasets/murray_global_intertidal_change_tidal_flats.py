"""Process the Murray Global Intertidal Change tidal-flat classification into open-set
segmentation label patches.

Source: Murray Global Intertidal Change Classification v1.2 (1999-2019); Murray et al.
2019 Nature "The global distribution and trajectory of tidal flats" (doi:10.1038/
s41586-018-0805-8) and the accompanying Scientific Data descriptor. Distributed via
Figshare (collection 5884598) and intertidal.app. A supervised per-pixel classification
of the Landsat archive over the global coastline (60S-60N) at 30 m, in 3-year epochs.

IMPORTANT (data reality vs manifest): the manifest lists three classes
[tidal flat, permanent water, other], but the *distributed* v1.2 classification raster is
a BINARY tidal-flat mask -- confirmed empirically across all 108 global tiles: the only
pixel values present are {0, 1}, where 1 = tidal flat and 0 = everything else (permanent
water and other land cover are NOT separated in the published GeoTIFF; this matches the
Earth Engine catalog, whose classification band is bit 0 = intertidal/non-intertidal).
We therefore produce a **2-class** classification: id 0 = tidal flat, id 1 = other
(non-tidal-flat, i.e. permanent water + all other cover merged). See summary/QUESTION.

We use the **2017-2019 epoch** (latest, matching manifest range 2016-2019). Inside the
56 GB product zip this is a 9.78 GB nested zip `2017-2019.zip` -> `global_intertidal/`
folder of 108 EPSG:4326 GeoTIFFs (each 74213x74213 uint16, 20deg x 20deg at 30 m, LZW).
Per the spec (global derived-product map) we do BOUNDED sampling: only this one epoch is
downloaded (not all 7 epochs / 56 GB), and windows are drawn from the 82 tiles that
contain tidal flat, spread across the global coastline with a per-tile cap. Native 30 m
EPSG:4326 windows are reprojected to a local UTM projection at 10 m with **nearest**
resampling (categorical). task_type=classification, 1-year time range in the epoch.

Sampling (tiles-per-class balanced): tidal-flat windows (any block with tidal-flat
coverage >= TF_MIN) carry both classes and are the rare/valuable positives; coastal
"other" windows (no tidal flat but adjacent to tidal flat, i.e. genuine coastline, not
open ocean) provide negatives. Balanced to <=1000 windows per class.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.murray_global_intertidal_change_tidal_flats
"""

import argparse
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "murray_global_intertidal_change_tidal_flats"

EPOCH = "2017-2019"
YEAR = 2018  # representative 1-year window inside the 2017-2019 epoch (center year)

FIGSHARE_URL = "https://ndownloader.figshare.com/files/34337744"
EPOCH_ZIP = io.raw_dir(SLUG) / f"{EPOCH}.zip"

PER_CLASS = 1000
BLOCK = 22  # native 30 m block ~= 660 m ~= a 64 px @ 10 m UTM tile footprint
TILE = 64
STRIP_BLOCKS = 88  # read the tile in strips of this many block-rows (memory bound)
TF_MIN_FRAC = 0.05  # a "tidal flat" window: >= 5% of the block is tidal flat
PER_TILE_CAP = 40  # cap candidates per class per tile for geographic diversity
NEG_NEIGHBORHOOD = (
    3  # coastal-other block must be within this many blocks of tidal flat
)

# Distributed raster is binary. source 1 -> tidal flat (id 0); source 0 -> other (id 1).
SRC_TO_ID = {1: 0, 0: 1}
CLASSES = [
    (
        "tidal flat",
        "Tidal flat ecosystems subject to regular tidal inundation: unconsolidated fine-grain "
        "sediments (tidal mudflats), coarse-grain sediments (tidal sand flats), and consolidated "
        "sediments/organic material/rock (wide tidal rock-platforms). Excludes vegetated "
        "intertidal systems such as mangroves and marshes. (Source value 1.)",
    ),
    (
        "other",
        "Non-tidal-flat: everything else in the global coastal analysis area. The distributed "
        "v1.2 raster is binary, so this merges the manifest's 'permanent water' and 'other' "
        "classes (open water, terrestrial land, vegetation, built-up). (Source value 0.)",
    ),
]


def download_epoch() -> None:
    """Download+decompress the single 2017-2019 nested zip via HTTP byte-range reads.

    The outer figshare zip stores each epoch as a DEFLATE member, so we range-read that
    member's compressed bytes and stream-decompress to disk (bounded: ~9.78 GB, not 56 GB).
    """
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    if EPOCH_ZIP.exists():
        print(f"  [skip] {EPOCH_ZIP.name} present")
        return
    import io as _io
    import struct
    import urllib.request
    import zlib

    def ranged(start, length, retries=6):
        end = start + length - 1
        req = urllib.request.Request(
            FIGSHARE_URL,
            headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"},
        )
        for a in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return r.read()
            except Exception:
                if a == retries - 1:
                    raise

    class _RR(_io.RawIOBase):
        def __init__(self):
            self._pos = 0
            req = urllib.request.Request(
                FIGSHARE_URL, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as r:
                self._size = int(r.headers["Content-Length"])

        def seekable(self):
            return True

        def readable(self):
            return True

        def seek(self, off, whence=0):
            self._pos = (
                off
                if whence == 0
                else (self._pos + off if whence == 1 else self._size + off)
            )
            return self._pos

        def tell(self):
            return self._pos

        def read(self, size=-1):
            if size < 0:
                size = self._size - self._pos
            if size <= 0:
                return b""
            d = ranged(self._pos, min(size, self._size - self._pos))
            self._pos += len(d)
            return d

        def readinto(self, b):
            d = self.read(len(b))
            b[: len(d)] = d
            return len(d)

    zf = zipfile.ZipFile(_RR())
    info = zf.getinfo(f"{EPOCH}.zip")
    lh = ranged(info.header_offset, 30)
    assert lh[:4] == b"PK\x03\x04"
    n = struct.unpack("<H", lh[26:28])[0]
    m = struct.unpack("<H", lh[28:30])[0]
    data_start = info.header_offset + 30 + n + m
    dec = zlib.decompressobj(-15) if info.compress_type == 8 else None
    CHUNK = 64 * 1024 * 1024
    read = 0
    tmp = EPOCH_ZIP.parent / (EPOCH_ZIP.name + ".tmp")
    print(f"  downloading epoch {EPOCH} ({info.compress_size / 1e9:.2f} GB compressed)")
    with tmp.open("wb") as f:
        while read < info.compress_size:
            io.check_disk()
            buf = ranged(data_start + read, min(CHUNK, info.compress_size - read))
            read += len(buf)
            f.write(dec.decompress(buf) if dec else buf)
        if dec:
            f.write(dec.flush())
    tmp.rename(EPOCH_ZIP)
    print(f"  wrote {EPOCH_ZIP.name}")


def list_tiles() -> list[str]:
    """Return zip member names of the global_intertidal GeoTIFF tiles."""
    with zipfile.ZipFile(EPOCH_ZIP.path) as zf:
        return sorted(
            m
            for m in zf.namelist()
            if "global_intertidal" in m and m.lower().endswith(".tif")
        )


def _vsipath(member: str) -> str:
    return f"/vsizip/{EPOCH_ZIP.path}/{member}"


def _scan_tile(member: str) -> list[dict[str, Any]]:
    """Scan one tile (streamed in native-res strips) for candidate 660 m blocks.

    Returns records for tidal-flat blocks (>= TF_MIN_FRAC tidal flat) and coastal-other
    blocks (no tidal flat but adjacent to tidal flat), capped per tile for diversity.
    """
    rng = np.random.default_rng(abs(hash(member)) % (2**32))
    with rasterio.open(_vsipath(member)) as ds:
        H, W = ds.height, ds.width
        st = ds.transform
        nby, nbx = H // BLOCK, W // BLOCK
        tf_frac = np.zeros((nby, nbx), np.float32)
        denom = float(BLOCK * BLOCK)
        step = STRIP_BLOCKS * BLOCK
        for r0 in range(0, nby * BLOCK, step):
            rows = min(step, nby * BLOCK - r0)
            nbr = rows // BLOCK
            if nbr == 0:
                continue
            a = ds.read(1, window=Window(0, r0, nbx * BLOCK, nbr * BLOCK))
            a = (a == 1).astype(np.float32).reshape(nbr, BLOCK, nbx, BLOCK)
            tf_frac[r0 // BLOCK : r0 // BLOCK + nbr] = a.sum(axis=(1, 3)) / denom
    is_tf = tf_frac >= TF_MIN_FRAC
    if not is_tf.any():
        return []
    # coastal-other blocks: no tidal flat, but within NEG_NEIGHBORHOOD blocks of a tf block.
    from scipy.ndimage import binary_dilation

    k = 2 * NEG_NEIGHBORHOOD + 1
    near_tf = binary_dilation(is_tf, structure=np.ones((k, k), bool))
    is_neg = near_tf & (tf_frac == 0.0)

    def sample(mask, label):
        brs, bcs = np.nonzero(mask)
        if len(brs) == 0:
            return []
        idx = np.arange(len(brs))
        if len(idx) > PER_TILE_CAP:
            idx = rng.choice(idx, PER_TILE_CAP, replace=False)
        recs = []
        for i in idx:
            br, bc = int(brs[i]), int(bcs[i])
            cx = bc * BLOCK + BLOCK / 2.0
            cy = br * BLOCK + BLOCK / 2.0
            lon = st.c + cx * st.a
            lat = st.f + cy * st.e
            recs.append(
                {
                    "member": member,
                    "lon": float(lon),
                    "lat": float(lat),
                    "label": label,
                    "tf_frac": float(tf_frac[br, bc]),
                    "source_id": f"{member.split('/')[-1]}_r{br}_c{bc}",
                }
            )
        return recs

    return sample(is_tf, 0) + sample(is_neg, 1)


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.01  # deg margin so tile is fully covered before nearest-resampling

    with rasterio.open(_vsipath(rec["member"])) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.zeros((TILE, TILE), np.uint16)
    reproject(
        source=src.astype(np.uint16),
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
    )
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    for src_v, cid in SRC_TO_ID.items():
        out[dst == src_v] = cid

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--scan-workers", type=int, default=24)
    args = parser.parse_args()

    io.check_disk()
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "Murray Global Intertidal Change Classification v1.2 (1999-2019)\n"
            "Figshare collection 5884598; epoch 2017-2019 (global_intertidal binary "
            "tidal-flat mask, EPSG:4326, 30 m).\n"
            f"{FIGSHARE_URL}\n"
        )

    print("Ensuring epoch download...")
    download_epoch()
    io.check_disk()

    tiles = list_tiles()
    print(f"{len(tiles)} global_intertidal tiles in epoch zip")

    print("Scanning tiles for candidate windows (native-res strips)...")
    with multiprocessing.Pool(args.scan_workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(member=m) for m in tiles]),
            total=len(tiles),
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate windows")
    print("  raw class-candidate counts:", Counter(r["label"] for r in all_recs))
    tiles_with = len({r["member"] for r in all_recs})
    print(f"  from {tiles_with} tiles")

    selected = balance_by_class(all_recs, "label", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (<= {PER_CLASS}/class)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    counts = Counter(r["label"] for r in selected)
    class_counts = {name: counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)}
    n_tiles_selected = len({r["member"] for r in selected})
    print("selected class counts (primary label):", class_counts)
    print("distinct source tiles in selection:", n_tiles_selected)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Murray Global Intertidal Change (tidal flats)",
            "task_type": "classification",
            "source": "Nature / intertidal.app (JCU/UQ, Murray et al.)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://www.intertidal.app/",
                "have_locally": False,
                "annotation_method": "manual training points + supervised Landsat classification",
                "citation": "Murray et al. 2019, Nature 565:222-225, doi:10.1038/s41586-018-0805-8",
                "product": "Global Intertidal Change Classification v1.2 (1999-2019)",
                "download": "Figshare collection 5884598",
                "epoch": EPOCH,
                "year": YEAR,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "BINARY product: the distributed v1.2 global_intertidal raster contains only "
                "values {0,1} across all 108 global tiles (verified), so this is a 2-class "
                "tidal-flat/other segmentation, NOT the 3-class [tidal flat, permanent water, "
                "other] the manifest anticipated -- permanent water is not separated in the "
                "published GeoTIFF and is merged into 'other'. Bounded sampling of the "
                "2017-2019 epoch (matches manifest range 2016-2019): only the single 9.78 GB "
                "epoch zip was downloaded (not the full 56 GB). 64x64 windows reprojected from "
                "native 30 m EPSG:4326 to local UTM at 10 m with nearest resampling. "
                "Tiles-per-class balanced across the global coastline (per-tile cap for "
                "diversity): tidal-flat windows (>=5% tidal flat) carry both classes; 'other' "
                "windows are coastal negatives adjacent to tidal flat (not open ocean). "
                "1-year time range anchored on the epoch center; task=classification."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
