"""Process the AI4Arctic / ASIP Sea Ice Dataset (v2) into open-set-segmentation labels.

Source: DTU Data (figshare) record 13011134, "AI4Arctic / ASIP Sea Ice Dataset -
version 2" (ASID-v2), CC-BY. 452 NetCDF scenes (~333 GB total): each pairs a
Sentinel-1 EW SAR scene (HH/HV, ~40 m, native swath geometry) with the operational
Greenland ice chart manually drawn by ice analysts at DMI, gridded to the SAR grid,
plus AMSR2 brightness temperatures. Scenes are 2018-2019, 9 regions around Greenland
(all post-2016).

PRIMARY PRODUCT = per-pixel sea-ice-CONCENTRATION REGRESSION (0-100 %).
------------------------------------------------------------------------
We regress the TOTAL sea-ice concentration (SIC), the cleanest AI4Arctic target: it maps
unambiguously from the single ``CT`` field of each ice-chart polygon (SIGRID-3 code).
Output is a single-band float32 label patch (percent 0-100), nodata ``io.REGRESSION_NODATA``
(-99999). Stage-of-development (SOD, ice type) and form/floe (FLOE) are ALSO encoded in the
polygon codes and could be produced later as a classification companion (they need
partial-concentration weighting), but SIC is the recommended primary target and the one we
build here. See the summary for the choice rationale.

We use ONLY the ice-chart label (``polygon_icechart`` + ``polygon_codes``) + the SAR
geolocation grid (``sar_grid_line/sample/latitude/longitude``) from each NetCDF. We do NOT
use the SAR or AMSR2 imagery -- pretraining supplies its own imagery. The full archive is
huge (~333 GB), so we do NOT bulk-download. NetCDF4 is HDF5, and the ice-chart variable is
gzip-compressed to ~30 KB, so we use **HTTP Range requests** (``download.HttpRangeFile`` +
h5py) to selectively read ONLY those few variables from each remote file -- ~60 KB per scene
instead of ~500 MB (spec 5/8 selective extraction). The extracted arrays are cached to a
small per-scene ``.npz`` in ``raw/{slug}/`` so re-runs are offline + idempotent. We sample a
bounded, seasonally- and regionally-stratified set of scenes (``N_SCENES``).

Concentration mapping (SIGRID-3 ``CT`` -> percent). ``polygon_icechart`` is a raster of
polygon ids; ``polygon_codes`` is the per-polygon SIGRID-3 attribute table
(``id;CT;CA;SA;FA;CB;SB;FB;CC;SC;FC;CN;CD;CF;POLY_TYPE``). We read the TOTAL concentration
``CT`` of each polygon:

    CT 00/01/02 (ice-free / <1/10 / bergy water) -> 0 %
    CT 10..90 (k/10)                             -> k*10 %
    CT 91 (9+/10 .. <10/10)                      -> 95 %
    CT 92 (10/10, incl. fast/compact ice)        -> 100 %
    CT "ab" range codes (a<=b, e.g. 46=4..6/10)  -> midpoint*10 %
    CT 99 / negative / undetermined              -> nodata

Resolution / label-generalization CAVEAT. Ice charts are drawn MANUALLY by ice analysts as
generalized polygons over large areas; the effective native resolution is coarse (many km,
not the SAR pixel), so a "10 m" label here is an UPSAMPLED coarse polygon field, not a fine
per-pixel measurement. We warp with **nearest** resampling (categorical polygon field) and
tile to 10 m only to meet the common output spec; the true information content is coarse.
This is documented in ``metadata.json`` and the summary.

Georeferencing. The ice chart is in SAR *swath* geometry (not a regular CRS grid). Each
NetCDF carries a coarse line/sample -> lon/lat geolocation grid. We build GCPs from that grid
and warp the concentration raster to a scene-local UTM projection at 10 m/pixel (UTM zone
from the scene-mean lon/lat), nearest resampling. We then tile the warped raster into 64x64
patches.

Time range. A sea-ice chart is valid only around its SAR acquisition, and ice is DYNAMIC, so
we treat it as state-at-time (spec 5): ``change_time=null`` with a TIGHT +/-3-day window
(6 days total) centered on the acquisition timestamp parsed from the filename (e.g.
``20190519T194908``). This is well under the 360-day pretraining cap and short enough that
the regional concentration field is roughly stable while still giving S1/S2 (very frequent
at Arctic latitudes) a chance to pair. All scenes are 2018-2019 (post-2016).

Sampling: regression, up to ``TOTAL`` (5000) tiles, **fixed-bucket balanced across the mean
concentration** (the SIC distribution is strongly bimodal: lots of open water 0 % and lots of
compact pack ice 100 %, fewer intermediate) so the corpus spans the full concentration range.
Tiles that are >50 % nodata are dropped.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ai4arctic_asip_sea_ice_dataset
"""

import argparse
import multiprocessing
import random
import re
import time as _time
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
from affine import Affine
from rasterio.control import GroundControlPoint as GCP
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "ai4arctic_asip_sea_ice_dataset"
NAME = "AI4Arctic / ASIP Sea Ice Dataset"

FIGSHARE_ARTICLE = "13011134"
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"

# Bounded scene sample: seasonally (12 months) x regionally stratified, preferring the
# smallest file per (month, region) cell to cap the download volume. ~3 scenes/month.
N_SCENES = 36
PER_MONTH = 3

TILE = 64
TOTAL = 5000  # regression per-dataset target (<= 25k cap)
CAND_PER_SCENE = 6000  # cap candidate tiles kept per scene (bounds memory; >> needed)
MAX_NODATA_FRAC = 0.5  # drop tiles that are more than half nodata
HALF_WINDOW_DAYS = 3  # tight +/-3-day window around the acquisition (dynamic ice)
SEED = 42

# Fixed mean-concentration bucket edges (percent). The SIC distribution is strongly bimodal
# (open water 0 % and compact pack ice 100 % dominate), so quantile buckets degenerate;
# fixed 10-% buckets give an even spread of concentration levels across the corpus.
BUCKET_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0001]

VALUE_NAME = "sea_ice_concentration"
VALUE_UNIT = "percent"

_NAME_RE = re.compile(r"(\d{8})T(\d{6})_(S1[AB])_AMSR2_Icechart-(.+)\.nc")


def ct_to_sic_percent(ct: int) -> float | None:
    """Map a SIGRID-3 total-concentration code (CT) to a percent (0-100), or None.

    None => undetermined/unknown (mapped to nodata). Handles clean deciles (10..90),
    open-water codes (0/1/2 -> 0 %), the 9+ and 10/10 codes (91 -> 95, 92 -> 100), and
    two-digit range codes "ab" (a<=b, e.g. 46 = 4/10..6/10 -> midpoint 50 %).
    """
    ct = int(ct)
    if ct < 0 or ct == 99:
        return None
    if ct in (0, 1, 2):
        return 0.0
    if ct == 91:
        return 95.0
    if ct == 92:
        return 100.0
    a, b = ct // 10, ct % 10
    if b == 0 and 1 <= a <= 9:
        return float(a * 10)
    if 1 <= a <= 9 and a <= b <= 9:
        return float((a + b) / 2 * 10)
    return None


def _acq_time(name: str) -> datetime:
    m = _NAME_RE.match(name)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=UTC
    )


def fetch_file_list() -> list[dict[str, Any]]:
    """Return the parsed NetCDF file list from the figshare record."""
    with urllib.request.urlopen(FIGSHARE_API, timeout=120) as r:
        import json

        meta = json.loads(r.read())
    out = []
    for f in meta["files"]:
        m = _NAME_RE.match(f["name"])
        if not m:
            continue
        out.append(
            {
                "name": f["name"],
                "size": f["size"],
                "url": f["download_url"],
                "date": m.group(1),
                "month": m.group(1)[4:6],
                "region": m.group(4),
            }
        )
    return out


def select_scenes(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic bounded sample: stratify by month, prefer smallest file per region.

    Within each calendar month, pick up to ``PER_MONTH`` scenes from distinct regions
    (smallest file first). This gives seasonal (concentration) + regional diversity while
    minimizing download volume.
    """
    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in files:
        by_month[f["month"]].append(f)
    selected: list[dict[str, Any]] = []
    for month in sorted(by_month):
        scenes = sorted(by_month[month], key=lambda r: (r["size"], r["name"]))
        seen: set[str] = set()
        count = 0
        for s in scenes:
            if s["region"] in seen:
                continue
            seen.add(s["region"])
            selected.append(s)
            count += 1
            if count >= PER_MONTH:
                break
    selected.sort(key=lambda r: r["name"])
    return selected[:N_SCENES]


_VARS = (
    "polygon_icechart",
    "polygon_codes",
    "sar_grid_line",
    "sar_grid_sample",
    "sar_grid_latitude",
    "sar_grid_longitude",
)


def _cache_path(name: str):
    return io.raw_dir(SLUG) / (name.replace(".nc", "") + ".labels.npz")


def _extract_one(rec: dict[str, Any]) -> str:
    """Selectively read the label vars from a remote NetCDF via HTTP Range; cache to npz.

    NetCDF4 == HDF5; the ice-chart variable is gzip-compressed to ~30 KB, so h5py over a
    ``HttpRangeFile`` fetches only ~60 KB rather than the ~500 MB scene. Idempotent.
    """
    import h5py

    out = _cache_path(rec["name"])
    if out.exists():
        return str(out)
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    rf = download.HttpRangeFile(rec["url"])
    try:
        f = h5py.File(rf, "r")
        pi = f["polygon_icechart"][:]
        pc = f["polygon_codes"][:]
        codes = [c.decode() if isinstance(c, bytes) else str(c) for c in pc]
        gl = f["sar_grid_line"][:]
        gs = f["sar_grid_sample"][:]
        la = f["sar_grid_latitude"][:]
        lo = f["sar_grid_longitude"][:]
        f.close()
    finally:
        rf.close()
    tmp = out.parent / (out.name + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            polygon_icechart=pi.astype(np.uint8),
            polygon_codes=np.array(codes, dtype=object),
            sar_grid_line=gl,
            sar_grid_sample=gs,
            sar_grid_latitude=la,
            sar_grid_longitude=lo,
        )
    tmp.rename(out)
    return str(out)


def _warp_scene(path: str):
    """Load a scene's cached ice chart and warp it to scene-local UTM 10 m.

    Returns (pct_array uint8 with 255 nodata, Projection, col0, row0), where each valid
    pixel is the sea-ice concentration percent (0..100) and (col0, row0) are the rslearn
    integer pixel coords of the warped array's top-left under the returned projection.
    (We keep the warped scene as uint8 -- percent fits 0..100, 255=nodata -- to halve
    memory vs float32; the small 64x64 output tiles are cast to float32 at write time.)
    """
    z = np.load(path, allow_pickle=True)
    pi = z["polygon_icechart"]  # uint8 polygon ids; fill value 0 = unobserved/land
    codes = z["polygon_codes"]
    gl = z["sar_grid_line"]
    gs = z["sar_grid_sample"]
    la = z["sar_grid_latitude"]
    lo = z["sar_grid_longitude"]

    # polygon id -> CT (total concentration) from the SIGRID-3 code table (row 0 = header).
    # Fill value 0 matches no polygon id, so unobserved pixels stay nodata automatically.
    id_to_ct: dict[int, int] = {}
    for c in codes[1:]:
        fields = str(c).split(";")
        id_to_ct[int(fields[0])] = int(fields[1])

    src = np.full(pi.shape, io.CLASS_NODATA, dtype=np.uint8)  # 255 = nodata here
    for pid, ct in id_to_ct.items():
        pct = ct_to_sic_percent(ct)
        if pct is not None:
            src[pi == pid] = int(round(pct))

    gcps = [
        GCP(row=float(ln), col=float(sm), x=float(x), y=float(y))
        for ln, sm, x, y in zip(gl, gs, lo, la)
    ]
    clon = float(np.mean(lo))
    clat = float(np.mean(la))
    zone = int((clon + 180) // 6) + 1
    epsg = (32600 if clat >= 0 else 32700) + zone
    utm = CRS.from_epsg(epsg)

    xs, ys = warp_transform(CRS.from_epsg(4326), utm, lo.tolist(), la.tolist())
    xmin = np.floor(min(xs) / io.RESOLUTION) * io.RESOLUTION
    xmax = np.ceil(max(xs) / io.RESOLUTION) * io.RESOLUTION
    ymin = np.floor(min(ys) / io.RESOLUTION) * io.RESOLUTION
    ymax = np.ceil(max(ys) / io.RESOLUTION) * io.RESOLUTION
    w = int((xmax - xmin) / io.RESOLUTION)
    h = int((ymax - ymin) / io.RESOLUTION)
    dst_t = Affine(io.RESOLUTION, 0, xmin, 0, -io.RESOLUTION, ymax)

    dst = np.full((h, w), io.CLASS_NODATA, dtype=np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_crs=CRS.from_epsg(4326),
        gcps=gcps,
        dst_crs=utm,
        dst_transform=dst_t,
        src_nodata=io.CLASS_NODATA,
        dst_nodata=io.CLASS_NODATA,
        resampling=Resampling.nearest,
    )
    proj = Projection(utm, io.RESOLUTION, -io.RESOLUTION)
    col0 = int(round(xmin / io.RESOLUTION))
    row0 = int(round(ymax / -io.RESOLUTION))
    return dst, proj, col0, row0


def _scan_scene(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate 64x64 tiles of a scene: (ti, tj, mean concentration).

    Vectorized per tile-row band (memory-light). A tile is a candidate if it is <=50 %
    nodata; its ``value`` is the mean concentration over valid pixels (used for the
    regression bucket-balance). Candidates are randomly subsampled to CAND_PER_SCENE.
    """
    arr, _proj, _c0, _r0 = _warp_scene(rec["path"])
    H, W = arr.shape
    nty, ntx = H // TILE, W // TILE
    if nty == 0 or ntx == 0:
        return []
    tot = TILE * TILE
    tis: list[np.ndarray] = []
    tjs: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for ti in range(nty):
        band = arr[ti * TILE : (ti + 1) * TILE, : ntx * TILE]
        # (TILE, ntx*TILE) -> (ntx, TILE*TILE): each row is one tile flattened.
        band = band.reshape(TILE, ntx, TILE).transpose(1, 0, 2).reshape(ntx, tot)
        nd = band == io.CLASS_NODATA
        nd_frac = nd.mean(axis=1)
        valid_counts = tot - nd.sum(axis=1)
        fv = band.astype(np.float32)
        fv[nd] = 0.0
        means = fv.sum(axis=1) / np.maximum(valid_counts, 1)
        keep = np.nonzero(nd_frac <= MAX_NODATA_FRAC)[0]
        if keep.size:
            tis.append(np.full(keep.size, ti, dtype=np.int32))
            tjs.append(keep.astype(np.int32))
            vals.append(means[keep].astype(np.float32))
    if not tis:
        return []
    ti_a = np.concatenate(tis)
    tj_a = np.concatenate(tjs)
    val_a = np.concatenate(vals)
    if ti_a.size > CAND_PER_SCENE:
        rng = np.random.default_rng(abs(hash(rec["name"])) % (2**32))
        sel = rng.choice(ti_a.size, size=CAND_PER_SCENE, replace=False)
        ti_a, tj_a, val_a = ti_a[sel], tj_a[sel], val_a[sel]
    return [
        {
            "name": rec["name"],
            "path": rec["path"],
            "ti": int(ti),
            "tj": int(tj),
            "value": float(v),
        }
        for ti, tj, v in zip(ti_a, tj_a, val_a)
    ]


def bucket_balance_fixed(
    records: list[dict[str, Any]], edges: list[float], total: int, seed: int = SEED
) -> list[dict[str, Any]]:
    """Balance across fixed [edge_i, edge_{i+1}) value buckets (bimodal SIC data).

    Take up to total//n_buckets per bucket, then top up from leftovers until ``total``.
    """
    n = len(edges) - 1
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for r in records:
        b = int(np.searchsorted(edges, r["value"], side="right")) - 1
        buckets[min(max(b, 0), n - 1)].append(r)
    rng = random.Random(seed)
    for b in buckets:
        rng.shuffle(b)
    per = max(1, total // n)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for b in buckets:
        selected.extend(b[:per])
        leftovers.extend(b[per:])
    if len(selected) < total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])
    return selected[:total]


def _write_scene(
    path: str, name: str, tiles: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Re-warp a scene and write its selected tiles + sidecars (idempotent).

    Returns per-tile stats {sample_id, n_valid, min, max, mean} for the metadata summary.
    """
    stats: list[dict[str, Any]] = []
    if not tiles:
        return stats
    acq = _acq_time(name)
    tr = io.centered_time_range(acq, half_window_days=HALF_WINDOW_DAYS)

    need_warp = not all(
        (io.locations_dir(SLUG) / f"{t['sample_id']}.tif").exists() for t in tiles
    )
    arr = proj = col0 = row0 = None
    if need_warp:
        arr, proj, col0, row0 = _warp_scene(path)

    for t in tiles:
        sample_id = t["sample_id"]
        tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
        ti, tj = t["ti"], t["tj"]
        if tif.exists():
            stats.append({"sample_id": sample_id, "value": t["value"]})
            continue
        sub_u8 = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
        nd = sub_u8 == io.CLASS_NODATA
        sub = sub_u8.astype(np.float32)
        sub[nd] = io.REGRESSION_NODATA
        good = sub_u8[~nd]
        if good.size == 0:
            continue
        x_min = col0 + tj * TILE
        y_min = row0 + ti * TILE
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        io.write_label_geotiff(
            SLUG, sample_id, sub, proj, bounds, nodata=io.REGRESSION_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            tr,
            change_time=None,
            source_id=f"{name}_r{ti}_c{tj}",
        )
        stats.append(
            {
                "sample_id": sample_id,
                "value": t["value"],
                "n_valid": int(good.size),
                "min": float(good.min()),
                "max": float(good.max()),
            }
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--scan-workers", type=int, default=24)
    parser.add_argument("--n-scenes", type=int, default=N_SCENES)
    parser.add_argument("--extract-retries", type=int, default=4)
    parser.add_argument("--extract-pause", type=float, default=6.0)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Fetching figshare file list...")
    try:
        files = fetch_file_list()
        print(f"  {len(files)} NetCDF scenes in record")
        scenes = select_scenes(files)[: args.n_scenes]
        print(
            f"  selected {len(scenes)} scenes "
            f"({sum(s['size'] for s in scenes) / 1e9:.1f} GB), stratified by month/region"
        )
    except Exception as e:  # noqa: BLE001 - figshare down: fall back to already-cached npz
        print(f"  figshare listing failed ({str(e)[:80]}); falling back to cached npz")
        cached = sorted(io.raw_dir(SLUG).glob("*.labels.npz"))
        scenes = [
            {"name": p.name.replace(".labels.npz", ".nc"), "url": None} for p in cached
        ][: args.n_scenes]
        if not scenes:
            manifest.write_registry_entry(
                SLUG,
                "temporary_failure",
                notes="figshare listing unreachable and no cached npz. Retry later.",
            )
            raise SystemExit(
                "figshare unreachable, no cache; recorded temporary_failure"
            )

    print("Extracting label vars via HTTP Range (h5py); caching to npz (idempotent)...")
    # HTTP Range extraction is done SERIALLY with pacing: NetCDF4=HDF5 and the DTU/figshare
    # host rate-limits many small range requests, so a paced serial loop avoids tripping the
    # per-IP quota. Scenes that still fail are skipped (not fatal); re-running resumes from
    # cache. We proceed with whatever is cached (>= a few scenes is plenty for 5000 tiles).
    ok_scenes: list[dict[str, Any]] = []
    for s in scenes:
        cp = _cache_path(s["name"])
        if not cp.exists():
            for attempt in range(args.extract_retries):
                try:
                    _extract_one(s)
                    break
                except Exception as e:  # noqa: BLE001 - transient host throttle
                    print(f"    extract failed ({s['name']}): {str(e)[:80]}")
                    _time.sleep(args.extract_pause * (attempt + 1))
            _time.sleep(args.extract_pause)
        if cp.exists():
            s["path"] = str(cp)
            ok_scenes.append(s)
    scenes = ok_scenes
    if not scenes:
        manifest.write_registry_entry(
            SLUG,
            "temporary_failure",
            notes="DTU/figshare host rate-limited HTTP Range extraction; no scenes cached. Retry later.",
        )
        raise SystemExit(
            "no scenes extracted (host throttling); recorded temporary_failure"
        )
    total_mb = sum(_cache_path(s["name"]).stat().st_size for s in scenes) / 1e6
    print(f"  {len(scenes)} scene label-caches on disk ({total_mb:.2f} MB total)")
    io.check_disk()

    print("Scanning scenes into 64x64 tiles (GCP warp -> UTM 10 m, vectorized)...")
    with multiprocessing.Pool(args.scan_workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in star_imap_unordered(p, _scan_scene, [dict(rec=s) for s in scenes]):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = bucket_balance_fixed(all_recs, BUCKET_EDGES, TOTAL)
    selected.sort(key=lambda r: (r["name"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_vals = np.array([r["value"] for r in selected], dtype=np.float64)
    bcounts = Counter(
        min(
            max(int(np.searchsorted(BUCKET_EDGES, v, side="right")) - 1, 0),
            len(BUCKET_EDGES) - 2,
        )
        for v in sel_vals
    )
    print(
        f"  selected {len(selected)} tiles (regression, fixed mean-concentration buckets);"
        f" bucket counts {dict(sorted(bcounts.items()))}"
    )

    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_scene[r["path"]].append(r)

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    io.check_disk()
    print(f"Writing tiles for {len(by_scene)} scenes...")
    write_args = [
        dict(path=pth, name=ts[0]["name"], tiles=ts) for pth, ts in by_scene.items()
    ]
    stats: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for st in star_imap_unordered(p, _write_scene, write_args):
            stats.extend(st)

    written = [s for s in stats if s.get("n_valid", 0) > 0]
    # On an idempotent re-run tiles already existed (no n_valid) -- count tifs on disk.
    n_on_disk = sum(1 for _ in io.locations_dir(SLUG).glob("*.tif"))
    n_written = len(written) if written else n_on_disk
    pix_min = min((s["min"] for s in written), default=0.0)
    pix_max = max((s["max"] for s in written), default=100.0)
    print(f"  wrote {len(written)} new tiles; {n_on_disk} tifs on disk total")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "DTU Data (figshare 13011134), AI4Arctic / ASIP Sea Ice Dataset v2 (ASID-v2)",
            "license": "CC-BY",
            "provenance": {
                "url": "https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134",
                "have_locally": False,
                "annotation_method": (
                    "manual operational ice charts (DMI ice analysts), SIGRID-3 polygon "
                    "codes co-registered to Sentinel-1 SAR grid"
                ),
                "access": (
                    "HTTP Range (h5py over NetCDF4/HDF5) selective read of the ice-chart "
                    "label vars only (~60 KB/scene vs ~500 MB); no imagery downloaded"
                ),
            },
            "sensors_relevant": ["sentinel1", "sentinel2"],
            "regression": {
                "name": VALUE_NAME,
                "description": (
                    "Per-pixel TOTAL sea-ice concentration (0-100 %) from the manually-drawn "
                    "Greenland operational ice charts of the AI4Arctic/ASIP v2 dataset "
                    "(SIGRID-3 CT code of each ice-chart polygon: 00/01/02->0, k0->k*10, "
                    "91->95, 92->100, range 'ab'->midpoint). Ice charts are generalized "
                    "analyst-drawn polygons; native effective resolution is coarse (km-scale), "
                    "upsampled to 10 m by nearest resampling -- the label is a coarse polygon "
                    "field, not a fine per-pixel measurement."
                ),
                "unit": VALUE_UNIT,
                "dtype": "float32",
                "value_range": [round(float(pix_min), 3), round(float(pix_max), 3)],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": BUCKET_EDGES,
            },
            "num_samples": n_written,
            "n_scenes": len(scenes),
            "notes": (
                "Per-pixel sea-ice CONCENTRATION regression (0-100 %) -- the recommended "
                "primary AI4Arctic target (maps unambiguously from the single CT field). "
                "Stage-of-development (SOD) and form (FLOE) are also in the polygon codes and "
                "could be produced later as a classification companion. Only the ice-chart "
                "label (polygon_icechart + polygon_codes) and the SAR geolocation grid were "
                "used; SAR/AMSR2 imagery was NOT downloaded. The 333 GB archive was NOT "
                "bulk-downloaded: for a bounded month/region-stratified scene sample we used "
                "HTTP Range requests (NetCDF4=HDF5, gzip chart ~30 KB) to read only the label "
                "vars via h5py (~60 KB/scene). Ice chart is in SAR swath geometry; warped to "
                "scene-local UTM 10 m via the geolocation grid as GCPs (nearest resampling; "
                "coarse native res upsampled to 10 m -- see resolution caveat). Tiled into "
                "64x64; regression fixed-bucket balanced across mean concentration (bimodal: "
                "open water 0 % and pack ice 100 % dominate); tiles >50 % nodata dropped. "
                "time_range = tight +/-3-day window centered on the SAR acquisition (sea ice "
                "is dynamic; state-at-time, change_time null). All scenes 2018-2019 (post-2016)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=n_written
    )
    print(f"num_samples={n_written} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
