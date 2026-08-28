"""Process CloudSEN12 (high-quality tier) into open-set-segmentation cloud/shadow tiles.

Source: CloudSEN12 / CloudSEN12+ (Aybar et al. 2022, 2024) -- a large global benchmark of
Sentinel-2 patches with hand-crafted pixel labels for cloud and cloud-shadow semantic
segmentation. Distributed as cloud-optimized "tortilla" files on the HuggingFace repo
``tacofoundation/cloudsen12`` and read lazily with the legacy ``tacoreader`` client
(``pip install 'tacoreader<1.0'``); mirrors on Zenodo (record 7034410) and ScienceDataBank.

We use only the **high-quality manual** label tier (``label_type == "high"``) at the
standard **509x509, 10 m** patch size (``real_proj_shape == 509``): 10,000 patches spread
across all continents except Antarctica, each a single Sentinel-2 L1C acquisition
(acquisition dates span 2018-2020, all Sentinel era) already stored in its native local
UTM zone at 10 m/pixel (padded to 512x512; the extra 3 rows/cols on the bottom/right are
0-fill and are never read).

**Download strategy (robust to HF rate limits).** The tortilla index is built once with
tacoreader and cached to ``raw/{slug}/index.parquet``; thereafter it is reconstructed from
that parquet with no HuggingFace request (the HF index build is heavy and the first thing
anonymous rate-limiting kills). For each patch we then issue a SINGLE HTTP byte-range GET
for its tortilla blob, parse the nested footer locally, slice out the tiny single-band
"target" label GeoTIFF, and discard the S2 imagery bytes (pretraining supplies imagery).
This keeps us to ~1 HF request/patch (vs tacoreader's multi-request vsicurl path) and is
trivially resumable via a per-patch ``.npy`` cache. Because CloudSEN12 is large, we
download only a bounded, class-diverse subset of patches (``--max-patches``, default 2500),
prioritizing the rarest class (thin cloud) so every class reaches its target (spec 5).

IMPORTANT indexing note: ``tacoreader.load`` sorts rows by ``tortilla:id`` but leaves the
pandas index labels randomized (they differ per load), while ``TortillaDataFrame.read(i)``
is positional. We therefore never rely on ``.read(idx)`` across processes; workers fetch
labels purely from the per-patch (url, blob_offset, blob_length) parsed from the index.
Every read is validated to contain only class ids {0,1,2,3} (the scribble tier's 0..6
scheme / 99 fill would indicate a wrong read and is rejected).

Label semantics (CloudSEN12 4-class manual scheme, Table 2 of the Data-in-Brief paper):
    0 = clear, 1 = thick cloud, 2 = thin cloud, 3 = cloud shadow.
These map directly to output class ids 0..3 (uint8). Every pixel in a "high" patch is
labeled, so no nodata occurs in practice; nodata sentinel 255 is still declared for the
open-set contract.

Recipe (spec 4 dense_raster, spec 5 tiles-per-class balanced): each 509x509 label is tiled
into 64x64 patches on an 8x8 grid that evenly covers the valid extent. A tile counts toward
every class present in it; tiles are selected rarest-class-first up to 1000 tiles/class,
25k total. This is a per-image (transient) label -- a cloud mask describes ONE Sentinel-2
acquisition -- so ``change_time`` is null and the time_range is a short window CENTERED on
the acquisition date (io.centered_time_range, +/-15 days -> ~1 month, well under the 1-year
cap; a tight window keeps paired imagery temporally near the labeled scene per spec 5's
"specific-image label" rule while respecting the prompt's "<=1-year, centered on
acquisition" instruction).

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cloudsen12
"""

import argparse
import math
import multiprocessing
import os
import random
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "cloudsen12"
NAME = "CloudSEN12"
TACO = "tacofoundation:cloudsen12-l1c"
URL = "https://huggingface.co/datasets/tacofoundation/cloudsen12"

TILE = io.MAX_TILE  # 64
VALID = 509  # native labeled extent (raster is padded to 512 with 0s)
RES = 10.0
PER_CLASS = 1000
SEED = 42
HALF_WINDOW_DAYS = 15  # +/-15d ~ 1 month window centered on the S2 acquisition

# CloudSEN12 4-class manual scheme -> output ids 0..3 (identity map).
CLASSES: list[tuple[int, str, str]] = [
    (
        0,
        "clear",
        "Pixels free of clouds and cloud shadows: clear land or clear water surfaces with "
        "unobstructed view of the ground.",
    ),
    (
        1,
        "thick cloud",
        "Opaque (thick) clouds that fully block the surface reflectance so no ground signal "
        "is observable through them.",
    ),
    (
        2,
        "thin cloud",
        "Semi-transparent / thin clouds (e.g. cirrus, haze) that partially transmit the "
        "surface signal; the ground is still partly visible through them.",
    ),
    (
        3,
        "cloud shadow",
        "Shadows cast on the surface by clouds, appearing as darkened regions adjacent to "
        "the clouds that produce them.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 4


def _labels_cache_dir() -> str:
    return os.path.join(io.raw_dir(SLUG).path, "labels")


def _cache_path(patch_id: str) -> str:
    return os.path.join(_labels_cache_dir(), f"{patch_id}.npy")


def _tile_offsets(size: int = VALID, tile: int = TILE) -> list[int]:
    """Evenly-spaced top-left offsets tiling [0, size) with tile-sized windows.

    For size=509, tile=64 -> 8 offsets [0,64,127,191,254,318,381,445] (last tile ends at
    509); windows overlap by a few px at most and never touch the padded border.
    """
    if size <= tile:
        return [0]
    n = math.ceil(size / tile)
    last = size - tile
    return sorted({int(round(i * last / (n - 1))) for i in range(n)})


OFFSETS = _tile_offsets()

# ---------------------------------------------------------------------------------------
# Worker: each process lazily loads the (remote) tortilla index once.
_DS = None


TACOS_JSON_URL = (
    "https://huggingface.co/datasets/tacofoundation/main/raw/main/tacos.json"
)


def _tacos_cache() -> str:
    return os.path.join(io.raw_dir(SLUG).path, "tacos.json")


def _ensure_tacos_json(retries: int = 8) -> str:
    """Download the tacofoundation registry (tacos.json) to a local cache (parent-side).

    64 workers each hitting the HF *raw* endpoint gets rate-limited (returns HTML, not
    JSON). Fetch it once here so workers read the local copy instead.
    """
    import json
    import time

    import requests

    p = _tacos_cache()
    if os.path.exists(p):
        return p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(TACOS_JSON_URL, timeout=60)
            data = r.json()
            with open(p + ".tmp", "w") as f:
                json.dump(data, f)
            os.replace(p + ".tmp", p)
            return p
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {TACOS_JSON_URL}: {last_err!r}")


def _index_parquet() -> str:
    return os.path.join(io.raw_dir(SLUG).path, "index.parquet")


def _load_taco(retries: int = 6):
    """Load the tortilla index, caching the fully-built dataframe to a local parquet.

    ``tacoreader.load`` reconstructs the index by range-reading the footer of every
    ``*.part.taco`` file on HuggingFace and concatenating them. That step is heavy and is
    the first thing HF anonymous rate-limiting (HTTP 429) kills. It is also completely
    static, so we cache the resulting dataframe to ``raw/{slug}/index.parquet`` on the
    first successful load and reconstruct the ``TortillaDataFrame`` from that parquet on
    every subsequent call -- no HF request needed. This makes the index load fast, robust,
    and offline once primed. (Note: ``tacoreader.load`` sorts rows by ``tortilla:id`` but
    keeps randomized pandas index labels; the ROW ORDER is what is stable/portable, so we
    always index positionally.)
    """
    import time

    import pandas as pd
    from tacoreader.v1.TortillaDataFrame import TortillaDataFrame

    cache = _index_parquet()
    if os.path.exists(cache):
        return TortillaDataFrame(pd.read_parquet(cache))

    import json

    import tacoreader
    import tacoreader.v1.loader_dataframe as ldf

    cached = json.load(open(_ensure_tacos_json()))
    ldf.load_tacofoundation_datasets = lambda: cached  # avoid the flaky raw fetch
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            ds = tacoreader.load(TACO)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            tmp = cache + ".tmp"
            pd.DataFrame(ds).to_parquet(tmp)
            os.replace(tmp, cache)
            return ds
        except Exception as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(
        f"tacoreader.load({TACO}) failed after {retries} tries: {last_err!r}"
    )


def _init_worker() -> None:
    """Workers fetch labels via direct HTTP range GETs (see _load_or_fetch_label) and do
    NOT need the tortilla index, so there is nothing to load here.
    """
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")


VALID_LABELS = frozenset({0, 1, 2, 3})  # high-tier scheme: clear/thick/thin/shadow only


def _validate_label(arr: np.ndarray) -> bool:
    """A correctly-read high-tier label contains only values in {0,1,2,3}.

    Any other value (4..6 => scribble-tier scheme, 99/255 => fill) means we read the
    WRONG asset/row and the array must be rejected (never written as a label).
    """
    return bool(set(np.unique(arr).tolist()) <= {0, 1, 2, 3})


_SUBFILE_RE = re.compile(r"/vsisubfile/(\d+)_(\d+),(.+)")


def parse_subfile(subfile: str) -> tuple[int, int, str]:
    """Parse a tortilla ``internal:subfile`` into (blob_offset, blob_length, url).

    e.g. ``/vsisubfile/8229850412_1477533,/vsicurl/https://.../cloudsen12-l1c.0001.part.taco``
    -> (8229850412, 1477533, "https://.../cloudsen12-l1c.0001.part.taco").
    """
    m = _SUBFILE_RE.match(subfile)
    if not m:
        raise ValueError(f"unrecognized internal:subfile {subfile!r}")
    off, length, path = m.groups()
    if path.startswith("/vsicurl/"):
        path = path[len("/vsicurl/") :]
    return int(off), int(length), path


def _http_range(
    url: str, start: int, length: int, retries: int = 6, timeout: int = 120
) -> bytes:
    """HTTP byte-range GET with exponential backoff, honoring HF 429 rate-limits.

    HuggingFace anonymous access is limited to ~3000 resolver requests / 300 s; on 429 we
    back off (respecting Retry-After / the fixed 300 s window) and retry, so a large run
    self-throttles instead of failing. Returns exactly the requested ``length`` bytes.
    """
    import time

    import requests

    headers = {"Range": f"bytes={start}-{start + length - 1}"}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 0)) or min(
                    60.0, 5.0 * (attempt + 1)
                )
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.content
            if len(data) < length:
                raise OSError(f"short read {len(data)} < {length} for {url}")
            return data[:length]
        except Exception as e:
            last_err = e
            time.sleep(min(30.0, 2.0 * (attempt + 1)))
    raise RuntimeError(
        f"range GET failed for {url} [{start}:{start + length}]: {last_err!r}"
    )


def _extract_label_from_blob(blob: bytes) -> np.ndarray:
    """Extract the 509x509 uint8 cloud label from a downloaded tortilla blob.

    The blob is itself a Tortilla (magic ``#y``/``WX``) whose footer (a parquet) lists its
    inner assets (``s2l1c`` image + ``target`` label) at offsets RELATIVE to the blob start.
    We parse the footer locally, slice the tiny ``target`` GeoTIFF out of the blob, and read
    it with a rasterio MemoryFile -- no further network I/O.
    """
    from pyarrow import BufferReader
    from pyarrow.parquet import read_table
    from rasterio.io import MemoryFile

    if blob[:2] not in (b"#y", b"WX"):
        raise ValueError("blob is not a Tortilla (bad magic)")
    footer_offset = int.from_bytes(blob[2:10], "little")
    footer_length = int.from_bytes(blob[10:18], "little")
    footer = read_table(
        BufferReader(blob[footer_offset : footer_offset + footer_length])
    ).to_pandas()
    tgt = footer[footer["tortilla:id"] == "target"]
    if len(tgt) == 0:
        raise ValueError("no 'target' asset in tortilla footer")
    row = tgt.iloc[0]
    off, length = int(row["tortilla:offset"]), int(row["tortilla:length"])
    with MemoryFile(blob[off : off + length]) as mf, mf.open() as src:
        arr = src.read(1)
    return np.ascontiguousarray(arr[:VALID, :VALID]).astype(np.uint8)


def _load_or_fetch_label(task: dict[str, Any], retries: int = 4) -> np.ndarray:
    """Return the 509x509 uint8 label for a patch, using a local .npy cache.

    Fetches the label with a SINGLE HTTP range GET of the patch's tortilla blob (then parses
    the label out locally), instead of tacoreader's multi-request vsicurl path -- this keeps
    us far under HuggingFace's per-request rate limit and is trivially resumable. Reads are
    validated against the 4-class scheme; a cached file that fails validation (e.g. from an
    older buggy run) is discarded and re-fetched.
    """
    import time

    patch_id = task["patch_id"]
    cp = _cache_path(patch_id)
    if os.path.exists(cp):
        cached = np.load(cp)
        if _validate_label(cached):
            return cached
        os.remove(cp)  # stale/corrupt cache from a buggy run -> re-fetch
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            blob = _http_range(task["url"], task["blob_offset"], task["blob_length"])
            arr = _extract_label_from_blob(blob)
            if not _validate_label(arr):
                raise ValueError(
                    f"label for {patch_id} has out-of-range values "
                    f"{np.unique(arr).tolist()} (expected subset of {{0,1,2,3}})"
                )
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            tmp = cp + f".{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, cp)
            return arr
        except Exception as e:  # transient network/decoding errors -> retry
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to read label for {patch_id}: {last_err!r}")


def _scan_patch(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Download+cache a patch label and emit one lightweight record per 64x64 tile.

    Never propagates an exception across the pool boundary (some remote-read exceptions
    are unpicklable). On persistent failure returns [] so the patch is simply skipped.
    """
    try:
        arr = _load_or_fetch_label(task)
    except Exception as e:
        print(f"WARN scan skip {task['patch_id']}: {e!r}", flush=True)
        return []
    out: list[dict[str, Any]] = []
    for tr in OFFSETS:
        for tc in OFFSETS:
            win = arr[tr : tr + TILE, tc : tc + TILE]
            if win.shape != (TILE, TILE):
                continue
            present = sorted(int(v) for v in np.unique(win))
            out.append(
                {
                    "patch_id": task["patch_id"],
                    "crs": task["crs"],
                    "x0": task["x0"],
                    "y0": task["y0"],
                    "ts": task["ts"],
                    "tr": tr,
                    "tc": tc,
                    "classes_present": present,
                    "source_id": f"{task['patch_id']}_r{tr}_c{tc}",
                }
            )
    return out


def _write_patch(task: dict[str, Any]) -> list[tuple[str, list[int]]]:
    """Write all selected tiles of one patch (one local cache read)."""
    tiles = task["tiles"]
    todo = [
        t
        for t in tiles
        if not (io.locations_dir(SLUG) / f"{t['sample_id']}.tif").exists()
    ]
    results = [(t["sample_id"], t["classes_present"]) for t in tiles]
    if not todo:
        return results
    arr = _load_or_fetch_label(task)
    proj = Projection(CRS.from_string(task["crs"]), RES, -RES)
    col_base = int(round(task["x0"] / RES))
    row_base = int(round(-task["y0"] / RES))
    center = datetime.fromtimestamp(task["ts"], tz=UTC)
    tr_range = io.centered_time_range(center, half_window_days=HALF_WINDOW_DAYS)
    for t in todo:
        r, c = t["tr"], t["tc"]
        win = np.ascontiguousarray(arr[r : r + TILE, c : c + TILE])
        col0 = col_base + c
        row0 = row_base + r
        bounds = (col0, row0, col0 + TILE, row0 + TILE)
        io.write_label_geotiff(
            SLUG, t["sample_id"], win, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            t["sample_id"],
            proj,
            bounds,
            tr_range,
            change_time=None,
            source_id=t["source_id"],
            classes_present=t["classes_present"],
        )
    return results


def _build_tasks() -> list[dict[str, Any]]:
    """Build one task per high-quality 509x509 patch.

    Each task carries everything a worker needs to fetch the label with a single HTTP range
    GET: the patch's tortilla blob (``url`` + ``blob_offset``/``blob_length`` parsed from the
    top-level ``internal:subfile``), plus georeferencing (``crs``, ``x0``, ``y0``) and the
    acquisition timestamp. Workers never touch the tortilla index, so the randomized/
    non-portable pandas index labels are irrelevant. Per-patch cloud-class percentages are
    included so ``main`` can select a class-diverse subset without reading any raster.
    """
    ds = _load_taco()
    high = ds[(ds["label_type"] == "high") & (ds["real_proj_shape"] == 509)]
    tasks: list[dict[str, Any]] = []

    def _pct(row, col):
        try:
            return float(row[col])
        except (TypeError, ValueError):
            return 0.0

    for _, row in high.iterrows():
        gt = row["stac:geotransform"]
        blob_offset, blob_length, url = parse_subfile(str(row["internal:subfile"]))
        tasks.append(
            {
                "patch_id": str(row["tortilla:id"]),
                "crs": str(row["stac:crs"]),
                "x0": float(gt[0]),
                "y0": float(gt[3]),
                "ts": float(row["stac:time_start"]),
                "url": url,
                "blob_offset": blob_offset,
                "blob_length": blob_length,
                "pct_clear": _pct(row, "clear_percentage"),
                "pct_thick": _pct(row, "thick_percentage"),
                "pct_thin": _pct(row, "thin_percentage"),
                "pct_shadow": _pct(row, "cloud_shadow_percentage"),
            }
        )
    return tasks


def select_patch_subset(
    tasks: list[dict[str, Any]], max_patches: int, seed: int = SEED
) -> list[dict[str, Any]]:
    """Choose a bounded, class-diverse subset of patches to download (spec 5: bounded
    sampling of a large product).

    CloudSEN12 is large (10k high patches); we only need enough tiles to fill <=1000/class.
    ``thin cloud`` is by far the rarest class (present in ~3.2k patches vs 6-9k for the
    others), so we greedily prioritize patches that contain the rarer classes: first those
    with thin cloud, then cloud shadow, then thick cloud, then the rest -- each stratum
    seed-shuffled for geographic/temporal spread. This guarantees the rare classes reach
    their target while keeping the total download (and HF requests) small.
    """
    if max_patches <= 0 or len(tasks) <= max_patches:
        return tasks
    rng = random.Random(seed)
    chosen: dict[str, dict[str, Any]] = {}
    strata = [
        [t for t in tasks if t["pct_thin"] > 0],
        [t for t in tasks if t["pct_shadow"] > 0],
        [t for t in tasks if t["pct_thick"] > 0],
        list(tasks),
    ]
    for stratum in strata:
        pool = [t for t in stratum if t["patch_id"] not in chosen]
        rng.shuffle(pool)
        for t in pool:
            if len(chosen) >= max_patches:
                break
            chosen[t["patch_id"]] = t
        if len(chosen) >= max_patches:
            break
    return list(chosen.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=0, help="debug: cap #patches scanned"
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=2500,
        help="bounded, class-diverse subset of high patches to download "
        "(0 = all 10k); default fills <=1000/class with a small download",
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    all_tasks = _build_tasks()
    tasks = select_patch_subset(all_tasks, args.max_patches)
    if args.limit:
        tasks = tasks[: args.limit]
    print(
        f"high-quality 509x509 patches: {len(all_tasks)} total -> {len(tasks)} selected "
        f"for download; {len(OFFSETS)}x{len(OFFSETS)} tiles/patch (offsets {OFFSETS})"
    )

    # ---- Phase 1: download+cache label rasters; emit candidate tile records -----------
    cands: list[dict[str, Any]] = []
    scanned_patches: set[str] = set()
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
        done = 0
        for recs in tqdm.tqdm(
            star_imap_unordered(pool, _scan_patch, [dict(task=t) for t in tasks]),
            total=len(tasks),
            desc="scan",
        ):
            cands.extend(recs)
            if recs:
                scanned_patches.add(recs[0]["patch_id"])
            done += 1
            if done % 2000 == 0:
                io.check_disk()
    n_failed = len(tasks) - len(scanned_patches)
    print(
        f"candidate tiles: {len(cands)} from {len(scanned_patches)} patches "
        f"({n_failed} patches skipped/failed)"
    )
    avail = Counter()
    for r in cands:
        for cid in r["classes_present"]:
            avail[cid] += 1
    print("candidate tiles per class:")
    for i, name, _d in CLASSES:
        print(f"  {i} {name:14} {avail.get(i, 0)}")

    # ---- Phase 2: tiles-per-class balanced selection (rare classes first) -------------
    selected = sampling.select_tiles_per_class(
        cands,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
        seed=SEED,
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (<= {PER_CLASS}/class, 25k cap)")

    # group selected tiles by patch to read each cached label at most once
    by_patch: dict[str, dict[str, Any]] = {}
    task_of = {t["patch_id"]: t for t in tasks}
    for r in selected:
        pid = r["patch_id"]
        t = task_of[pid]
        g = by_patch.setdefault(
            pid,
            {
                "patch_id": pid,
                "crs": r["crs"],
                "x0": r["x0"],
                "y0": r["y0"],
                "ts": r["ts"],
                "url": t["url"],
                "blob_offset": t["blob_offset"],
                "blob_length": t["blob_length"],
                "tiles": [],
            },
        )
        g["tiles"].append(r)

    # ---- Phase 3: write patches in parallel ------------------------------------------
    tile_counts = Counter()
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
        done = 0
        for results in tqdm.tqdm(
            star_imap_unordered(
                pool, _write_patch, [dict(task=g) for g in by_patch.values()]
            ),
            total=len(by_patch),
            desc="write",
        ):
            for _sid, present in results:
                for cid in present:
                    tile_counts[cid] += 1
            done += 1
            if done % 500 == 0:
                io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "CloudSEN12 / CloudSEN12+ (tacofoundation/cloudsen12 on HuggingFace, via tacoreader)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "zenodo": "https://zenodo.org/records/7034410",
                "have_locally": False,
                "annotation_method": (
                    "manual pixel-level expert annotation (high-quality tier) with IRIS, "
                    "reviewed following the CloudSEN12 labeling protocol"
                ),
                "product": "cloudsen12-l1c high-quality manual labels (band 'target'), 509x509 @ 10 m",
                "citation": "Aybar et al. 2022 (Sci Data); Aybar et al. 2024 CloudSEN12+ (Data in Brief)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[i][1]: int(tile_counts.get(i, 0)) for i in range(NUM_CLASSES)
            },
            "notes": (
                "High-quality manual tier only (label_type=='high'), 509x509 @ 10 m native UTM "
                "Sentinel-2 L1C patches; acquisition dates span 2018-2020 (all Sentinel-2 era, "
                "post-2016). Class ids 0=clear,1=thick cloud,2=thin cloud,3=cloud shadow map 1:1 "
                "from the source (values verified strictly in {0,1,2,3}; the 7-value scribble "
                "scheme and the 99/fill values were rejected by an explicit validator). Only the "
                "labels are used: for each patch a SINGLE HTTP range GET fetches its tortilla "
                "blob, the tiny 'target' label GeoTIFF is parsed out locally, and the S2 imagery "
                "bytes are discarded (never written) -- pretraining supplies imagery. This keeps "
                "us to ~1 HuggingFace request/patch (the source is public but anonymously "
                "rate-limited to ~3000 requests/300s, so tacoreader's multi-request vsicurl path "
                "is avoided). CloudSEN12 is large (10,000 high 509x509 patches); we downloaded a "
                "bounded, class-diverse subset of 2,500 patches (prioritizing the rarest class, "
                "thin cloud) -- more than enough to fill >=1000 tiles/class (spec 5 bounded "
                "sampling). Each downloaded 509x509 patch is tiled into 64x64 windows on an 8x8 "
                "grid; tiles-per-class balanced (rarest first), <=1000 tiles/class, 25k cap. "
                "Cloud masks are per-image/transient labels: change_time=null and time_range is a "
                "short window (+/-15 days) centered on the S2 acquisition date, keeping paired "
                "pretraining imagery temporally near the labeled scene (spec 5 specific-image "
                "rule) while <= the 1-year cap. Every 'high' pixel is labeled so no nodata "
                "occurs; 255 is the declared nodata sentinel. The 2000x2000 'high' patches and "
                "the scribble/nolabel tiers were not used."
            ),
        },
    )
    print("tile counts per class:")
    for i, name, _d in CLASSES:
        print(f"  {i} {name:14} {tile_counts.get(i, 0)}")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
