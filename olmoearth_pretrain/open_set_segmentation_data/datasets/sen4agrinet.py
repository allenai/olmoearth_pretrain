"""Process Sen4AgriNet into open-set-segmentation crop-type label patches (dense_raster).

Source: Sen4AgriNet (National Observatory of Athens), a Sentinel-2 multi-year, multi-country
crop-type benchmark annotated from LPIS farmer declarations. Hugging Face dataset
``orion-ai-lab/S4A`` (also github.com/Orion-AI-Lab/S4A). Covers 2019-2020 for **Catalonia
(ES)** and 2019 for **France (FR)** over 11 Sentinel-2 MGRS tiles (all UTM zone 31). Note:
the manifest/task blurb mentions "Greece/Catalonia", but the actual published S4A data are
Catalonia + France (no Greece) -- see summary. License: CC-BY / MIT (open LPIS + S4A code).

On-disk form: one NetCDF (.nc) per 366x366 patch (a subwindow of a 10 m S2 tile). Each patch
carries the S2 bands plus two georeferenced 366x366 rasters at 10 m in the tile's UTM CRS
(EPSG:32631): ``labels`` (FAO ICC crop taxonomy codes, uint32; 0 = no declaration / nodata)
and ``parcels`` (per-parcel ids). Every raster has an affine ``transform`` + ``crs`` attr, so
patches are fully georeferenced (the georeferencing check passes; not coordinate-free).

Task: per-pixel **classification** (crop type). Recipe (spec 4, dense_raster): each patch's
``labels`` raster is already UTM 10 m, so we reuse the source CRS and cut non-overlapping
64x64 windows (a 5x5 grid over the 320x320 top-left; the 46 px remainder is dropped). Crop
codes are mapped to class ids (0..N-1) in **descending global pixel frequency**; code 0 and
any code outside the taxonomy -> 255 (nodata/ignore). We do NOT invent a background class:
non-declared pixels are ignore, per spec (assembly adds negatives from other datasets).

Class map: the FAO/ICC "harmonized" taxonomy has 168 codes (S4A ``CROP_ENCODING``). That is
< 254, so all fit in uint8 and no 254-class-cap truncation is needed; ids are still assigned
by descending frequency. Names come from the S4A repo's ``encodings_en.py`` (vendored below).

Sampling: bounded download of the huge product -- up to ``MAX_PATCHES_PER_COMBO`` patches per
(year, tile) combo (16 combos), spread evenly, to draw the target counts from representative
regions rather than pulling all 10,013 patches. Candidate 64x64 windows with too few labeled
pixels are dropped (``MIN_LABELED_FRAC``). Final selection is **tiles-per-class balanced**
(each tile counts toward every crop id present), rare classes first, up to per_class =
min(1000, 25000 // N) tiles/class and <= 25k total.

Time range: crop labels are seasonal/annual -> a 1-year window on the patch year (2019/2020).

Reproduce (idempotent; skips already-written {sample_id}.tif):
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sen4agrinet
"""

import argparse
import multiprocessing
import re
from collections import Counter, defaultdict
from typing import Any

import netCDF4
import numpy as np
import tqdm
from huggingface_hub import HfApi, hf_hub_download
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.datasets._sen4agrinet_encoding import (
    CROP_ENCODING,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
    select_tiles_per_class,
)

SLUG = "sen4agrinet"
NAME = "Sen4AgriNet"
HF_REPO = "orion-ai-lab/S4A"

TILE = 64  # output patch size (<= MAX_TILE)
GRID = 366 // TILE  # 5 non-overlapping 64x64 windows per axis (covers 320 px)
RES = io.RESOLUTION  # 10 m
MIN_LABELED_FRAC = (
    0.05  # keep windows with >= this fraction of declared (non-zero) pixels
)
MAX_PATCHES_PER_COMBO = 120  # bounded download per (year, tile)
PER_CLASS = 1000

# code -> crop name (invert the S4A name->code taxonomy). Code 0 is NOT in the taxonomy
# (it is the "no declaration"/nodata background) and is mapped to 255 on write.
CODE_TO_NAME = {code: name for name, code in CROP_ENCODING.items()}


# ---------------------------------------------------------------------------
# Enumerate + sample patches
# ---------------------------------------------------------------------------
def list_patches() -> list[str]:
    """All .nc patch paths in the HF repo (data/{year}/{tile}/{name}.nc)."""
    api = HfApi()
    return sorted(
        f
        for f in api.list_repo_files(HF_REPO, repo_type="dataset")
        if f.endswith(".nc")
    )


def sample_patches(paths: list[str], per_combo: int) -> list[str]:
    """Evenly-spaced deterministic sample of up to ``per_combo`` patches per (year, tile)."""
    by_combo: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in paths:
        parts = p.split("/")  # data/2019/31TBF/2019_31TBF_patch_17_11.nc
        by_combo[(parts[1], parts[2])].append(p)
    out: list[str] = []
    for combo in sorted(by_combo):
        lst = sorted(by_combo[combo])
        if len(lst) <= per_combo:
            out.extend(lst)
        else:
            step = len(lst) / per_combo
            out.extend(lst[int(i * step)] for i in range(per_combo))
    return sorted(set(out))


def _download_one(rel_path: str) -> str | None:
    """Download one patch; return local path, or None if HF throttled/errored it.

    HF rate-limits (HTTP 429) unauthenticated pulls; we tolerate persistent failures so a
    stray 429 does not abort the whole run. Cached files make re-runs resume/accumulate.
    """
    dst_dir = io.raw_dir(SLUG)
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        return hf_hub_download(
            HF_REPO, rel_path, repo_type="dataset", local_dir=dst_dir.path
        )
    except Exception as e:  # noqa: BLE001 - transient HF 429/network; keep going
        print(f"  download failed ({rel_path}): {type(e).__name__}")
        return None


# ---------------------------------------------------------------------------
# Scan pass: enumerate candidate windows + per-code pixel frequency
# ---------------------------------------------------------------------------
def _read_labels(nc_path: str) -> tuple[np.ndarray, float, float, int, int]:
    """Return (labels uint32 366x366, transform.c, transform.f, epsg, year)."""
    ds = netCDF4.Dataset(nc_path)
    try:
        g = ds.groups["labels"]
        arr = np.asarray(g.variables["labels"][:]).astype(np.uint32)
        tr = list(g.variables["labels"].transform)
        crs_str = str(g.variables["labels"].crs)
        year = int(ds.patch_year)
    finally:
        ds.close()
    epsg = int(re.search(r"(\d+)$", crs_str.strip()).group(1))
    return arr, float(tr[2]), float(tr[5]), epsg, year


def _scan_one(nc_path: str) -> tuple[list[dict[str, Any]], dict[int, int]]:
    arr, ox, oy, epsg, year = _read_labels(nc_path)
    records: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for wi in range(GRID):
        for wj in range(GRID):
            r0, c0 = wi * TILE, wj * TILE
            sub = arr[r0 : r0 + TILE, c0 : c0 + TILE]
            labeled = sub != 0
            frac = float(labeled.mean())
            codes = [int(c) for c in np.unique(sub[labeled])] if labeled.any() else []
            for c in codes:
                counts[c] += int((sub == c).sum())
            if frac < MIN_LABELED_FRAC or not codes:
                continue
            records.append(
                {
                    "nc_path": nc_path,
                    "year": year,
                    "epsg": epsg,
                    "ox": ox,
                    "oy": oy,
                    "r0": r0,
                    "c0": c0,
                    "codes": codes,
                }
            )
    return records, dict(counts)


# ---------------------------------------------------------------------------
# Write pass
# ---------------------------------------------------------------------------
def _make_lut(code_to_id: dict[int, int]) -> np.ndarray:
    max_code = max(max(code_to_id), 0) + 1
    lut = np.full(max_code + 1, io.CLASS_NODATA, dtype=np.uint8)
    for code, cid in code_to_id.items():
        lut[code] = cid
    return lut


def _write_group(file_recs: dict[str, Any]) -> list[tuple[str, list[int]]]:
    """Write all selected windows from a single .nc file (opened once)."""
    nc_path = file_recs["nc_path"]
    recs = file_recs["recs"]
    lut = file_recs["lut"]
    # Idempotency: skip whole file if all its tifs exist.
    if all((io.locations_dir(SLUG) / f"{r['sid']}.tif").exists() for r in recs):
        return [(r["sid"], r["class_ids"]) for r in recs]
    arr, ox, oy, epsg, year = _read_labels(nc_path)
    proj = Projection(CRS.from_epsg(epsg), RES, -RES)
    out: list[tuple[str, list[int]]] = []
    for r in recs:
        sid = r["sid"]
        r0, c0 = r["r0"], r["c0"]
        sub = arr[r0 : r0 + TILE, c0 : c0 + TILE]
        mapped = lut[np.clip(sub, 0, len(lut) - 1)]
        x_min = int(round((ox + c0 * RES) / RES))
        y_min = int(round(-(oy - r0 * RES) / RES))
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        present = sorted(int(v) for v in np.unique(mapped) if v != io.CLASS_NODATA)
        tif = io.locations_dir(SLUG) / f"{sid}.tif"
        if not tif.exists():
            io.write_label_geotiff(
                SLUG, sid, mapped, proj, bounds, nodata=io.CLASS_NODATA
            )
            io.write_sample_json(
                SLUG,
                sid,
                proj,
                bounds,
                io.year_range(year),
                source_id=r["source_id"],
                classes_present=present,
            )
        out.append((sid, present))
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    global MIN_LABELED_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    # HF throttles (HTTP 429) unauthenticated parallel pulls; keep download concurrency low.
    ap.add_argument("--dl-workers", type=int, default=4)
    ap.add_argument("--per-combo", type=int, default=MAX_PATCHES_PER_COMBO)
    ap.add_argument("--min-frac", type=float, default=MIN_LABELED_FRAC)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="skip HF download; process whatever .nc patches are already in raw_dir "
        "(used because the unauthenticated HF rate limit makes a full pull impractical).",
    )
    args = ap.parse_args()
    MIN_LABELED_FRAC = args.min_frac

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    if args.offline:
        raw = io.raw_dir(SLUG)
        local_paths = [str(p) for p in raw.glob("**/*.nc")]
        print(f"offline: processing {len(local_paths)} cached patches from {raw}")
    else:
        print("listing patches on HF...")
        all_paths = list_patches()
        sampled = sample_patches(all_paths, args.per_combo)
        print(f"{len(all_paths)} total patches; sampled {len(sampled)}")
        # Download (bounded, parallel, idempotent).
        print("downloading sampled patches...")
        dl_workers = max(1, args.dl_workers)
        local_paths = []
        with multiprocessing.Pool(dl_workers) as p:
            for lp in tqdm.tqdm(
                star_imap_unordered(
                    p, _download_one, [dict(rel_path=r) for r in sampled]
                ),
                total=len(sampled),
            ):
                if lp is not None:
                    local_paths.append(lp)
        print(f"downloaded/cached {len(local_paths)} of {len(sampled)} sampled patches")
    io.check_disk()

    # Scan pass: candidate windows + global per-code pixel frequency.
    print("scanning label windows...")
    records: list[dict[str, Any]] = []
    pix_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for recs, counts in tqdm.tqdm(
            star_imap_unordered(p, _scan_one, [dict(nc_path=lp) for lp in local_paths]),
            total=len(local_paths),
        ):
            records.extend(recs)
            pix_counts.update(counts)

    pix_counts.pop(0, None)  # code 0 is background/nodata, never a class
    if not records:
        raise RuntimeError("no candidate windows found; lower --min-frac")

    # Class map: descending pixel frequency -> ids 0..N-1. 168 taxonomy codes < 254,
    # so no truncation; but honor the cap defensively (top-254 by frequency).
    ordered = [c for c, _ in pix_counts.most_common()]
    dropped = ordered[254:]
    ordered = ordered[:254]
    code_to_id = {code: i for i, code in enumerate(ordered)}
    n_classes = len(ordered)
    print(f"{n_classes} crop classes (dropped {len(dropped)} over 254-cap)")

    # Attach mapped class-ids to each record; drop windows with no in-map class.
    for r in records:
        r["class_ids"] = sorted({code_to_id[c] for c in r["codes"] if c in code_to_id})
    records = [r for r in records if r["class_ids"]]
    print(f"{len(records)} candidate windows after class mapping")

    per_class = min(PER_CLASS, max(1, MAX_SAMPLES_PER_DATASET // n_classes))
    selected = select_tiles_per_class(
        records,
        classes_key=lambda r: r["class_ids"],
        per_class=per_class,
        total_cap=MAX_SAMPLES_PER_DATASET,
    )
    print(f"selected {len(selected)} tiles (per_class={per_class})")

    # Assign sample ids (deterministic order) + provenance source_id.
    for i, r in enumerate(selected):
        r["sid"] = f"{i:06d}"
        rel = r["nc_path"].split("/data/")[-1]
        r["source_id"] = f"{rel}:r{r['r0']}c{r['c0']}"

    # Group by file for one-open-per-file write.
    lut = _make_lut(code_to_id)
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_file[r["nc_path"]].append(r)
    groups = [dict(nc_path=k, recs=v, lut=lut) for k, v in by_file.items()]

    print(f"writing {len(selected)} tiles from {len(groups)} files...")
    written_counts: Counter = Counter()
    n_written = 0
    with multiprocessing.Pool(args.workers) as p:
        for out in tqdm.tqdm(
            star_imap_unordered(p, _write_group, [dict(file_recs=g) for g in groups]),
            total=len(groups),
        ):
            for sid, present in out:
                n_written += 1
                for cid in present:
                    written_counts[cid] += 1

    # Patch coverage actually processed (year, tile -> count), for provenance.
    combo_counts: Counter = Counter()
    for lp in local_paths:
        m = re.search(r"/data/(\d+)/([0-9A-Z]+)/", lp)
        if m:
            combo_counts[f"{m.group(1)}/{m.group(2)}"] += 1
    coverage = ", ".join(f"{k}:{v}" for k, v in sorted(combo_counts.items()))

    # Dataset metadata.
    classes = [
        {"id": cid, "name": CODE_TO_NAME.get(code, f"code_{code}"), "description": None}
        for code, cid in sorted(code_to_id.items(), key=lambda kv: kv[1])
    ]
    metadata = {
        "dataset": SLUG,
        "name": NAME,
        "task_type": "classification",
        "source": "Sen4AgriNet (orion-ai-lab/S4A, Hugging Face)",
        "license": "CC-BY-4.0",
        "provenance": {
            "url": "https://github.com/Orion-AI-Lab/S4A",
            "hf_repo": HF_REPO,
            "have_locally": False,
            "annotation_method": "farmer declaration (LPIS), FAO ICC crop taxonomy",
        },
        "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
        "classes": classes,
        "nodata_value": io.CLASS_NODATA,
        "num_samples": n_written,
        "num_patches_processed": len(local_paths),
        "patch_coverage": coverage,
        "notes": (
            "Catalonia (ES) 2019-2020 + France (FR) 2019 (no Greece, despite the task blurb). "
            "11 UTM-zone-31 S2 tiles in the full product; this run is a bounded sample of "
            f"{len(local_paths)} patches (per (year,tile): {coverage}). 64x64 windows at native "
            "10 m UTM (EPSG:32631); code 0 (no LPIS declaration) -> nodata 255 (no fabricated "
            f"background). tiles-per-class balanced, per_class={per_class}. Class ids by "
            "descending pixel frequency; 168-code FAO/ICC taxonomy < 254 so no cap truncation. "
            "Sample skew (a few tiles dominate) is due to HF unauthenticated rate-limiting during "
            "download; re-running with an HF_TOKEN or after the quota resets pulls the full even "
            "sample and expands coverage (idempotent)."
        ),
    }
    io.write_dataset_metadata(SLUG, metadata)

    # Report class balance (top/bottom).
    print(f"WROTE {n_written} tiles across {n_classes} classes")
    common = written_counts.most_common()
    for cid, cnt in common[:8]:
        print(f"  id {cid:3d} {CODE_TO_NAME.get(ordered[cid], '?'):30s} {cnt}")
    if len(common) > 8:
        print("  ...")
        for cid, cnt in common[-5:]:
            print(f"  id {cid:3d} {CODE_TO_NAME.get(ordered[cid], '?'):30s} {cnt}")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done.")


if __name__ == "__main__":
    main()
