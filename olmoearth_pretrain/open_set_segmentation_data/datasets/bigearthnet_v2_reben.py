"""Process BigEarthNet v2 (reBEN) CORINE reference maps into open-set-segmentation labels.

Source: BigEarthNet v2 / "reBEN" (Clasen et al. 2024), Zenodo record 10891137
(https://zenodo.org/records/10891137), license CDLA-Permissive-1.0. reBEN is a large
multi-modal Sentinel-1 + Sentinel-2 patch archive over 10 European countries (imagery
2017-2018) with CORINE Land Cover 2018 annotations. Each 120x120 @ 10 m patch ships a
per-pixel **reference map** GeoTIFF (`*_reference_map.tif`) carrying the underlying CORINE
CLC Level-3 codes (e.g. 112 discontinuous urban fabric, 312 coniferous forest, 512 water
bodies). This is the dense_raster component we use.

We do NOT download the two huge patch archives (BigEarthNet-S1.tar.zst 54 GB,
BigEarthNet-S2.tar.zst 63 GB) -- only `Reference_Maps.tar.zst` (0.28 GB) and
`metadata.parquet`. The reference maps carry real georeferencing (local UTM CRS + 10 m
geotransform), verified at read time.

Task: per-pixel **classification** using the official BigEarthNet-19 nomenclature (the 19
land-cover classes into which CORINE Level-3 codes are grouped; matches the manifest
"19/43 CORINE" and the per-patch multi-labels in metadata.parquet). Each source CLC code
is remapped to a BEN-19 class id 0..18; CLC codes not part of the 19-class nomenclature
(roads/rail 122, airports 124, mineral extraction 131, dumps 132, construction 133, green
urban 141, sport/leisure 142, bare rocks 332, burnt areas 334, glaciers 335, nodata 999)
become 255 (nodata/ignore) -- exactly as the original BigEarthNet dropped those classes.

Sampling: dense multi-class **tiles-per-class balanced** (spec 5). Patches are indexed by
metadata.parquet (480,038 patches with 19-class labels + split + country); we greedily
select patches rarest-class-first to reach up to PER_CLASS tiles per class under the 25k
total cap, WITHOUT scanning every raster. For each selected patch we read its reference
map, remap to BEN-19 ids, and crop the 64x64 window (from a 3x3 offset grid) that best
covers the class the patch was selected for -- so each tile actually contains its target
class. classes_present is recorded from the written crop. Time range is a 1-year window on
the S2 acquisition year parsed from the patch id.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.bigearthnet_v2_reben
"""

import argparse
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
)

SLUG = "bigearthnet_v2_reben"
NAME = "BigEarthNet v2 (reBEN)"
ZENODO_RECORD = "10891137"
URL = "https://zenodo.org/records/10891137"

PER_CLASS = 1000
TILE = 64
PATCH = 120
# 3x3 grid of candidate top-left offsets within the 120x120 patch (0, 28, 56).
OFFSETS = [0, (PATCH - TILE) // 2, PATCH - TILE]
SEED = 42

# BigEarthNet-19 nomenclature. (name, description, [constituent CORINE CLC Level-3 codes]).
# Class id = index. Descriptions note the CLC codes grouped into each BEN-19 class.
CLASSES: list[tuple[str, str, list[int]]] = [
    (
        "Urban fabric",
        "Continuous (CLC 111) and discontinuous (112) urban fabric: residential built-up areas.",
        [111, 112],
    ),
    (
        "Industrial or commercial units",
        "Industrial or commercial units (CLC 121): factories, commercial and public facilities.",
        [121],
    ),
    (
        "Arable land",
        "Non-irrigated arable land (CLC 211), permanently irrigated land (212) and rice fields (213).",
        [211, 212, 213],
    ),
    (
        "Permanent crops",
        "Vineyards (CLC 221), fruit trees and berry plantations (222), olive groves (223) and "
        "annual crops associated with permanent crops (241).",
        [221, 222, 223, 241],
    ),
    ("Pastures", "Pastures (CLC 231): permanent grassland used for grazing.", [231]),
    (
        "Complex cultivation patterns",
        "Complex cultivation patterns (CLC 242): mosaics of small annual crops, pasture and "
        "permanent crops.",
        [242],
    ),
    (
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
        "CLC 243: agricultural land interspersed with significant areas of natural vegetation.",
        [243],
    ),
    (
        "Agro-forestry areas",
        "Agro-forestry areas (CLC 244): annual crops or grazing under woody (often oak) cover.",
        [244],
    ),
    (
        "Broad-leaved forest",
        "Broad-leaved forest (CLC 311): deciduous/evergreen broad-leaved tree cover.",
        [311],
    ),
    (
        "Coniferous forest",
        "Coniferous forest (CLC 312): needle-leaved evergreen/deciduous tree cover.",
        [312],
    ),
    (
        "Mixed forest",
        "Mixed forest (CLC 313): stands with both broad-leaved and coniferous trees.",
        [313],
    ),
    (
        "Natural grassland and sparsely vegetated areas",
        "Natural grassland (CLC 321) and sparsely vegetated areas (333).",
        [321, 333],
    ),
    (
        "Moors, heathland and sclerophyllous vegetation",
        "Moors and heathland (CLC 322) and sclerophyllous vegetation (323).",
        [322, 323],
    ),
    (
        "Transitional woodland, shrub",
        "Transitional woodland-shrub (CLC 324): bushy or herbaceous vegetation with scattered "
        "trees, incl. forest regeneration/degradation.",
        [324],
    ),
    ("Beaches, dunes, sands", "Beaches, dunes and sand plains (CLC 331).", [331]),
    ("Inland wetlands", "Inland marshes (CLC 411) and peat bogs (412).", [411, 412]),
    (
        "Coastal wetlands",
        "Salt marshes (CLC 421), salines (422) and intertidal flats (423).",
        [421, 422, 423],
    ),
    (
        "Inland waters",
        "Water courses (CLC 511) and water bodies (512): rivers, canals, lakes, reservoirs.",
        [511, 512],
    ),
    (
        "Marine waters",
        "Coastal lagoons (CLC 521), estuaries (522) and sea and ocean (523).",
        [521, 522, 523],
    ),
]

NAME_TO_ID: dict[str, int] = {name: i for i, (name, _d, _c) in enumerate(CLASSES)}
CLC_TO_ID: dict[int, int] = {
    clc: i for i, (_n, _d, codes) in enumerate(CLASSES) for clc in codes
}
N_CLASSES = len(CLASSES)


def ref_map_path(patch_id: str):
    """Reference-map GeoTIFF path for a patch id (parent dir = id minus the _col_row suffix)."""
    parent = patch_id.rsplit("_", 2)[0]
    return (
        io.raw_dir(SLUG)
        / "Reference_Maps"
        / parent
        / patch_id
        / f"{patch_id}_reference_map.tif"
    )


def acquisition_year(patch_id: str) -> int:
    """Parse the S2 acquisition year from the patch id (token like 20170613T101031)."""
    for tok in patch_id.split("_"):
        if len(tok) >= 8 and tok[:8].isdigit():
            return int(tok[:4])
    return 2017


def _remap_clc(arr: np.ndarray) -> np.ndarray:
    """Remap a CLC-code raster to BEN-19 class ids (uint8); unmapped codes -> 255."""
    out = np.full(arr.shape, io.CLASS_NODATA, dtype=np.uint8)
    for clc, cid in CLC_TO_ID.items():
        out[arr == clc] = cid
    return out


def _write_one(rec: dict[str, Any]) -> tuple[str, list[int]] | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None
    path = ref_map_path(rec["patch_id"])
    if not path.exists():
        return None
    with rasterio.open(str(path)) as ds:
        full = ds.read(1)
        t = ds.transform
        src_crs = ds.crs
    if full.shape != (PATCH, PATCH):
        return None
    ids = _remap_clc(full)

    # Choose the 64x64 window (3x3 offset grid) that best covers the target class.
    target = rec["primary_id"]
    best = None
    best_score = -1
    for roff in OFFSETS:
        for coff in OFFSETS:
            win = ids[roff : roff + TILE, coff : coff + TILE]
            score = int((win == target).sum())
            if score > best_score:
                best_score = score
                best = (roff, coff)
    roff, coff = best
    win = ids[roff : roff + TILE, coff : coff + TILE]

    # Build projection/bounds directly from the source UTM geotransform (no reprojection:
    # source is already local UTM at 10 m). Pixel bounds top-left corner in proj units.
    proj = Projection(src_crs, io.RESOLUTION, -io.RESOLUTION)
    x_min = int(round(t.c / io.RESOLUTION)) + coff
    y_min = int(round(t.f / -io.RESOLUTION)) + roff
    bounds = (x_min, y_min, x_min + TILE, y_min + TILE)

    io.write_label_geotiff(SLUG, sample_id, win, proj, bounds, nodata=io.CLASS_NODATA)
    present = sorted(int(x) for x in np.unique(win) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["patch_id"],
        classes_present=present,
    )
    return sample_id, present


def select_patches() -> list[dict[str, Any]]:
    """Greedy rarest-class-first, multi-label tiles-per-class balancing from metadata."""
    meta_path = io.raw_dir(SLUG) / "metadata.parquet"
    df = pd.read_parquet(meta_path.path, columns=["patch_id", "labels"])
    print(f"metadata patches: {len(df)}")

    # patch_id -> set of class ids present (from the 19-class multi-labels).
    patch_ids: list[str] = []
    patch_classes: list[list[int]] = []
    class_to_patch_idx: dict[int, list[int]] = defaultdict(list)
    for pid, labs in zip(df["patch_id"].tolist(), df["labels"].tolist()):
        cids = sorted({NAME_TO_ID[l] for l in labs if l in NAME_TO_ID})
        if not cids:
            continue
        idx = len(patch_ids)
        patch_ids.append(pid)
        patch_classes.append(cids)
        for c in cids:
            class_to_patch_idx[c].append(idx)

    avail = {c: len(v) for c, v in class_to_patch_idx.items()}
    print(
        "patches per class (available):",
        {CLASSES[c][0]: avail.get(c, 0) for c in range(N_CLASSES)},
    )

    per_class = min(PER_CLASS, max(1, MAX_SAMPLES_PER_DATASET // N_CLASSES))
    rng = random.Random(SEED)
    selected: dict[int, int] = {}  # patch idx -> primary class id
    counts: Counter = Counter()

    # Rarest class first so rare classes reach target before the budget fills.
    for c in sorted(range(N_CLASSES), key=lambda x: avail.get(x, 0)):
        if len(selected) >= MAX_SAMPLES_PER_DATASET:
            break
        cand = class_to_patch_idx.get(c, [])[:]
        rng.shuffle(cand)
        for idx in cand:
            if counts[c] >= per_class:
                break
            if len(selected) >= MAX_SAMPLES_PER_DATASET:
                break
            if idx in selected:
                continue
            selected[idx] = c
            for cc in patch_classes[idx]:
                counts[cc] += 1

    recs = []
    for i, (idx, primary) in enumerate(sorted(selected.items())):
        pid = patch_ids[idx]
        recs.append(
            {
                "sample_id": f"{i:06d}",
                "patch_id": pid,
                "primary_id": primary,
                "year": acquisition_year(pid),
            }
        )
    print(
        f"selected {len(recs)} patches (per_class target={per_class}); "
        f"metadata-label class coverage: "
        f"{ {CLASSES[c][0]: counts.get(c, 0) for c in range(N_CLASSES)} }"
    )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    manifest.write_registry_entry(SLUG, "in_progress")
    io.check_disk()

    recs = select_patches()

    io.check_disk()
    present_counts: Counter = Counter()  # tiles actually containing each class id
    written = 0
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in recs]),
            total=len(recs),
        ):
            if res is None:
                continue
            written += 1
            _sid, present = res
            for c in present:
                present_counts[c] += 1

    io.check_disk()
    # Count actual outputs on disk (idempotent re-runs).
    n_tif = sum(1 for _ in io.locations_dir(SLUG).glob("*.tif"))
    class_counts = {CLASSES[c][0]: present_counts.get(c, 0) for c in range(N_CLASSES)}
    print(f"wrote {written} new tiles this run; total tiles on disk: {n_tif}")
    print("per-class tile counts (actual, from written crops):")
    for name, cnt in class_counts.items():
        print(f"  {cnt:>6}  {name}")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo",
            "license": "CDLA-Permissive-1.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "derived-product (CORINE Land Cover 2018)",
                "citation": "Clasen et al. 2024, BigEarthNet v2 (reBEN); Zenodo 10891137",
                "component": "Reference_Maps (per-pixel CORINE CLC Level-3 reference maps)",
                "nomenclature": "BigEarthNet-19 (CLC Level-3 grouped into 19 land-cover classes)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc, _codes) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_tif,
            "class_counts": class_counts,
            "notes": (
                "Dense per-pixel classification from reBEN CORINE reference maps. Only the "
                "0.28 GB Reference_Maps archive + metadata.parquet were downloaded (not the "
                "54+63 GB S1/S2 patch archives). Source CLC Level-3 codes remapped to the "
                "BigEarthNet-19 nomenclature; CLC codes outside the 19-class scheme "
                "(122/124/131/132/133/141/142/332/334/335 and nodata 999) -> 255. Patches "
                "are 120x120 @ 10 m in local UTM (real geotransform preserved, no "
                "reprojection); each label tile is a 64x64 crop chosen (from a 3x3 offset "
                "grid) to best cover the class the patch was selected for. Tiles-per-class "
                "balanced greedily rarest-first from metadata multi-labels, per_class=1000, "
                "25k total cap. Time range: 1-year window on the S2 acquisition year "
                "(2017/2018) parsed from the patch id. All source splits (train/val/test) "
                "used as pretraining labels."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_tif
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
