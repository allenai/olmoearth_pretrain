"""Process So2Sat LCZ42 (v4.2, with geolocation) into open-set-segmentation labels.

Source: So2Sat LCZ42 (Zhu et al. 2020), TUM mediaTUM record 1836598 / doi:10.14459/
2025mp1836598.002 ("v4: data with geolocation"), CC-BY-4.0. ~400k co-registered
Sentinel-1/2 32x32 patches over 42 global cities, each hand-labeled by experts into one
of the 17 Local Climate Zones (Stewart & Oke 2012). Distributed as ML-ready HDF5 patch
stacks (sen1/sen2/label one-hot) that historically STRIPPED geocoordinates; the v4.2
release adds per-patch geolocation (``*_geo.h5``: EPSG code + a worldfile ``tfw`` affine +
city), which is exactly what lets us place each patch on the S2 grid.

Triage: ACCEPT. Georeferencing is recoverable (v4.2 geo files), labels are 2017 (post-
2016), and LCZ patches are drawn from expert-delineated homogeneous LCZ polygons, i.e.
genuinely coherent land-cover / urban-morphology patches -> per spec section 4 (scene-
level) we emit a uniform-class tile per patch rather than rejecting as patch
classification. Each patch is 32x32 @ 10 m (320 m) in a local UTM CRS already, so we reuse
the source CRS/resolution directly (no reprojection) and fill the tile with the single LCZ
class id. Sparse-point rules do not apply (label footprint is 320 m, > 1 px).

Download strategy (label-only; NO imagery pulled): the geo files are small and hosted
uncompressed on Hugging Face (corrected EPSG + city, verified identical EPSG to the
mediaTUM originals for val/test and consistent corner coordinates), so we download those
fully. The LCZ ``label`` array lives only inside the big sen1/sen2/label HDF5, but mediaTUM
serves those UNCOMPRESSED with HTTP Range support, so we read just the contiguous ``label``
dataset via a byte-range read -- never touching the tens of GB of imagery.

Splits used: validation + testing only (48,307 patches, 10 cities across continents:
guangzhou, jakarta, moscow, mumbai, munich, nairobi, sanfrancisco, santiago, sydney,
tehran). The 42-city ``training.h5`` (52 GB) is EXCLUDED here: the mediaTUM data server
would not serve HTTP Range requests on that single 52 GB file within a workable time
budget (even a 1-byte probe did not return in >6 min, while the 3.5 GB val/test files read
in ~3 min each). This is a source-server throughput limit, not a data issue; to add the
training patches later, drop a ``raw/so2sat_lcz42/training_labels.npy`` (uint8 argmax of
the one-hot ``training.h5`` label) and add "training" to ``SPLITS`` -- the rest is idempotent.
All 17 LCZ classes are already present in val+test; some rare built classes (e.g. LCZ E
bare rock/paved, LCZ 1 compact high-rise) fall short of the 1000/class target, which spec
section 5 permits (keep sparse classes; downstream assembly drops the too-small ones).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.so2sat_lcz42
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import h5py
import numpy as np
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "so2sat_lcz42"
NAME = "So2Sat LCZ42"

# mediaTUM v4.2 share (uncompressed .h5, Range-capable) -> label arrays.
MEDIATUM_BASE = "https://dataserv.ub.tum.de/public.php/dav/files/m1836598"
MEDIATUM_AUTH = ("m1836598", "m1836598")
# Hugging Face mirror of v4.2 -> small corrected geolocation files (epsg/tfw/city).
HF_GEO_URL = "https://huggingface.co/datasets/zhu-xlab/So2Sat-LCZ42/resolve/main/v4/{split}_geo.h5"

# training.h5 (52 GB) is excluded -- the mediaTUM server will not serve HTTP Range
# requests on it in a workable time (see module docstring). It is available for a future
# retry: cache raw/<slug>/training_labels.npy and prepend "training" here.
SPLITS = ["validation", "testing"]
TILE = 32  # 32x32 @ 10 m == the native 320 m patch footprint (<= 64 cap)
PER_CLASS = 1000
YEAR = 2017  # So2Sat S2 imagery acquired 2017; 1-year window (spec 5)

# The 17 Local Climate Zones, in the So2Sat one-hot column order (LCZ 1-10 built types,
# then LCZ A-G natural types), with Stewart & Oke (2012) definitions.
CLASSES = [
    (
        "compact_high_rise",
        "LCZ 1: dense mix of tall buildings (tens of storeys); few/no trees; mostly paved; "
        "concrete/steel/stone/glass construction.",
    ),
    (
        "compact_mid_rise",
        "LCZ 2: dense mix of midrise buildings (3-9 storeys); few/no trees; mostly paved; "
        "stone/brick/tile/concrete construction.",
    ),
    (
        "compact_low_rise",
        "LCZ 3: dense mix of low-rise buildings (1-3 storeys); few/no trees; mostly paved; "
        "stone/brick/tile/concrete construction.",
    ),
    (
        "open_high_rise",
        "LCZ 4: open arrangement of tall buildings (tens of storeys); abundant pervious land "
        "(low plants, scattered trees).",
    ),
    (
        "open_mid_rise",
        "LCZ 5: open arrangement of midrise buildings (3-9 storeys); abundant pervious land "
        "(low plants, scattered trees).",
    ),
    (
        "open_low_rise",
        "LCZ 6: open arrangement of low-rise buildings (1-3 storeys); abundant pervious land "
        "(low plants, scattered trees).",
    ),
    (
        "lightweight_low_rise",
        "LCZ 7: dense mix of single-storey lightweight buildings (wood/thatch/corrugated "
        "metal); few/no trees; hard-packed ground; informal settlements.",
    ),
    (
        "large_low_rise",
        "LCZ 8: open arrangement of large low-rise buildings (1-3 storeys); few/no trees; "
        "mostly paved; steel/concrete/metal construction (warehouses, retail, industry).",
    ),
    (
        "sparsely_built",
        "LCZ 9: sparse arrangement of small/medium buildings in a natural setting; abundant "
        "pervious land (low plants, scattered trees).",
    ),
    (
        "heavy_industry",
        "LCZ 10: low- and mid-rise industrial structures (towers, tanks, stacks); few/no "
        "trees; mostly paved or hard-packed; metal/steel/concrete construction.",
    ),
    (
        "dense_trees",
        "LCZ A: heavily wooded landscape of deciduous/evergreen trees; land cover mostly "
        "pervious (low plants); natural forest, tree cultivation, urban park.",
    ),
    (
        "scattered_trees",
        "LCZ B: lightly wooded landscape of scattered deciduous/evergreen trees; land cover "
        "mostly pervious (low plants); natural forest, tree cultivation, urban park.",
    ),
    (
        "bush_scrub",
        "LCZ C: open arrangement of bushes, shrubs and short woody trees; land cover mostly "
        "pervious (bare soil/sand); scrubland, agriculture.",
    ),
    (
        "low_plants",
        "LCZ D: featureless landscape of grass or herbaceous plants/crops; few/no trees; "
        "natural grassland, agriculture, urban park.",
    ),
    (
        "bare_rock_or_paved",
        "LCZ E: featureless landscape of rock or paved cover; few/no trees or plants; natural "
        "desert (rock) or urban transportation.",
    ),
    (
        "bare_soil_or_sand",
        "LCZ F: featureless landscape of soil or sand cover; few/no trees or plants; natural "
        "desert or agriculture.",
    ),
    (
        "water",
        "LCZ G: large, open water bodies (seas, lakes) or small ones (rivers, reservoirs, "
        "lagoons).",
    ),
]
N_CLASSES = len(CLASSES)  # 17


def _label_npy(split: str):
    return io.raw_dir(SLUG) / f"{split}_labels.npy"


def _geo_h5(split: str):
    return io.raw_dir(SLUG) / f"{split}_geo.h5"


def load_split_classes(split: str) -> np.ndarray:
    """Per-patch LCZ class id (argmax of the one-hot label), cached to raw/ as .npy.

    Reads only the contiguous ``label`` dataset from the uncompressed mediaTUM HDF5 via
    HTTP Range requests -- the ~52 GB of sen1/sen2 imagery is never fetched.
    """
    cache = _label_npy(split)
    if cache.exists():
        with cache.open("rb") as f:
            return np.load(f)
    url = f"{MEDIATUM_BASE}/{split}.h5"
    print(f"  range-reading label array from {url} ...", flush=True)
    onehot = download.read_remote_h5_dataset(url, "label", auth=MEDIATUM_AUTH)
    classes = onehot.argmax(axis=1).astype(np.uint8)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.parent / (cache.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, classes)
    tmp.rename(cache)
    print(f"  {split}: {len(classes)} patch labels", flush=True)
    return classes


def load_split_geo(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (epsg[N], tfw[N,6]) from the corrected HF geolocation file (cached to raw/)."""
    dst = _geo_h5(split)
    download.download_http(HF_GEO_URL.format(split=split), dst)
    with h5py.File(dst.path, "r") as f:
        epsg = np.squeeze(f["epsg"][:]).astype(int)
        tfw = f["tfw"][:].astype(float)
    return epsg, tfw


def build_records() -> list[dict[str, Any]]:
    """One record per patch across all splits: class id + UTM CRS + upper-left corner."""
    recs: list[dict[str, Any]] = []
    for split in SPLITS:
        classes = load_split_classes(split)
        epsg, tfw = load_split_geo(split)
        if len(classes) != len(epsg):
            raise RuntimeError(
                f"{split}: label/geo count mismatch {len(classes)} != {len(epsg)}"
            )
        # Corrected worldfile order [A, D, B, E, C, F]: A=x_res(10), E=y_res(-10),
        # C=x of upper-left pixel corner, F=y of upper-left pixel corner.
        x_ul = tfw[:, 4]
        y_ul = tfw[:, 5]
        for i in range(len(classes)):
            recs.append(
                {
                    "cls": int(classes[i]),
                    "epsg": int(epsg[i]),
                    "x_ul": float(x_ul[i]),
                    "y_ul": float(y_ul[i]),
                    "source_id": f"{split}/{i}",
                }
            )
    return recs


def _projection_and_bounds(
    rec: dict[str, Any],
) -> tuple[Projection, tuple[int, int, int, int]]:
    proj = Projection(CRS.from_epsg(rec["epsg"]), io.RESOLUTION, -io.RESOLUTION)
    # Source is already UTM @ 10 m; snap the (sub-metre off-grid) corner to the pixel grid.
    col0 = int(round(rec["x_ul"] / io.RESOLUTION))
    row0 = int(round(rec["y_ul"] / -io.RESOLUTION))
    bounds = (col0, row0, col0 + TILE, row0 + TILE)
    return proj, bounds


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj, bounds = _projection_and_bounds(rec)
    arr = np.full((TILE, TILE), rec["cls"], dtype=np.uint8)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=[rec["cls"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "So2Sat LCZ42 v4.2 (with geolocation), doi:10.14459/2025mp1836598.002.\n"
            f"Labels: contiguous 'label' dataset range-read from {MEDIATUM_BASE}/"
            "{validation,testing}.h5 (uncompressed; imagery never downloaded).\n"
            "training.h5 (52 GB) excluded: server would not serve Range requests on it "
            "in a workable time (retryable; see script docstring).\n"
            "Geolocation (epsg/tfw/city): corrected v4.2 files from "
            "https://huggingface.co/datasets/zhu-xlab/So2Sat-LCZ42 (v4/*_geo.h5).\n"
        )

    print("Loading per-patch labels + geolocation...")
    recs = build_records()
    print(f"  {len(recs)} total labeled patches")
    io.check_disk()

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "cls", per_class=PER_CLASS)
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    counts = Counter(r["cls"] for r in selected)
    print(
        f"  selected {len(selected)} patches (<= {PER_CLASS}/class over {N_CLASSES} classes)"
    )
    print("  per-class counts:", {CLASSES[c][0]: counts[c] for c in range(N_CLASSES)})

    print(f"Writing {len(selected)} uniform-class {TILE}x{TILE} tiles...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]):
            pass

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "mediaTUM (TUM) / Hugging Face zhu-xlab/So2Sat-LCZ42",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://mediatum.ub.tum.de/1836598",
                "doi": "10.14459/2025mp1836598.002",
                "have_locally": False,
                "annotation_method": "manual (expert-labeled Local Climate Zones)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {CLASSES[c][0]: counts[c] for c in range(N_CLASSES)},
            "notes": (
                f"Local Climate Zone (LCZ) patch classification, 17 classes. Each So2Sat "
                f"patch is a 32x32 @ 10 m (320 m) window hand-labeled with one LCZ from an "
                f"expert-delineated homogeneous LCZ polygon; emitted as a uniform-class "
                f"{TILE}x{TILE} tile (spec 4 scene-level). Patches are already in a local "
                f"UTM CRS at 10 m, reused directly; label<->geo alignment uses the v4.2 "
                f"corrected geolocation files. Splits used: validation + testing (10 "
                f"cities); the 52 GB training.h5 was excluded because the mediaTUM server "
                f"would not serve HTTP Range requests on it in a workable time (source "
                f"throughput limit, retryable). Tiles-per-class balanced to <= {PER_CLASS}/"
                f"class; rare built classes (e.g. LCZ E, LCZ 1) fall short of 1000, which "
                f"is allowed (spec 5). Time range: 1-year window at {YEAR} (So2Sat S2 "
                f"imagery, no per-patch date). Only the label arrays were downloaded "
                f"(byte-range read of the uncompressed HDF5); no Sentinel-1/2 imagery."
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
