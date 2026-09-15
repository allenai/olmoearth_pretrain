"""Process EuroCropsML into open-set-segmentation label points (crop-type points).

Source: EuroCropsML (Zenodo concept DOI 10.5281/zenodo.10629609; latest record
15095445, v13, 2025-03-31), the ML-ready benchmark built on EuroCrops v9. It provides
706,683 agricultural parcels from **Estonia, Latvia and Portugal** for the year **2021**,
each labeled with a harmonized **HCAT** (Hierarchical Crop and Agriculture Taxonomy)
10-digit code and pre-processed into a per-parcel ``.npz`` (Sentinel-2 median time series
+ metadata). Licensed CC-BY-4.0.

This is the ML-ready **points** product (registry ``label_type: points``); the sibling
``eurocrops`` dataset already covers the full parcel-polygon → dense-raster product, so to
avoid duplication we emit EuroCropsML as the **sparse-point** product per spec 2a/4:
each parcel is one WGS84 point at its precomputed centroid, carrying its crop class.

Why points (not rasterized polygons): EuroCropsML distributes only per-parcel median time
series + a centroid ``[lon, lat]`` (in each ``.npz``'s ``center`` array) — the parcel
polygons live in the much larger ``raw_data.zip``. The centroid + HCAT code + reference
year 2021 is exactly the sparse-point signal, and crop parcels are clearly observable at
10 m S2, so a 1x1 point label placed at the parcel centroid pairs cleanly with imagery.

Access: download ``preprocess.zip`` (~1.47 GB). Every parcel's HCAT code and NUTS3 region
are encoded in the member filename ``preprocess/{NUTS3}_{parcelid}_{EC_hcat_c}.npz``, so
the full class distribution is read from the zip's central directory with **no
extraction**; only the sampled subset of ``.npz`` files is read (from the local zip) to
pull each parcel's centroid.

Classes: HCAT codes present in the sampled parcels, named via the repo's HCAT3 mapping
(data/eurocrops_hcat3_mapping.json). EuroCropsML has 176 distinct HCAT codes (< the 254
uint8 cap), so all are kept; ids are assigned 0..N-1 in **descending global frequency**.

Sampling: class-balanced with the 25k per-dataset cap. With N classes the effective
per-class limit is min(1000, 25000 // N) (``balance_by_class`` default). "pasture meadow
grassland grass" dominates (~45% of parcels) but is capped like every other class.

Time range: 1-year window anchored on 2021 (all EuroCropsML parcels are year 2021).
change_time is null (crop type is a static seasonal label, not a change event).

Run (idempotent; skips writing if points.geojson already exists):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurocropsml
"""

import argparse
import io as _io
import json
import multiprocessing
import urllib.request
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "eurocropsml"
NAME = "EuroCropsML"
ZENODO_RECORD = "15095445"  # v13 (2025-03-31); concept DOI 10.5281/zenodo.10629609
YEAR = 2021  # all EuroCropsML parcels are year 2021

HCAT_MAPPING_PATH = "data/eurocrops_hcat3_mapping.json"

# uint8 class labels -> at most 254 classes (255 = nodata).
MAX_CLASSES = 254
PER_CLASS = (
    1000  # spec target; lowered automatically to 25000 // N by balance_by_class.
)

COUNTRY_NAMES = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}


def load_hcat_names() -> dict[int, str]:
    """Return {hcat_code(int): hcat_name} from the repo HCAT3 mapping."""
    with open(HCAT_MAPPING_PATH) as f:
        entries = json.load(f)
    return {int(e["hcat_code"]): e["hcat_name"] for e in entries}


def ensure_data() -> str:
    """Download preprocess.zip; return the local path. Writes SOURCE.txt."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "preprocess.zip"
    if not dst.exists():
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{ZENODO_RECORD}"
        ) as r:
            meta = json.loads(r.read())
        urls = {
            f["key"]: (f["links"].get("self") or f["links"].get("download"))
            for f in meta["files"]
        }
        io.check_disk()
        download.download_http(urls["preprocess.zip"], dst)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "EuroCropsML (Zenodo record 15095445; concept DOI 10.5281/zenodo.10629609), "
            "CC-BY-4.0.\n"
            "https://zenodo.org/records/15095445 ; https://github.com/dida-do/eurocropsml\n"
            "Downloaded: preprocess.zip (ML-ready per-parcel .npz; Estonia/Latvia/Portugal, "
            "year 2021). Centroids read from each .npz 'center' array ([lon, lat] WGS84); "
            "HCAT code + NUTS3 region parsed from member filenames.\n"
        )
    return str(dst.path)


def parse_members(zip_path: str) -> list[dict[str, Any]]:
    """Parse every npz member filename into a record.

    Member name: ``preprocess/{NUTS3}_{parcelid}_{EC_hcat_c}.npz``. NUTS3 codes contain
    no underscore, so rsplit('_', 2) recovers (region, parcel_id, hcat_code).
    """
    recs: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".npz"):
                continue
            base = name.rsplit("/", 1)[-1][: -len(".npz")]
            parts = base.rsplit("_", 2)
            if len(parts) != 3:
                continue
            region, parcel_id, hcat_str = parts
            try:
                hcat = int(hcat_str)
            except ValueError:
                continue
            recs.append(
                {
                    "member": name,
                    "region": region,
                    "country": COUNTRY_NAMES.get(region[:2], region[:2]),
                    "parcel_id": parcel_id,
                    "hcat": hcat,
                    "source_id": f"{region}/{parcel_id}",
                }
            )
    return recs


_ZIP_PATH: str | None = None
_ZF: zipfile.ZipFile | None = None


def _init_worker(zip_path: str) -> None:
    global _ZIP_PATH, _ZF
    _ZIP_PATH = zip_path
    _ZF = zipfile.ZipFile(zip_path)


def _read_center(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Read one parcel centroid ([lon, lat]) from the local zip."""
    try:
        buf = _ZF.read(rec["member"])  # type: ignore[union-attr]
        with np.load(_io.BytesIO(buf), allow_pickle=True) as d:
            center = d["center"]
        lon, lat = float(center[0]), float(center[1])
        if not (np.isfinite(lon) and np.isfinite(lat)):
            return None
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            return None
        rec = dict(rec)
        rec["lon"] = lon
        rec["lat"] = lat
        return rec
    except Exception as e:  # noqa: BLE001
        print(f"error reading {rec['member']}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    hcat_names = load_hcat_names()
    zip_path = ensure_data()

    # ---- Parse all member filenames -> full class distribution (no extraction) ----
    recs = parse_members(zip_path)
    print(f"parsed {len(recs)} parcels")
    by_country = Counter(r["country"] for r in recs)
    print("  by country:", dict(by_country))

    global_freq: Counter = Counter(r["hcat"] for r in recs)
    in_map = sum(1 for c in global_freq if c in hcat_names)
    print(f"  distinct HCAT codes: {len(global_freq)}; in HCAT3 mapping: {in_map}")

    # ---- Keep top-N codes by frequency, ids 0..N-1 (descending freq) --------------
    ranked = [code for code, _ in global_freq.most_common()]
    kept = ranked[:MAX_CLASSES]
    dropped = ranked[MAX_CLASSES:]
    code_to_id = {code: i for i, code in enumerate(kept)}
    print(f"kept {len(kept)} classes; dropped {len(dropped)}")

    for r in recs:
        r["class_id"] = code_to_id.get(r["hcat"])

    # ---- Class-balanced selection under the 25k cap -------------------------------
    selected = balance_by_class(
        recs, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    n_classes = len(code_to_id)
    eff = max(1, min(PER_CLASS, 25000 // n_classes))
    print(f"selected {len(selected)} parcels (eff per-class cap = {eff})")

    # ---- Read centroids for the selected subset (parallel, local zip) -------------
    points: list[dict[str, Any]] = []
    written_by_class: Counter = Counter()
    with multiprocessing.Pool(
        args.workers, initializer=_init_worker, initargs=(zip_path,)
    ) as p:
        for out in tqdm.tqdm(
            star_imap_unordered(p, _read_center, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            if out is None:
                continue
            points.append(out)

    io.check_disk()

    # Assign running ids in a stable (country, source_id) order for reproducibility.
    points.sort(key=lambda r: (r["country"], r["source_id"]))
    out_points = []
    for i, r in enumerate(points):
        written_by_class[r["class_id"]] += 1
        out_points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": int(r["class_id"]),
                "time_range": io.year_range(YEAR),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", out_points)
    num = len(out_points)
    print(f"wrote {num} points")

    # ---- Metadata -----------------------------------------------------------------
    classes = [
        {
            "id": cid,
            "name": hcat_names.get(code, f"hcat_{code}"),
            "description": (
                f"EuroCrops HCAT3 code {code} "
                f"({hcat_names.get(code, 'code not in HCAT3 mapping')})."
            ),
        }
        for code, cid in sorted(code_to_id.items(), key=lambda kv: kv[1])
    ]
    class_counts = {
        hcat_names.get(code, f"hcat_{code}"): int(written_by_class.get(cid, 0))
        for code, cid in sorted(code_to_id.items(), key=lambda kv: kv[1])
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/15095445",
                "have_locally": False,
                "annotation_method": (
                    "farmer self-declaration (CAP/LPIS), harmonized to HCAT; "
                    "ML-ready per-parcel centroids from EuroCropsML preprocess.zip"
                ),
                "zenodo_record": ZENODO_RECORD,
                "concept_doi": "10.5281/zenodo.10629609",
                "countries": ["Estonia", "Latvia", "Portugal"],
                "reference_year": YEAR,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num,
            "class_counts": class_counts,
            "parcels_by_country": dict(by_country),
            "dropped_hcat_codes": [int(x) for x in dropped],
            "notes": (
                "Sparse crop-type POINTS product of EuroCropsML (Estonia/Latvia/Portugal, "
                "year 2021). One WGS84 point per agricultural parcel at its precomputed "
                "centroid ([lon, lat] from each .npz 'center' array); label = HCAT crop "
                "class. Sibling dataset 'eurocrops' covers the parcel-polygon dense-raster "
                "product. Class ids assigned 0..N-1 by descending global HCAT-code "
                f"frequency; kept all {len(kept)} of {len(ranked)} codes "
                f"(dropped {len(dropped)}; < 254 uint8 cap). Class-balanced with the 25k "
                "cap (effective per-class cap = min(1000, 25000 // n_classes)); the "
                "dominant 'pasture meadow grassland grass' class is capped like the rest. "
                "Time range = 1-year window on 2021; change_time null (static seasonal "
                "label)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num
    )
    print(f"done: {num} samples across {n_classes} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
