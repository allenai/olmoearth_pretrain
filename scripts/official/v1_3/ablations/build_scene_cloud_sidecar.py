"""Scene-level cloud cover for every Landsat or S2 capture of the AllCaptures corpus (CPU job).

The h5s keep no scene IDs, but the source tiles' metadata CSVs
(``osm_allcaptures/10_<modality>_allcap_meta/<crs>_10_<col>_<row>.csv``) list the
scene of every capture. A sample's tile is recovered from its ``latlon`` (the tile
centre, ``dataset/sample.py``), its captures are matched to the tile's scenes by date,
and each scene's cloud cover comes from Planetary Computer's STAC (Landsat:
``landsat:cloud_cover_land``, else ``eo:cloud_cover``; S2: ``eo:cloud_cover``). On a
day with two scenes the cloudier one is used.

With ``--parquet`` the cloud cover is read from Planetary Computer's GeoParquet
snapshot of the collection (monthly partitions downloaded whole, two columns read)
instead of the STAC search API, which throttles a few million S2 IDs to ~13 h.

Writes one h5: ``offsets`` [N + 1] and ``cloud_cover`` (percent, NaN = unknown)
aligned with each sample's ``timestamps_<modality>``, read through
``OlmoEarthDatasetConfig.scene_cloud_cover_paths``.

Usage: ``python build_scene_cloud_sidecar.py {landsat,s2} OUT.h5 [--parquet]
[--workers 32] [--limit N]``.
"""

import argparse
import csv
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from multiprocessing import get_context

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import requests
from pyproj import Transformer

SOURCE = "/weka/dfive-default/helios/dataset/osm_allcaptures"
H5_DIR = (
    f"{SOURCE}/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_l2_openstreetmap_raster_sentinel1_sentinel2_l2a_sentinel2_scl_"
    "srtm_worldcereal_worldcover_wri_canopy_height_map/395660"
)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
ITEMS_BLOB = "https://pcstacitems.blob.core.windows.net/items"
ITEMS_SAS_URL = (
    "https://planetarycomputer.microsoft.com/api/sas/v1/token/pcstacitems/items"
)
TILE_M = 10 * 256  # metres per 256 px grid tile at 10 m
# sensor -> (modality, STAC collection, cloud properties in order of preference)
SENSORS = {
    "landsat": (
        "landsat_l2",
        "landsat-c2-l2",
        ("landsat:cloud_cover_land", "eo:cloud_cover"),
    ),
    "s2": ("sentinel2_l2a", "sentinel-2-l2a", ("eo:cloud_cover",)),
}


def _parse_csv(name: str) -> tuple[str, dict[int, list[str]]]:
    by_day: dict[int, list[str]] = defaultdict(list)
    with open(os.path.join(_SENSOR["meta"], name)) as f:
        for row in csv.DictReader(f):
            day = date.fromisoformat(row["start_time"][:10]).toordinal()
            by_day[day].append(row["scene_id"])
    return name.removesuffix(".csv"), dict(by_day)


def _cloud_covers(ids: list[str]) -> dict[str, float]:
    for attempt in range(6):
        try:
            r = requests.post(
                STAC,
                json={
                    "collections": [_SENSOR["collection"]],
                    "ids": ids,
                    "limit": len(ids),
                },
                timeout=60,
            )
            r.raise_for_status()
            out = {}
            for feat in r.json()["features"]:
                p = feat["properties"]
                for key in _SENSOR["properties"]:
                    cc = p.get(key)
                    if cc is not None and cc >= 0:
                        out[feat["id"]] = float(cc)
                        break
            return out
        except Exception:  # noqa: BLE001 - retried, then reported missing
            time.sleep(2**attempt)
    return {}


def _items_token() -> str:
    return requests.get(ITEMS_SAS_URL, timeout=60).json()["token"]


def _partitions(collection: str) -> list[tuple[str, date, date]]:
    """(blob name, first day, last day) of each GeoParquet partition."""
    body = requests.get(
        f"{ITEMS_BLOB}?restype=container&comp=list&prefix={collection}.parquet/"
        f"&{_items_token()}",
        timeout=120,
    ).text
    out = []
    for name in re.findall(r"<Name>([^<]+)</Name>", body):
        days = re.findall(r"_(\d{4}-\d{2}-\d{2})T", name)
        if len(days) == 2:
            out.append((name, date.fromisoformat(days[0]), date.fromisoformat(days[1])))
    return out


def _parquet_covers(name: str, wanted: frozenset[str]) -> dict[str, float]:
    """Cloud cover of the ``wanted`` ids in one partition (download, read, delete)."""
    import pyarrow.parquet as pq

    for attempt in range(5):
        try:
            with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
                with requests.get(
                    f"{ITEMS_BLOB}/{name}?{_items_token()}", stream=True, timeout=300
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(2**24):
                        tmp.write(chunk)
                tmp.flush()
                table = pq.read_table(tmp.name, columns=["id", *_SENSOR["properties"]])
            out = {}
            cols = [table.column(p).to_pylist() for p in _SENSOR["properties"]]
            for i, item_id in enumerate(table.column("id").to_pylist()):
                if item_id not in wanted:
                    continue
                for col in cols:
                    if col[i] is not None and col[i] >= 0:
                        out[item_id] = float(col[i])
                        break
            return out
        except Exception as e:  # noqa: BLE001 - retried, then reported missing
            print(json.dumps({"partition_retry": name, "error": str(e)[:200]}))
            time.sleep(2**attempt)
    return {}


_SENSOR: dict = {}
_TILES: dict = {}
_CLOUD: dict = {}
_TRANSFORMERS: dict[int, Transformer] = {}


def _transformer(epsg: int) -> Transformer:
    if epsg not in _TRANSFORMERS:
        _TRANSFORMERS[epsg] = Transformer.from_crs(4326, epsg, always_xy=True)
    return _TRANSFORMERS[epsg]


def _tile_keys(lat: float, lon: float) -> list[str]:
    """Candidate tile names for a tile centre (its UTM zone first, then neighbours)."""
    zone = int((lon + 180) // 6) + 1
    keys = []
    for z in (zone, zone - 1, zone + 1):
        if not 1 <= z <= 60:
            continue
        for epsg in (
            (32600 if lat >= 0 else 32700) + z,
            (32700 if lat >= 0 else 32600) + z,
        ):
            x, y = _transformer(epsg).transform(lon, lat)
            col = round(x / TILE_M - 0.5)
            row = round(-y / TILE_M - 0.5)
            keys.append(f"EPSG:{epsg}_10_{col}_{row}")
    return keys


def _sample(i: int) -> tuple[np.ndarray, int, int]:
    """(cloud cover per Landsat capture, matched?, captures with a value)."""
    with h5py.File(f"{H5_DIR}/sample_{i}.h5", "r") as h5:
        key = f"timestamps_{_SENSOR['modality']}"
        if key not in h5:
            return np.zeros(0, np.float32), 1, 0
        ts = h5[key][()]
        lat, lon = (float(v) for v in h5["latlon"][()])
    days = [date(int(y), int(m) + 1, int(d)).toordinal() for d, m, y in ts]
    for key in _tile_keys(lat, lon):
        scenes = _TILES.get(key)
        if scenes is None or not all(d in scenes for d in days):
            continue
        cc = np.full(len(days), np.nan, np.float32)
        for j, d in enumerate(days):
            vals = [_CLOUD[s] for s in scenes[d] if s in _CLOUD]
            if vals:
                cc[j] = max(vals)
        return cc, 1, int(np.isfinite(cc).sum())
    return np.full(len(days), np.nan, np.float32), 0, 0


def main() -> None:
    """Build the sidecar (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("sensor", choices=sorted(SENSORS))
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parquet", action="store_true")
    args = ap.parse_args()
    modality, collection, properties = SENSORS[args.sensor]
    _SENSOR.update(
        modality=modality,
        collection=collection,
        properties=properties,
        meta=f"{SOURCE}/10_{modality}_allcap_meta",
    )

    names = sorted(os.listdir(_SENSOR["meta"]))
    with get_context("fork").Pool(args.workers) as pool:
        tiles = dict(pool.map(_parse_csv, names, chunksize=64))
    ids = sorted({s for t in tiles.values() for v in t.values() for s in v})
    print(json.dumps({"tiles": len(tiles), "unique_scenes": len(ids)}), flush=True)

    cloud: dict[str, float] = {}
    if args.parquet:
        days = [
            date(int(d[:4]), int(d[4:6]), int(d[6:8]))
            for d in (re.search(r"_(\d{8})T", s).group(1) for s in ids)  # type: ignore[union-attr]
        ]
        lo, hi = min(days), max(days)
        parts = [n for n, a, b in _partitions(collection) if b >= lo and a <= hi]
        wanted = frozenset(ids)
        print(json.dumps({"partitions": len(parts), "from": str(lo), "to": str(hi)}))
        with ThreadPoolExecutor(8) as ex:
            futures = [ex.submit(_parquet_covers, n, wanted) for n in parts]
            for n, fut in enumerate(futures):
                cloud.update(fut.result())
                print(json.dumps({"partitions_done": n + 1, "found": len(cloud)}))
    else:
        batches = [ids[k : k + 200] for k in range(0, len(ids), 200)]
        with ThreadPoolExecutor(16) as ex:
            for n, part in enumerate(ex.map(_cloud_covers, batches)):
                cloud.update(part)
                if n % 200 == 0:
                    print(
                        json.dumps({"stac_batches": n, "of": len(batches)}), flush=True
                    )
    print(
        json.dumps({"scenes_with_cloud_cover": len(cloud), "of": len(ids)}), flush=True
    )

    _TILES.update(tiles)
    _CLOUD.update(cloud)
    n = len(
        [f for f in os.listdir(H5_DIR) if f.startswith("sample_") and f.endswith(".h5")]
    )
    n = min(n, args.limit) if args.limit else n
    values, offsets, matched, with_value = [], [0], 0, 0
    with get_context("fork").Pool(args.workers) as pool:
        for k, (cc, ok, nv) in enumerate(pool.imap(_sample, range(n), chunksize=64)):
            values.append(cc)
            offsets.append(offsets[-1] + len(cc))
            matched += ok
            with_value += nv
            if k % 20000 == 0:
                print(
                    json.dumps(
                        {
                            "samples": k,
                            "of": n,
                            "tile_matched_so_far": round(matched / (k + 1), 4),
                            "with_cloud_cover_so_far": round(
                                with_value / max(offsets[-1], 1), 4
                            ),
                        }
                    ),
                    flush=True,
                )
    flat = np.concatenate(values) if values else np.zeros(0, np.float32)
    with h5py.File(args.out + ".tmp", "w") as h5:
        h5["offsets"] = np.array(offsets, dtype=np.int64)
        h5["cloud_cover"] = flat.astype(np.float16)
    os.replace(args.out + ".tmp", args.out)
    finite = flat[np.isfinite(flat)]
    print(
        json.dumps(
            {
                "samples": n,
                "tile_matched": round(matched / max(n, 1), 4),
                "captures": int(len(flat)),
                "with_cloud_cover": round(with_value / max(len(flat), 1), 4),
                "cloud_cover_pct_p25_50_75_90": [
                    round(float(v), 1) for v in np.percentile(finite, [25, 50, 75, 90])
                ]
                if len(finite)
                else None,
                "frac_over_50": round(float((finite > 50).mean()), 3)
                if len(finite)
                else None,
                "frac_over_20": round(float((finite > 20).mean()), 3)
                if len(finite)
                else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
