"""Process fMoW-Sentinel (Functional Map of the World) into an open-set-segmentation
point table.

fMoW-Sentinel (Cong et al. 2022, SatMAE; Stanford Digital Repository
https://purl.stanford.edu/vg497cb6002) pairs the Functional Map of the World locations
with Sentinel-2 imagery. Each metadata row is one Sentinel-2 composite image for a
facility location, with columns ``category`` (one of 62 functional land-use classes),
``location_id`` (fMoW location index within a category+split), ``image_id``
(``<100`` = a real fMoW acquisition, ``>=100`` = a synthetic 6-month-interval composite),
``timestamp`` (UTC center of the composite interval), and ``polygon`` (WGS84 lat/long
bbox of the location). Metadata CSVs (train/val/test_gt) are small (~220 MB total); the
77 GB image tarball is NOT downloaded — pretraining supplies its own imagery.

Encoding decision (spec 2a/4 "scene-level" + task guidance): a fMoW category labels a
**facility at a location** (a golf course, stadium, port, gas station, ...). The bbox is
the facility footprint (median ~0.4 km), and the pixels surrounding the facility are not
reliably the same class, so a uniform-class tile would overclaim. We therefore emit one
**1x1 sparse point classification** label at the facility bbox **centroid** with the
category class, written to a single dataset-wide ``points.geojson`` (spec 2a) rather than
tens of thousands of tiny per-facility GeoTIFFs. The 62 fMoW categories map to class ids
0..61 (descending unique-location frequency; well under the 254-class uint8 cap).

Deduplication: the raw CSVs are a per-location image time series (882,779 rows over
82,012 locations); we collapse each unique (split, category, location_id) location to ONE
point (the facility land use is static across its time series). Balanced to <=1000/class,
subject to the 25k per-dataset cap (spec 5) => ~403/class effective across 62 classes.

Time range (spec 5): the facility is a static land use. For each location we pick a
representative **post-2016** image — preferring a real fMoW acquisition (image_id<100),
else a synthetic composite center — and set a 1-year window centered on that acquisition
timestamp (image-acquisition-based ~1-year window). Post-2016 rule: rows before 2016 are
dropped; the 11 locations with no post-2016 image are dropped entirely.

Note on parallelism: the scan is three bulk CSV reads (not the tens-of-thousands of small
weka files the mp.Pool guidance targets) and the write is a single points.geojson, so
vectorized pandas is used directly; no multiprocessing.Pool is needed here.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.fmow_sentinel_functional_map_of_the_world``
"""

import argparse
import re
from collections import Counter
from datetime import UTC, datetime

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "fmow_sentinel_functional_map_of_the_world"
NAME = "fMoW-Sentinel (Functional Map of the World)"
PER_CLASS = 1000
STACKS_BASE = "https://stacks.stanford.edu/file/druid:vg497cb6002/"
SPLITS = ["train", "val", "test_gt"]
HALF_WINDOW_DAYS = 180  # +/-180d => 360-day window, within the pretraining cap

_NUM_RE = re.compile(r"-?\d+\.?\d*")

# Concise definitions for the 62 fMoW functional/land-use categories (from the fMoW
# challenge taxonomy). Populated into metadata.json class descriptions (spec 3).
CATEGORY_DESCRIPTIONS = {
    "airport": "Airport complex: runways, taxiways, terminals and apron areas.",
    "airport_hangar": "Large hangar building for aircraft storage/maintenance at an airfield.",
    "airport_terminal": "Passenger terminal building of an airport.",
    "amusement_park": "Amusement/theme park with rides, coasters and attractions.",
    "aquaculture": "Aquaculture site: fish/shellfish ponds, pens or enclosures.",
    "archaeological_site": "Archaeological site with ruins or excavated structures.",
    "barn": "Agricultural barn / livestock or storage building on farmland.",
    "border_checkpoint": "Border crossing / checkpoint facility with inspection lanes.",
    "burial_site": "Cemetery or burial ground.",
    "car_dealership": "Vehicle dealership lot with rows of parked cars and a showroom.",
    "construction_site": "Active construction site with bare ground, equipment or partial structures.",
    "crop_field": "Cultivated agricultural crop field.",
    "dam": "Dam impounding a river or reservoir.",
    "debris_or_rubble": "Area of debris or rubble (e.g. post-disaster or demolition).",
    "educational_institution": "School, college or university campus.",
    "electric_substation": "Electrical substation with transformers and switchgear.",
    "factory_or_powerplant": "Factory, industrial plant or power-generation station.",
    "fire_station": "Fire station with apparatus bays.",
    "flooded_road": "Roadway inundated by flood water.",
    "fountain": "Ornamental fountain / water feature.",
    "gas_station": "Fuel/gas station with pump canopy.",
    "golf_course": "Golf course: fairways, greens and bunkers.",
    "ground_transportation_station": "Bus/train/ground-transport station.",
    "helipad": "Helicopter landing pad.",
    "hospital": "Hospital or medical-center complex.",
    "impoverished_settlement": "Informal / impoverished settlement (dense low-rise dwellings).",
    "interchange": "Highway interchange with ramps and overpasses.",
    "lake_or_pond": "Inland lake or pond.",
    "lighthouse": "Coastal lighthouse tower.",
    "military_facility": "Military base or installation.",
    "multi-unit_residential": "Multi-unit residential building(s) (apartments/condos).",
    "nuclear_powerplant": "Nuclear power plant with reactor containment and cooling structures.",
    "office_building": "Commercial office building.",
    "oil_or_gas_facility": "Oil/gas processing, refining or storage facility.",
    "park": "Public park / green space.",
    "parking_lot_or_garage": "Surface parking lot or parking garage.",
    "place_of_worship": "Church, mosque, temple or other place of worship.",
    "police_station": "Police station.",
    "port": "Seaport/harbor with quays, cranes and container yards.",
    "prison": "Prison / correctional facility with perimeter walls.",
    "race_track": "Motor or horse race track / oval circuit.",
    "railway_bridge": "Railway bridge carrying tracks over water or terrain.",
    "recreational_facility": "Sports/recreational facility (fields, courts, stadium-like venues).",
    "road_bridge": "Road bridge carrying a roadway over water or terrain.",
    "runway": "Airport runway strip.",
    "shipyard": "Shipyard with drydocks and shipbuilding berths.",
    "shopping_mall": "Shopping mall / large retail complex with parking.",
    "single-unit_residential": "Detached single-family residential building.",
    "smokestack": "Industrial smokestack / chimney.",
    "solar_farm": "Solar photovoltaic farm with panel arrays.",
    "space_facility": "Spaceport / space launch or research facility.",
    "stadium": "Large stadium / arena with seating bowl.",
    "storage_tank": "Storage tank(s) for liquids or gas.",
    "surface_mine": "Open-pit / surface mine or quarry.",
    "swimming_pool": "Swimming pool.",
    "toll_booth": "Highway toll booth / toll plaza.",
    "tower": "Tall freestanding tower (communications/observation).",
    "tunnel_opening": "Tunnel portal / opening.",
    "waste_disposal": "Waste disposal / landfill site.",
    "water_treatment_facility": "Water or wastewater treatment plant with circular/rectangular basins.",
    "wind_farm": "Wind farm with turbine arrays.",
    "zoo": "Zoo / animal park with enclosures.",
}


def parse_timestamp(ts: str) -> datetime:
    """Parse an fMoW UTC timestamp; handles both plain and fractional-second forms
    (e.g. ``2016-10-01T00:00:00Z`` and ``2016-04-16T18:44:27.405Z``).
    """
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp: {ts!r}")


def bbox_centroid(polygon_wkt: str) -> tuple[float, float]:
    """Centroid (lon, lat) of an axis-aligned WGS84 bbox WKT POLYGON string."""
    nums = [float(x) for x in _NUM_RE.findall(polygon_wkt)]
    lons = nums[0::2]
    lats = nums[1::2]
    return (min(lons) + max(lons)) / 2.0, (min(lats) + max(lats)) / 2.0


def download_metadata(raw_dir) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fn in ["README.md", *[f"{s}.csv" for s in SPLITS]]:
        download.download_http(STACKS_BASE + fn, raw_dir / fn)


def load_locations(raw_dir) -> pd.DataFrame:
    """Return one representative post-2016 row per (split, category, location_id) location."""
    frames = []
    for split in SPLITS:
        df = pd.read_csv(raw_dir / f"{split}.csv")
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = df["timestamp"].str[:4].astype(int)
    df = df[df["year"] >= 2016].copy()  # post-2016 (Sentinel era) only
    df["real"] = df["image_id"] < 100  # real fMoW acquisition vs synthetic composite
    # Prefer real acquisitions, then earliest timestamp, deterministically.
    df = df.sort_values(
        ["split", "category", "location_id", "real", "timestamp"],
        ascending=[True, True, True, False, True],
    )
    reps = df.groupby(["split", "category", "location_id"], as_index=False).first()
    return reps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing outputs"
    )
    args = parser.parse_args()

    io.check_disk()

    out = io.dataset_dir(SLUG) / "points.geojson"
    if out.exists() and not args.force:
        import json

        n = json.load(out.open()).get("count")
        print(f"{out} exists; skipping (use --force to regenerate)")
        manifest.write_registry_entry(
            SLUG, "completed", task_type="classification", num_samples=n
        )
        return

    manifest.write_registry_entry(SLUG, "in_progress")
    raw = io.raw_dir(SLUG)
    download_metadata(raw)
    io.check_disk()

    reps = load_locations(raw)
    print(f"unique post-2016 locations: {len(reps)}")

    # Class ids 0..N-1 by descending unique-location frequency (62 categories << 254 cap).
    freq = Counter(reps["category"])
    ordered = sorted(freq, key=lambda c: (-freq[c], c))
    cat2id = {c: i for i, c in enumerate(ordered)}

    lons, lats = zip(*(bbox_centroid(p) for p in reps["polygon"]))
    reps = reps.assign(lon=lons, lat=lats, label=reps["category"].map(cat2id))

    records = reps.to_dict("records")
    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class, 25k cap)")

    points = []
    for i, r in enumerate(selected):
        center = parse_timestamp(r["timestamp"])
        points.append(
            {
                "id": f"{i:06d}",
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "label": int(r["label"]),
                "time_range": io.centered_time_range(center, HALF_WINDOW_DAYS),
                "source_id": (
                    f"{r['split']}/{r['category']}/{r['location_id']}/img{r['image_id']}"
                ),
            }
        )
    io.write_points_table(SLUG, "classification", points)

    sel_counts = Counter(r["category"] for r in selected)
    classes = [
        {
            "id": cat2id[c],
            "name": c,
            "description": CATEGORY_DESCRIPTIONS.get(c),
        }
        for c in ordered
    ]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Stanford / IARPA (fMoW-Sentinel, Cong et al. 2022)",
            "license": "fMoW Challenge Public License (metadata: locations/categories)",
            "provenance": {
                "url": "https://purl.stanford.edu/vg497cb6002",
                "have_locally": False,
                "annotation_method": "manual/crowdsourced fMoW annotations + expert QA",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {c: sel_counts.get(c, 0) for c in ordered},
            "notes": (
                "fMoW-Sentinel 62 functional land-use categories. Each label is a 1x1 "
                "sparse point at a facility's bbox centroid (WGS84), written to "
                "points.geojson (spec 2a) -- facilities are point-like land uses, not "
                "uniform-class tiles. One point per unique (split, category, location_id) "
                "location (per-image time series collapsed). Post-2016 images only; 1-year "
                "window centered on a representative acquisition timestamp (real fMoW "
                "acquisition preferred, else synthetic 90-day composite center). Balanced "
                "to <=1000/class under the 25k cap (~403/class effective)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
