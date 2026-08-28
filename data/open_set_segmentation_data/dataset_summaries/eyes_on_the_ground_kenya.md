# Eyes on the Ground (Kenya)

- **slug**: `eyes_on_the_ground_kenya`
- **status**: **rejected** — **no recoverable geocoordinates** (SOP §8.2). Exact field GPS
  is deliberately removed for farmer privacy; every record's geometry is the **GADM36
  village/ward bounding box** in which the field lies (one polygon shared by all records
  in that village), not a per-field point. Median village box is **~45 km × ~35 km
  (~1,500 km²)** — thousands of Sentinel-2 pixels across — so a 10 m crop-type label
  cannot be placed on the S2 grid. This is a fundamental, permanent block (the
  fuzzing is intrinsic to the public release), not a `temporary_failure`.
- **task_type** (intended, had coordinates been usable): classification — sparse crop-type
  points (maize / beans / sorghum / other) → `points.geojson` per §2a/§4.
- **num_samples**: 0

## Source

- Manifest name: `Eyes on the Ground (Kenya)`; source **Source Cooperative (Lacuna Fund)**,
  <https://source.coop/lacuna/eyes-on-the-ground>; family `crop_type`, label_type `points`,
  region Kenya, time_range `[2020, 2022]`, license CC-BY-4.0 (dataset Documentation states
  **CC-BY-SA-4.0**), have_locally: false. DOI `10.34911/rdnt.1bs2jw` (orig. Radiant MLHub).
- Open S3-compatible access (no credential needed) via the Source Cooperative data proxy:
  `boto3` unsigned client, `endpoint_url=https://data.source.coop`, bucket `lacuna`, prefix
  `eyes-on-the-ground/`. Repo composition (252,926 objects): 112,308 field photos (`.jpg`,
  4.6 GB), 112,446 STAC/label `.json` (7.4 GB, dominated by per-site ERA5/ARC/TAMSAT/S2
  time-series subsets), a 2.7 GB `EotG_data_final.tar.gz` bundle, and `Documentation.pdf`.
- The clean label table is `data/labels/*.json` — **28,077 files, 57 MB total**, one
  per photo. Only these (not the multi-GB photos/ancillary series) would have been needed;
  per §8 the impractical photo download was correctly avoided.

## What the label records actually contain

Each `data/labels/<img>.json` is a one-feature GeoJSON `FeatureCollection`. Example
(`100_initial_1_2638_2638.json`):

```json
{"type":"FeatureCollection","features":[{"type":"Feature",
 "geometry":{"type":"Polygon","coordinates":[[[37.30895996,-0.45111084],
   [37.30895996,-0.15012503],[37.89470673,-0.15012503],
   [37.89470673,-0.45111084],[37.30895996,-0.45111084]]]},
 "properties":{"farmer_unique_id":"TN2114","site_id":2638,"crop_name":"sorghum",
   "sowing_date":"2020-04-17","expected_yield":250,"season":"LR2020",
   "spatial_location":"Chuka/Igambang'Ombe","spatial_unit":"gadm36",
   "datetime":"2020-05-17T00:00:00Z","filename":"..."}}]}
```

The **label content is excellent** — manually agronomist-assigned `crop_name` (maize
dominant; also sorghum, green gram, beans, etc.), plus `growth_stage` phenology, `damage`
(drought/weed/pest/disease/flooding), extent, sowing date, season (SR/LR + year), all
timestamped in 2020–2022 (post-2016, Sentinel era). Crop type at 10 m would be a standard,
in-scope signal. The blocker is purely spatial.

## Why rejected — coordinate fuzzing (SOP §8.2)

The published geometry is **not the field**. The dataset Documentation is explicit:

- **Appendix A:** *"All original images were geo-located using the GPS internal to
  smallholder farmer cellphones. To ensure privacy exact locations are removed and only
  bounding boxes of the village in which the field is located are reported (along with the
  village name). Village names are sourced from the GADM36 dataset."*
- **Appendix D:** *"Due to the lack of precise geolocation of the cellphone images (for
  privacy reasons) we provide point based subsets of Sentinel 2 / ARC2 / TAMSAT / ERA5 …"*

Empirically verified from the data itself (no full download needed): in a random 400-record
sample the 400 records collapse to only **41 distinct geometries**, exactly matching the 41
distinct `spatial_location` village names — i.e. **one shared GADM36 polygon per village**.
Measured box sizes (300-record sample): width min 16 km / median 45 km / max 112 km; height
min 13 km / median 35 km / max 113 km; median area ~1,538 km². A crop label attached to a
40 km box cannot be localized to a 10 m S2 pixel, and the box is not a homogeneous
land-cover region (villages contain mixed cropland, settlement, water, etc.), so no
aggregate/mask representation salvages it either (§8.2). Per §8.2, "a per-sample
tile/region id alone … is not sufficient"; the village box is a coarse region id.

The authors' own workaround (pre-extracted point Sentinel-2/S1/climate time series per
`site_id`) confirms the true coordinates were used internally but are **withheld** — those
ancillary files carry band values + dates only, **no lon/lat** — so precise locations cannot
be recovered from anything in the release.

## Judgment calls

- **Rejected, not accepted.** The manifest note steered toward acceptance ("crop type at
  10 m is a standard signal — accept") *but conditioned on* reconsidering "if coords are
  withheld/fuzzed." Triage found exactly that fuzzing, documented by the source, so the
  spec-correct outcome (§8.2 no-recoverable-geocoordinates) is rejection. This mirrors prior
  fuzzed-point rejections (e.g. FIA ~1 mi).
- **Rejected, not `temporary_failure`.** The source is open and reachable; the block is the
  intrinsic privacy fuzzing of the release, which re-running cannot fix.
- **Not `needs-credential`.** Access is fully open (unsigned S3); no gate.
- **No bulk download performed.** Georeferencing was checked cheaply first (§8.2): the
  label schema + Documentation established the village-box fuzzing from ~700 small JSON reads
  (~a few MB), so the 2.5 GB tarball / multi-GB photos were never pulled.
- **Secondary labels (damage/phenology).** Even kept as a single crop-type classification
  (with damage/health as a documented secondary signal), the spatial blocker is unchanged.

## Reproduce / revisit

No outputs written to weka `datasets/eyes_on_the_ground_kenya/` beyond `registry_entry.json`.
To reconfirm the blocker (open access, no credential), from the repo root:

```python
import boto3, botocore, json
s3 = boto3.client("s3", endpoint_url="https://data.source.coop",
                  config=botocore.config.Config(signature_version=botocore.UNSIGNED))
o = s3.get_object(Bucket="lacuna",
                  Key="eyes-on-the-ground/data/labels/100_initial_1_2638_2638.json")
print(json.load(o["Body"]))  # geometry is a ~40 km GADM36 village box, not a field point
```

This dataset would become usable only if a **coordinate-bearing** version were released
(true per-field lon/lat + acquisition date), at which point it is a strong 2020–2022 Kenya
smallholder crop-type point dataset (maize/beans/sorghum/other), ~1-year time window
anchored on each record's season, honoring the 25k cap.
