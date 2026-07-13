# CAL FIRE FRAP Fire Perimeters — `cal_fire_frap_fire_perimeters`

**Status:** completed · **task_type:** classification (binary burned-area segmentation) ·
**num_samples:** 25,000

## Source

CAL FIRE Fire and Resource Assessment Program (FRAP) **"California Fire Perimeters
(all)"** — the authoritative historical wildland-fire perimeter polygon layer for
California, updated annually. Public domain.

- Manifest landing page (legacy File-GDB download): `https://frap.fire.ca.gov/data/frapgisdata-sw-fireperimeters_download`
- **Access used:** the legacy `frap.fire.ca.gov/media/fire-perimeters/fire*.gdb.zip`
  downloads now 301-redirect to a landing page and 403 (gated). CAL FIRE's own
  `egis.fire.ca.gov` ArcGIS service requires a token. So I pulled the **identical layer**
  from CAL FIRE-Forestry's public **hosted** ArcGIS Feature Service (no credentials):
  `https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/California_Historic_Fire_Perimeters/FeatureServer/0`
  (layer name "California Fire Perimeters (all)"). Downloaded all `YEAR_ >= 2016` features
  as GeoJSON (EPSG:4326) via paginated queries → `raw/.../perimeters_2016plus.geojson`
  (4,266 perimeters).

Each feature is one fire perimeter with attributes `YEAR_`, `ALARM_DATE` (ignition date,
epoch ms), `CONT_DATE`, `CAUSE` (ignition-cause code), `GIS_ACRES`, `FIRE_NAME`, etc.
All 4,266 post-2016 records have a non-null `ALARM_DATE`. Acreage spans 0.001 →
~1,032,700 acres (2020 August Complex); mean ~2,982 acres.

## Label design

**Binary burned-area segmentation** (uint8):
- `0` = background (outside the perimeter — unburned in this fire's window)
- `1` = fire (burned area inside a FRAP perimeter)
- `255` = nodata (declared, unused)

Ignition **CAUSE** and **acreage** are per-fire attributes that are **not observable
per-pixel** from 10–30 m S2/S1/Landsat (a burn scar looks the same regardless of ignition
cause), so they are kept as **provenance metadata only** (`provenance.cause_codes` maps the
19 FRAP CAUSE codes), not as label classes. This is why the manifest's single "fire
perimeter (year, cause, acreage)" class becomes a background/fire binary mask.

Per spec §5, **no synthetic far negatives** are fabricated: background (0) appears only as
genuine out-of-perimeter context inside fire tiles (the perimeter authoritatively delimits
the burned extent). Downstream assembly supplies additional negatives.

## Change semantics (this is a change/event dataset)

A fire is a dated change event. Each sample carries `change_time = ALARM_DATE` and a
`time_range` of **±180 days (360-day, ≤1-year window) centered on `change_time`** (spec
§5), so pretraining pairs the burned-area mask with imagery spanning the fire (before +
after the scar appears). Metadata flags `is_change_dataset: true`.

**Pre-2016 filtering:** only `YEAR_ >= 2016` fires are used (Sentinel era); FRAP's large
pre-2016 back-catalogue is filtered out.

## Tiling & sampling

- Perimeters reprojected to local **UTM at 10 m/pixel** (CA falls in EPSG:32610 / 32611).
- Small fire (footprint ≤ 64×64 px = 640 m): **one centered 64×64 tile**.
- Large fire: gridded into **non-overlapping 64×64 windows**; windows that actually
  intersect the perimeter are kept, and up to **`MAX_TILES_PER_FIRE = 40`** are randomly
  sampled per fire for geographic spread.
- Inside polygon → 1, outside → 0 (`rasterize_shapes`, `all_touched=False`).
- **Selection:** round-robin across fires (every fire contributes ≥1 tile before big fires
  add more), capped at **25,000** tiles total (`sampling.MAX_SAMPLES_PER_DATASET`).
  Candidate pool was 26,821 across all 4,266 fires → 25,000 selected.

**Counts:** 25,000 tiles. Tiles with background present: 19,398; fire-only
(large-fire interiors): 5,602. Per-year spread: 2016:2222, 2017:3928, 2018:2686,
2019:1844, 2020:3656, 2021:2045, 2022:1403, 2023:1652, 2024:3202, 2025:2362.

## Verification (§9)

- 5 tifs: single-band, `(1,64,64)`, uint8, UTM 10 m (EPSG:32610/32611), values {0,1}. ✓
- Every `.tif` has a matching `.json` (25,000 / 25,000); `time_range` span = 360 days,
  `change_time` set, `classes_present` recorded. ✓
- metadata.json: `task_type=classification`, `num_samples=25000`, `nodata_value=255`,
  classes = [(0, background), (1, fire)]. ✓
- Geographic sanity: 200 random tile centroids all fall inside the California lon/lat box
  (0 outliers). ✓
- Fire-fraction over 300 random tiles: mean 0.48, min 0.00, max 1.00, 18% fully-fire
  (large-fire interiors) — consistent with a mix of boundary and interior tiles. ✓
- A full Sentinel-2 image overlay was **not** performed (rslearn S2 fetch is
  heavy/out-of-band); georeferencing is exact because tiles are written via
  `GeotiffRasterFormat` in the same UTM projection the perimeter was rasterized in.

## Judgment calls / caveats

- Dropped per-fire CAUSE/acreage as classes (not per-pixel observable) → binary mask.
- Used the public hosted ArcGIS layer instead of the gated legacy File-GDB (identical
  data). If the user prefers the official GDB, credentials for `egis.fire.ca.gov` or a
  manual download from the landing page would be needed.
- `time_range` centered on ignition gives ~6 months of post-fire imagery; if finer
  temporal precision is later wanted (scar strictly post-fire), narrow to a forward window.
- Perimeters can overlap across years; within a single fire's ±180 d window, out-of-
  perimeter pixels labeled background may (rarely) have burned in a different-year fire.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cal_fire_frap_fire_perimeters
```
Idempotent: existing `locations/{id}.tif` are skipped. Raw GeoJSON is cached at
`raw/cal_fire_frap_fire_perimeters/perimeters_2016plus.geojson`.
