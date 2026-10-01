# CropSight-US

- **Slug**: `cropsight_us`
- **Task**: classification (per-pixel crop type)
- **Label type**: polygons (per-field crop-type ground truth)
- **Family / region**: crop_type / Contiguous United States
- **License**: CC-BY-4.0
- **Status**: completed — 14,193 label tiles

## Source & access

Manifest source is Zenodo record
[15702415](https://zenodo.org/records/15702415) (CROPSIGHT-US v1.0.0). That record's
files are **access-restricted** (Zenodo `access_right: restricted`; the files API and all
download links return HTTP 403), so it cannot be fetched unauthenticated. However, the
latest open-access version of the same record concept — recid **19501943, v1.0.1,
CC-BY-4.0** — publishes the identical product as a single open ZIP. We downloaded that
instead:

```
https://zenodo.org/api/records/19501943/files/cropsight-us_app_dat_v1.0.1.zip/content
```

Raw files (297 MB ZIP + extracted shapefile) live at
`raw/cropsight_us/`; provenance recorded in `raw/cropsight_us/SOURCE.txt`.

## What the source is

One WGS84 (EPSG:4326) ESRI shapefile, `cropsight-us_app_dat_v1.0.1.shp`, of **124,419
cropland field polygons** across CONUS. Each polygon was delineated from Sentinel-2 field
boundaries and its crop type assigned by a virtual audit of Google Street View imagery.
Attributes: `Label` (crop type), `Year`/`Month` (of the street-view image), and
confidence metrics (`Entropy`, `Variance`, `Confidence`). Geometry is all `Polygon`.

## Class mapping (17 crops)

Class ids follow manifest order. The shapefile spells two crops differently; these were
mapped to the manifest names:
- `peanuts` → `peanut`
- `potatoes` → `potato`

| id | class | id | class | id | class |
|----|-------|----|-------|----|-------|
| 0 | alfalfa | 6 | grape | 12 | soybean |
| 1 | almond | 7 | orange | 13 | sugarbeet |
| 2 | canola | 8 | peanut | 14 | sugarcane |
| 3 | cereal | 9 | pistachio | 15 | sunflower |
| 4 | corn | 10 | potato | 16 | walnut |
| 5 | cotton | 11 | sorghum | | |

## Processing

- **Time filtering**: crop-type labels are year-specific (fields rotate crops), so only
  fields with **`Year >= 2016`** (the Sentinel era, matching the manifest's 2016–2023
  range) were kept. This drops the 52,188 pre-2016 fields (2013–2015), leaving 72,231
  candidate fields. All confidence levels were retained.
- **Rasterization**: each field polygon is reprojected to its local UTM zone and
  rasterized at 10 m/pixel (`all_touched=True`). The tile is sized to the field's pixel
  footprint and **hard-capped at 64×64**; fields larger than 640 m are cropped to a 64×64
  window centered on the field. Value = class id **inside** the polygon; **255 (nodata)**
  outside, since the neighboring land use is unknown / unobserved.
- **Sampling**: tiles-per-class balanced to **≤1000 per class** (`balance_by_class`,
  seed 42). 12 abundant classes hit the 1000 cap; 5 rare classes keep all their fields.
- **Time range**: 1-year window `[Jan 1 YEAR, Jan 1 YEAR+1)` anchored on the labeled year.
  No change labels.

## Outputs

- `datasets/cropsight_us/metadata.json`
- `datasets/cropsight_us/locations/{000000..014192}.tif` (+ `.json`)

Each `.tif`: single-band uint8, local UTM at 10 m, ≤64×64, nodata=255.

## Sample counts per class (14,193 total)

| class | n | class | n | class | n |
|-------|---|-------|---|-------|---|
| alfalfa | 1000 | grape | 1000 | soybean | 1000 |
| almond | 1000 | orange | 1000 | sugarbeet | 709 |
| canola | 281 | peanut | 1000 | sugarcane | 1000 |
| cereal | 1000 | pistachio | 278 | sunflower | 185 |
| corn | 1000 | potato | 740 | walnut | 1000 |
| cotton | 1000 | sorghum | 1000 | | |

## Verification

- 14,193 `.tif` files, each with a matching `.json`; all single-band uint8, UTM CRS
  (EPSG:326xx), 10 m resolution, nodata 255.
- Dimension/value audit over 800 random tiles: max size 64×64 (cap respected); all pixel
  values in {0..16, 255}; zero out-of-range values.
- `metadata.json` class ids (0–16) cover all values appearing in the tiles.
- `classes_present` in each sample JSON matches the tile's class value; all `time_range`s
  are exactly 1 year.

## Caveats

- Used the open v1.0.1 mirror because the manifest's v1.0.0 record is access-restricted.
- Crop labels are model-audited from street view (per-field entropy/variance/confidence
  available in the source but not filtered on here); described as near-ground-survey
  quality. Confidence attributes are dropped in the output but could be used to filter
  low-confidence fields if desired.
- Outside-field pixels are nodata (255), not a background class — tiles carry only the
  audited field, so effective supervision is the field footprint.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cropsight_us
```

Idempotent: existing `{sample_id}.tif` are skipped on re-run.
