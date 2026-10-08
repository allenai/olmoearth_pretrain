# South Africa Crop Type (Spot the Crop)

- **Slug:** `south_africa_crop_type_spot_the_crop`
- **Status:** completed
- **Task type:** classification (per-pixel crop type)
- **Samples:** 9,000 label patches (9 classes x 1,000 fields)

## Source

"Crop Type Classification Dataset for Western Cape, South Africa" (Western Cape Department
of Agriculture + Radiant Earth Foundation, 2021), produced for the **Radiant Earth Spot
the Crop Challenge**. Manual government field surveys over the Western Cape, paired with
2017 Sentinel-1/2 time series.

- Manifest name: `South Africa Crop Type (Spot the Crop)`
- URL: https://source.coop/radiantearth/south-africa-crops-competition
- License: **CC-BY-4.0**
- DOI: 10.34911/rdnt.j0co8q
- `have_locally: false`

## Access method

Radiant MLHub is deprecated; the dataset is mirrored on **Source Cooperative** and served
through its S3-compatible data proxy at `https://data.source.coop` (account `radiantearth`,
repo `south-africa-crops-competition`). Downloaded unsigned (no credentials) via
`download.download_s3_unsigned(bucket="radiantearth", key=..., endpoint_url="https://data.source.coop")`.
This required a small reusable extension to `download.py` (added an `endpoint_url`
parameter so S3-compatible hosts like Source Cooperative work).

## Georeferencing (checked per spec 8.2 — ACCEPTED)

The label layer is **fully georeferenced**, not coordinate-free chips:
`train/labels/{tile}.tif` is a per-pixel crop-code raster and `{tile}_field_ids.tif` a
per-pixel field-id raster. Both are **EPSG:32634** (UTM zone 34, with negative northings
for the southern hemisphere), **10 m/pixel**, north-up, 256x256. There are **2,650 train
tiles**.

Verified: tile 1000 center reprojects to lon/lat ~ (18.51, -32.24) and output patch
`000000` to (18.34, -31.78) — Western Cape. The source CRS and pixel grid are kept
**verbatim** (no resampling): the spec permits reusing a source window's CRS when it is
already UTM at 10 m, and pyproj transforms the negative-northing EPSG:32634 coordinates to
the correct S-hemisphere lon/lat, so downstream geographic pairing is exact. (Judgment
call: kept EPSG:32634 rather than reprojecting to the canonical southern zone 32734 —
that reprojection is a pure integer false-northing offset and would only add resampling
risk for no gain.)

## Label / class mapping

Classes follow the dataset's authoritative `labels.json`. **The manifest class list is
slightly wrong** — it lists "barley", but the real legend has "Weeds" (and both "Planted
pastures" and "Small grain grazing"). We follow `labels.json`. Crop code 0 = "No Data" ->
nodata (255). Crop codes 1-9 -> class ids 0-8:

| class id | name | code | fields (candidates) | samples |
|---|---|---|---|---|
| 0 | Lucerne/Medics | 1 | 8,340 | 1,000 |
| 1 | Planted pastures (perennial) | 2 | 13,917 | 1,000 |
| 2 | Fallow | 3 | 7,915 | 1,000 |
| 3 | Wine grapes | 4 | 24,225 | 1,000 |
| 4 | Weeds | 5 | 8,137 | 1,000 |
| 5 | Small grain grazing | 6 | 8,249 | 1,000 |
| 6 | Wheat | 7 | 10,712 | 1,000 |
| 7 | Canola | 8 | 1,494 | 1,000 |
| 8 | Rooibos | 9 | 4,124 | 1,000 |

Each field is painted with a **single uniform crop code** in the raster (verified against
`field_info_train.csv` — 100% match on a sample tile), so the per-field class is the
raster mode over the field's pixels.

## Processing

Per-field label patches, EuroCrops / CV4A-Kenya style. For each surveyed field we build a
**<=64x64 UTM 10 m** patch sized to the field footprint and centered on it: the crop class
id is burned at every labeled pixel in the window (neighboring labeled fields included),
and **255 (nodata/ignore)** fills unlabeled land — we only have a ground-truth crop label
inside surveyed fields, so unlabeled land is ignore, not a background class.

- **dtype/nodata:** uint8, 255 = nodata.
- **Sampling:** tiles-per-class balanced (`balance_by_class`, `per_class=1000`,
  `total_cap=25000`). All 9 classes had > 1,000 candidate fields, so every class hit
  exactly 1,000 and the 25k cap is not binding. 9,000 total.
- **Splits:** train only. The **test** split ships field-id rasters but the crop labels
  are **withheld** (competition holdout) — `test/labels/{tile}.tif` does not exist — so
  test carries no usable labels and is skipped.
- **Time range:** 1-year window on **2017** (`[2017-01-01, 2018-01-01)`), the survey /
  growing season. No change labels.

## Verification

- 9,000 `.tif` + 9,000 matching `.json`; all single-band uint8, EPSG:32634 at 10 m,
  <=64x64, nodata 255, values in {0..8, 255}.
- Sample JSON `time_range` is exactly 1 year; `metadata.json` class ids cover all label
  values.
- Georeferencing sanity: output patch centers reproject into the Western Cape; source grid
  kept verbatim so label/imagery alignment is inherited from the source (validated at
  source: tile centers land on Western Cape farmland).

## Caveats / judgment calls

- Manifest class list is inaccurate ("barley" not present; real legend used instead).
- Test crop labels withheld -> train split only.
- Kept source EPSG:32634 (negative-northing UTM 34) rather than reprojecting to 32734;
  lossless and geographically correct.
- "Weeds" and "Fallow" are legitimate scored classes in the challenge (Crop_ID_1..9), kept
  as normal classes; "No Data" (code 0) is treated as nodata/ignore.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.south_africa_crop_type_spot_the_crop
```
Idempotent: skips already-written `locations/{sample_id}.tif`. Raw label rasters are
mirrored under `raw/south_africa_crop_type_spot_the_crop/train/labels/`.
