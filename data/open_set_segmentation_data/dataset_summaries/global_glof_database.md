# Global GLOF Database

- **Slug:** `global_glof_database`
- **Status:** completed — `classification`, **249 samples** (sparse-point table)
- **Source:** Glacier Lake Outburst Flood Database V3.0 (ESSD), Zenodo record
  [7330345](https://doi.org/10.5281/zenodo.7330345), license **CC-BY-4.0**.
- **Access:** unauthenticated HTTP download of `glofdatabase_V3.ods` (+ `Parameter_Readme.ods`)
  from the Zenodo record. No credentials required.

## What the source is

A single OpenDocument spreadsheet with **8 regional sheets** (Andes, European Alps,
NW North America, High Mountain Asia, Scandinavia, Other, Iceland, Greenland),
~3,150 documented glacier lake outburst flood (GLOF) events, ~57 attributes each. Two
secondary header rows are embedded per sheet (dropped by requiring a numeric `ID`). Key
fields used: `Longitude`/`Latitude` (source glacier-lake point), `Date`/`Date_Min`/`Date_Max`
(event date), `Lake_type` (impounding dam type).

## Key judgment calls

- **"points + polygons" → points only.** The manifest and prior notes describe "manually
  mapped lake polygons", but the V3.0 Zenodo release publishes **no polygon geometry** — only
  scalar `Lake_area_before/after` (m²) and `Perimeter` values derived from unpublished polygons.
  There is nothing larger-than-a-pixel to rasterize, so this is processed as a **pure
  sparse-point dataset** (spec §2a): one `points.geojson`, no per-sample GeoTIFFs.
- **Post-2016 filter.** Event dates span 1100–2022. Per the Sentinel-era rule, only events
  with a usable year ≥ 2016 **and** valid coordinates are kept: **249** of ~3,150 (verifies the
  prior run's ~259 estimate). The ~2,900 pre-2016 / coordinate-less events are dropped.
- **Change labels.** A GLOF is a sudden dated event. `change_time` = the event date when a
  full `YYYY-MM-DD` is known (**182 of 249**), else `null`. `time_range` = a 1-year window
  **centered** on the event date (or the calendar year for year-only events). Pretraining uses
  a sample when the input window spans the event (lake drainage before/after).
- **Positive-only.** These are presence points with no "no-GLOF" class; no negatives are
  fabricated (assembly supplies them per spec §5).

## Class scheme (dam type, from `Lake_type`)

12 canonical dam-type classes; `Lake_type` variants normalized (`ice – volc` → `ice_volcanic`;
slashed mixes like `ice/moraine`, `moraine/bedrock` → `combined`; en-dash normalized).
Post-2016 counts:

| id | class | count |
|----|-------|-------|
| 0 | ice_dammed | 181 |
| 1 | ice_volcanic | 31 |
| 2 | moraine_dammed | 16 |
| 3 | water_pocket | 4 |
| 4 | combined | 4 |
| 5 | supraglacial | 3 |
| 6 | bedrock_dammed | 2 |
| 7 | subglacial | 1 |
| 8 | volcanic | 1 |
| 9 | snow | 1 |
| 10 | other | 1 |
| 11 | unknown | 4 |

uint8, `nodata_value` = 255 (unused — 1×1 point labels). Distribution is heavily skewed to
`ice_dammed`; several classes are single-sample. Rare classes are kept per spec §5 (downstream
assembly drops too-small ones). Year distribution: 2016:46, 2017:48, 2018:57, 2019:41, 2020:31,
2021:16, 2022:10.

## Outputs

- `datasets/global_glof_database/points.geojson` — FeatureCollection, 249 Point features
  (WGS84 lon/lat), each with `id`, `label` (dam-type class id), `time_range`, `change_time`,
  `source_id` (`{sheet}/{ID}`).
- `datasets/global_glof_database/metadata.json` — class map + counts + notes.
- `raw/global_glof_database/` — cached `glofdatabase_V3.ods`, `Parameter_Readme.ods`, `SOURCE.txt`.

## Verification

- `points.geojson` structure valid; all 249 coordinates in range and (spot-checked) in
  glaciated regions (e.g. Greenland ~77 N/74 N); every `time_range` is ~365 days.
- Class ids in features match `metadata.json` (0–11). No S2 overlay was rendered; point
  locations are the source-published lake coordinates.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_glof_database
```
Idempotent: re-downloads only if the raw ODS is absent, then rewrites the point table + metadata.
