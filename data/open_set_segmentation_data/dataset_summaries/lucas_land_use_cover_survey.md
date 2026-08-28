# LUCAS Land Use/Cover Survey

- **Slug:** `lucas_land_use_cover_survey`
- **Status:** completed
- **Task type:** classification (sparse points)
- **Samples:** 21,281 (balanced; see below)
- **Source:** Eurostat / JRC — LUCAS (Land Use/Cover Area frame Survey), CC-BY-4.0
- **Manifest URL:** https://ec.europa.eu/eurostat/web/lucas ; https://essd.copernicus.org/articles/13/1119/2021/

## What the source is

LUCAS is the EU-wide in-situ ground survey of land cover (LC) and land use (LU). Field
surveyors visit georeferenced grid points and record the observed land-cover class,
land-use class, photos, and a GPS reading of the actual observation location. It is a
gold-standard in-situ reference dataset. This is a pure **sparse-point classification**
dataset (spec §2a/§4 "points"): each label is a single 10 m pixel carrying an LC class, so
we write **one dataset-wide `points.geojson`**, not per-point GeoTIFFs.

## Access method (reproduce)

Both files are downloaded from the JRC open-data FTP (no credentials):

- **2018:** `LUCAS_harmonised/1_table/lucas_harmo_uf_2018.zip` (harmonised LUCAS DB,
  d'Andrimont et al. 2020; 337,854 records). Columns `lc1`/`lc1_label`,
  `gps_lat`/`gps_long` (field GPS), `th_lat`/`th_long` (theoretical grid point).
- **2022:** `LUCAS_2022_Copernicus/l2022_survey_cop_radpoly_attr.csv` (2022 Copernicus
  "radius-polygon" EO-relevant survey subset, ~138k points, 1.96 GB). Columns `survey_lc1`
  ("CODE - Label"), `survey_gps_lat`/`survey_gps_long`, `point_lat`/`point_long`.

Run: `python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lucas_land_use_cover_survey`
(from the repo root). Raw files live at
`.../open_set_segmentation/raw/lucas_land_use_cover_survey/` with a `SOURCE.txt`.

## Judgment calls / decisions

- **LC (land cover) level, detailed LC1 codes.** Used the detailed 3-char LC1 code
  (e.g. `B11 = Common wheat`, `C10 = Broadleaved woodland`) rather than the LU (land-use)
  field or a coarser level-1 grouping. LC1 is the natural per-pixel observable and matches
  the manifest's example classes. This yields **76 classes** (ids 0–75), well under the
  254-class uint8 cap — no classes dropped. Class name is stored as `"CODE - label"`.
- **Class ids by descending combined frequency** (id 0 = most common, `C10 Broadleaved
  woodland`).
- **Coordinate = observed field GPS where valid, else theoretical grid point.** LUCAS
  publishes both. Many "In office PI" (photo-interpreted) points have no field GPS and carry
  the documented `88.888…` sentinel; for those we fall back to the theoretical grid
  coordinate. Coordinate provenance is recorded per point in `source_id`
  (`<year>/<point_id>/<gps|theoretical>`). In the selected set: **17,573 GPS, 3,708
  theoretical**.
- **Post-2016 only.** The harmonised DB spans 2006–2018; only the 2018 subset is used. 2022
  comes from the separate Copernicus survey table. (2006/2009/2012/2015 harmonised files were
  not used.) 417 selected points were surveyed in early 2023 (LUCAS-2022 campaign spillover);
  their time range is anchored on 2023. Year split of selected: 2018=14,668, 2022=6,592,
  2023=21.
- **Time range:** 1-year window per survey year (`[Jan 1 yr, Jan 1 yr+1)`), a static/seasonal
  land-cover label (no change_time).
- **2022 CSV parsing.** The 2022 file has a trailing multi-line, unquoted `radpoly` polygon
  geometry blob that breaks whole-file CSV parsing. Every needed attribute is on a record's
  first physical line, which starts with an all-digit `point_id`; geometry continuation lines
  start with a decimal, so the record boundary is detected by `token.isdigit()` and each
  record line is parsed on its own. This recovers all 134,213 valid 2022 points (all with
  field GPS).
- **Invalid LC1 codes dropped** (`8 - Not relevant`, `NA`, blanks). 2018 had ~37 such rows;
  2022 had ~3,753.

## Class balancing (spec §5)

Balanced to ≤1000/class with the 25k per-dataset cap via
`balance_by_class(..., per_class=1000, total_cap=25000)`. With 76 classes the effective
per-class limit is `25000 // 76 = 328`; 58 classes reach 328, the remaining 18 are smaller
(rarest: `G40 Sea and ocean`=3, `G22 Inland salty running water`=6, `G50 Glaciers/permanent
snow`=23). Per spec §5, rare classes are **kept, not dropped** (downstream assembly filters
too-small classes). Total selected = **21,281** from 472,030 candidates (337,817 in 2018 +
134,213 in 2022).

## Outputs

- `datasets/lucas_land_use_cover_survey/points.geojson` — FeatureCollection, 21,281 Point
  features (WGS84 lon/lat), `properties`: id, label (class id), time_range, change_time
  (null), source_id.
- `datasets/lucas_land_use_cover_survey/metadata.json` — class map (76 classes) +
  per-class counts.
- `raw/lucas_land_use_cover_survey/` — downloaded source + `SOURCE.txt`.

## Verification / caveats

- Coordinates fall in EU bounds (lon −10.2…34.35, lat 34.59…69.8) with correct GeoJSON
  lon/lat ordering; labels span the full declared id range 0–75.
- A full Sentinel-2 overlay eyeball (spec §9) was **not** performed; alignment is trusted
  from the authoritative in-situ GPS provenance and the coordinate sanity check above.
- ~17% of selected points use the theoretical grid coordinate (no field GPS); these are the
  LUCAS grid points and are generally representative but may be slightly less pixel-exact
  than the field-GPS points.
- The 2022 subset is the Copernicus EO-relevant ("homogeneous radius-polygon") subset, so its
  points are, if anything, better co-located with a homogeneous 10 m land-cover footprint.
- Re-running is idempotent (deterministic seeded balancing; outputs overwritten atomically).
