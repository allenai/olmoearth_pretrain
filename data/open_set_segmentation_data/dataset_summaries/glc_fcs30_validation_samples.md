# GLC_FCS30 Validation Samples

- **Slug**: `glc_fcs30_validation_samples`
- **Status**: completed
- **Task type**: classification (sparse points)
- **Num samples**: 16,872 (of 44,514 available; balanced to ≤1000/class)
- **Family / region**: land_cover / Global
- **License**: CC-BY-4.0

## Source & access

- Zenodo concept DOI [10.5281/zenodo.3551994](https://doi.org/10.5281/zenodo.3551994)
  → version record `3551995`, "A Dataset of Global Land Cover Validation Samples"
  (the reference/validation set used to assess the **GLC_FCS30** 30 m global land-cover
  product, Zhang et al., ESSD). Open access, no credentials required.
- Downloaded via `download.download_zenodo("3551995", raw_dir)`:
  - `GLC_ValidationSampleSet_v1.rar` (~0.9 MB) — an ESRI **point shapefile**.
  - `Data description.docx` — the LCCS codebook (Table 1) and source-data notes (Table 2).
- The RAR is extracted with `bsdtar` into `raw/{slug}/extracted/`. The shapefile
  `GLC_ValidationSampleSet.shp` is **EPSG:4326** with **44,514** point features and a single
  attribute of interest, `sample_lab` = the LCCS **fine** land-cover code (plus redundant
  `lon`/`lat` columns that match the geometry).

## Label / class mapping

Each point is a Google-Earth-interpreted, homogeneous land-cover reference pixel. `label` =
the LCCS **fine classification** class. The docx Table 1 fine system has exactly **24 codes**,
and all 24 appear in the data. Class ids are assigned 0–23 by ascending LCCS code:

| id | LCCS code | name | id | LCCS code | name |
|----|-----------|------|----|-----------|------|
| 0 | 10 | Rainfed cropland | 12 | 130 | Grassland |
| 1 | 11 | Herbaceous cover cropland | 13 | 140 | Lichens and mosses |
| 2 | 12 | Tree/shrub cover (orchard) cropland | 14 | 150 | Sparse vegetation |
| 3 | 20 | Irrigated cropland | 15 | 152 | Sparse shrubland |
| 4 | 50 | Evergreen broadleaved forest | 16 | 153 | Sparse herbaceous cover |
| 5 | 60 | Deciduous broadleaved forest | 17 | 180 | Wetlands |
| 6 | 70 | Evergreen needleleaved forest | 18 | 190 | Impervious surfaces |
| 7 | 80 | Deciduous needleleaved forest | 19 | 200 | Bare areas |
| 8 | 90 | Mixed leaf forest | 20 | 201 | Consolidated bare areas |
| 9 | 120 | Shrubland | 21 | 202 | Unconsolidated bare areas |
| 10 | 121 | Evergreen shrubland | 22 | 210 | Water body |
| 11 | 122 | Deciduous shrubland | 23 | 220 | Permanent ice and snow |

The raw LCCS code is also retained per point as `properties.lccs_code` in `points.geojson`.
Per-class descriptions (with the LCCS Level-1 group) are in `metadata.json`.

## Output format (spec §2a)

Pure sparse-point dataset (each label is one 10 m pixel), so a **single dataset-wide
GeoJSON point table** `datasets/{slug}/points.geojson` is written (no per-point GeoTIFFs).
Coordinates are WGS84 `[lon, lat]`; `properties` carry `id`, `label`, `time_range`,
`change_time` (null), `source_id` (original shapefile FID), and `lccs_code`. `metadata.json`
holds the 24-class map.

## Sampling

- `sampling.balance_by_class(records, "label", per_class=1000)` with the default 25k cap.
  24 classes → effective per-class ≤ min(1000, 25000//24) = 1000, so up to 24k, well under
  the cap. **All 24 classes are kept.** Selected class counts (≤1000 each):

  Rainfed cropland 1000, Herbaceous cover cropland 1000, Orchard cropland 246, Irrigated
  cropland 810, Evergreen broadleaved 1000, Deciduous broadleaved 1000, Evergreen
  needleleaved 1000, Deciduous needleleaved 973, Mixed leaf 1000, Shrubland 1000, Evergreen
  shrubland 214, Deciduous shrubland 275, Grassland 1000, Lichens/mosses 304, Sparse veg 1000,
  Sparse shrubland 183, Sparse herbaceous 149, Wetlands 1000, Impervious 498, Bare areas 1000,
  Consolidated bare 104, Unconsolidated bare 116, Water body 1000, Ice/snow 1000. **Total
  16,872.** Nine classes are naturally sparse (<1000); none dropped (downstream assembly
  filters too-small classes per §5).

## Time range & change handling

The shapefile has **no per-point acquisition date**, and these are static/stable reference
points (chosen to be homogeneous, persistent land cover and visually re-checked). GLC_FCS30
spans the ~2015–2020 epochs, so per spec §5 (static labels → representative 1-year Sentinel-era
window) every point is assigned a single **2018** window `[2018-01-01, 2019-01-01)` (mid-point
of 2015–2020, firmly inside the Sentinel-2 era). `change_time` is null (no dated event).

## Verification (spec §9)

- `points.geojson`: `FeatureCollection`, `task_type=classification`, `count=16872`, 24 distinct
  label ids 0–23, all with a ≤1-year `time_range`. Global extent (lon −179.8…179.3,
  lat −54.7…81.6).
- `metadata.json` class ids (0–23) cover every label value present.
- Coordinate/class plausibility spot-check: permanent ice/snow points concentrate at high
  latitudes (62% above |55°|; the ~28°N minimum corresponds to Himalaya/Tibet high-mountain
  glaciers), impervious surfaces sit mostly in mid-latitudes — consistent with reality. (No
  full S2 raster overlay was pulled; the source is already high-resolution-imagery-verified
  reference data.)
- Re-running the script is idempotent (skips the existing download; deterministic seeded
  selection yields the same 16,872 points).

## Caveats

- Representative-year choice (2018) is a judgment call because the source carries no per-point
  date; land cover here is deliberately stable, so any Sentinel-era year is defensible.
- A few LCCS fine classes are semantically close at 10–30 m (e.g. consolidated vs
  unconsolidated bare areas; sparse shrubland vs sparse herbaceous) and are sparse; kept as-is.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glc_fcs30_validation_samples
```
