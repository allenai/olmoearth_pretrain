# SATFiD (Synthesized Alaskan Tundra Field Database)

- **slug:** `satfid_synthesized_alaskan_tundra_field_database`
- **manifest name:** SATFiD (Synthesized Alaskan Tundra Field Database)
- **status:** **completed**
- **task type:** classification (dominant tundra plant-functional-type per plot)
- **num_samples:** **350** sparse 1×1 points (`points.geojson`, spec §2a)
- **source:** ORNL DAAC (ABoVE), DOI [10.3334/ORNLDAAC/2177](https://doi.org/10.3334/ORNLDAAC/2177) —
  "Field Data on Soils, Vegetation, and Fire History for Alaska Tundra Sites, 1972-2020"
- **data paper:** Chen et al. 2024, ESSD 16, 3687 — [essd.copernicus.org/articles/16/3687/2024](https://essd.copernicus.org/articles/16/3687/2024/)
- **CMR collection:** `C2756289636-ORNL_CLOUD` (short name `FieldData_Alaska_Tundra_2177`)
- **license:** CC-BY-4.0

## What the dataset is

SATFiD is an in-situ field database harmonized from **37 real Alaskan-tundra field
campaigns**, 1972-2020 ("synthesized" = harmonized/compiled, **NOT synthetic**). It ships
as three CSVs; only `Tundra_field_database.csv` (197,830 rows, 34 cols) carries the label
signal. Each row is a georeferenced plot with `latitude`/`longitude` (decimal degrees),
`date` (`YYYYMMDD`), `yr_data` (`YYYY`), and per-plant-functional-type **percent-cover**
columns: `shrub_cover, lichen_cover, moss_cover, graminoid_cover, forb_cover, litter_cover`
(and `bare_cover`), nodata `-999`. Other columns (active layer, soil, biomass, fire
history) are out of scope for this label effort.

**Georeferencing (§8.2): PASSES.** Real decimal-degree lat/lon per plot (~5 dp ≈ 1 m
precision); all output points fall inside the documented study bbox
(lon −166.41..−141.68, lat 61.14..71.33) — verified output range lon −164.8..−148.6,
lat 61.14..70.32. Not coordinate-fuzzed, not tile-id-only.

## Access

Data files are behind the NASA Earthdata / URS **protected** download (public paths 404,
bundle 401 unauthenticated). Downloaded the protected bundle
`https://data.ornldaac.earthdata.nasa.gov/protected/bundle/FieldData_Alaska_Tundra_2177.zip`
(3.3 MB, HTTP 200) using authorized Earthdata credentials written to `~/.netrc`
(`machine urs.earthdata.nasa.gov`, chmod 600; creds sourced from
`.env` `NASA_EARTHDATA_USERNAME`/`_PASSWORD`). The 3 CSVs are
unzipped into `raw/{slug}/`. No open mirror exists (ORNL DAAC is the sole archive).

## Label mapping

Sparse-point **classification by dominant plant-functional-type cover**. For each plot,
`argmax` over the six manifest PFT cover columns → class id (nodata `-999` cover ignored):

| id | class | source column |
|----|-------|---------------|
| 0 | shrubs | shrub_cover |
| 1 | lichens | lichen_cover |
| 2 | mosses | moss_cover |
| 3 | graminoids | graminoid_cover |
| 4 | forbs | forb_cover |
| 5 | litter | litter_cover |

The cover columns are **independent per-layer estimates** (values can exceed 100%; row
sums range 1-346), so they are not a partition summing to 100 — `argmax`-dominant is the
faithful single-scalar encoding for the §2a point table (a multi-target regression of all
6 covers is not expressible as one dataset). `nodata_value = 255` (uint8 class raster
convention; not used in the point table itself).

**Filters applied** (row kept only if all hold): valid lat/lon; `yr_data >= 2016`; at least
one non-nodata PFT cover with a positive dominant value; `bare_cover` does not exceed the
dominant PFT cover (drops clearly bare-dominated plots — 4 such in the 2016+ subset).

## Sample counts (350 points)

| class | count |
|-------|-------|
| shrubs (0) | 102 |
| lichens (1) | 6 |
| mosses (2) | 24 |
| graminoids (3) | 218 |
| forbs (4) | 0 |
| litter (5) | 0 |

`forbs` and `litter` are retained in the class map but are never the dominant PFT in the
2016+ subset (they exist as valid cover values — 105 and 43 non-nodata in 2016+ — but always
as minor components). Kept per §5 (do not drop classes for sparsity; downstream assembly
filters too-small classes). Balancing (`balance_by_class`, ≤1000/class, 25k cap) does not
bind at this size. Points come from 3 source campaigns in the 2016+ window
(Loboda_2022 = 246, AKVEG_2022 = 65, Frost_2020 = 43 cover-bearing rows).

## Time handling

Seasonal/annual field cover → **1-year `time_range`** anchored on `yr_data`
(`io.year_range`); `change_time = null` (not a change/event label). Years present among the
350 points: 2016 (81), 2017 (127), 2018 (102), 2019 (40).

## Verification (§9)

- `points.geojson`: `FeatureCollection`, `task_type=classification`, `count=350`, 350 Point
  features; every `label` an int in 0-5; every `time_range` ≤ 1 year; every `change_time`
  null; `id`/`source_id` populated.
- `metadata.json` class ids (0-5) cover all label values present (0-3). `num_samples=350`.
- Spatial sanity: all points inside the ORNL-documented Alaska tundra study bbox (land,
  correct region). A full S2 RGB overlay was not performed because dominant-PFT distinctions
  (e.g. shrub- vs graminoid-tundra) are not reliably separable by eye in S2 imagery; the
  bbox/land check is the meaningful georeferencing sanity for this point-label type.
- Idempotent: re-running skips the CSV download (kept in `raw/`) and regenerates
  `points.geojson` deterministically (seeded balancing).

## Caveats / judgment calls

- **Classification, not regression** — source is multi-target fractional cover, but the §2a
  point schema stores one scalar and the manifest lists the 6 PFTs as `classes`; dominant-PFT
  is the faithful single-label encoding.
- **Sentinel-era subset only** — full DB is 1972-2020, heavily weighted to 2013-2014 (and the
  2016 bulk is mostly soil/biomass records with `-999` cover). Only **350** of the 46,194
  post-2016 rows carry vegetation cover; per §8.2 only the post-2016 subset is processed
  (pre/post-2016 mix → not era-rejected). Small but valid.
- **Bare soil** kept off-manifest: no `bare` class was invented; bare-dominated plots are
  dropped instead (only 4 in-subset), keeping the class set exactly the manifest's 6.
- **Observability at 10-30 m (minor)** — plots are small field footprints, but tundra
  vegetation is spatially coherent over tens of metres; treated as observable.

## Reproduce

```
# creds: ~/.netrc with `machine urs.earthdata.nasa.gov login <user> password <pass>` (chmod 600)
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.satfid_synthesized_alaskan_tundra_field_database
```
Script: `olmoearth_pretrain/open_set_segmentation_data/datasets/satfid_synthesized_alaskan_tundra_field_database.py`
(auto-downloads the protected bundle via netrc if the CSVs are not already in `raw/{slug}/`).
