# USGS Closed Depressions in Karst Regions

- **Slug:** `usgs_closed_depressions_in_karst_regions`
- **Status:** `temporary_failure` (retry candidate — see "Access / blocker")
- **Task type (planned):** classification (binary closed-depression segmentation)
- **Source:** Jones, Doctor, Wood, Falgout & Rapstine (2021), "Closed depression density in
  karst regions of the conterminous United States: features and grid data", USGS
  ScienceBase, [doi:10.5066/P9EV2I12](https://doi.org/10.5066/P9EV2I12)
  (item `60f79cb0d34e9143a4ba4f4e`). **License: public domain.**

## What the source is

Closed depressions (sinkholes / karst depressions) extracted by automated algorithms from
the 1/3 arc-second (~10 m) National Elevation Dataset (NED/3DEP) across the conterminous
US. The DEM was first hydro-conditioned (breaching digital dams at road/stream crossings),
then depressions were restricted to karst-prone geologic units and screened against
developed land, open water, wetlands, and glacial/alluvial cover. The item ships several
attached files:

| File | Used? | Reason |
|------|-------|--------|
| `karst_depression_polys_conus.zip` (shapefile, 25 MB) | **Yes** | The individual closed-depression footprints — the observable phenomenon and this dataset's target. |
| `sink_density_1km_conus.zip`, `sink_density_classified_1km_conus.zip`, `sink_density_classified_polys_1km_conus.zip` | No | The manifest "sink-density classes". These are a **derived 1 km aggregate** (count/density of depressions per km²) — a regional landscape statistic, ~100× coarser than our 10 m grid, and density is not a per-pixel land feature a 10–30 m S2/S1/Landsat patch can resolve. **Judgment call: excluded** as not observable at 10–30 m. |
| `USGS_karst_depression_density_conus.gdb.zip` | No | Same content as a file geodatabase; redundant with the shapefile. |

## Access / blocker (why `temporary_failure`, not `rejected`)

The ScienceBase file-delivery endpoint `https://www.sciencebase.gov/catalog/file/get/...`
returned **HTTP 404 for every attached file** at processing time, while the catalog HTML
page (`/catalog/item/{id}`) and the metadata JSON API (`?format=json`) both returned 200
and advertised these exact download URLs as current. The 404 was reproduced across
**multiple different ScienceBase item IDs**, encoded and unencoded query forms, browser
User-Agents, cookies, `http`/`https`, and the `api.sciencebase.gov` host, and the S3
content buckets do not exist under the guessable names. This is a **source-side
file-delivery outage** (the metadata plane is healthy, the file plane is down), not a
permanent access gate and **not credential-gated** (rights are public domain). Per spec
§1a this is a transient failure and a retry candidate.

**Retry:** once the ScienceBase file endpoint recovers, simply re-run

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_closed_depressions_in_karst_regions
```

No credentials, no manual steps. Quick health check:
`curl -A '<browser UA>' 'https://www.sciencebase.gov/catalog/file/get/60f79cb0d34e9143a4ba4f4e?f=__disk__0a%2F43%2F6a%2F0a436a5eeeaec4c07eb7f5b229c406d6b1d0b8a7'`
should return a ~25 MB zip (not an HTML 404 page). The script detects a non-zip response
and raises a clear `TRANSIENT:` error, so it fails fast and idempotently until the source
is back.

## Planned processing (implemented and validated up to the download)

Class scheme (binary segmentation):

- `0 = background` — terrain outside a mapped depression: the genuine, observed
  non-depression context surrounding a sinkhole within the 640 m tile. The inventory
  delimits each depression footprint, so out-of-polygon pixels are real negatives (no
  synthetic far negatives added).
- `1 = closed_depression` — a karst closed depression / sinkhole footprint.
- `255` nodata declared but unused (every tile pixel is observed).

Steps:

1. Download + unzip `karst_depression_polys_conus.shp`.
2. Compute each polygon's area in an equal-area CRS (EPSG:5070 CONUS Albers); log the full
   area distribution.
3. **Observability filter (spec §8):** keep depressions with area ≥ `MIN_AREA_M2`
   (default **900 m²**, ≈ one Landsat 30 m pixel / a 3×3 S2 block); drop smaller ones as
   unresolvable at 10–30 m. `all_touched=True` rasterization guarantees a kept depression
   yields ≥1 positive pixel. `MIN_AREA_M2` is a CLI parameter (`--min_area_m2`) and should
   be re-checked against the logged distribution on the retry run — many depressions are
   tiny (10 m DEM origin), so the default may need tuning.
4. Reproject each kept depression to local UTM at 10 m and center it in a **64×64** (640 m)
   context tile (inside → 1, outside → 0). Depressions whose footprint exceeds 64 px on an
   axis (rare for sinkholes) are gridded into non-overlapping 64×64 windows, keeping
   intersecting windows (≤ `MAX_TILES_PER_FEATURE` = 16).
5. Round-robin selection across depressions (every depression contributes ≥1 tile before
   extras are added), capped at **25,000** tiles total (spec §5). Sinkholes are static
   topographic features with no per-feature date, so each sample gets a representative
   1-year window (`REP_YEAR = 2020`, Sentinel era); `change_time = null`.
6. Write `locations/{id}.tif` (single-band uint8, UTM 10 m, ≤64×64) + `.json`,
   `metadata.json` (records `area_filter_m2`, `area_distribution_m2`, source/kept counts),
   and mark the registry entry `completed`.

## Caveats

- Source includes some **false positives** (DEM processing artifacts) and some **non-karst
  depressions**; the authors note per-feature validation would be needed to confirm each as
  a true karst landform. Recorded in `metadata.json` / class description; downstream users
  should treat class 1 as "DEM-derived closed depression", not verified sinkhole.
- The "sink-density classes" manifest class is intentionally not produced (see table).
- Sample counts, per-class tile counts, and the spatial/temporal sanity check (§9) will be
  filled in on the successful retry run — no label outputs exist yet.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_closed_depressions_in_karst_regions
# optional: --min_area_m2 <float>  --workers <n>
```
Script: `olmoearth_pretrain/open_set_segmentation_data/datasets/usgs_closed_depressions_in_karst_regions.py`
