# USDA Cropland Data Layer (CDL)

- **Slug**: `usda_cropland_data_layer_cdl`
- **Task type**: classification (dense_raster)
- **Status**: completed
- **Samples**: 24,321 label patches (64×64, uint8, local UTM @ 10 m)
- **Classes**: 105 (CDL codes remapped to compact ids 0–104)

## Source

USDA NASS Cropland Data Layer — an annual 30 m crop-specific land-cover raster for the
conterminous US (CONUS) with ~130 active categories, produced by a decision-tree classifier
trained on FSA farm-program ground truth plus NASA/USGS imagery.
<https://www.nass.usda.gov/Research_and_Science/Cropland/>. License: public domain.
`annotation_method`: derived-product (trained on FSA ground truth) — a MAP, but the major
crop classes are high-accuracy, so it is used directly (spec §4/§5 allow derived-product
maps sampled at high-confidence/homogeneous windows).

## Access (frugal — no national download)

The full CONUS CDL is only distributed as ~2 GB/year national archives, but only bounded
label windows are needed. We therefore pulled **regional clips** via the NASS
CroplandCROS / CropScape `GetCDLFile` web service, which clips the CDL to an arbitrary
EPSG:5070 (CONUS Albers) bounding box and returns a small GeoTIFF (a 45 km region ≈ 2 MB).
One clip per (region, year) was fetched — **~74 MB total raw**, versus ~4 GB for two
national rasters. No national raster was downloaded (spec §5 bounded-tile sampling).

Endpoint: `https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year=<Y>&bbox=<x1,y1,x2,y2>`
(bbox in EPSG:5070 meters); the XML response gives a `returnURL` to the clipped tif.

## Regions & years

**16 representative CONUS agricultural regions × 2 years (2021, 2024)** = 34 clips (one
region-year, Idaho 2021 vs 2024, produced identical geographies; all 34 fetched fine). Each
region is a 45 km EPSG:5070 box centered on a major crop geography, chosen to span the CDL
taxonomy:

| Region | Crops emphasized |
|---|---|
| Iowa Corn Belt, Central Illinois | corn, soybeans |
| Central Kansas | winter wheat, sorghum, corn |
| North Dakota | spring wheat, canola, sunflower, dry beans, barley |
| Red River Valley MN/ND | sugarbeets, potatoes, spring wheat, soybeans |
| Arkansas/Mississippi Delta, South Louisiana | rice, cotton, soybeans, sugarcane |
| N & S California Central Valley | rice, tomatoes, almonds, pistachios, walnuts, grapes, citrus, cotton |
| California Central Coast | grapes, strawberries, lettuce/vegetables |
| Texas High Plains | cotton, sorghum, corn, winter wheat |
| South Georgia, E North Carolina | peanuts, cotton, tobacco, soybeans |
| Washington Columbia Basin, Idaho Snake River | potatoes, apples, sugarbeets, barley, alfalfa, wheat |
| Michigan/Wisconsin (Great Lakes) | corn, alfalfa, cherries, blueberries, cranberries, sugarbeets |
| South Florida | citrus/oranges, sugarcane, vegetables |

Tile centers were verified to fall in the correct geography and UTM zone for every region.

## Class scheme (254-class uint8 cap)

Raw CDL codes were remapped to a compact uint8 id space by **descending frequency across
the sampled windows** (id 0 = most frequent). CDL has ~130 defined codes; **105 appeared**
in the sampled windows, so all kept — **0 dropped** by the 254-class cap. Each class's
original CDL code is preserved in `metadata.json` `classes[].cdl_code` (e.g. id 0 =
Grassland/Pasture (176), id 1 = Soybeans (5), id 3 = Corn (1)).

- **CDL code 0 (Background / out-of-CONUS)** and **81 (Clouds/No Data)** are mapped to
  **nodata (255)** and never become classes.
- Category names come from the official USDA NASS CDL legend (public domain), embedded in
  the script.

## Sampling

- **dense_raster, tiles-per-class balanced** (spec §4/§5). Non-overlapping ~64 px-footprint
  (630 m = 21 native 30 m px) windows scanned across all clips (167,284 candidates). A CDL
  code counts as **present** in a window when it covers **≥ 10 %** of the block — a
  high-confidence-presence filter appropriate for a derived-product map.
- `select_tiles_per_class` (rarest class first) up to **1000 tiles/class**, capped at the
  per-dataset **25,000** limit → **24,321 samples** selected. A tile counts toward every
  class it contains, so ubiquitous classes (Grassland/Pasture 5,639; Woody Wetlands 4,422)
  accumulate more than 1000 while rare classes are filled first.
- **No rare classes dropped and no synthetic negatives fabricated** (spec §5). Sparse
  classes are kept as-is (e.g. Dbl Crop Barley/Soybeans = 1, Speltz = 2, Hops = 2); the
  downstream assembly step filters classes below its minimum count.

## GeoTIFF / time range

- Single-band **uint8**, local **UTM @ 10 m**, 64×64. Native 30 m EPSG:5070 windows
  reprojected to UTM with **nearest** resampling (categorical). nodata = **255**.
- Each sample's `time_range` is the **1-year window of its CDL year** (annual crop label);
  `change_time = null`. `source_id` records region_year and block indices for provenance.

## Verification (spec §9)

- 24,321 `.tif` each with a matching `.json`; all 64×64, single-band uint8, UTM @ 10 m.
- Sampled 400 tifs: class ids in **0–103**, no values outside 0–104 (255 = nodata) — matches
  the 105-class map.
- Region-center lon/lat land in the correct geography and UTM zone for all 16 regions.
- Idempotent: re-running skips existing `{sample_id}.tif`.

## Judgment calls / caveats

- **Derived-product map used directly** — the manifest notes "Model product but many
  high-accuracy classes; sample confident/homogeneous pixels." Handled via the ≥10 %
  presence threshold rather than a per-class accuracy filter (CDL does not ship a
  per-pixel confidence layer at this access path).
- **CropScape instead of national download** — chosen for frugality; ~74 MB total vs
  multi-GB national rasters. A full-CONUS 2024 CDL GeoTIFF also exists on weka at
  `/path/to/cdl/`
  (verified identical CRS/resolution) and could be windowed as an alternative if CropScape
  is unavailable.
- **2 years (2021, 2024)** sampled for temporal diversity within the manifest 2016–2024
  range; more years could be added by extending `YEARS`.
- The `cdl.py` precedent
  (`olmoearth_pretrain/dataset_creation/rslearn_to_olmoearth/cdl.py`) is a different pipeline
  (rslearn-window → OlmoEarth Modality.CDL converter, requires a pre-ingested local rslearn
  CDL dataset); it informed the CDL semantics (code 0 = background nodata, uint8) but was
  not directly reusable for this external bounded-sampling task.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usda_cropland_data_layer_cdl --workers 64
```
