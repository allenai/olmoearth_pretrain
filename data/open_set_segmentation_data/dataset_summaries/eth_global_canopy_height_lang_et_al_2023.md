# ETH Global Canopy Height (Lang et al. 2023)

- **Slug:** `eth_global_canopy_height_lang_et_al_2023`
- **Status:** completed
- **Task type:** regression (`dense_raster`)
- **Regressed quantity:** `canopy_height` — top-of-canopy height in **metres**
- **num_samples:** 5000 (64×64 float32 label patches)
- **Family / region:** biomass / Global

## Source

ETH Zurich EcoVision Lab, *"A high-resolution canopy height model of the Earth"*
(Lang, Jetz, Schindler & Wegner, *Nature Ecology & Evolution* 2023). A global, wall-to-wall
canopy **top** height map for the year **2020** at 10 m ground sampling distance, produced
by a probabilistic deep-learning CNN ensemble that fuses NASA **GEDI** spaceborne lidar
(RH98 canopy-height reference) with **Sentinel-2** optical imagery, and also emits a
per-pixel predictive standard deviation (uncertainty).

- Project page: https://langnico.github.io/globalcanopyheight/
- Data DOI (global map): https://doi.org/10.3929/ethz-b-000609802
- License: **CC-BY-4.0** ("free of charge, without restriction of use"; attribution required).

## Access method (no credentials)

The product is released as **3°×3° tiles** on the ESA-WorldCover grid, as Cloud-Optimized
GeoTIFFs on ETH's public **libdrive** share (no login). The official tile browser
(`langnico.github.io/globalcanopyheight/assets/tile_index.html`) enumerates 2651 land
tiles; each tile's canopy-height ("`_Map`") download URL follows a fixed template on the
share, which the script builds from the tile name:

```
https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_{TILE}_Map.tif
```

Each `_Map` tile is **EPSG:4326**, ~10 m (1/12000°), single-band **uint8** = canopy top
height in **metres**, with **255 = no-data** (ocean, permanent snow/ice, masked pixels).
COG overviews are present. (The companion `_Map_SD.tif` uncertainty layer was **not**
downloaded — see below.)

## Bounded sampling (global derived product, spec §5)

This is a global derived-product raster, so we do **bounded-tile** `dense_raster` sampling
rather than pulling the planet. We resolved a curated, **cross-biome set of 35 tiles** from
biome seed points and downloaded only those (`raw/{slug}/`, ~9.4 GB). Biomes covered:

- **Tropical rainforest:** Amazon (Brazil, Peru), Congo (DRC, Gabon), Borneo, Sumatra,
  Papua New Guinea, Western Ghats.
- **Temperate forest:** US Pacific NW, US Appalachia, Central Europe (Germany), Japan,
  SE China, SE Australia, Chile Valdivian.
- **Boreal:** Canada, Siberia, Fennoscandia (Finland), Alaska.
- **Savanna / woodland:** Central Africa, Tanzania, Brazilian Cerrado, N. Australia,
  Madagascar dry forest, India dry deciduous.
- **Mediterranean / shrubland:** California, Iberia.
- **Grassland / steppe:** Sahel, US Great Plains, Central Asian steppe.
- **Tundra:** N. Canada, N. Siberia.
- Plus: SE US pine, Central Mexico highland, Myanmar mixed forest.

## Label construction

- **Scan:** each downloaded tile is scanned in parallel native-pixel row-chunks in 64×64
  blocks. A block is a candidate if ≥60 % of its pixels are observed land (≠255); its
  center lon/lat and mean-over-valid canopy height are recorded. Blocks within 260 px of a
  tile edge are skipped so the per-window reprojection never runs off the tile. 167,067
  candidate windows were gathered.
- **Balancing:** the raw height distribution is heavily **zero-inflated** (deserts,
  grassland, water edges) and **tall canopy is globally rare** (~5 % of land >30 m), so we
  **bucket-balance across fixed height buckets** `[0,1,3,5,10,15,20,25,30,40,300)` m,
  drawing 500 windows per bucket → **5000** windows with an even spread from 0 m to the
  tallest canopies (all 10 buckets filled).
- **Patch write:** each selected window is reprojected from EPSG:4326 ~10 m to a **local
  UTM 64×64 tile at 10 m** (≈640 m). The continuous height field is warped **bilinear**,
  and a validity mask is warped alongside and thresholded so the 255 no-data never blends
  into valid output pixels. The **uint8 metres** source is written as **float32 metres**;
  source no-data **255 → -99999** (`io.REGRESSION_NODATA`). Windows landing >70 % on
  no-data are skipped.
- **Time range:** the map is an **annual 2020** product → every sample gets a **1-year**
  window `[2020-01-01, 2021-01-01)`. No `change_time`.

### metadata.json regression block

`name=canopy_height`, `unit=meters`, `dtype=float32`, `nodata_value=-99999`,
`source_nodata_value=255`, `value_range=[0.0, 59.952]`, `buckets=[0,1,3,5,10,15,20,25,30,40,300]`.

## Why the SD/uncertainty layer is *not* used as a filter

Spec §4 suggests preferring high-confidence windows for derived products. We deliberately
did **not** filter by the SD layer here: model uncertainty in this product is strongly
**correlated with canopy height**, so an SD threshold would preferentially discard the
tall-canopy windows we most want to represent, defeating the height bucket-balancing.
Quality is instead ensured via the ≥60 % valid-fraction gate and the height balancing.

## Verification (§9)

- 5000 `.tif` + 5000 matching `.json`; every `.tif` single-band **float32**, **UTM** CRS
  (many zones, e.g. 32610/32630/32647/32719), **10 m** resolution, **64×64**, nodata
  **-99999**; per-pixel values in-range (sampled global range 0–59.95 m).
- All sample `time_range`s are the 1-year 2020 window (0 samples exceed 1 year).
- **Spatial/biome sanity:** mean per-region height matches ecology — Sahel/steppe/tundra
  ≈ 0–4 m; boreal/temperate 8–25 m; tropical rainforest (Congo, Amazon, PNG, Borneo)
  30–40 m; the single tallest sample (53 m) is in the US Pacific NW conifer tile. All 35
  regions contribute samples. A full Sentinel-2 image overlay was not run (would need heavy
  data-source setup); it is unnecessary here because the product is itself an
  exactly-georeferenced, Sentinel-2-derived raster and reprojection uses exact affine
  transforms — the biome-consistency check confirms geographic sensibility.
- Re-running the script is idempotent (existing `{sample_id}.tif` are skipped).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eth_global_canopy_height_lang_et_al_2023 --workers 64
```

Downloads the 35 curated tiles to `raw/{slug}/` (idempotent), scans + bucket-balances, and
writes `metadata.json` + `locations/{id}.tif|.json` under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/{slug}/`.

## Caveats

- Bounded, curated tile set (35 of 2651 land tiles): broad biome coverage but not exhaustive
  global sampling. `N00E114` (Borneo) contributes disproportionately (423 samples) because it
  supplies most of the scarce tall-canopy (>40 m) windows.
- Labels are a **model-derived** product (GEDI-referenced Sentinel-2 regression), not in-situ
  measurements; canopy top height in the product saturates/has higher uncertainty for very
  tall canopies. Included as a dense auxiliary regression label, as intended by the manifest.
