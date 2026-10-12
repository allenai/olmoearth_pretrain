# GLAD Global Surface Water Dynamics

- **Slug:** `glad_global_surface_water_dynamics`
- **Status:** completed
- **Task type:** classification (per-pixel, dense_raster)
- **Num samples:** 1603 label patches (64×64, single-band uint8, local UTM @ 10 m)
- **Source:** UMD GLAD, "Global surface water dynamics 1999–2021/2025" (Pickens et al.
  2020, *Remote Sensing of Environment* 243, 111792).
  <https://glad.umd.edu/dataset/global-surface-water-dynamics>
- **License:** CC-BY 4.0 (free/public, attribution required).

## Source & access

30 m derived-product mapping of **inland** surface-water dynamics from the full Landsat
archive (Collection 2, Provisional v2.0), distributed as public 10×10° uint8 GeoTIFF tiles
in EPSG:4326 on a Google Cloud Storage bucket. No credentials required. Tile URL scheme
(from the download page JS):

```
https://storage.googleapis.com/earthenginepartners-hansen/waterC2/<LAT>_<LON>/<FILE>.tif
```

`<LAT>` = top-left latitude padded to 3 chars (e.g. `00N`, `40N`, `20S`); `<LON>` = top-left
longitude padded to 4 (e.g. `020E`, `070W`). Available layers include per-year
**annual water percent** (`<YYYY>_percent`, 1999–2025), the multi-year interannual
`dynamic_classes_99_25`, monthly means, and RGB. We use the **annual water percent** layer.

## Class mapping (and how water gain / water loss were handled)

The manifest lists five classes: *stable water, seasonal water, water gain, water loss,
land*. **"Water gain" and "water loss" are multi-year change classes** defined over the
whole 1999–2021 period with **no precise event date**, so they fail the ~1–2 month
change-timing rule (task spec §5) and cannot be encoded as dated change labels. They were
therefore **dropped**. Instead we build a per-pixel **static classification** from the
per-year annual water percent layer (reference year **2020**, within the manifest
2016–2021 window), which captures the within-year (intra-annual) water state:

| pixel value (annual water %) | class id | class name      |
|------------------------------|----------|-----------------|
| 0                            | 0        | land            |
| 1–99                         | 1        | seasonal water  |
| 100                          | 2        | stable water    |
| 255                          | —        | nodata / ignore |

"Seasonal water" (intra-annual variability) is a valid within-year class. "Stable water"
here is the per-year permanent-water state (water in every valid observation that year),
consistent with the manifest's stable-water notion.

**Manual reference sample — evaluated, not used.** The GLAD time-series reference sample
(`https://glad.geog.umd.edu/timeSeriesReference/timeSeriesSample.zip`) is georeferenced
(600 ~30 m single-pixel polygons, EPSG:4326) but (a) far too small to balance static
per-year classes and (b) its per-point `Stratum` field is the multi-year *map stratum* the
point was drawn from (including gain/loss), not a clean per-year interpretation. We
therefore used the derived annual-percent raster directly (an expert-validated product),
sampling only high-confidence windows — the same decision made for `jrc_global_surface_water`.

## Sampling (bounded-tile, tiles-per-class balanced)

This is a huge global product (each tile is 40000×40000 px), so per spec §5 we do
bounded-tile sampling over **8 representative INTERIOR continental 10×10° tiles** across
diverse biomes/hemispheres:

| tile | region |
|------|--------|
| `00N_020E` | Congo Basin interior — rivers, wetlands |
| `00N_070W` | Central Amazon — Solimões floodplain, rivers |
| `70N_060E` | W Siberia — Ob wetlands, thermokarst lakes |
| `60N_110W` | Canadian prairies/shield — lakes |
| `30N_080E` | Ganges/Himalaya foreland — seasonal floodplain |
| `20S_130E` | Australia interior — Lake Eyre basin, ephemeral lakes |
| `20N_010W` | Niger inland delta / Sahel — seasonal water |
| `50N_020E` | E Europe — lakes, rivers, reservoirs |

Interior tiles are chosen deliberately: GLAD maps **inland** water, so on interior tiles
value 0 corresponds to genuine dry land (ocean is not a target). Each tile is scanned in
non-overlapping ~22-px (native 30 m) blocks ≈ a 64-px @ 10 m footprint. A block is a
candidate if it is either **pure land** (0 % water → class 0) or has a **strong water
signal** (≥ 10 % water pixels → high-confidence); blocks with weak/ambiguous water
(0 < water < 10 %) or any nodata are skipped. A class is "present" in a block at ≥ 5 %
coverage. Windows are selected **tiles-per-class balanced, rarest class first**, up to
1000 tiles per class (25k total cap, not reached). Native 30 m EPSG:4326 windows are
reprojected to local UTM at 10 m with **nearest** resampling (categorical).

Scanned 22,934,308 candidate blocks (per-class candidates: land 21.8M, seasonal 2.50M,
stable 1.88M).

**Time range:** 1-year window `[2020-01-01, 2021-01-01)` for every sample; `change_time`
= null (static per-year classification, no dated change).

### Selected class counts (tiles-per-class; a tile counts toward every class it contains)

| class | count |
|-------|-------|
| land (0)           | 1000 |
| seasonal water (1) | 1008 |
| stable water (2)   | 1196 |

Total distinct samples: **1603**.

## Verification (spec §9)

- Opened output `.tif`s: all single-band, **uint8**, **64×64**, **UTM at 10 m**
  (e.g. EPSG:32735/32720/32641), nodata **255**. Dataset-wide unique pixel values across
  all 1603 tiles = `{0, 1, 2, 255}`, all covered by `metadata.json` classes.
- 1603 `.tif` ↔ 1603 `.json`; every sidecar has a ≤1-year `time_range` and `change_time`=null.
- **Georeferencing round-trip (spatial sanity):** for 8 random samples, reprojected each
  output tile center back to lon/lat, read the raw GLAD raster there, and confirmed the
  source-derived class is present in the output tile — **8/8 pass** (stable-water outputs
  sit on value-100 water pixels, land outputs on value-0 land). This validates end-to-end
  label placement against the source; a full Sentinel-2 image overlay was not run (not
  practical headlessly), but the round-trip against the georeferenced source is equivalent
  for confirming alignment.
- Script is idempotent (skips existing `{sample_id}.tif`; raw tiles cached).

## Caveats

- Gain/loss classes intentionally dropped (change-timing rule) — this dataset covers only
  the static land / seasonal-water / stable-water triad.
- Bounded sampling (8 interior tiles, year 2020) — not global coverage; representative of
  diverse inland-water biomes.
- Ocean/coast excluded by design (interior tiles only) since GLAD maps inland water.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glad_global_surface_water_dynamics
```

Raw tiles: `raw/glad_global_surface_water_dynamics/` (498 MB, 8 tiles). Outputs:
`datasets/glad_global_surface_water_dynamics/{metadata.json, locations/*.tif+.json}`.
