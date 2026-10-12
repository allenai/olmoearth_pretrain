# MapBiomas Brasil (annual LULC) — `mapbiomas_brasil_annual_lulc`

- **Status:** completed
- **Task type:** classification (dense_raster)
- **Samples:** 7,828 label tiles (64×64, UTM, 10 m)
- **Family / region:** land_cover / Brazil
- **License:** CC BY-SA 4.0

## Source

MapBiomas Project — Brazil annual land-use/land-cover, **Collection 9** (1985–2023),
30 m, Landsat-based. Landing page: https://brasil.mapbiomas.org/en/downloads/

The per-year national coverage mosaics are public single-band `uint8` COGs
(EPSG:4326, ~0.00027°/px ≈ 30 m) on Google Cloud Storage:

```
https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection_9/lclu/coverage/brasil_coverage_{YEAR}.tif
```

No credentials required. We read the **2022** national COG (~158,828 × 155,241 px,
≈24.6 Gpx) via windowed HTTP range requests only — the full mosaic is never downloaded.
`raw/mapbiomas_brasil_annual_lulc/` holds `SOURCE.txt` + the small cached 0.6° region
windows under `regions/` (~17 MB).

**Year choice:** 2022 (post-2016 rule; a recent, fully-consolidated collection year).

## Resolution: 30 m → 10 m (documented deviation)

MapBiomas is a **Landsat-based 30 m** product, **not** 10 m. Per spec §2/§4 we resample
the categorical label to the pretraining 10 m grid with **NEAREST** resampling: each
selected ~22×22 native block (≈640 m) is reprojected from native EPSG:4326 (~30 m) to a
local UTM projection at 10 m as a 64×64 tile. Each ~30 m native pixel therefore expands to
roughly a 3×3 block of 10 m pixels (blocky boundaries — expected for a 30 m source).
Categorical labels are never bilinearly resampled.

## Legend collapse (16 classes)

MapBiomas Collection 9 uses a deep hierarchical legend (~30 codes, incl. per-crop
subclasses: soybean, sugar cane, rice, cotton, coffee, citrus, palm oil, …). Most crop and
fine natural subclasses are not reliably separable at 30 m, so the legend is collapsed to
**16 level-1/level-2 classes** that are observable at 30 m, while keeping the distinctive
Brazilian ecosystem classes (Cerrado savanna, floodable forest, mangrove, wetland,
grassland). Output ids are `uint8`; source **0 / 27** (no-data / not observed) and any
unmapped code → **nodata = 255**.

| id | class | MapBiomas source codes |
|----|-------|------------------------|
| 0  | Forest Formation | 3 |
| 1  | Savanna Formation (Cerrado) | 4 |
| 2  | Mangrove | 5 |
| 3  | Floodable Forest | 6 |
| 4  | Forest Plantation (Silviculture) | 9 |
| 5  | Wetland | 11 |
| 6  | Grassland | 12 |
| 7  | Other Non-Forest Natural Formation | 13, 29, 32, 49, 50 |
| 8  | Pasture | 15 |
| 9  | Temporary Crop | 18, 19, 39, 20, 40, 62, 41 |
| 10 | Perennial Crop | 36, 46, 47, 35, 48 |
| 11 | Mosaic of Uses | 21 |
| 12 | Urban Area | 24 |
| 13 | Mining | 30 |
| 14 | Other Non-Vegetated Area | 22, 23, 25 |
| 15 | Water (incl. aquaculture) | 26, 33, 31 |

The full raw code→name legend and the collapse map are recorded in `metadata.json`
(`provenance.source_value_legend`, `provenance.class_collapse`).

## Sampling (bounded-tile, tiles-per-class balanced)

Per spec §5, this is a large derived-product raster with no in-situ reference alternative,
so we do **bounded-tile sampling** — never global coverage. 42 spatially-distributed 0.6°
region windows span all six biomes (**Amazon, Cerrado, Atlantic Forest, Caatinga,
Pantanal, Pampa**) plus targeted regions for rare classes: mangrove/aquaculture coasts
(Maranhão/Pará, NE shrimp farms, Cananéia, Bahia), mining districts (Carajás, Tapajós
garimpo, Minas iron quadrilateral), and coffee / citrus / silviculture / rice belts.

Each cached region is scanned for **mostly-observed** 22×22 native blocks (≥90% of pixels
map to a real class). A class counts as "present" in a block when it covers ≥5% of it.
Blocks are then selected **tiles-per-class balanced, rarest class first**
(`sampling.select_tiles_per_class`, `per_class=1000`, `total_cap=25000`) so rare classes
(mangrove, mining) reach the target while common ones are capped. 411,452 candidate blocks
→ 7,828 selected tiles. Selected native windows are reprojected to local UTM at 10 m
(nearest). Full multi-class label tiles are written (not just the dominant class), so each
tile is a genuine dense LULC patch.

**Per-class selected-tile counts** (a tile counts toward every class present in it):

```
Forest Formation                    3832   Pasture                   2423
Savanna Formation                   1125   Temporary Crop            1000
Mangrove                            1094   Perennial Crop            1059
Floodable Forest                    1084   Mosaic of Uses            2762
Forest Plantation (Silviculture)    1004   Urban Area                1010
Wetland                             1030   Mining                    1014
Grassland                           1192   Other Non-Vegetated Area  1050
Other Non-Forest Natural Formation  1061   Water                     1386
```

Total unique tiles = 7,828 (well under the 25k cap; 16 classes ≤ 254-class uint8 cap).
Classes >1000 occur because those classes co-appear in tiles selected to cover rarer
classes.

## Time range / change

Static per-year state label: `change_time = null`, `time_range` = the 1-year window
`[2022-01-01, 2023-01-01)` on every sample (spec §5, seasonal/annual labels).

## Output contract

- `datasets/mapbiomas_brasil_annual_lulc/metadata.json` (16 classes, `nodata_value=255`)
- `datasets/mapbiomas_brasil_annual_lulc/locations/{000000..}.tif` — single-band uint8,
  UTM, 10 m, 64×64, `nodata=255`
- `datasets/mapbiomas_brasil_annual_lulc/locations/{id}.json` — `crs`, `pixel_bounds`,
  `time_range` (≤1 yr), `change_time=null`, `source_id`, `classes_present`
- `raw/mapbiomas_brasil_annual_lulc/SOURCE.txt` + `regions/*.tif`

## Verification (spec §9)

- All 7,828 tiles: single band, `uint8`, UTM CRS (zones 32720–32723), 10 m, 64×64,
  `nodata=255`. 0 tiles fail the spec; no out-of-range values (only ids 0–15 + 255).
- All 7,828 sidecar JSONs present with a 1-year `time_range` and `change_time=null`.
- `metadata.json` class ids cover every value appearing in the tiles.
- Spatial sanity: dominant-class centroids land correctly — mangrove on the Maranhão/Pará
  and Bahia coasts, mining at Carajás and the Minas iron quadrilateral, urban near
  Brasília/Goiânia, wetland in the Pantanal — all inside Brazil across biomes.

## Caveats

- 30 m native → 10 m output (blocky boundaries; nearest resampling).
- Fine crop/natural subclasses are intentionally collapsed and are **not** recoverable
  from these labels; use `metadata.json` legend maps for the exact grouping.
- "Mosaic of Uses" (mixed agri/pasture) is an inherently ambiguous MapBiomas class kept
  as-is.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mapbiomas_brasil_annual_lulc
```

Idempotent: existing `locations/{id}.tif` are skipped; cached region windows are reused.
