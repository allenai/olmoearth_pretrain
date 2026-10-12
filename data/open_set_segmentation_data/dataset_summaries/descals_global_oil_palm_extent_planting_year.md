# Descals Global Oil Palm Extent & Planting Year

- **Slug**: `descals_global_oil_palm_extent_planting_year`
- **Status**: completed
- **Task type**: classification (dense_raster, tiles-per-class balanced)
- **Num samples**: 1692 label patches (64×64, 10 m, local UTM)

## Source

Descals et al., *"Global oil palm extent and planting year from 1990 to 2021"* (ESSD),
Zenodo record [10.5281/zenodo.13379129](https://doi.org/10.5281/zenodo.13379129),
**CC-BY-4.0** (open access — no credentials needed). Derived from Sentinel-1 + Landsat
time-series classification, validated against photo-interpreted reference points.

The record ships four files; we downloaded all four to `raw/{slug}/` and extracted them:

- `GlobalOilPalm_OP-extent.zip` — **609** tiled GeoTIFFs (`extent/`), the **10 m 2021
  oil-palm EXTENT** layer, EPSG:4326, `uint8`, ~0.9° tiles over the global tropics:
  - `0` = background / not oil palm
  - `1` = industrial oil palm
  - `2` = smallholder oil palm
- `GlobalOilPalm_OP-YoP.zip` — 609 tiled GeoTIFFs (`yop/`), the **30 m planting-year**
  layer, `uint16`: `0` = none, `1989..2021` = year of first planting.
- `Grid_OilPalm2016-2021.zip` — the tile grid shapefile (tile ID → footprint).
- `Validation_points_GlobalOP2016-2021.zip` — the product's validation points.

## Key decisions

- **Primary label = oil-palm TYPE classification** (industrial vs smallholder) from the
  10 m extent layer. Native ids are preserved as a **3-class** scheme:
  - `0` = *other* — in-context non-oil-palm land inside an oil-palm-centered tile
    (spatially-meaningful background/negative, not fabricated).
  - `1` = *industrial oil palm*
  - `2` = *smallholder oil palm*
- **Planting-year (YoP) layer NOT emitted.** It was downloaded and is documented here as
  the "age dimension" the manifest notes, but kept as an **auxiliary** only. Emitting it
  would mean a coarse 30 m regression target that co-registers poorly with the 10 m type
  signal and would double the dataset for a weaker label; per the task guidance, keeping
  **one clean oil-palm-type classification dataset** is cleanest. The raw YoP tiles remain
  in `raw/{slug}/yop/` for anyone who wants to add a regression variant later.
- **Background kept as a real class** (rather than positive-only/255) because the extent
  product has an explicit `0` and, since every tile is centered on oil palm (which is on
  land), the surrounding `0` pixels are genuine near-plantation land — meaningful negatives
  within the tile (same precedent as `global_sugarcane_10_m`). Caveat: for coastal
  plantations a small fraction of a tile can be ocean, which the product also codes as `0`;
  this is rare and diluted by the oil-palm-centering.
- **Bounded-tile dense_raster sampling (§5).** The whole product is only ~750 MB
  uncompressed, so rather than sub-sample the *download*, we downloaded it fully and
  **scanned all 609 extent tiles** (they already ARE the representative oil-palm regions).
  Scanned in 64×64 native-pixel blocks (≈640 m) with a per-chunk reservoir. Candidate
  blocks require **≥15 % oil-palm pixels** (strong, contiguous signal — avoids speckle);
  a type "counts" toward a tile if it is **≥25 % of that block's oil-palm pixels** (so
  tiles are labeled by their dominant type; mixed tiles carry both). Blocks with no
  dominant type are dropped.
- **Tiles-per-class balanced**, rarest-first (`sampling.select_tiles_per_class`,
  `per_class=1000`, `total_cap=25000`) so the rarer smallholder class is prioritized.
- **Reprojection**: each selected block center → local UTM at 10 m; 64×64 patch produced
  with `rasterio` `reproject` **nearest** resampling (categorical). Values keep native ids
  (0/1/2); `255` = nodata/ignore. Written with `io.write_label_geotiff` (exact
  georeferencing, atomic).
- **Time range**: the extent map is the 2021 product, so every sample gets the **2021
  one-year window** `[2021-01-01, 2022-01-01)`. Oil palm is a persistent perennial crop,
  so this is a static presence/state label — `change_time = null` (no change semantics).

## Sample counts

Tiles-per-class (a tile counts toward every class present in it):

| class id | name                 | tiles |
|----------|----------------------|-------|
| 0        | other (background)   | 1692  |
| 1        | industrial oil palm  | 1102  |
| 2        | smallholder oil palm | 1000  |

Total distinct label patches: **1692** (well under the 25 k cap).

Geographic distribution of selected tiles (by tile-center longitude):
SE Asia / Pacific **1235**, Latin America **249**, Africa **199**, South Asia **9** — this
matches the real-world concentration of oil palm in SE Asia while still covering all three
requested regions.

## Verification (§9)

- 1692 `.tif` + 1692 `.json`, fully paired (0 unpaired).
- Sampled 200 tiles: **0 nonconforming** — every patch is single-band `uint8`, UTM
  (`EPSG:326xx`), 10 m, ≤64×64, nodata `255`; pixel values ∈ {0,1,2,255}.
- `metadata.json` class ids {0,1,2} cover all values in the tifs; `time_range` is the
  ≤1-year 2021 window; `change_time = null`.
- **Georeferencing back-projection check**: for sampled tiles, the UTM tile center
  re-projected to WGS84 matches the source extent-pixel lon/lat to **<10 m (sub-pixel)**,
  confirming the reproject pipeline places labels correctly. (Full Sentinel-2 overlay was
  not run in this session, but georeferencing is exact-by-construction — labels are written
  with an explicit UTM projection + pixel bounds.)
- Script is **idempotent**: re-running skips existing `{sample_id}.tif`.

## Reproduce

```bash
# (raw zips already downloaded+extracted under raw/{slug}/{extent,yop,grid,val})
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.descals_global_oil_palm_extent_planting_year --workers 64
```

Outputs on weka under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/descals_global_oil_palm_extent_planting_year/`
(`metadata.json`, `locations/{id}.tif`+`.json`, `registry_entry.json`).
