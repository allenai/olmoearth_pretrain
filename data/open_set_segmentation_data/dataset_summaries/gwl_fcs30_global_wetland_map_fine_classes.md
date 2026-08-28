# GWL_FCS30 Global Wetland Map (fine classes)

- **Slug**: `gwl_fcs30_global_wetland_map_fine_classes`
- **Status**: completed
- **Task type**: classification (dense_raster)
- **Num samples**: 8000 (1000 per class × 8 classes)

## Source

Zhang et al. 2023, *ESSD* 15, 265–293, "GWL_FCS30: a global 30 m wetland map with a fine
classification system using multi-sourced and time-series remote sensing imagery in 2020"
(doi:10.5194/essd-15-265-2023). Data: Zenodo record **7340516**
(https://doi.org/10.5281/zenodo.7340516), license **CC-BY**. The map was produced on Google
Earth Engine by fusing pre-existing wetland products with time-series Landsat/Sentinel
observations for the **2020** epoch.

Distribution: 12 longitude-band ZIPs (~2.4 GB total) of 5°×5° GeoTIFF tiles, EPSG:4326,
~0.00027° (~30 m/px), uint8. Accessed with `download.download_zenodo`; all 12 bands were
fetched (2.4 GB is well under any bulk-download concern), giving a tile index of 962 global
5°×5° tiles.

## Access method

`python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gwl_fcs30_global_wetland_map_fine_classes`

Downloads the band ZIPs to `raw/{slug}/`, builds a tile→ZIP index from the ZIP namelists,
extracts only the curated wetland-rich 5°×5° tiles into `raw/{slug}/tiles/`, then scans and
writes label patches. Idempotent (skips existing zips, tiles, and `locations/*.tif`).

## Value legend → output classes

Source raster values (verified by reading tiles + the GEE-community catalog / paper):

| source value | class | output id |
|---|---|---|
| 0 | non-wetland / background | → nodata (255) |
| 180 | permanent water | 0 |
| 181 | swamp (woody/forested wetland) | 1 |
| 182 | marsh (herbaceous freshwater) | 2 |
| 183 | flooded flat (seasonally inundated) | 3 |
| 184 | saline (inland salt lakes/pans/marsh) | 4 |
| 185 | mangrove forest | 5 |
| 186 | salt marsh (coastal herbaceous) | 6 |
| 187 | tidal flat (coastal mud/sand) | 7 |

This is a wetland-only product (0 = everything else), so non-wetland is set to **nodata=255**
rather than a fabricated background class (spec §2/§5). Per-class descriptions are in
`metadata.json`.

## Sampling (bounded global derived-product, §5)

58 curated wetland-rich `(lon, lat)` regions across all continents (mapped to **56 unique
5°×5° tiles**; 0 regions missing a tile) were scanned. Regions were chosen to cover every
class:

- **mangrove**: Sundarbans, Sumatra/Kalimantan coasts, Papua, Amazon/Guianas coast, Niger
  Delta, Florida, N. Australia, Mozambique, Philippines
- **salt marsh**: Georgia/Chesapeake/Louisiana coasts, San Francisco Bay, Wadden Sea, UK
  Wash, Jiangsu coast, Patagonia
- **tidal flat**: Yellow Sea (Korea), Bohai, NW Australia, Amazon (Pará) coast, UK
- **swamp**: Congo cuvette, Amazon, Kalimantan/Sumatra peat, Atchafalaya, Pantanal
- **marsh**: Everglades, prairie potholes, West Siberia/Ob, Sudd, Poyang/Dongting, Camargue,
  Biebrza
- **flooded flat**: Amazon floodplain, Ganges/Brahmaputra deltas, Mekong, Okavango, Niger
  inland delta
- **saline**: Kazakhstan/Caspian salt flats, Andean altiplano salars, Australian salt lakes,
  Chott (Tunisia), Iran playa, Great Salt Lake, Etosha, Qinghai
- **permanent water**: Great Lakes, Lake Victoria, Finnish/Canadian-shield lakes, Amazon

Each tile is scanned on its native 30 m grid over a non-overlapping ~660 m block grid
(BLOCK=22 px). A block qualifies as a homogeneous, high-confidence window when **≥15 % of
its pixels are wetland** and **≥80 % of the wetland pixels are a single class** (§4 guidance
to prefer spatially-homogeneous windows for derived-product maps). The block's dominant
wetland class is used as the balancing key. Scan produced **3,074,828** candidate windows;
candidate distribution (id→count): permanent water 1.09M, swamp 872k, marsh 819k, flooded
flat 24.3k, saline 73.0k, mangrove 91.4k, salt marsh 34.7k, tidal flat 73.2k.

`balance_by_class(per_class=1000)` → **1000 tiles/class, 8000 total** (all classes reached
the target; well under the 25k cap). All source train/val splits are irrelevant (single map).

## GeoTIFF / reprojection

Each selected window is reprojected from native EPSG:4326 (~30 m) to a **local UTM
projection at 10 m** using **nearest** resampling (categorical labels), cropped to **64×64**
(640 m). All wetland values are mapped to output class ids; non-wetland → 255. Output tifs
are single-band uint8, north-up, nodata=255.

## Time range / change handling

Static 2020 map → per-sample `time_range` is the 1-year window **2020-01-01 … 2021-01-01**;
`change_time = null`. (The manifest's [2016, 2022] is the product's validity envelope; the
map itself is the 2020 epoch.)

## Verification (§9)

- Sampled tifs: single band, uint8, EPSG:326xx (UTM), res (10, 10), 64×64, nodata 255;
  pixel values ∈ {0–7} ∪ {255}. ✓
- All **8000** `.tif` have a matching `.json`; sampled `time_range`s are exactly 1 year
  (0 samples > 366 days); `change_time = null`. ✓
- `metadata.json` class ids (0–7) cover all values appearing in the tiles. ✓
- Class balance: 1000 each × 8 classes. ✓
- Reprojection is taken directly from the source label raster (label overlays itself by
  construction); windows were pre-filtered for single-class homogeneity so patches are clean.

## Caveats

- Derived-product map (30 m, model-generated), not in-situ reference — accuracy is that of
  GWL_FCS30 (paper reports ~86 % OA). Homogeneity filtering favors confident, spatially-clean
  wetland patches, reducing mixed-pixel label noise.
- Coastal classes (mangrove, salt marsh, tidal flat) and inland saline are naturally rarer
  and patchier than permanent water / marsh, but all reached 1000 tiles from the curated
  coastal/salt-lake regions.
- Bounded regional sample by design (§5): not global wall-to-wall coverage.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gwl_fcs30_global_wetland_map_fine_classes
```
