# JRC Global Forest Types 2020 (GFT2020)

- **Slug**: `jrc_global_forest_types_2020_gft2020`
- **Status**: completed
- **Task type**: classification (dense_raster)
- **Samples**: 3000 (1000 / class × 3 classes)

## Source

EC JRC "Global map of forest types 2020 — version 1" (GFT2020 V1; Bourgoin, Ameztoy,
Verhegghen, Carboni, Achard, Colditz, 2026, doi:10.2905/JRC.C760PNG). A global **10 m**
derived-product raster (EPSG:4326, ~8.333e-5°/px, ≈9.26 m) that classifies the forest
area of the JRC Global Forest Cover 2020 v3 mask into the main forest types set out by the
EU Deforestation Regulation (EUDR, Reg. (EU) 2023/1115). The product is 2020-anchored (the
EUDR cut-off year).

- Landing page: https://forobs.jrc.ec.europa.eu/GFT
- License: **CC BY 4.0** (free with attribution).
- Access used: the single **global COG** on the JRC Big Data Platform (JEODPP), read via
  windowed HTTP range requests (`/vsicurl/`) — the ~50 GB mosaic is **never fully
  downloaded**:
  `https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFT2020/LATEST/single-cog/JRC_GFT2020_V1_cog.tif`
  (10°×10° per-tile GeoTIFFs are also published under `.../LATEST/tiles/`.)

## Raster legend and class mapping

The manifest lists four EUDR forest types (primary, naturally regenerating, planted,
plantation), but the **V1 raster merges planted + plantation into a single value (20)**.
Verified by reading the COG, the source values are:

| source value | meaning | output class id |
|---|---|---|
| 0 | non-forest / outside the forest mask | → **nodata (255)** |
| 10 | primary forest | 0 |
| 1 | naturally regenerating forest | 1 |
| 20 | planted / plantation forest | 2 |

This is a forest-type-only product defined *inside* a forest mask, so per spec §2 the
non-forest / no-data value (source 0) is written as **nodata=255** rather than as a
fabricated background class. Three classes are therefore emitted (ids 0–2), not four.

- `nodata_value = 255`; single-band **uint8**; local UTM at **10 m**; **64×64** tiles.

## Method

Global 10 m product → **bounded-tile sampling** (spec §5). We range-read a
spatially-distributed set of **59** small (0.4°≈4800 px) windows across all continents and
forest biomes (Amazon, Congo, SE-Asia/Oceania tropics, boreal Siberia/Canada/Scandinavia,
temperate forests, and plantation regions in the SE-US, Brazil, Chile, NZ, Iberia, France,
Sweden/Germany, China, Japan, South Africa, India, Vietnam, Australia, Uruguay). Each
window is cached under `raw/{slug}/regions/`.

Each cached region is scanned for spatially-**homogeneous** ≥64×64 blocks (BLOCK=76 native
px): a block qualifies if a single forest class covers ≥50 % of it (`DOMINANCE_FLOOR`) and
non-forest is ≤20 % (`FOREST_FLOOR≥0.8`) — the §4 guidance to prefer
homogeneous/high-confidence windows for derived-product maps. Qualifying blocks give
candidate records (region, centre lon/lat, dominant-class label). Candidates were:
primary 48 427, naturally regenerating 50 574, planted/plantation 24 696.

`balance_by_class(per_class=1000)` selects **1000 tiles per class** (3000 total, well under
the 25 k cap). Each selected native EPSG:4326 window is reprojected to a local UTM
projection at 10 m with **nearest** resampling (categorical), values remapped to ids 0–2,
outside-mask → 255.

## Time range & change handling

Static 2020 per-year state → `change_time = null`, `time_range = [2020-01-01, 2021-01-01)`
(1-year window on 2020, §5). No change/event semantics.

## Verification (spec §9)

- 3000 `.tif` + 3000 matching `.json`. Sampled tiles: single-band uint8, UTM (EPSG:326xx/
  327xx), 10 m, 64×64, values in {0,1,2,255}, nodata=255. metadata class ids cover all tile
  values.
- Sidecar JSONs: 1-year 2020 `time_range`, `change_time=null`, `classes_present` set.
- **Georeferencing sanity**: for 6 random samples across continents, the tile's majority
  class id matched the source COG value read at the tile centroid **6/6** (georef exact).

## Caveats

- Planted vs plantation forest cannot be distinguished (merged as value 20 in V1); recorded
  as one class "planted / plantation forest".
- Homogeneous-window selection biases toward interior/pure forest patches (high confidence);
  mixed forest-type edges are under-represented by design.
- Classes are well-balanced (1000 each); no rare-class truncation. The 254-class cap is not
  a concern (3 classes).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jrc_global_forest_types_2020_gft2020
```
Idempotent: cached region reads and already-written `locations/{id}.tif` are skipped.
