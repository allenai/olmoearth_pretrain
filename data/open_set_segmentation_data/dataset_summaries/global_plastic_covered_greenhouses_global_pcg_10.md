# Global Plastic-Covered Greenhouses (Global-PCG-10)

- **Slug:** `global_plastic_covered_greenhouses_global_pcg_10`
- **Status:** completed
- **Task type:** classification (dense_raster, binary)
- **Num samples:** 2000 (1000 PCG tiles + 1000 non-PCG tiles)
- **Time range:** 2020 (annual product) → 1-year window per tile; no change label.

## Source

- **Paper:** ESSD 2025, "Global-PCG-10: a 10-m global map of plastic-covered greenhouses
  derived from Sentinel-2 in 2020" (https://essd.copernicus.org/articles/17/5065/2025/).
- **Data:** figshare DOI `10.6084/m9.figshare.27731148` (v2), file
  `Global_PCG_10_Dataset.zip` (326 MB), license **CC-BY-4.0**.
- **Access method:** unauthenticated HTTPS via `download.download_http` from
  `https://ndownloader.figshare.com/files/50488200` (resolved through the figshare API
  `https://api.figshare.com/v2/articles/27731148`). No credentials required.
- **Format:** ~1639 GeoTIFFs, each a 1°×1° tile (11133×11133 px) at ~10 m in **EPSG:4326**
  (WGS84). Only tiles that contain PCG are released. Files named
  `{gridID}/{gridID}_{subgridID}_PCG_Result.tif`. Raster is binary **uint8**, no nodata
  band: `0 = non-PCG`, `1 = plastic-covered greenhouse (PCG)`.
- Also downloaded (not used for labels): `Classification_Grid.zip`, `PCG_Grid.zip`
  (SHP indexing grids).

## Processing

Global derived-product map → **bounded-tile dense_raster** sampling
(`olmoearth_pretrain/open_set_segmentation_data/datasets/global_plastic_covered_greenhouses_global_pcg_10.py`):

1. **Scan** every source tile in 64×64 native-pixel blocks (mp.Pool 64 workers). Classify
   each block by its PCG fraction:
   - **PCG tile** if `>= 5%` of the block is class 1 (~205 px) — a confident, resolvable,
     dense-plasticulture window. (Block-fraction survey of 25 random tiles: ~2000
     blocks ≥5% PCG each 25 tiles → far more than needed to reach 1000.)
   - **non-PCG tile** if the block is pure background (0 PCG pixels).
   - Per-tile reservoir caps (200 PCG / 25 non-PCG) bound memory across ~1639 tiles.
   - Candidates found: PCG = 78,845, non-PCG = 40,975.
2. **Balance & select:** shuffle (seed 42), take up to 1000 per class → 1000 PCG + 1000
   non-PCG = 2000 tiles, spread globally.
3. **Write** each block: reproject a 220-px native window centered on the block to local
   UTM at 10 m (`Resampling.nearest`, categorical) into a 64×64 patch; anything not 0/1
   (reprojection fill at tile edges) → 255. Atomic single-band uint8 GeoTIFF +
   sidecar JSON. Idempotent (skips existing `{id}.tif`).

## Class mapping

Native raster ids are kept unchanged as output class ids:

| id | name | notes |
|----|------|-------|
| 0 | non-PCG | any non-greenhouse pixel (other land cover / water); the map's 0 value |
| 1 | plastic-covered greenhouse | film-covered protected cultivation, Sentinel-2 2020; the map's 1 value |

`255 = nodata/ignore`. PCG tiles carry **both** classes per-pixel (0 and 1), so class 0 is
abundantly covered; the 1000 pure non-PCG tiles add background diversity.

## Time range

Annual 2020 product → each sample gets the 1-year window `[2020-01-01, 2021-01-01)`.
Not a dated event, so `change_time` is null.

## Verification (§9)

- 2000 `.tif` + 2000 `.json`, fully paired; single-band **uint8**, local **UTM @ 10 m**,
  **64×64**, `nodata=255`. Pixel values ∈ {0, 1, 255}.
- Global pixel tally over all 2000 tiles: `0 → 7,524,602`, `1 → 661,244`,
  `255 → 6,154` (0.07% — reprojection edge fill only).
- 1002 tiles contain class 1 (the 1000 PCG tiles + 2 non-PCG tiles that picked up a stray
  PCG pixel via nearest reprojection from a neighbor — negligible).
- `metadata.json` class ids {0,1} cover all non-nodata values.
- **Spatial sanity:** PCG sample centers land in well-known plasticulture regions — North
  China Plain (116°E, 38°N; 116°E, 39.7°N), Turkey/Mediterranean coast (33°E, 36°N),
  Gansu China (104°E, 36°N), NE China / Jilin (124°E, 43°N). Georeferencing confirmed
  correct (rslearn Projection uses `y_resolution=-10`).
- Did **not** overlay live Sentinel-2 imagery (no configured S2 source in this
  environment); relied on the published, exactly-georeferenced 10 m product + the
  coordinate check above.

## Judgment calls / caveats

- **PCG threshold = 5% per 64×64 block.** PCG is a small, clustered target; 5% (~205 px)
  gives clearly-present, high-confidence greenhouse windows while leaving ample candidates
  (78k) to sample 1000 from globally.
- **No nodata band in the source.** Pure non-PCG tiles are sampled from within
  PCG-containing 1° tiles and may occasionally fall on **water** in coastal tiles (e.g.
  Almería, Antalya are coastal). There is no land mask in the product to exclude these;
  noted as a minor caveat. Non-PCG is a genuine source label (value 0), not fabricated.
- **Global spread, not global coverage.** Per §5 for large global maps, we sample a
  bounded 2000 tiles rather than the full ~1639-tile product footprint.

## Reproduce

```
# (raw already downloaded+extracted under raw/<slug>/Global_PCG_10_Dataset/)
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_plastic_covered_greenhouses_global_pcg_10 --workers 64
```
Raw source: `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/global_plastic_covered_greenhouses_global_pcg_10/`
Outputs: `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/global_plastic_covered_greenhouses_global_pcg_10/`
