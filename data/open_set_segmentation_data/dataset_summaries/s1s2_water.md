# S1S2-Water — COMPLETED (classification, dense_raster)

- **Slug**: `s1s2_water`
- **Name**: S1S2-Water
- **Source**: Zenodo record [11278238](https://zenodo.org/records/11278238) /
  IEEE JSTARS.
- **Citation**: Wieland et al. 2024, IEEE JSTARS — "S1S2-Water: A global dataset
  for semantic segmentation of water bodies from Sentinel-1 and Sentinel-2
  satellite images".
- **Family / region**: water / global (65 scenes, 29 countries).
- **License**: CC-BY-4.0.
- **Label type**: dense_raster → per-pixel **classification** (binary water).
- **Task type**: classification. **num_samples**: 1011 tiles (64×64).

## Source

65 globally distributed ~100×100 km scenes, each a Sentinel-1 / Sentinel-2 pair
with **quality-checked binary water masks**, released as per-scene STAC items
inside 6 zip parts (part1–part6). Per scene the archive stores (as cloud-optimized
GeoTIFFs):

- `s2_msk` (uint8): **Sentinel-2-derived** binary water mask — native **UTM,
  10 m/px**, 10980×10980. `0` = no-water, `1` = water. **(used)**
- `s2_valid` (uint8): S2 validity mask — `1` = valid, `0` = invalid/nodata. **(used)**
- `s1_msk` (uint8, 9 m/px) + `s1_valid`: the Sentinel-1 counterpart. (not used)
- `s2_img`, `s1_img`, `copdem30_elevation`, `copdem30_slope`: imagery + Copernicus
  DEM. (not used)

The **S2 mask** was chosen as the label because it is already in the target
projection/resolution (local UTM at 10 m), so no reprojection is needed and
georeferencing is exact. All 65 scenes have `flood == False` (permanent / static
water), so no flood-event handling is required.

## Access method

Public, no credentials (CC-BY-4.0). The 6 zip parts total ~163 GB, but only the
`s2_msk`, `s2_valid` and per-scene `meta.json` are needed (~2.3 MB/scene, ~150 MB
total). These were pulled **selectively via HTTP range requests** against the
Zenodo zip parts using `remotezip`, so the large imagery/DEM assets were never
downloaded. Files land in `raw/s1s2_water/{scene}/`. Idempotent (skips scenes
already extracted). The scene→zip-part map is hard-coded in the script (derived
from the zip central directories).

## Class mapping (2 classes, manifest order)

| id | name | definition |
|----|------|-----------|
| 0 | water | S2 binary water mask == 1 (rivers, lakes, reservoirs, coastal water) |
| 1 | no-water | S2 binary water mask == 0 (land) |
| 255 | nodata/ignore | `s2_valid == 0` (outside valid swath / no data) |

Water is class 0 (phenomenon of interest); no-water is class 1 (co-occurring land
class — a real class in binary water segmentation, not a fabricated negative).

## Processing

Each 10980×10980 UTM 10 m S2 mask is cut, **without reprojection**, into 64×64
tiles aligned to the source pixel grid (top-left origin taken from the source
transform). Tiles >50% nodata are dropped; a tile counts toward a class only with
≥32 px of it. **Tiles-per-class balanced** selection (spec §5): water (the rare
class) is filled first up to 1000 tiles; no-water co-occurs in nearly every water
tile. As a safety margin so no-water can reach its target even if some water tiles
are pure water, a small deterministic per-scene sample (≤50) of land-only tiles is
also emitted as candidates. Candidates are sorted deterministically
(`scene, ti, tj`) before the seeded shuffle, so re-runs are reproducible regardless
of multiprocessing completion order.

- 450,449 candidate tiles → **1011 selected**, from 62 of the 65 scenes.
- **Tiles containing each class** (a tile can count for both):
  - water: 1000
  - no-water: 1009

Well under the 25k per-dataset cap.

## Time range & change handling

The masks are **static water** (not dated events; `flood == False` for all
scenes), so **no `change_time`** is set. Each scene has a Sentinel-2 acquisition
date parsed from the source product id in `properties.s2_srcids` (the STAC
`datetime` field is a placeholder `2020-01-01`); `time_range` is a **1-year window
centered** on that date (±182/183 days). Acquisition dates span **2018–2020**, all
within the Sentinel era (post-2016).

## Caveats / judgment calls

- **S2 vs S1 mask**: both masks are provided; the S2 mask was used because it is
  native UTM 10 m (our target grid) and needs no reprojection. The S1 mask (9 m,
  possibly a slightly different acquisition date) was not used. Either sensor's
  imagery can still be paired downstream — both are recorded in `sensors_relevant`.
- **Class ordering**: `water` is id 0, `no-water` is id 1 (matches the manifest
  class order and puts the phenomenon of interest at 0). The source raster encodes
  the opposite (0=no-water, 1=water); the remap is applied at write time.
- **num_samples is ~1011**, not larger: this is a binary dataset, so the
  ≤1000/class balancing caps it near 1000 (water tiles, which dominate the
  selection, also supply the no-water class). This is expected for a 2-class dense
  raster and honors the "up to 1000 locations per class" rule.
- **Selective download** via `remotezip` range requests avoids fetching ~163 GB of
  unused imagery/DEM. If Zenodo range support ever regresses, the full zip parts can
  be downloaded and unzipped instead.

## Verification

- 1011 `.tif` + 1011 matching `.json`; every tile single-band **uint8**, local UTM,
  **10 m**, **64×64**, values ⊆ {0, 1, 255} with nodata = 255. `metadata.json`
  class ids {0, 1} cover all non-nodata tif values.
- `time_range` ≤ 1 year on every sample; `change_time` is null on all (static).
- **Georeferencing round-trip** (5 samples across scenes): each written tile
  reproduced the source `s2_msk`+`s2_valid` remap with **100% pixel agreement**, and
  the tile world bounds / CRS matched the source window exactly. Tile centers land
  globally as expected (Japan 130.6°E/33.3°N, Argentina −58.8/−33.7, Canada
  −95.1/59.6, Australia 116.0/−29.9, Argentina −59.8/−29.6). Because the label is
  the S2 mask in its native projection with no reprojection, label/imagery overlay
  is exact by construction.
- Re-running skips already-written tiles (idempotent).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.s1s2_water
```
