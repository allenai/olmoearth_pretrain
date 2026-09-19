# USGS USMIN Mine Features

- **Slug:** `usgs_usmin_mine_features`
- **Status:** completed · classification (polygon segmentation) · **7,076 samples**
- **Source:** USGS Mineral Resources "Prospect- and Mine-Related Features from USGS 7.5- and
  15-Minute Topographic Quadrangle Maps" (USMIN), version 10.0 (May 2023). Public domain.
  <https://mrdata.usgs.gov/usmin/>
- **Annotation method:** manual digitizing of mine symbols from historical USGS topographic
  quadrangle maps.

## Scope — polygon features only (dataset split)

This dataset is now **polygon mine footprints only**. The **point-marker feature types were split
out** into the sibling presence-only point dataset **`usgs_usmin_mine_features_points`** (6,427
mine-marker points, including the point-only classes `mine_shaft` and `adit`). This summary covers
only the polygon footprints.

## Source & access

National File Geodatabase downloaded from ScienceBase item `5a1492c3e4b09fc93dcfd574`
(`USGS_TopoMineSymbols_ver10_Geodatabase.zip`); no credentials. Layers used: 24k + 48k **polygon**
(the 625k layer is dropped for poor positional accuracy). No imagery downloaded.

## Label type — polygon footprints

Polygon mine footprints rasterized into ≤ 64×64 local-UTM 10 m tiles (footprint centered;
footprints > 640 m keep the central 64×64). Class scheme (uint8): `0 = background`, `255 =
nodata`. Written to `datasets/usgs_usmin_mine_features/locations/{id}.tif` (+ `.json`).

## Classes / counts

Only feature types that occur as polygons are kept (the point-only marker classes moved to the
sibling `_points` dataset). Balanced to ≤ 1000 polygons/class. **7,076 samples total.**

| id | name | samples |
|----|------|---------|
| 0 | background | (in-tile fill) |
| 1 | prospect_pit | 76 |
| 2 | quarry_open_pit | 1000 |
| 3 | gravel_borrow_pit | 1000 |
| 4 | strip_mine | 1000 |
| 5 | tailings_pile | 1000 |
| 6 | tailings_pond | 1000 |
| 7 | mine_dump | 1000 |
| 8 | disturbed_surface | 1000 |

Unmapped minor `Ftr_Type` values dropped (clay/cinder/shale/… pits, generic Mine, coal/uranium/
placer/hydraulic mines, mill site, tipple).

## Time handling

Persistent, map-digitized features → 1-year `time_range` at a representative Sentinel-era year
(spread across **2016–2022**). `change_time = null`.

## Output

- `datasets/usgs_usmin_mine_features/locations/{id}.tif` (+ `.json`)
- `datasets/usgs_usmin_mine_features/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_usmin_mine_features
```

Idempotent (skips already-written tiles).

## Caveats

- Polygons are genuine 10 m segmentation targets (median max-extent ≈ 289 m; 87% > 100 m).
- `prospect_pit` has only 76 polygon footprints (most prospect pits are point-only → `_points`).
