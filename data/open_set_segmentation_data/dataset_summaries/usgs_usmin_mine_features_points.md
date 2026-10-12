# USGS USMIN Mine Features (points)

- **Slug:** `usgs_usmin_mine_features_points`
- **Status:** completed · classification (presence-only points, multi-class) · **6,427 points**
- **Source:** USGS Mineral Resources "Prospect- and Mine-Related Features from USGS 7.5- and
  15-Minute Topographic Quadrangle Maps" (USMIN), version 10.0 (May 2023). Public domain.
  <https://mrdata.usgs.gov/usmin/>
- **Annotation method:** manual digitizing of mine symbols from historical USGS topographic
  quadrangle maps.

## Scope — point markers (dataset split)

Companion to the polygon dataset **`usgs_usmin_mine_features`**. This dataset holds the
**point-marker feature types** (including the point-only classes `mine_shaft` and `adit`); the
polygon mine footprints live in the sibling `usgs_usmin_mine_features` dataset. The raw File
Geodatabase is reused from the shared `usgs_usmin_mine_features` raw dir (not re-downloaded).
Layers used: 24k + 48k **point** (the 625k layer is dropped for poor positional accuracy).

## Label type — presence-only points

Each mapped mine symbol is one labeled location written to a dataset-wide `points.geojson`
(spec §2a). There is **NO background class** — every point is a positive presence of its feature
type — and no fabricated GeoTIFF context, buffer, or negative tiles. Negatives are supplied
downstream by the assembly step.

## Classes / counts

Distinct feature types that occur as points, ids 0–8. Balanced to ≤ 1000 points/class. **6,427
points total.**

| id | name | pts | id | name | pts |
|----|------|-----|----|------|-----|
| 0 | prospect_pit | 1000 | 5 | strip_mine | 1000 |
| 1 | mine_shaft | 1000 | 6 | tailings_pile | 65 |
| 2 | adit | 1000 | 7 | tailings_pond | 3 |
| 3 | quarry_open_pit | 1000 | 8 | mine_dump | 359 |
| 4 | gravel_borrow_pit | 1000 | | | |

Unmapped minor `Ftr_Type` values dropped; the `disturbed_surface` type does not occur as points.
Sparse classes kept per spec §5.

## Time handling

Persistent, map-digitized features → static 1-year `time_range` at a representative Sentinel-era
year (spread across **2016–2022**). `change_time = null`.

## Output

- `datasets/usgs_usmin_mine_features_points/points.geojson`
- `datasets/usgs_usmin_mine_features_points/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_usmin_mine_features_points
```

Idempotent (rewrites `points.geojson`).

## Caveats

- `prospect_pit`, `mine_shaft`, `adit` are typically sub-10 m — weak presence markers, not
  resolvable footprints. For their polygon footprints see `usgs_usmin_mine_features`.
