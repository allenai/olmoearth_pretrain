# Global Renewables Watch (solar PV)

- **Slug:** `global_renewables_watch`
- **Status:** completed · classification (polygon segmentation) · **1,000 samples**
- **Source:** GitHub (Microsoft / Planet / TNC), v1.0 (2024 Q2).
  <https://github.com/microsoft/global-renewables-watch> · **License:** MIT
- **Annotation method:** deep learning (PlanetScope) + human QC.

## Scope — solar-PV polygons only (dataset split)

This dataset is now **solar-PV polygons only**. The product's **wind-turbine point detections
were split out** into the sibling presence-only point dataset **`global_renewables_watch_points`**
(1,000 wind-turbine points). This summary covers only the solar-PV polygon footprints.

## Label type — polygon footprints

Ground-mounted / large solar PV installation footprints (polygons) rasterized at 10 m into
variable ≤ 64×64 local-UTM tiles. Class scheme (uint8): `0 = background`, `1 = solar_pv`;
nodata = 255. Written to `datasets/global_renewables_watch/locations/{id}.tif` (+ `.json`).

## Classes / counts

| id | name | samples |
|----|------|---------|
| 0 | background | (in-tile fill) |
| 1 | solar_pv | 1000 |

`solar_pv` capped at 1000 (`balance_by_class`). **1,000 samples total.**

## Time handling

1-year `time_range` on each feature's `construction_year` (2017–2024). `change_time = null`
(construction year is year-granular).

## Output

- `datasets/global_renewables_watch/locations/{id}.tif` (+ `.json`)
- `datasets/global_renewables_watch/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_renewables_watch
```

Idempotent (skips already-written tiles).

## Caveats

- Derived product (DL on PlanetScope + QC), not in-situ reference.
- Wind turbines from the same product live in `global_renewables_watch_points`.
