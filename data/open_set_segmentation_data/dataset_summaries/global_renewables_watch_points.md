# Global Renewables Watch (wind turbines)

- **Slug:** `global_renewables_watch_points`
- **Status:** completed · classification (presence-only points) · **1,000 points**
- **Source:** GitHub (Microsoft / Planet / TNC), v1.0 (2024 Q2).
  <https://github.com/microsoft/global-renewables-watch> · **License:** MIT
- **Annotation method:** deep learning (PlanetScope) + human QC.

## Scope — wind-turbine points (dataset split)

The Global Renewables Watch product carries two geometries. This dataset holds the **wind-turbine
point detections**; the product's solar-PV polygon footprints live in the sibling dataset
**`global_renewables_watch`** (solar PV). This split separates the point and polygon geometries
into two clean datasets.

## Label type — presence-only points

Emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a): each detected
wind turbine is one point of the single foreground class. **Converted from the old detection-tile
encoding** — there is **no fabricated GeoTIFF context, and no background / buffer / negative
tiles**. This dataset carries **no fabricated negatives**; negatives are supplied downstream by
the assembly step.

## Classes / counts

Single class `0 = wind_turbine`. **1,000 points** (up to 1000/class, `balance_by_class`).

## Time handling

1-year `time_range` on each turbine's `construction_year` (2017–2024). `change_time = null`
(construction year is year-granular).

## Output

- `datasets/global_renewables_watch_points/points.geojson`
- `datasets/global_renewables_watch_points/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_renewables_watch_points
```

Idempotent (rewrites `points.geojson`).

## Caveats

- Derived product (DL on PlanetScope + QC), not in-situ reference; a point marks turbine
  presence, not a resolvable footprint at 10 m.
