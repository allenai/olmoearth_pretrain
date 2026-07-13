# Global Renewables Watch

- **Slug**: `global_renewables_watch`
- **Status**: completed
- **Task type**: classification (open-set segmentation, mixed polygon + point/detection)
- **Family / region**: energy / Global
- **License**: MIT
- **Source**: GitHub (Microsoft / Planet / The Nature Conservancy),
  <https://github.com/microsoft/global-renewables-watch>

## What the source is

Global Renewables Watch (GRW) is a quarterly global inventory of solar photovoltaic (PV)
installations and wind turbines detected from PlanetScope imagery with deep learning and
human QC. Each feature carries an estimated construction date (`construction_year`,
`construction_quarter`) and preceding land use. We use the **v1.0 (2024 Q2) release**
GeoPackages from GitHub Releases:

- `solar_all_2024q2_v1.gpkg` — 86,345 PV installation **polygons** (EPSG:3857).
- `wind_all_2024q2_v1.gpkg` — 375,197 wind turbine **points** (EPSG:3857).

Fields: `construction_year` (2017–2024, no nulls), `construction_quarter`, `COUNTRY`,
`landcover_in_2018`, `area` (solar, m²), and `local_utm_*` (wind).

## Access method

Downloaded unauthenticated over HTTPS via `download.download_http` from the v1.0 release
assets (see `raw/global_renewables_watch/SOURCE.txt`). No credentials required.

## Class / label mapping

Two label kinds are combined into a single classification dataset with a shared scheme:

| id | name          | encoding |
|----|---------------|----------|
| 0  | background    | negative / non-target land (fill + detection background) |
| 1  | solar_pv      | PV polygon rasterized (all_touched) into a variable ≤64×64 UTM tile |
| 2  | wind_turbine  | turbine point via the tunable **detection encoding** |
| 255| nodata/ignore | detection buffer rings around turbine positives |

- **solar_pv (polygons)**: each polygon is reprojected to its local UTM zone (10 m) and
  rasterized into a tile sized to the polygon footprint, capped at 64×64 and centered on
  the polygon bbox center. Inside-polygon = 1, outside = 0 (background). Polygons larger
  than 640 m (~7% of them) are captured by a central 64×64 crop. Every solar tile contains
  ≥1 solar pixel.
- **wind_turbine (points → detection)**: `sampling.encode_detection_tile` with
  `tile_size=32`, `positive_size=1`, `buffer_size=10`. Each turbine is a single positive
  pixel (class 2) at tile center, ringed by a 10 px nodata (255) buffer (21×21 ignore
  region, since coordinates aren't pixel-exact), with background (0) filling the rest of
  the 320 m × 320 m context tile. Neighboring turbines that fall within the tile
  are also marked positive (135 of 1000 tiles are multi-turbine), so no in-tile turbine is
  mislabeled as background.
- **background negatives**: 300 background-only 32×32 tiles placed 5–15 km from a random
  turbine and verified turbine-free (no turbine within 1 km in EPSG:3857, ≫ the tile
  half-diagonal even at high latitude), so the detection class has explicit negatives.

## Time range and change handling

Each sample's `time_range` is a 1-year window (`io.year_range`) anchored on the feature's
`construction_year`. Note `2017` is the release's earliest-baseline bucket (features built
in or before 2017-Q1 are lumped there); it is the largest bucket. No `change_time` is set —
these are presence labels, not dated change events. Caveat: at the construction year a PV
site may be only partially built; the label footprint is from the 2024-Q2 observation.

## Sampling

- Bounded to **≤1000 tiles per target class**, stratified by construction year
  (`balance_by_class` on `year`, ~125/year × 8 years) for temporal diversity, then capped
  at 1000. All years 2017–2024 have far more than 125 features.
- Result: **1000 solar_pv**, **1000 wind_turbine**, **300 background negatives**
  = **2300 samples total**.

## Output GeoTIFF spec

Single-band uint8, local UTM at 10 m/pixel, north-up, nodata=255. Solar tiles are variable
size ≤64×64 (footprint-sized); wind and negative tiles are 32×32.

## Verification performed

- 6 sample tifs inspected: single band, uint8, UTM CRS, 10 m resolution, size ≤64.
- Solar tiles: values ⊆ {0,1}; all 1000 contain ≥1 solar pixel.
- Wind tiles: values ⊆ {0,2,255}; detection layout confirmed (center positive `2`, 10 px
  `255` ring = 21×21 ignore / 440 nodata px, `0` background); all 1000 contain ≥1 positive.
- Negative tiles: all-zero (background) 32×32.
- Every `.tif` has a matching `.json` with a 1-year `time_range`; metadata class IDs cover
  all values present in the tifs.
- Not done: full Sentinel-2 overlay eyeball (georeferencing is exact via rslearn write and
  coordinates come directly from the source vectors).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_renewables_watch
```

Idempotent: re-running skips any `locations/{id}.tif` already written. All logic lives in
`olmoearth_pretrain/open_set_segmentation_data/datasets/global_renewables_watch.py`; no
shared module or `registry.json` was modified by the script.

## Caveats

- Derived product (deep learning + human QC), not in-situ reference; still usable per the
  task spec.
- Wind turbines are ~1 px at 10 m — near the resolution limit for S2/S1/Landsat; the
  detection encoding (small positive + nodata buffer) is the intended representation.
- The `2017` construction bucket conflates "built in 2017" with "present at baseline".
