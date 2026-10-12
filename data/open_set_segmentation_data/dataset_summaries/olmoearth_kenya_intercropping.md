# OlmoEarth Kenya intercropping

- **Slug:** `olmoearth_kenya_intercropping`
- **Status:** completed
- **Task type:** classification (cropping system: intercrop / monocrop / other)
- **Family / label_type:** crop_type / dense_raster (registry) → processed as **sparse points** (see decision below)
- **Region:** Kenya (smallholder farmland; western Kenya + coastal/Tana areas)
- **Num samples:** 3,000 (1,000 per class, balanced)
- **Output:** `datasets/olmoearth_kenya_intercropping/points.geojson` + `metadata.json`

## Source

Local rslearn eval dataset (`have_locally: true`), no raw copy made; `raw/olmoearth_kenya_intercropping/SOURCE.txt` points at:

```
/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/kenya_intercropping/
```

Kenyan smallholder-field cropping-system labels from a manual field survey
(Copernicus4GEOGLAM ground points, "relabeled with original"). 8,285 windows across
`train` (4,071) / `val` (2,125) / `test` (2,089). Each window is a 64×64 patch at 10 m in
UTM zone 36S (EPSG:32736), with a `label_raster` layer (uint8, nodata 255) and a `label`
vector layer. License: internal.

## Key finding & processing decision (dense_raster → sparse points)

The registry tags this `dense_raster`, but inspection showed **every window's
`label_raster` labels exactly one pixel** — the surveyed point at the window center
(row 32, col 32) — with all other 4,095 pixels = 255 (nodata). The `label` vector layer is
only the window-footprint rectangle, not a real field boundary. Intercropping vs
monocropping is a field-level property observed at a single survey point; the surrounding
pixels are not guaranteed to be the same field.

Therefore this is a **pure sparse-point dataset** (spec §2/§2a): each label is a single
10 m pixel with a class id. Per the SOP, sparse 1×1 labels are written to **one dataset-wide
GeoJSON point table** (`points.geojson`), NOT per-sample GeoTIFFs — writing 64×64 tiles
would fabricate labels for unobserved neighboring pixels, which spec §2 forbids. This
mirrors the `olmoearth_ethiopia_crops` point-table treatment (also manual field survey).

## Class mapping

Class ids follow the source `label_raster` encoding verbatim (1:1 with the window
`category` string):

| id | name       | source category | raw windows |
|----|------------|-----------------|-------------|
| 0  | intercrop  | `intercrop`     | 2,438       |
| 1  | monocrop   | `monocrop`      | 3,332       |
| 2  | other      | `other`         | 2,515       |

The manifest blurb listed `["background","monocrop","intercrop"]`, but the on-disk
categories use `other` in place of `background`. `other` is a real surveyed residual class
(neither mono- nor intercropped), so it is kept as a normal class; no synthetic negatives
are fabricated (assembly adds cross-dataset negatives, spec §5). All three classes fit well
under the 254-class uint8 cap.

## Sampling, time range

- **Sampling:** classification, `balance_by_class(..., per_class=1000)` → 1,000 per class,
  3,000 total. All three classes had >1,000 candidates, so each was truncated to 1,000
  (well under the 25k per-dataset cap). All train/val/test splits used (no split filtering).
- **Time range:** all windows carry the identical growing-season window
  `[2022-10-01, 2023-03-31)` (~6 months, ≤ 1 year), preserved verbatim; `change_time` null.
  Post-2016 ✓. (The manifest's `2019–2021` hint does **not** match the on-disk windows,
  which are the 2022/23 short-rains season; the on-disk `time_range` is trusted.)
- **Point location:** exact center of the single labeled pixel, transformed from the
  window's UTM projection to WGS84 lon/lat (GeoJSON native CRS).

## Verification (spec §9)

- `points.geojson`: FeatureCollection, 3,000 Point features, `task_type=classification`,
  label counts `{intercrop:1000, monocrop:1000, other:1000}`.
- All features: valid `Point` geometry, `label ∈ {0,1,2}`, coords within Kenya
  (lon 34.00–40.15, lat −4.66–1.27), `time_range` ≤ 1 year and post-2016, `change_time`
  null, `source_id` set.
- `metadata.json` class ids cover all label values present.
- **Georeferencing round-trip / spatial sanity:** a sampled point's lon/lat maps back to
  its source window's labeled pixel (32,32) with the matching class value (verified for
  `test/...point_111..._-2.761391_40.148925`, label 0 → source value 0).

## Reproduce (idempotent)

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_kenya_intercropping
```

## Caveats

- Single-pixel field-survey labels: usable as sparse points, not dense segmentation.
- The `label` vector layer is a window-footprint box (not a field polygon); the
  `label_raster` center pixel is the authoritative label used here.
- `other` semantics are the survey's residual bucket (non-mono/intercrop land), retained as
  a class per spec §5.
