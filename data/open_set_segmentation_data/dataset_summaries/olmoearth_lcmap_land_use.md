# olmoearth_lcmap_land_use

**Status:** completed · classification · 5,643 samples (point table)

## Source
Local rslearn eval `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/lcmap_lu`
(USGS LCMAP land-use, derived product). Each window is one interpreted land-use point with
the class name, lon/lat, and a ~1-year time range stored in window `metadata.json`
`options` (`lon`, `lat`, `label`, `split`). Groups: `train` (1,800) + `test` (24,713) =
26,513 labeled points; all splits used.

## Processing
- Parallel-scanned all window `metadata.json` (`Pool(64)`, ~18 s) to collect
  (lon, lat, label, year, source_id).
- Classes mapped in manifest order → ids 0–5: Developed, Agriculture, Rangeland, Forest,
  Non-forest Wetland, Other (short LCMAP definitions in `metadata.json` `classes[].description`).
- Sparse point segmentation → **point table** (`points.json`, spec §2a), not per-point
  GeoTIFFs. Each point: `{lon, lat, label=class_id, time_range, source_id}`.
- Time range = the point's LCMAP labeled year (2017–2021), as a 1-year window.
- Balanced to ≤1000 per class (seeded shuffle).

## Output
- `datasets/olmoearth_lcmap_land_use/points.json` — 5,643 points.
- `datasets/olmoearth_lcmap_land_use/metadata.json` — class map + counts.
- `raw/olmoearth_lcmap_land_use/SOURCE.txt` — pointer to source (local; not copied).

## Class counts
Developed 1000 · Agriculture 1000 · Rangeland 1000 · Forest 1000 ·
Non-forest Wetland 643 · Other 1000.

## Reproduce
`python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_lcmap_land_use`
(idempotent; rewrites `points.json`).

## Notes
- 1×1 point labels carry no spatial context by design; paired with S2/S1/Landsat at
  pretraining time by lon/lat + time overlap.
- This was the pipeline's bootstrap/worked-example dataset.
