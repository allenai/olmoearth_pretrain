# OlmoEarth Descals oil palm

- **Slug:** `olmoearth_descals_oil_palm`
- **Task type:** classification (sparse point segmentation)
- **Status:** completed
- **Num samples:** 1954

## Source

Local rslearn eval dataset at
`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/descals` (`have_locally: true`,
so no copy made — `raw/olmoearth_descals_oil_palm/SOURCE.txt` points at it). Derived from
the Descals et al. (2021) global oil-palm map with photo-interpreted validation. License
CC-BY-4.0.

Each window is a 32×32 px UTM tile at 10 m carrying **one** photo-interpreted validation
**point** (in `layers/label/data.geojson` and mirrored in `metadata.json` `options`), with
`options.lon`, `options.lat`, `options.label` (class name), and a 1-year `time_range`. There
is also a `label_raster` layer and Sentinel-2 imagery per window, but the annotation unit is
a single point → this is treated as sparse point classification per the manifest
(`label_type: points`).

- Total windows: **17,477** (all with lon/lat, all 32×32, years 2019/2020/2021, all
  post-2016).
- Splits: test 16,877 + train 600 — **both used** (all splits are fair game per spec §5).

## Class mapping (manifest order → id)

| id | name | raw count | selected |
|----|------|-----------|----------|
| 0 | Industrial oil palm | 661 | 661 |
| 1 | Smallholder oil palm | 293 | 293 |
| 2 | Other (background/non-oil-palm) | 16,523 | 1000 |

Balanced with `balance_by_class(per_class=1000)`: the dominant `Other` class is capped at
1000; the two rare oil-palm classes are kept in full (rare-class preservation, spec §5 — not
dropped). Total selected = **1954**.

## Output

- `datasets/olmoearth_descals_oil_palm/points.geojson` — one `Point` feature per location
  (WGS84 lon/lat), `properties.label` = class id, `properties.time_range` = 1-year window
  anchored on the Descals labeled year, `source_id` = `{split}/{window}`.
- `datasets/olmoearth_descals_oil_palm/metadata.json` — dataset metadata (class map + counts).
- No per-sample GeoTIFFs (sparse 1×1 points use the point table, spec §2a).

## Time range

Seasonal/annual labels → 1-year window (`io.year_range`) anchored on each window's labeled
year (2019–2021). No change labels.

## Judgment calls / caveats

- Sparse-point path chosen (points.geojson) over per-window rasters: annotation is a single
  point, matching manifest `label_type: points`. The source's 32×32 `label_raster` was not
  used as the label footprint.
- `Other` is a genuine negative class in the source, so no synthetic negatives fabricated;
  simply capped at the per-class limit.
- Class distribution is heavily skewed toward `Other`; `Smallholder oil palm` (293) is
  sparse but retained. Downstream assembly applies its own rare-class minimum filter.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_descals_oil_palm
```
Idempotent: re-running overwrites `points.geojson`/`metadata.json` deterministically
(`balance_by_class` is seeded).
