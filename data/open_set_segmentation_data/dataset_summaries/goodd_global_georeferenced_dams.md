# GOODD (Global Georeferenced Dams)

- **Slug:** `goodd_global_georeferenced_dams`
- **Status:** completed · classification (presence-only points) · **1,000 points**
- **License:** CC0
- **Annotation method:** manual photointerpretation of Landsat/SPOT imagery.

## Source & access

Mulligan, van Soesbergen & Sáenz, *GOODD, a global dataset of more than 38,000 georeferenced
dams*, Scientific Data 7, 31 (2020), <https://doi.org/10.1038/s41597-020-0362-5>; distributed by
Global Dam Watch (<https://www.globaldamwatch.org/goodd>). `GOODD_data.zip` (~18.9 MB) →
`raw/goodd_global_georeferenced_dams/`. Uses `GOOD2_dams.shp` (38,667 dam-wall points,
EPSG:4326). The `GOOD2_catchments.shp` polygons (upstream drainage basins, not observable at the
dam location) are dropped.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (32×32 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each selected dam wall is one point of the single foreground class. There is **no fabricated
GeoTIFF context, and no background / buffer / negative tiles** — this dataset carries **no
fabricated negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

Single class `0 = dam` (source records only a point, no dam-type attribute). **1,000 of 38,667
dams** sampled (`balance_by_class`, spec §5 per-class cap).

## Time handling

Persistent, undated features → 1-year `time_range` at a representative Sentinel-era year
(spread pseudo-randomly across **2016–2022**). `change_time = null`.

## Output

- `datasets/goodd_global_georeferenced_dams/points.geojson`
- `datasets/goodd_global_georeferenced_dams/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.goodd_global_georeferenced_dams
```

Idempotent. Raw zip staged at `raw/goodd_global_georeferenced_dams/GOODD_data.zip`.

## Caveats

- Dam walls of small dams may be sub-10 m; a point marks presence, not a precise footprint.
- Leaves ~37.7k dams unused (per-class cap); re-run with a higher cap to scale up.
