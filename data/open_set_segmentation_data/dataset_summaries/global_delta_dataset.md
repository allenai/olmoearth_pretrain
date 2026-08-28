# Global Delta Dataset

- **Slug**: `global_delta_dataset`
- **Status**: completed
- **Task type**: classification (sparse points)
- **Num samples**: 2778
- **Source**: https://github.com/jhnienhuis/GlobalDeltaChange (Nienhuis et al. 2020,
  *Nature* — "Global-scale human impact on delta morphology has led to net land area gain").
- **License**: MIT / CC-BY-4.0 (open GitHub repo; no credentials needed).

## What the source is

A global inventory of **10,848 river deltas**. The core file `GlobalDeltaData.mat`
(MATLAB v7.3 / HDF5, ~8 MB) stores, per delta, a river-mouth `MouthLon`/`MouthLat` and
three modeled sediment/energy fluxes: `QWave` (WaveWatch), `QRiver_prist` (pristine
fluvial, WBMSED), and `QTide` (TOPEX). The delta **morphology class** is the dominant
flux. This is exactly the published classification
(`validation/global_delta_validation.m`):

```matlab
[~, morphology] = max([QWave, QRiver_prist, QTide], [], 2)
% -> ["Wave dominated", "River dominated", "Tide dominated"]
```

## Class mapping

Ids assigned by descending global frequency (matches the paper's global totals):

| id | class | count (full inventory) | count (selected) |
|----|-------|------------------------|------------------|
| 0  | wave-dominated  | 8245 | 1000 |
| 1  | river-dominated | 1825 | 1000 |
| 2  | tide-dominated  | 778  | 778  |

The manifest also lists a generic `delta` class. Every point **is** a delta and is exactly
one dominance type, so a per-point label uses only the three morphology classes; `delta`
is the umbrella over all points, not a separate id. Documented rather than encoded.

## Encoding decisions

- **Point encoding (points.geojson, spec §2a).** The morphology dominance is a single
  per-delta attribute attached to the river-mouth location. The full inventory provides
  reliable *points* for all deltas; polygons exist only for the 100 largest deltas
  (`land_area_change/GlobalDeltaMax100_poly.kml`) and are not worth a hybrid scheme. Each
  delta is therefore one sparse WGS84 point at its river mouth. Deltas are large landforms,
  so the S2/S1/Landsat context around the mouth carries the morphological signal at 10-30 m;
  pretraining projects the point onto the S2 grid.
- **Longitude fix.** `MouthLon` is stored in `[0, 360)`; converted to `[-180, 180)`.
- **Time range.** Static morphology -> representative 1-year Sentinel-era window
  (2016-01-01 .. 2017-01-01), `change_time=null`.
- **Land/water change NOT used as a change label.** The repo's Aquamonitor/GSW land-area
  change is a multi-year comparison with no precise event date (fails the §5 ~1-2 month
  timing requirement). Per the task instructions it is intentionally excluded; only the
  static morphology classification is encoded.
- **Balancing.** `balance_by_class(per_class=1000)` -> wave 1000, river 1000, tide 778
  (total 2778, well under the 25k cap). Wave and river deltas are downsampled from their
  full inventory counts.

## Verification

- `points.geojson`: FeatureCollection, count 2778, task_type classification; label counts
  {0:1000, 1:1000, 2:778}. Lon in [-178.6, 178.7], lat in [-54.9, 79.0] (global coastal).
- `metadata.json`: 3 classes with descriptions, nodata 255.
- **Spatial sanity check** (nearest inventory delta to known mouths, on the raw fluxes):
  Mississippi -> (-89.26, 29.11), 0.04° off, **river-dominated** (correct, birdsfoot);
  Amazon -> tide-dominated (correct); Ganges-Brahmaputra -> tide-dominated (correct);
  Nile -> nearest point 1.1° off, model says river (the model's own tide/river accuracy is
  limited — see repo confusion matrix; tide precision ~23%). Coordinate conversion and
  placement verified good.

## Caveats

- Labels are **model predictions**, not field observations. The repo reports per-class
  prediction accuracy: wave 89%, river 65%, tide 23%. Tide-dominated is the noisiest class.
- The morphology is a whole-delta property represented at a single river-mouth pixel; the
  surrounding delta footprint is not masked (point, not polygon).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_delta_dataset
```

Idempotent: re-downloads `GlobalDeltaData.mat` only if missing and rewrites the single
`points.geojson` + `metadata.json`.
