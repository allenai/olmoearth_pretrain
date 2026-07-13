# Sen1Floods11 — COMPLETED (classification, dense_raster)

- **Slug**: `sen1floods11`
- **Name**: Sen1Floods11
- **Source**: Cloud to Street / Google — GitHub `cloudtostreet/Sen1Floods11`;
  data on public GCS bucket `gs://sen1floods11/v1.1`.
- **Citation**: Bonafilia et al. 2020, CVPRW.
- **Family / region**: flood / global (11 events, 6 continents).
- **License**: CC-BY-4.0.
- **Label type**: dense_raster → per-pixel **classification**.
- **Task type**: classification. **num_samples**: 1212 tiles.

## Source

The hand-labeled subset `data/flood_events/HandLabeled/` (446 georeferenced
512×512 ~10 m chips, EPSG:4326) over 11 flood events. Two source rasters per
chip are used:

- `LabelHand` (int16): manually annotated surface-water extent at the flood
  Sentinel-1 acquisition. `-1` = no data / not analyzed, `0` = not water,
  `1` = water.
- `JRCWaterHand` (uint8): JRC Global Surface Water permanent-water mask
  co-registered to the chip. `0` = not permanent, `1` = permanent water.

(The WeaklyLabeled subset and the S1/S2 imagery layers are not used — only the
high-quality hand labels, as the task directs.)

## Access method

Public, no credentials. Downloaded with `gsutil -m rsync` from
`gs://sen1floods11/v1.1/data/flood_events/HandLabeled/{LabelHand,JRCWaterHand}/`
into `raw/sen1floods11/`, plus `Sen1Floods11_Metadata.geojson` for event dates.

## Class mapping (3 classes, manifest order)

| id | name | definition |
|----|------|-----------|
| 0 | flood water | LabelHand water AND not JRC permanent water (flood inundation) |
| 1 | permanent water | JRC permanent water, restricted to observed (non-nodata) pixels; wins over flood/non-water |
| 2 | non-water | LabelHand not-water (land) |
| 255 | nodata/ignore | LabelHand == -1 (unanalyzed) |

Fusion order per pixel: nodata where `LabelHand==-1`; else non-water/flood from
`LabelHand`; then permanent water overrides where `JRCWaterHand==1`.

## Processing

Each 512×512 EPSG:4326 chip is reprojected **once** to its local UTM zone at
10 m using **nearest** resampling (categorical labels), then cut into 64×64
tiles. Tiles that are >50% nodata are dropped; a tile counts toward a class only
if it holds ≥32 px of it. **Tiles-per-class balanced** selection (spec §5): rare
classes filled first up to 1000 tiles/class; a tile contributes to every class
it contains. 22,242 candidate tiles → 1212 selected.

**Tiles containing each class** (a tile can count for several):

- flood water: 1004
- permanent water: 1022
- non-water: 1000

Well under the 25k per-dataset cap.

## Time range & change handling

The flood mask is an **event** label. Each chip's event date is the Sentinel-1
acquisition from `Sen1Floods11_Metadata.geojson`; `change_time` is set to that
date and `time_range` is a **1-year window centered** on it (±182/183 days).
Event dates:

Bolivia 2018-02-15 · Ghana 2018-09-18 · India 2016-08-12 · Cambodia/Mekong
2018-08-05 · Nigeria 2018-09-21 · Pakistan 2017-06-28 · Paraguay 2018-10-31 ·
Somalia 2018-05-07 · Spain 2019-09-17 · Sri-Lanka 2017-05-30 · USA 2019-05-22.

## Caveats

- The Colombia event (metadata ID 12) has **no hand labels** and is absent from
  the hand-labeled subset (11 events represented, not 12).
- Hand-labeled chips prefix the Cambodia event as **`Mekong`** (the Mekong
  river); this is mapped to the Cambodia (KHM) acquisition date.
- All source dates fall 2016–2019, within the Sentinel era.
- Permanent-water/flood split relies on the JRC mask co-registered by the
  authors; a pixel that is permanent water but read as land at flood time is
  still labeled permanent water (JRC priority).

## Verification

- 1212 `.tif` + 1212 matching `.json`; every tile single-band uint8, local UTM,
  10 m, 64×64, values ⊆ {0,1,2,255} with nodata=255.
- `time_range` is a 1-year window and `change_time` set on every sample.
- Spatial/label round-trip: sampled UTM label pixels reprojected back to WGS84
  and compared to the source LabelHand+JRC rasters — 64/64 agreement on a Bolivia
  permanent-water tile; tile center coordinates land in Bolivia as expected.
- Re-running skips already-written tiles (idempotent).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sen1floods11
```
