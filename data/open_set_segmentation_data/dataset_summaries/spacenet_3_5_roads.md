# SpaceNet 3/5 Roads (`spacenet_3_5_roads`)

**Status:** completed · **task_type:** classification (positive-only line mask) ·
**num_samples:** 4488

## Source
SpaceNet Roads challenges, hosted on the public AWS Open Data bucket
`s3://spacenet-dataset/spacenet/` (unsigned/anonymous S3; license **CC-BY-SA-4.0**).
Roads are covered by two challenges:
- **SpaceNet 3** (`SN3_roads`): AOI_2_Vegas, AOI_3_Paris, AOI_4_Shanghai, AOI_5_Khartoum.
- **SpaceNet 5** (`SN5_roads`): AOI_7_Moscow, AOI_8_Mumbai. (San Juan exists only as a
  test/imagery split, no labels.)

Labels are hand-digitized road **centerlines** (LineStrings), one small GeoJSON per
~400 m image chip under `train/{AOI}/geojson_roads_speed/`, carrying route/type + inferred
speed attributes (SN3: `road_type`, `lane_number`, `paved`, `bridge_type`,
`inferred_speed_mph`; SN5: OSM-style `highway`, `surface`, `lanes`, `inferred_speed_mph`).
Geometries are WGS84 (CRS84 lon/lat). Paired imagery is DigitalGlobe/Maxar VHR (~0.3 m).

## Access method (label-only)
Only the per-chip `geojson_roads_speed/*.geojson` label files from the labeled `train/`
splits are downloaded (via `download.download_s3_unsigned`) to
`raw/spacenet_3_5_roads/`. The multi-GB VHR imagery tarballs are **not** pulled
(pretraining supplies its own S2/S1/Landsat imagery), and the unlabeled `test_public`
splits are skipped. Total label download ≈ 68 MB across 4918 chips.

## Recipe (spec §4 "lines")
Each road centerline is reprojected WGS84 → local UTM at 10 m/pixel, dilated by ~1 px
radius (`shapely.buffer(1.0)`) → ~2–3 px (20–30 m) wide, and rasterized (`all_touched`)
into a **64×64** UTM 10 m tile centered on the chip's road-union centroid. One tile per
chip (~400 m chips fit inside one 640 m tile). Chips whose road mask has `< 3` road pixels
(empty / trivial slivers) are dropped (430 of 4918 chips dropped → 4488 tiles).

## Class mapping
Single foreground class — **positive-only** (spec §5): non-road pixels are left as
nodata/ignore. No synthetic background class is fabricated; the assembly step supplies
negatives from other datasets.

| id | name | meaning |
|----|------|---------|
| 0 | road | a mapped road centerline, dilated to ~20–30 m so it registers at 10 m |
| 255 | (nodata) | non-road / unobserved |

Type / surface / lane / inferred-speed attributes exist in the source but are collapsed
into the single `road` class per the task spec.

## Time range / change handling
Road networks are static/persistent features, so a static-label 1-year window (spec §5) is
used; `change_time` is null. SpaceNet 3 VHR was collected ~2015–2016 and SpaceNet 5
~2017–2018, so each program is anchored to a representative Sentinel-era year within the
manifest range [2016, 2019]: **SN3 → 2017**, **SN5 → 2018** (`time_range` = that calendar
year).

## Sample counts (tiles per AOI)
- AOI_2_Vegas: 981
- AOI_3_Paris: 257
- AOI_4_Shanghai: 1028
- AOI_5_Khartoum: 283
- AOI_7_Moscow: 1298
- AOI_8_Mumbai: 641
- **Total: 4488** (well under the 25k cap).

## Verification (spec §9)
- Sampled tifs: single band, `uint8`, 64×64, UTM CRS (e.g. EPSG:32611) at 10 m, nodata
  255, pixel values ∈ {0, 255} only — matches the declared class map.
- Every `.tif` has a matching `.json`; all `time_range`s are exactly 1 year; `change_time`
  null; `classes_present` = [0].
- Georeferencing sanity: tile-center lon/lat for one sample per AOI lands in the correct
  city (Vegas, Paris outskirts, Shanghai, Khartoum, Moscow, Mumbai).

## Caveats
- Narrow residential streets are under-resolved at 10 m; they aggregate into the road-
  network signal but individual thin streets may be faint. Major/arterial roads are clearly
  resolvable. Retained per spec §5.
- The 1-year time window is a representative-era assignment (roads are static), not the VHR
  acquisition date.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spacenet_3_5_roads
```
Idempotent: re-running skips already-written `locations/{id}.tif`. Use `--probe` to
scan/report counts without writing.
