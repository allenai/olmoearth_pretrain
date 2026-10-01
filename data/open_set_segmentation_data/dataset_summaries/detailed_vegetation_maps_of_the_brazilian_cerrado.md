# Detailed Vegetation Maps of the Brazilian Cerrado

- **Slug**: `detailed_vegetation_maps_of_the_brazilian_cerrado`
- **Status**: completed
- **Task type**: classification (sparse points → `points.geojson`, spec §2a)
- **Num samples**: 2,829 in-situ field points, 12 classes
- **Source**: PANGAEA — Bendini, H.N. et al. (2021), *Detailed vegetation maps of the
  Brazilian Savanna (Cerrado) biome produced with a semi-automatic approach*,
  doi:[10.1594/PANGAEA.932642](https://doi.org/10.1594/PANGAEA.932642)
- **License**: CC-BY-4.0 (attribution: cite Bendini et al. 2021)
- **Family / region**: savanna / Cerrado, central Brazil

## Source contents & access

Direct file download (no credential) from
`https://download.pangaea.de/dataset/932642/files/<filename>`:

| File | What |
| --- | --- |
| `Cerrado_Vegetation_Map_level1_Bendini-etal_2021.tif` | 30 m random-forest map, 3 level-1 classes (EPSG:4326, uint16, nodata=0) |
| `Cerrado_Vegetation_Map_level2_Bendini-etal_2021.tif` | 30 m random-forest map, 12 level-2 classes (EPSG:4326, uint16, nodata=0, values 1–12) |
| `Samples_for_Vegetation_Mapping_Bendini-etal_2021.csv` | **2,828** field ground samples (WGS84 `Point(lon lat)`, level-1 + level-2 class, origin) — *used here* |
| `Style_Vegetation_level{1,2}_Bendini-etal_2021.qml` | QGIS style files |

All five files are archived in `raw/detailed_vegetation_maps_of_the_brazilian_cerrado/`.
The level-2 map is `85674 × 77367 px` (~0.000268°, ~30 m), spanning lon −60.99…−40.25,
lat −25.09…−2.12 (the Cerrado biome). Verified: raster values are exactly 1–12 with
nodata 0.

## Key decision: in-situ field samples, not the derived RF map

The manifest `label_type` is *"dense_raster + 2,828 ground samples"*. The two rasters are
**derived random-forest products** (Landsat ARD + environmental covariates; reported
level-2 overall accuracy 0.77, with weak per-class F1 for Vereda 0.36 and Campo rupestre
0.53). The CSV holds the **actual in-situ field observations** (WGS84 lon/lat + level-1 and
level-2 physiognomy class + provenance) that were used to train/validate that RF model.

Spec §0 explicitly **prefers manual/in-situ reference data over derived-product maps**
(maps are a fallback, and only homogeneous/high-confidence windows). The field samples
carry lon/lat + class, so per the task's §1 preference this dataset is built from the
**ground samples as sparse points** → one dataset-wide `points.geojson` (spec §2a), rather
than cropping windows from the RF map. The RF maps are retained in `raw/` for a possible
future dense-raster reprocess but were **not** used for labels.

## Class mapping

Label = the **level-2** hierarchy (12 Cerrado physiognomies, the "fine
savanna-physiognomy classes" the manifest highlights), following the Ribeiro & Walter
(2008) classification. Class id = source level-2 code − 1 (ids 0–11). Each point also
carries `level1_id`, `level1_name`, `level2_name`, and `origin` as auxiliary properties.

| id | source code | Class | Field-sample count |
| --- | --- | --- | --- |
| 0 | 1 | Campo limpo | 276 |
| 1 | 2 | Campo rupestre | 210 |
| 2 | 3 | Campo sujo | 319 |
| 3 | 4 | Cerradao | 160 |
| 4 | 5 | Cerrado rupestre | 162 |
| 5 | 6 | Cerrado sensu stricto | 580 |
| 6 | 7 | Ipuca | 91 |
| 7 | 8 | Mata riparia | 447 |
| 8 | 9 | Mata seca | 76 |
| 9 | 10 | Palmeiral | 135 |
| 10 | 11 | Parque de cerrado | 246 |
| 11 | 12 | Vereda | 127 |

Level-1 (auxiliary): 1 = Nat. Arbustivo (savanna), 2 = Nat. Campestre (grassland),
3 = Nat. Florestal (forest).

All 12 classes are well under the 1000/class and 25k total caps, so **no truncation** —
every field sample is kept (spec §5: keep sparse classes; downstream assembly handles
rare-class filtering and negatives). Per-class descriptions from Ribeiro & Walter (2008)
are stored in `metadata.json`.

## Time-range & change handling

Cerrado vegetation physiognomy is a **persistent (static) land-cover type**. Per spec §5
(static labels) each point gets a representative 1-year Sentinel-era window; the maps and
study period fall in 2016–2020, so we anchor on **2018** (`time_range` =
2018-01-01…2019-01-01, 365 days). `change_time = null`. This static-label treatment also
satisfies the ≥2016 rule regardless of the original field-visit dates (physiognomy
persists across the window).

## Sample counts, verification (§9)

- `points.geojson`: `FeatureCollection`, `count = 2829`, `task_type = classification`;
  every feature is a WGS84 `Point`. Labels span exactly ids 0–11; `metadata.json` class
  ids cover all label values. `time_range` span = 365 days on all points.
- Coordinates all within central Brazil (lon −59.4…−42.3, lat −22.7…−2.8).
- **Spatial/label sanity check**: sampling the level-2 RF map raster at the 2,829 point
  coordinates, **94.1%** (2642/2807) of points matched the map's level-2 class exactly, and
  only 22 points landed on map nodata. High agreement (above the map's 0.77 OA, since these
  are the RF's own training/validation samples) confirms the point georeferencing is
  correct and the labels are sensible.
- No per-sample `.tif`/`.json` files (point-only dataset uses `points.geojson`, spec §2a).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.detailed_vegetation_maps_of_the_brazilian_cerrado
```

Idempotent: raw files are skipped if present; parsing + seeded `balance_by_class` are
deterministic; `points.geojson`/`metadata.json` are rewritten atomically.

## Caveats

- Labels are point (1×1) samples; pretraining projects them onto the S2 grid.
- Field samples were compiled from many sources over multiple years (SEMA, FIP fieldwork,
  LAPIG, IFN-DF, older inventories); treated as persistent physiognomy with a fixed 2018
  window (see above).
- The derived RF maps (level-1/level-2, 30 m) are available in `raw/` if a future
  dense-raster (homogeneous-window) variant is desired; they were intentionally not used
  in favor of the in-situ reference per spec §0.
