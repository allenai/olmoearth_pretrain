# DETER-B (near-real-time deforestation & degradation alerts)

- **Slug**: `deter_b_near_real_time_deforestation_degradation_alerts`
- **Status**: completed
- **Task type**: classification (open-set segmentation; change/event labels, change_time scheme)
- **Family / region**: deforestation / Amazon + Cerrado (Brazil)
- **License**: CC-BY-SA-4.0
- **Source**: INPE TerraBrasilis DETER-B, <https://terrabrasilis.dpi.inpe.br/downloads/>
- **num_samples**: 5991

## What the source is

DETER-B is INPE's near-real-time forest-change alert system. Analysts photointerpret
medium-resolution imagery (CBERS-4/4A AWFI/WPM, Amazonia-1 WFI, etc.) and hand-digitize
polygons of newly detected change, each tagged with a change class (`classname`) and an
observation date (`view_date`). Alerts are served as WFS layers from the TerraBrasilis
GeoServer:

- `deter-amz:deter_amz` — Legal Amazon, all classes (~451k polygons, 2016-08 onward)
- `deter-cerrado-nb:deter_cerrado` — Cerrado, clearcut only (~129k polygons, 2018-05 onward)

There is **no standalone DETER Pantanal layer** on the GeoServer (Pantanal has PRODES but
not DETER), so despite the manifest region string this dataset covers Amazon + Cerrado.

## Access method

Public WFS GetFeature (no credentials). Per (layer, classname, year) query filtered with
`CQL_FILTER=classname='X' AND view_date DURING <year>`, `outputFormat=application/json`,
`srsName=EPSG:4326` (GeoServer reprojects the native EPSG:4674 / SIRGAS 2000 to WGS84).
Up to 300 candidate polygons fetched per query-year (years 2016–2025) and cached as
GeoJSON in `raw/<slug>/<layer>__<classname>__<year>.geojson`. See `raw/<slug>/SOURCE.txt`.

## Class / label mapping

Manifest DETER classnames are unified into one uint8 class scheme:

| id | name | DETER classname(s) |
|----|------|--------------------|
| 0 | background | (no alert in-tile) |
| 1 | clearcut | DESMATAMENTO_CR (Amazon + Cerrado) |
| 2 | deforestation_with_vegetation | DESMATAMENTO_VEG |
| 3 | degradation | DEGRADACAO |
| 4 | selective_logging | CS_DESORDENADO + CS_GEOMETRICO + CORTE_SELETIVO |
| 5 | mining | MINERACAO |
| 6 | fire_scar | CICATRIZ_DE_QUEIMADA |

Full-layer class availability (hits): clearcut 394,874 (265,732 Amazon + 129,142 Cerrado);
fire_scar 116,334; degradation 39,260; selective_logging 11,290; deforestation_with_vegetation
9,867; mining 8,551.

## Encoding, tiling, and time

- Each selected alert polygon → one **64×64 UTM 10 m** tile centered on the polygon
  centroid. The polygon is rasterized (`all_touched=True`) as its class id; everything
  else is background (0). nodata sentinel = 255 (unused — background is a real class).
- **change_time scheme** (spec §5): `change_time` = the alert `view_date`;
  `time_range` = ±180 days (360-day window) centered on that date. Deforestation,
  degradation, fire scars and mining persist in imagery, so a ~1-year window centered on
  the alert is well-posed; no rejection needed.
- **Only the target polygon is drawn per tile.** Co-located alerts of a different
  date/class are left as background (matches the "label = mask of the alert polygon"
  instruction). Because only a sampled subset of alerts is downloaded, exhaustive
  multi-polygon labeling was not attempted.
- Many polygons exceed 640 m (fraction with max-dimension > 640 m in a 400-polygon probe:
  clearcut 0.36, degradation 0.62, fire_scar 0.74, selective_logging 0.95). For those the
  64×64 tile crops to the central 640 m; the resulting "change occurred here" mask is
  still valid (often fully positive).

## Sampling

Up to 1000 tiles per alert class (spec §5), sampled round-robin across years for temporal
diversity, seed 42. Candidates loaded per class: clearcut 5393, deforestation_with_vegetation
2749, degradation 3000, selective_logging 5382, mining 2810, fire_scar 3000.

Final per-class tile counts (background co-occurs in most tiles):

| class | tiles |
|-------|------:|
| clearcut | 1000 |
| deforestation_with_vegetation | 1000 |
| degradation | 1000 |
| selective_logging | 995 |
| mining | 1000 |
| fire_scar | 996 |
| **total** | **5991** |

9 candidate tiles were dropped as degenerate (polygon missed all pixel centers of the
centered tile).

## Verification

- Sampled tiles: single band, 64×64, uint8, local UTM (EPSG:327xx) at 10 m, nodata 255.
- Pixel values across a 200-tile sample: {0,1,2,3,4,5,6} — all covered by `metadata.json`.
- Every `.tif` has a matching `.json`; `time_range` span = 360 days; `change_time` set.
- Sample centroids fall in Brazil (lon ≈ −45…−65, lat ≈ −3…−15), UTM zones 19–22 S,
  consistent with the Amazon/Cerrado biomes.

## Caveats

- Alerts are analyst-digitized on medium-resolution sensors; polygon edges are approximate
  at 10 m. Only the target polygon is labeled per tile (see above).
- Cerrado contributes clearcut only; all other classes are Amazon-only.
- Source native CRS is EPSG:4674 (SIRGAS 2000); WFS-reprojected to WGS84 (sub-meter
  difference, negligible at 10 m).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.deter_b_near_real_time_deforestation_degradation_alerts
```

Idempotent: existing `raw/*.geojson` and `locations/*.tif` are skipped on re-run.
