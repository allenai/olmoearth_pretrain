# OlmoEarth forest loss driver

- **Slug**: `olmoearth_forest_loss_driver`
- **Task type**: classification (change segmentation)
- **Source**: local rslearn dataset (`have_locally: true`), `olmoearth` internal eval,
  deployed as forest-loss.allen.ai.
- **Source path**: `/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/combined`
- **Region / time**: Amazon basin (Peru, Brazil, Colombia), 2019–2025 events.
- **Num samples**: 4387

## Source

7883 rslearn windows across 11 groups (`peru3`, `nadia2`, `20250428_brazil/colombia_phaseN`,
`20260112_peru`, `*_interesting`, ...). Each window is one manually-annotated GLAD
forest-loss event. The `label` vector layer (`layers/label/data.geojson`, WGS84) holds a
**polygon footprint** of the loss patch and a `new_label` driver class. The event date
comes from `info.json` (`pixel_date` or `date`); windows lacking `info.json` use the
midpoint of the window `metadata.json` `time_range`. All source splits/groups are used.

## Access

Local; no download. `raw/olmoearth_forest_loss_driver/SOURCE.txt` points at the source
path (nothing copied).

## Points vs tiles decision

The label carries a **real multi-pixel footprint** (median ~7×7 px at 10 m, 90th pct
~16 px, occasional >64 px) and is a **dated change event**, so per spec §5 ("the label is
a mask of where the change occurred") it is emitted as **small rasterized-polygon GeoTIFF
tiles**, not a point table. Each tile: driver class id inside the reprojected footprint,
`255` (nodata/ignore) elsewhere. Tile size = footprint bbox + 10 px context ring, clamped
to [32, 64]. Single-band uint8, local UTM at 10 m/px (nearest-touch rasterization,
`all_touched=True`). Outside-footprint is nodata (not a class) because surrounding pixels
are unlabeled; there is no background/no-change class in the driver scheme.

## Change handling

- `change_time` = event date (see above).
- `time_range` = 1-year window **centered** on `change_time` (±180 days = 360 days ≤ 1 yr).
- The 1-year window is appropriate: the pre/post-S2 driver signal is a persistent land
  change (clearing, mine, road, burn scar), observable across a full year around the event.

## Class mapping

Source `new_label` maps 1:1 onto the manifest's 10-class scheme (id : name):

| id | class | source count | selected |
|----|-------|-------------|----------|
| 0 | agriculture | 1110 | 1000 (capped) |
| 1 | mining | 266 | 266 |
| 2 | airstrip | 40 | 40 |
| 3 | road | 354 | 354 |
| 4 | logging | 183 | 183 |
| 5 | burned | 340 | 340 |
| 6 | landslide | 962 | 962 |
| 7 | hurricane | 402 | 402 |
| 8 | river | 355 | 355 |
| 9 | none | 485 | 485 |

**Dropped** ambiguous / free-text source labels (not part of the manifest taxonomy):
`unlabeled` (3041), `unknown` (283), `Natural - Unknown` (33), `General deforestation
(Clearing)` (15), `Anthropic - Unknown` (9), `natural` (5). These were dropped rather than
force-mapped to keep the class scheme clean and 1:1 with the manifest. `agriculture` was
truncated from 1110 to 1000 (per-class cap); all other classes kept in full. Total 4387,
well under the 25k cap.

## Verification

- 4387 `.tif` each with a matching `.json`; single-band uint8, EPSG:327xx/326xx UTM,
  10 m/px, ≤64×64. Pixel values observed ⊆ {0..9, 255}; metadata class ids cover them.
- Every sample `time_range` = 360 days with `change_time` set and inside the range.
- Georeferencing spot-check: sample `001068` source label reads "at -3.2908, -76.3476";
  computed tile-center lon/lat = (-76.3477, -3.2908) — exact match. All samples fall in
  the Amazon basin.
- Idempotent: re-running skips existing `.tif` (full re-run ~4 s).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_forest_loss_driver
```

## Caveats

- Tiles are mostly nodata with a small labeled footprint (change-mask semantics); this is
  expected for sparse dated events.
- Rare footprints exceed 64 px and are clipped to the centered 64×64 tile; a fallback sets
  the center pixel if a footprint is fully clipped out (none observed in practice).
- Driver classes with rich real-world variation (agriculture subtypes: rice / smallholder
  / Mennonite; coca) are collapsed into `agriculture` in the source `new_label` already.
