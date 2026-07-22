# OlmoEarth GLanCE land cover

- **Slug:** `olmoearth_glance_land_cover`
- **Task type:** classification (sparse point segmentation)
- **Status:** completed
- **Num samples:** 10,591 points (across 11 classes)

## Source

Local rslearn eval dataset at
`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/glance` (`have_locally=true`;
`raw/olmoearth_glance_land_cover/SOURCE.txt` points at it, nothing copied).

Each window is a 32×32 UTM (10 m) context tile whose **single center pixel** carries one
land-cover class. The class is stored three consistent ways: `options.label` (name) in the
window `metadata.json`, a single `Point` feature in the `label` vector layer, and a 32×32
`label_raster` GeoTIFF with one valued pixel (the class id) and the rest = 255 nodata.
Verified on 300 random windows + 6 selected points that `options.label` name, the manifest
class order, and the `label_raster` pixel value all agree (0 mismatches). So this is a
**pure sparse-point** dataset → written as one dataset-wide `points.geojson` (spec §2a),
not per-point GeoTIFFs.

Windows: 34,885 total (`train` 3,300 + `test` 31,585); **all splits used** as pretraining
labels. All labels fall in **2017-2020** (fully Sentinel-era; the per-window `time_range`
is already a ~1-year window and is preserved verbatim in the output).

## Class mapping (manifest order = class id = source raster value)

| id | name | count |
|----|------|-------|
| 0 | water | 1000 |
| 1 | evergreen | 1000 |
| 2 | deciduous | 1000 |
| 3 | agriculture | 1000 |
| 4 | grassland | 1000 |
| 5 | mixed | 1000 |
| 6 | developed | 1000 |
| 7 | sand | 1000 |
| 8 | shrub | 1000 |
| 9 | rock | 1000 |
| 10 | soil | 591 |

Balanced to ≤1000 per class (`balance_by_class`, per_class=1000, 25k cap not binding).
Ten classes hit the 1000 cap; `soil` is the only under-target class (591 = all available)
and is kept in full per §5 (no dropping sparse classes). All 11 manifest classes present.

## Time range / change handling

Seasonal/annual land-cover labels → 1-year window per point, taken directly from the source
window `time_range` (anchored on the labeled year, 2017-2020). No change labels.

## Judgment calls / caveats

- **Map vs. reference pairing.** The manifest also lists the upstream manual reference,
  "GLanCE Global Land Cover Training Data" (`have_locally=false`, Source Cooperative,
  CC-BY-4.0, standard **7-class** GLanCE Level-1 legend), noted as "prefer over the map."
  This local eval is the **derived/map product** but uses a **different, finer 11-class
  OlmoEarth legend** (water / evergreen / deciduous / agriculture / grassland / mixed /
  developed / sand / shrub / rock / soil), so it is a distinct product, not a redundant
  copy of that reference. It is `have_locally=true` and was explicitly assigned for
  processing; I processed it and record the pairing here. The 7-class manual reference
  remains a separate external dataset. (This mirrors the precedent of
  `olmoearth_lcmap_land_use`, also a derived-product local that was processed.)
- `annotation_method` recorded as "derived-product (GLanCE land cover)"; homogeneity
  filtering (§4 dense-raster note) is not applicable here since labels are single
  photointerpreted/derived points, not sampled from a wall-to-wall map.
- Point label ids equal the manifest class order (0-10); confirmed against source
  `label_raster` values.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_glance_land_cover
```
Idempotent: rewrites `points.geojson` / `metadata.json` / `registry_entry.json` from the
local source each run.
