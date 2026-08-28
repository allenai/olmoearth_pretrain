# OlmoEarth vessel attributes (type)

- **Slug:** `olmoearth_vessel_attributes_type`
- **Status:** completed · classification (presence-only points, multi-class by vessel type) ·
  **8,000 points**
- **Source:** OlmoEarth (internal) · **License:** internal
- **Annotation method:** AIS-matched vessel attributes.

## Source & access

Local rslearn dataset (`have_locally=true`, not copied — see `raw/{slug}/SOURCE.txt`):
`/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20250205`. The
OlmoEarth vessel-attribute eval, 584,432 windows total; each window is a 128×128 per-vessel
Sentinel-2 crop (local UTM @ 10 m) centered on one AIS-matched vessel, with a `~2-hour` S2
acquisition `time_range` and a label layer `info` holding one Point whose `properties.type` is
the vessel category (9 eval categories; unknown/other omitted → skipped).

## Label type — presence-only points

**Converted from the old per-vessel object-detection tile encoding** (32×32 tiles). Now emitted
as **presence-only points** in a dataset-wide `points.geojson` (spec §2a): each source crop
yields exactly one typed presence point at the labeled vessel's lon/lat (converted from its pixel
coordinate in the window's UTM projection). There is **no fabricated GeoTIFF context, and no
background / buffer / negative tiles** — because neighboring vessels in a crop are unlabeled, the
crop background is not a genuine negative, so it is not emitted. Negatives are supplied downstream
by the assembly step; this dataset carries **no fabricated negatives**.

## Classes / counts

Vessel-type classes only, ids 0–8. Class-balanced up to 1000 points/type → **8,000 points**:

| id | name | pts | id | name | pts |
|----|------|-----|----|------|-----|
| 0 | cargo | 1000 | 5 | pleasure | 1000 |
| 1 | tanker | 1000 | 6 | fishing | 1000 |
| 2 | passenger | 1000 | 7 | enforcement | 1000 |
| 3 | service | 1000 | 8 | sar | 1000 |
| 4 | tug | 0 | | | |

`tug` is in the class map (id 4) but no vessel in this release maps to it (0 points); kept per
spec §5 (empty/sparse classes retained). Length/width/course/speed attributes are regression and
excluded per the manifest.

## Time handling

Each point uses its **source window's own ~2-hour S2 acquisition `time_range`** (specific-image,
spec §5). No `change_time`.

## Output

- `datasets/olmoearth_vessel_attributes_type/points.geojson`
- `datasets/olmoearth_vessel_attributes_type/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_vessel_attributes_type --workers 64
```

Idempotent (deterministic selection).

## Caveats

- `unknown`/`other` vessels (~26% of windows) skipped — not valid type labels.
