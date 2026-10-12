# Geo-Wiki Global 10 m Land Cover Reference (2015)

- **Slug:** `geo_wiki_global_10_m_land_cover_reference_2015`
- **Task type:** classification (sparse point segmentation)
- **Family / region:** land_cover / Global
- **Source:** Zenodo / ESSD record **14871659** — "Global land cover data set at 10m for 2015 (Geo-Wiki)"
- **URL:** https://doi.org/10.5281/zenodo.14871659
- **License:** CC-BY-4.0
- **Annotation method:** manual photointerpretation (Geo-Wiki expert/crowd). This is the
  reference set behind CGLS-LC100 and ESA WorldCover.

## Source data

Single CSV, `final_reference_data.csv` (~1.97 GB, downloaded via `download.download_zenodo`
to `raw/.../`). It contains **16,569,600 rows** over **165,696 unique sample locations**
(~100 interpreted 10 m points per location). Relevant columns:

- `class_id` / `class_name` — land-cover class of the interpreted 10 m pixel.
- `center_x`, `center_y` — WGS84 lon/lat of the pixel center.
- `reference_year` — the reference year (all rows are **2015**).

Each row is one individually interpreted 10 m pixel carrying its own class, so the data
maps directly onto the sparse-point recipe: one **1×1** uint8 label patch per point.

## Class mapping (derived from the data)

Source `class_name` values and their global row counts:

| id | name | source class_id | source row count |
|----|------|-----------------|------------------|
| 0 | tree | 3024 | 4,081,740 |
| 1 | shrub | 3025 | 2,100,394 |
| 2 | grassland | 3026 | 5,094,973 |
| 3 | crops | 3027 | 1,847,882 |
| 4 | urban/built-up | 3028 | 170,003 |
| 5 | bare | 3029 | 1,142,880 |
| 6 | burnt | 3030 | 12,644 |
| 7 | water | 3031 | 578,826 |
| 8 | snow and ice | 3032 | 45,112 |
| 9 | fallow/shifting cultivation | 3033 | 95,141 |
| 10 | wetland (herbaceous) | 3471 | 643,586 |
| 11 | Lichen and moss | 4074 | 78,680 |

These 12 classes match the manifest list exactly (trees→tree, herbaceous wetland→wetland
(herbaceous), cultivated/crops→crops, snow & ice→snow and ice, lichen & moss→Lichen and
moss). A 13th source value, **"Not sure" (class_id 3034, 677,739 rows)**, is an uncertainty
label rather than a land-cover class and is **dropped** (treated as ignore).

Class IDs are assigned 0–11 in ascending source `class_id` order. Value 255 = nodata/ignore
(never emitted in these 1×1 patches).

## Sampling

- Up to **1000 points per class**, balanced with `sampling.balance_by_class`. Every class
  has ≥5000 source points, so all 12 reach the full 1000 → **12,000 samples total**.
- Selection is reproducible: the CSV is read once, each class is shuffled (seed 42) and
  capped at a 5000-point pre-sample, then `balance_by_class` (seed 42) trims to 1000.
- All Geo-Wiki records are used as candidates (no source train/val/test split to filter).

## GeoTIFF spec

Single-band uint8, local UTM at 10 m/pixel, north-up, **1×1** pixel. Projection chosen per
point via `io.lonlat_to_utm_pixel`; pixel value = class id.

## Time range

Reference year is **2015**, but global Sentinel-2 coverage in 2015 is sparse (S2A launched
mid-2015; S2B in 2017). Per task guidance the 1-year window is anchored on **2016** (first
full Sentinel-2 year: `time_range = [2016-01-01, 2017-01-01)`) so labeled points can be
co-located with imagery. Land cover at these classes is stable enough year-to-year that the
2016 window is appropriate. No `change_time` (not a change dataset).

## Outputs

- `raw/geo_wiki_global_10_m_land_cover_reference_2015/final_reference_data.csv`
- `datasets/geo_wiki_global_10_m_land_cover_reference_2015/metadata.json`
- `datasets/geo_wiki_global_10_m_land_cover_reference_2015/locations/{000000..011999}.tif` + `.json`

Per-class counts: 1000 each for all 12 classes (12,000 total).

## Caveats

- Labels are single 10 m pixels; a 1×1 patch has no spatial context by design (sparse
  point segmentation).
- Points within a source location can be tens of meters apart; independent sampling per
  class means a few selected points may be spatial neighbors, but this is rare given the
  global spread and per-class caps.
- "Not sure" points excluded (see above).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.geo_wiki_global_10_m_land_cover_reference_2015
```

Idempotent: existing `{sample_id}.tif` outputs are skipped on re-run.
