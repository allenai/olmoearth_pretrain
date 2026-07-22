# CoastBench

- **Slug:** `coastbench`
- **Status:** completed
- **Task type:** classification (sparse coastal-transect points → GeoJSON point table, spec §2a)
- **Num samples:** 1763
- **Source:** CoastBench — Deltares / TU Delft. Zenodo record
  [15800285](https://zenodo.org/records/15800285), CC-BY-4.0.

## What the source is

~1,763 expert-labeled coastal transects, globally distributed, anchored to the Global
Coastal Transect System (GCTS) grid. Each transect is an expert web-labeling of coastal
character along a cross-shore profile, with a WGS84 lon/lat origin (`lon`/`lat` columns).
Labels were created Aug 2024 – Apr 2025. Each transect carries several attribute
dimensions: `coastal_type` (landform/type), `shore_type` (sediment/shore material),
`is_built_environment` (bool), `has_defense` (bool), plus a `confidence` field and various
geometry/metadata columns.

## Access method

The Zenodo release is a single 16.6 GB zip (`coastbench-release-2025-04-09.zip`) containing
24,850 entries — mostly 10 m imagery tiles (`imgs/…`) and model checkpoints (`models/…`).
Only the labels are needed. The labels live in one **`labels.parquet`** (~340 KB) at the
zip root. The script selectively extracts *just that member* via HTTP range requests
(`download.HttpRangeFile` + `zipfile`), fetching ~3 MB total instead of the 16.6 GB
archive. `raw/coastbench/labels.parquet` + `SOURCE.txt` are written to weka.

## Class mapping (primary target: `coastal_type`)

The dataset is multi-attribute; the **primary per-point class is `coastal_type`** (the
coastal landform/type classification), which directly covers the manifest classes (cliffed
coast, sediment plain, dune, wetland, …). Class ids are assigned by descending frequency:

| id | name | count |
|----|------|-------|
| 0 | sediment_plain | 366 |
| 1 | cliffed_or_steep | 351 |
| 2 | moderately_sloped | 222 |
| 3 | engineered_structures | 167 |
| 4 | bedrock_plain | 159 |
| 5 | dune | 157 |
| 6 | wetland | 156 |
| 7 | inlet | 154 |
| 8 | coral | 31 |

All 1,763 transects have a valid `coastal_type` and non-null lon/lat, so all are kept. Max
class (366) is below the 1000/class cap → no truncation; total (1,763) well under the 25k cap.

### Secondary attributes (auxiliary, carried per-point, not the primary class)

Per spec §5 multi-target guidance, the several attribute dimensions describe the *same*
point, so rather than splitting into separate datasets we keep one dataset with
`coastal_type` as the primary `label` and attach the other dimensions as auxiliary
properties on each `points.geojson` feature (documented in `metadata.json.auxiliary_attributes`):

- `shore_type` — sediment/shore material class (sandy_gravel_or_small_boulder_sediments,
  no_sediment_or_shore_platform, rocky_shore_platform_or_large_boulders, muddy_sediments,
  ice_or_tundra).
- `has_defense` — human-made coastal defense present (bool; 486 true / 1277 false).
- `is_built_environment` — built environment present (bool; 715 true / 1048 false).
- `confidence` — annotator confidence (medium 1670 / high 74 / low 19).

`landform_type` (mostly N/A) was not carried — too sparse (1,582 of 1,763 are N/A).

## Time-range & change handling

Coastal type is quasi-static. Per §5 (static labels), each point gets a representative
1-year Sentinel-era window **2023-01-01 → 2024-01-01** and `change_time=null`. The
`datetime_created`/`datetime_updated` fields are annotation timestamps (2024–2025), not
observation dates, so they are not used for the window. All samples are effectively
post-2016.

## Verification

- `points.geojson`: FeatureCollection, count=1763, all label ids in 0–8, all coordinates
  within valid lon/lat bounds. Each feature has a ≤1-year `time_range` and `change_time=null`.
- `metadata.json`: 9 classes with descriptions; class_counts cover every label present.
- **Georeferencing sanity check:** spot-checked lon/lat against the source `country`/
  `common_region_name` columns for a global spread — all consistent (e.g. -135.5,68.7 →
  CA Northwest Territories; -7.85,43.7 → ES Galicia; 174.66,-36.8 → NZ Auckland). The
  `lon`/`lat` columns also match the transect geometry (WKB) origin to ~a pixel.
- Idempotent: re-running skips the already-extracted parquet and re-derives identical
  outputs (deterministic seeded balancing).

## Caveats

- Sparse point labels (no negatives) — downstream assembly supplies negatives (§5).
- `coral` class is sparse (31); kept per §5 (downstream filtering removes too-small classes).
- Labels are point/transect origins; the full cross-shore profile geometry is not encoded
  (a single 10 m pixel per transect on the S2 grid, spec §2a).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coastbench
```
