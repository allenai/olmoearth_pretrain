# GLanCE Global Land Cover Training Data

- **Slug:** `glance_global_land_cover_training_data`
- **Status:** completed
- **Task type:** classification (sparse points → `points.geojson`, spec §2a)
- **Num samples:** 6041
- **Source:** Boston University GLanCE project, via Source Cooperative repo
  `boston-university/bu-glance` (`https://source.coop/boston-university/bu-glance`).
- **License:** CC-BY-4.0.
- **Reference:** Stanimirova et al. (2023), "A global land cover training dataset from 1984
  to 2020", *Scientific Data* 10, 879. DOI 10.1038/s41597-023-02798-5.

## What the source is

~1,874,995 globally distributed 30 m training units, each labeled by **manual on-screen
photointerpretation** for the NASA MEaSUREs GLanCE land-cover product. Distributed as a
GeoParquet (73 MB) and a GeoJSON (1.1 GB); we used the parquet
(`bu_glance_training_dataV1.parquet`) — a single columnar read, no per-file weka I/O.

Each row has `Lat`/`Lon` (WGS84), a time segment `[Start_Year, End_Year]` (1984–2020), a
GLanCE **Level-1** land-cover class (1–7), plus attributes (`Change`, `Segment_Type`,
`LC_Confidence`, ecoregion/continent codes, Level-2 class, etc.). This is the manually
photointerpreted **reference** the manifest flags as "prefer over the map".

## Access method

Public unsigned S3 via the Source Cooperative data proxy
(`download.download_s3_unsigned("boston-university", "bu-glance/bu_glance_training_dataV1.parquet", ..., endpoint_url="https://data.source.coop")`).
No credentials required.

## Class mapping (GLanCE Level-1 → our ids)

| our id | name | GLanCE L1 | selected count |
|---|---|---|---|
| 0 | Water | 1 | 1000 |
| 1 | Ice/Snow | 2 | 41 |
| 2 | Developed | 3 | 1000 |
| 3 | Barren | 4 | 1000 |
| 4 | Trees | 5 | 1000 |
| 5 | Shrub | 6 | 1000 |
| 6 | Herbaceous | 7 | 1000 |

Ids/order match the manifest class list. Per-class descriptions (from the README legend)
are stored in `metadata.json`. **Ice/Snow is sparse** (only 41 stable post-2016 points in
the whole global table) — kept per spec §5 (rare classes are not dropped; downstream
assembly filters too-small classes).

## Time-range and change handling (spec §5)

- **Post-2016 rule:** kept only records whose segment reaches the Sentinel era
  (`End_Year >= 2016`): 1,280,452 of the stable rows. Each label is a stable land-cover
  **state** over its multi-year segment, so we assign a **1-year window uniformly sampled**
  from the post-2016 span `[max(Start_Year, 2016), min(End_Year, 2020)]`, deterministic
  (seeded per `Glance_ID`). Selected-sample window-start years span 2016–2020.
- **Change labels dropped.** Rows with `Change==1` (388,527) denote land-cover change
  somewhere inside a multi-year segment; the change date is **not resolvable to within
  ~1–2 months**, so per spec §5 they are not usable as change labels and are excluded. We
  keep only stable segments (`Change==0`, 1,486,468 rows), whose recorded class holds
  across the whole segment — safe for any in-segment 1-year window. No `change_time` is set.

## Sampling

`sampling.balance_by_class(records, "label", per_class=1000)` (25k total cap). Result:
6041 points = 1000 each for the six common classes + 41 for Ice/Snow. Well under the 25k
per-dataset and 254-class caps.

## Relationship to `olmoearth_glance_land_cover`

Not a duplicate. `olmoearth_glance_land_cover` is the local OlmoEarth **derived-product**
eval subset with a distinct 11-class OlmoEarth legend. This dataset is the **upstream
public manual reference** (full V1 release, ~1.9M points, GLanCE Level-1 7-class legend)
that the manifest prefers over the map. Different releases, different legends, processed
independently; geographic overlap is expected but the class schemes and provenance differ.

## Verification (spec §9)

- `points.geojson`: 6041 `Point` features, WGS84 coords spanning the globe
  (lon −160.4…175.5, lat −51.9…79.1); `task_type=classification`, `count=6041`.
- All `time_range`s are ≤1 year and start ≥2016; all `change_time` are null.
- Labels 0–6 present and fully covered by `metadata.json` `classes`.
- Coordinates are source-native WGS84 from the authoritative manual-reference table, so
  georeferencing is exact by construction (no reprojection applied); a live Sentinel-2
  overlay was therefore not required.
- Idempotent: re-running skips the raw download and regenerates the deterministic outputs.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glance_global_land_cover_training_data
```

## Caveats

- Only Level-1 (7-class) labels used; the richer Level-2 (13-class) and modifier
  attributes are available in the source but not exported here.
- Change/transitional segments excluded (timing not resolvable); if a future need arises,
  the persistent post-change state could potentially be recast as a static presence label
  per §5, but that was not done here.
