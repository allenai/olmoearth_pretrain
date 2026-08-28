# FORWIND (European forest wind disturbance) — `forwind_european_forest_wind_disturbance`

**Status:** completed · **task_type:** classification (windthrow presence segmentation,
single foreground class) · **num_samples:** 15,859

## Source

FORWIND — *"A spatially-explicit database of wind disturbances in European forests over the
period 2000–2018"* (Forzieri et al., Earth System Science Data, 2020). figshare record
v2, CC-BY-4.0.

- Landing page (DOI): `https://doi.org/10.6084/m9.figshare.9555008`
- **Access used:** open HTTP from figshare's `ndownloader` (no credentials). A Firefox
  User-Agent was used to avoid CDN 403s. Downloaded the single shapefile
  `FORWIND_v2.{shp,shx,dbf,prj}` (+ `readme.txt`) to
  `raw/forwind_european_forest_wind_disturbance/`.

The shapefile holds **89,743** polygons (geometry in WGS84 / EPSG:4326), each a
spatially-delineated forest area disturbed by wind. Attributes: `Id_poly`, `EventDate`,
`StormName`, `EventType`, `Country`, `Area` [ha], `Perimeter` [m], `Damage_deg` [fraction
0–1], `Methods`, `Dataprovid`, `Source` (missing = −999). Annotation: aerial/satellite
photointerpretation + field survey.

## Label design

**Windthrow presence segmentation** (uint8), positive-only:
- `0` = wind_disturbance (inside a FORWIND wind-damaged forest polygon)
- `255` = nodata/ignore (everything outside the polygon)

**Single foreground class** rather than damage-degree classes. `Damage_deg` is a
continuous, **stand-level (per-polygon)** value that is (a) present for only ~57% of the
post-2016 polygons (7,386 of 13,068) and (b) **constant within each polygon**, so it does
not create meaningful within-tile class structure. It is preserved as per-sample
provenance in `source_id` (e.g. `Id_poly=87401:Vaia:2018-10-28:IT:dmg=0.9500`), analogous
to how `cal_fire` keeps per-fire CAUSE as metadata rather than a per-pixel class.

**Background is nodata, not class 0.** FORWIND is a *compilation of mapped disturbances*,
not an exhaustive damage/no-damage map of every forest, so out-of-polygon pixels are
"unmapped", not authoritatively undamaged. Per spec §5 positive-only handling, no synthetic
negatives are fabricated; downstream assembly supplies negatives from other datasets. (This
differs from `cal_fire`, where a fire perimeter authoritatively delimits burned vs unburned
and background=0 is justified.)

## Change semantics (this is a change/event dataset)

Each polygon is dated to a named windstorm event. **All post-2016 FORWIND `EventDate`s are
day-precision**, well within spec §5's ~1–2 month change-timing requirement:

| Storm | Date | polygons |
|-------|------|----------|
| Vaia | 2018-10-28 / 2018-10-29 | 7,416 |
| Friederike | 2018-01-18 | 2,990 |
| Xavier | 2017-11-10 | 2,073 |
| unknown-name (still day-dated) | 2017-08-17 / 2017-08-02 / 2017-11-10 | 589 |

Each sample carries `change_time = EventDate`, which splits it into two adjacent six-month
windows (via `io.pre_post_time_ranges`): **`pre_time_range`** = the ~6 months (≤183 days)
immediately before the storm and **`post_time_range`** = the ~6 months (≤183 days)
immediately after, with **`time_range` = null** (total span still ~1 year). Pretraining pairs
the "before" image stack with the "after" stack and probes on their difference (forest before
vs blowdown after). Metadata flags `is_change_dataset: true`. The blowdown gap is also a
persistent post-event state, but because exact dates are available we use the change-label
scheme (change_time set) rather than a static presence label.

**Pre-2016 filtering (spec §8):** only records with `EventDate` year ≥ 2016 are used
(13,068 of 89,743). FORWIND's 2000–2015 back-catalogue (incl. a few malformed dates) is
filtered out. Note: FORWIND has no 2016 events, so realized samples are 2017 + 2018 only.

## Tiling & sampling

- Each polygon reprojected to a local **UTM projection at 10 m/pixel** (data lands in
  EPSG:32632/32633/32634 — central Europe).
- **Small polygon** (footprint ≤ 64×64 px): **one tile tightly framed on the polygon's
  bounding box** (variable size down to a few px — spec §2 "size down to the label's real
  footprint"), giving dense foreground.
- **Large polygon** (>64 px either axis): gridded into non-overlapping 64×64 windows;
  windows intersecting the polygon are kept, up to `MAX_TILES_PER_POLY = 40` sampled.
- Inside polygon → 0, outside → 255 (`rasterize_shapes`, `all_touched=True` so thin/small
  polygons still register at 10 m).
- **Selection:** round-robin across polygons (every polygon contributes ≥1 tile) capped at
  `MAX_SAMPLES_PER_DATASET = 25,000`. The candidate pool came in under the cap, so all
  13,068 polygons are represented → **15,859 tiles**.

**Counts:** 15,859 tiles, all containing foreground (class 0). Per year: 2017 = 3,934;
2018 = 11,925. (Single class, so no per-class balancing needed; positive-only.)

## Verification (§9)

- 6 sampled tifs + 500-tif scan: single band, uint8, UTM at 10 m
  (EPSG:32632/32633/32634), all sizes ≤ 64×64 (mix of tight small tiles e.g. 5×6, 9×11,
  and full 64×64 grid tiles), pixel values ⊆ {0, 255}, nodata = 255. 0 tifs with
  out-of-range values. Foreground fraction ≈ 35% over sampled tiles. ✓
- All 15,859 `.tif` have a matching `.json` (0 unmatched either way). ✓
- 2,000 sampled JSONs: `change_time` always set; each carries adjacent ≤183-day
  `pre_time_range`/`post_time_range` windows split exactly at `change_time` with
  `time_range` = null. ✓
- metadata.json: `task_type=classification`, `num_samples=15859`, `nodata_value=255`,
  classes = [(0, wind_disturbance)], `is_change_dataset=true`, license CC-BY-4.0. All tif
  values (0, 255) are covered by the class map + nodata. ✓
- Geographic sanity: 300 random tile centroids all fall in a Europe box (lon 8.6–17.9°E,
  lat 45.9–54.4°N) — consistent with Vaia (Italian Alps), Friederike & Xavier (Germany/
  Poland); 0 outliers. A full Sentinel-2 image overlay was not run (heavy/out-of-band);
  georeferencing is exact because tiles are written via `GeotiffRasterFormat` in the same
  UTM projection the polygon was rasterized in. ✓
- Idempotent: existing `locations/{id}.tif` are skipped on re-run.

## Judgment calls / caveats

- Damage_deg dropped as a class (continuous, per-polygon, ~43% missing, not per-pixel
  meaningful) → single foreground class; damage kept in `source_id`.
- Positive-only with nodata background (FORWIND is not an exhaustive no-damage map).
- `EventType` for the post-2016 subset is uniformly "Windstorm" (the pre-2016 tornado
  records are filtered out), so no event-type sub-classing was warranted.
- The `post_time_range` already gives ~6 months of strictly post-event imagery (and
  `pre_time_range` the ~6 months before), so the before/after split needed for change
  probing is built in.
- Only 2017–2018 realized (FORWIND ends at 2018 and has no 2016 events).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.forwind_european_forest_wind_disturbance
```
Idempotent: existing `locations/{id}.tif` are skipped. Raw shapefile cached at
`raw/forwind_european_forest_wind_disturbance/FORWIND_v2.shp` (+ sidecars).
