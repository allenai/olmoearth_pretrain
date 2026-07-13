# DeepOWT (Global Offshore Wind Turbines)

- **Slug:** `deepowt_global_offshore_wind_turbines`
- **Manifest name:** DeepOWT (Global Offshore Wind Turbines)
- **Task type:** classification (positive-only object **detection**, encoded as per-pixel classes)
- **Status:** completed — **2561 samples**
- **Family:** wind · **Region:** global offshore · **License:** CC-BY-4.0

## Source

DeepOWT, Zhang et al., *Earth System Science Data* (ESSD), published on Zenodo
(https://doi.org/10.5281/zenodo.5933967, CC-BY-4.0). A global inventory of offshore
wind-energy infrastructure with per-quarter deployment status derived from Sentinel-1
time series using deep learning, with validation against reference data.

**Access:** direct unauthenticated HTTP download of the main file
`DeepOWT.geojson` (4.1 MB) from Zenodo — no credentials. Saved to
`raw/deepowt_global_offshore_wind_turbines/DeepOWT.geojson` (+ `SOURCE.txt`). The ground-truth
`gt_*` GeoJSONs (NSB/ECS regional validation polygons) were not needed — the main file already
carries the global point inventory with full temporal status.

## Data structure

`DeepOWT.geojson` is a WGS84 (CRS84) `FeatureCollection` of **9,941 Point** features. Each
feature has **20 quarterly status columns** `Y2016Q3 … Y2021Q2`, each valued with DeepOWT's
semantic class:

| status | meaning |
|--------|---------|
| 0 | open sea |
| 1 | under construction |
| 2 | offshore wind turbine |
| 3 | offshore wind farm substation |

Observed point counts: 8,885 points are ever a turbine, 5,415 ever under construction,
204 ever a substation, 5,495 ever open sea (points transition through these states over time).

## Class scheme (kept = DeepOWT native ids)

| id | name | notes |
|----|------|-------|
| 0 | background | open sea / open water (DeepOWT status 0) |
| 1 | under_construction | foundation/platform present, turbine not yet operational |
| 2 | offshore_turbine | installed operational turbine |
| 3 | substation | offshore substation / transformer platform |
| 255 | nodata/ignore | detection buffer rings; ambiguous (transitioning) in-tile neighbors |

DeepOWT status 0 = open sea maps directly onto the detection **background** class, so the
class ids were kept unchanged rather than remapped. Turbines are small (~monopile + rotor)
but resolvable at 10 m against open water — DeepOWT itself is derived at Sentinel-1 10 m.

## Time / change handling (spec §5) — key judgment call

DeepOWT resolves each structure's appearance/state only to a **quarter (~3 months)**, which
is **coarser** than the §5 change-timing requirement (~1–2 months). Per the task instruction,
this is therefore **not** encoded as a dated change label. Instead each structure is treated
as a **persistent structure**:

- A positive for class `c` is emitted for a point **only in a full calendar year (2017–2020)
  in which all four quarters equal `c`**, guaranteeing the state is genuinely persistent across
  the entire 1-year label window (satisfying §5's persistent-state condition).
- `change_time = null`; `time_range` = that calendar year (`io.year_range`).
- 2016 (Q3–Q4 only) and 2021 (Q1–Q2 only) are partial and excluded from the all-4-quarters
  rule — everything used is 2017–2020, comfortably in the Sentinel era.

This contrasts with a dated-change encoding: we know *that* a turbine exists in a year, not the
month it appeared, so we assert presence over a static window rather than centering a window on
an install event.

## Detection encoding (spec §4)

Shared `sampling.encode_detection_tile` (same as the vessel and Global Renewables Watch
precedents):
- 32×32 (`DET_TILE`) UTM 10 m context tile per selected structure, centered on its point.
- 1 px positive (`positive_size=1`) of the class id, ringed by a **10 px nodata buffer**
  (`buffer_size=10` → 21×21 ignore region — turbine coordinates are not pixel-exact; the
  round-trip check below shows ~5–15 m / up-to-~1 px offset), rest background.
- **Every other DeepOWT point inside the tile** is also encoded by its status *in the same
  year*: 1/2/3 → that positive class; status 0 (open sea) → left as background; transitioning
  that year (not all-four-equal → ambiguous) → a **255 nodata** marker so an ambiguous neighbor
  is ignored, never a false label. (Offshore farms are dense; a 320 m tile can contain
  several structures.)
- **Negatives:** background tiles centered on points that are open sea for a full calendar year
  — real, geolocated open-water sites (many are future turbine locations), with any in-tile
  structures still encoded correctly. These are the detection exception in §5 (spatially
  meaningful in-tile negatives).

## Sampling

Up to **1000 tiles per positive class**, stratified across calendar years (250/year) for
temporal diversity via `sampling.balance_by_class(key="year")`, one tile per physical point
per class (random stable year chosen). Plus up to **1000 background negatives** (250/year).
Well under the 25k per-dataset cap.

**Final counts (2561 samples):**

| class | tiles |
|-------|-------|
| offshore_turbine (2) | 1000 |
| under_construction (1) | 448 |
| substation (3) | 167 |
| background negatives (0) | 946 |

`under_construction` and `substation` are **sparse** — construction rarely persists a full
calendar year (only 122–265 points/year stable) and substations are rare (204 points total).
Per §5, sparse classes are **kept in full** (not dropped); downstream assembly filters
too-small classes. (Every tile also contains background(0); turbine(2) appears in 1001 tiles
because one negative/other tile picked up a neighboring turbine.)

## Verification (spec §9)

- 2561 `.tif` each with a matching `.json`; all single-band **uint8**, **32×32**, local UTM
  (EPSG:326xx), **10 m** resolution.
- Pixel values ∈ {0,1,2,3,255}; metadata `classes` (ids 0–3) + nodata 255 cover all values.
- Turbine tile center pixel = 2 (positive laid correctly); ambiguous neighbors → 255.
- All `time_range` = 1 year, `change_time` = null.
- **Georeferencing round-trip:** reconstructed tile-center lon/lat vs source point differ by
  ~5–15 m (sub-pixel to ~1 px) — consistent with pixel-quantization and exactly why the 10 px
  buffer exists.
- A live Sentinel-2 overlay was **not** run (would require standing up an imagery data source
  disproportionately); geolocation is validated by the round-trip, DeepOWT is a peer-reviewed
  validated product, and all coordinates fall in known offshore wind basins (Yellow/East China
  Sea, Baltic, North Sea, off Vietnam).
- Idempotent: `_write_tile` skips any already-written `{sample_id}.tif`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.deepowt_global_offshore_wind_turbines --workers 64
```

## Caveats

- Derived product (DL on Sentinel-1) with validation, not in-situ reference — acceptable per
  the manifest (no reference alternative for global offshore turbines).
- Quarterly timing forces the persistent-structure model (no dated-change labels).
- `under_construction` / `substation` are sparse and may be dropped downstream by the
  min-count filter.
- Precedents: encoding mirrors `global_renewables_watch` (wind turbine points) and
  `olmoearth_sentinel_2_vessels` (positive-only marine detection).
```
