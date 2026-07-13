# Global Fishing Watch SAR Fixed Infrastructure

- **Slug**: `global_fishing_watch_sar_fixed_infrastructure`
- **Status**: completed
- **Task type**: classification (positive-only object **detection**, encoded as per-pixel classes)
- **Family / label_type**: infrastructure / points (object-detection encoding, spec §4)
- **Num samples**: 4000 (3000 positive detection tiles + 1000 background negatives)
- **License**: CC-BY-NC-4.0

## Source

Global Fishing Watch, from Paolo et al. 2024, *Nature*, "Satellite mapping reveals
extensive industrial activity at sea". We use the paper's public **analysis-data**
repository on figshare (<https://doi.org/10.6084/m9.figshare.24309475>), downloading only
the label file `offshore_infrastructure_v20231106.csv.zip` (11.4 MB). **No imagery is
downloaded** — the pretraining pipeline supplies its own Sentinel-1/-2/Landsat; only the
label coordinates are needed.

The CSV holds 1,441,242 detection-months of offshore fixed infrastructure (2017–2021),
detected on **monthly Sentinel-1 SAR** median composites and classified with deep learning
(SAR + optical). Fields: `structure_id` (unique per physical structure; its lon/lat is
constant across all its detection-months — verified std=0), `composite_date` (center of the
6-month composite), `lat`, `lon`, `label`.

This figshare release is the georeferenced public CSV; it is equivalent to (a static
snapshot of) the GFW Data Download Portal / Bulk Download API dataset
`public-fixed-infrastructure-data:v1.1`, which would otherwise require an API token. Using
the figshare CSV avoids any credential requirement.

## Access / reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_fishing_watch_sar_fixed_infrastructure
```

Idempotent (skips already-written `locations/{id}.tif`). Raw label CSV lands in
`raw/global_fishing_watch_sar_fixed_infrastructure/` with a `SOURCE.txt` pointer.

## Class mapping

Manifest three-class scheme + a `background` class for the detection encoding. GFW
confidence tiers are folded into the coarse manifest class:

| id | name | GFW source labels |
|----|------|-------------------|
| 0 | background | (open water — detection encoding fill / negatives) |
| 1 | oil | `oil`, `probable_oil`, `possible_oil`, `lake_maracaibo` |
| 2 | wind | `wind`, `probable_wind`, `possible_wind` |
| 3 | other | `unknown` (piers, bridges, power lines, aquaculture, etc.) |
| 255 | nodata/ignore | detection buffer rings; not used as a class |

`lake_maracaibo` is folded into `oil` (README: "likely an oil structure in Lake
Maracaibo"). Within each `structure_id`-year the coarse label is **100% consistent**
(verified), so there is no per-year label ambiguity.

## Encoding (detection, spec §4)

- One **32×32** UTM 10 m context tile per selected structure, centered on the structure.
- The structure is a **1 px positive** of its coarse class, ringed by a **10 px nodata (255)
  buffer** (21×21 ignore region — offshore detections are not pixel-exact), all other pixels
  **background (0)**.
- Every **other** structure detected in the **same calendar year** that falls inside a tile
  is also painted with its coarse class (structures cluster in oil fields / wind farms).
- **Negatives**: 1000 background-only open-water tiles obtained by offsetting a random real
  structure by 3–8 km in a random bearing and confirming (KD-tree) that no structure lies
  within ~1.1 km — real offshore open water in the same regions as the positives.

## Time-range & change handling (spec §5)

Fixed infrastructure is **persistent**, not a change event. Detection timing is only monthly
on 6-month composites (coarser than the ~1–2 month change-timing bar), so **no dated change
labels** are emitted. Each structure is treated as a persistent structure: a positive is
emitted for a structure only in a calendar year (2017–2021) in which it is detected
persistently across the **whole** year — **≥ 6 monthly detections spanning both the first
quarter (month ≤ 3) and the last quarter (month ≥ 10)** — guaranteeing the state is present
across the entire 1-year label window. `change_time = null`; `time_range` is that calendar
year. This mirrors the DeepOWT persistent-structure precedent. All labels are post-2016.

## Sampling

Up to **1000 tiles per positive class**, stratified across the 5 years (200/year) for
temporal diversity, plus up to **1000 negatives** (also year-stratified). Persistent
candidate structures available: oil ≈ 15,878, wind ≈ 9,631, other ≈ 2,652 — all well above
1000, so each positive class hit its 1000 cap.

Final counts: oil=1000, wind=1000, other=1000, background_negative=1000; **4000 total**,
800 per year. `other/unknown` is comparatively sparse in the source but easily reaches 1000
here; per spec §5 no class is dropped, and downstream assembly filters any too-small classes.

## Verification (spec §9)

- 4000 `.tif` each with a matching `.json`; all **32×32, single-band uint8**, projected UTM
  at **10 m** (positive/negative resolution 10/−10), nodata=255. Pixel values ⊆ {0,1,2,3,255}.
- All `time_range`s are exactly 1 year, all post-2016; no `change_time` set; years balanced
  (800 each 2017–2021). `metadata.json` class ids {0,1,2,3} cover all raster values.
- **Georeferencing round-trip**: for sampled positive tiles the positive pixel re-projects
  back to the true GFW structure lon/lat within **1–5 m** (sub-pixel), on real offshore oil
  (Gulf of Thailand, Lake Maracaibo), wind (North Sea), and other (Indonesia) sites.

## Caveats

- Coordinates are GFW model detections (>98% classification accuracy in the paper), not
  in-situ surveys; low-confidence tiers (`possible_*`) are included but the persistence rule
  (present most of a year at a stable `structure_id`) strongly filters transient noise.
- The figshare snapshot covers 2017–2021 (a 2022-01 composite tail exists but is excluded by
  the whole-year persistence rule). More recent detections are available via the GFW Bulk
  Download API (token required) if a refresh is ever wanted.
- Detection footprints are 1 px (platforms/turbines are ~15–50 m objects at 10 m); the 10 px
  ignore buffer absorbs localization error.
