# Cal-FF (California CAFOs)

- **Slug**: `cal_ff_california_cafos`
- **Status**: completed
- **Task type**: classification (positive-only, animal-type polygon segmentation)
- **Num samples**: 1543 (one 64×64 tile per selected facility)
- **Source**: Hugging Face `reglab/cal-ff` (accepted at *Nature Scientific Data*, 2025), <https://huggingface.co/datasets/reglab/cal-ff>
- **License**: CC0-1.0 (public domain dedication)

## What the source is

Cal-FF is a human-validated, near-complete census of **Concentrated Animal Feeding
Operations (CAFOs) in California**, compiled with satellite imagery + a computer-vision
detector + human-in-the-loop validation (Magesh, Rothbacher, Comess, Maneri, Rodolfa,
Tartof, Casey, Nachman, Ho). The label file `facilities.geojson` holds **2,121 facilities**,
each a **MultiPolygon building/pen footprint** in WGS84 (lon/lat) with:

- `animal_types` — list of animal-type tags (`cattle`, `dairy`, `poultry`, `swine`, `sheep`,
  `goat(s)`, `unknown`);
- construction/destruction date annotations (`construction_upper`, `destruction_upper`, …);
- parcel, permit, census-block, and building-count metadata.

Facility addresses are redacted for privacy, but the geospatial footprints and animal-type
labels — the parts we need — are fully public.

## Access method (label-only)

Downloaded only the label file from the public HF dataset repo (no imagery pulled —
pretraining supplies imagery) via `download.hf_download("reglab/cal-ff",
"facilities.geojson", raw_dir)`. Public CC0 repo; no credentials/gating (the manifest's
"needs-credential if gated" caveat does not apply). Raw provenance recorded in
`raw/cal_ff_california_cafos/SOURCE.txt`.

## Triage decision — accept

- **Georeferencing exact**: WGS84 MultiPolygons, so footprints place cleanly on the S2 grid
  (checked cheaply before any bulk work — a single ~small geojson, no multi-GB archives).
- **Observable at 10 m**: CAFO footprints are large, high-contrast built structures — median
  bounding box ≈ 22 px (~220 m) across at 10 m, 90th percentile ≈ 56–59 px, ~13 % exceed a
  640 m tile. Clearly resolvable from Sentinel-2 / Landsat.
- **Expressible as per-pixel classification**: the animal-type tags give a natural unified
  class scheme. Accepted as **positive-only polygon segmentation**.

### Presence vs change / time-range (spec §5)

CAFO barns, pens, and manure lagoons are **persistent structures**. Every facility is
present in the 2016–2017 reference imagery the dataset was built from: all `destruction_upper`
dates are **2018 or later**, and `construction_upper` bounds are almost all ≤ 1998 (latest
2017). So this is a **presence/state** label, not a change label: `change_time=null` and a
**static 1-year window anchored on 2017** (a year in which every facility is both built and
not-yet-destroyed; consistent with the manifest's 2016–2017 span). No dated construction
event is used (dates are year-granular at best, not resolvable to ~1–2 months).

## Class / label mapping

Unified scheme derived from `animal_types`. Per-facility class =
`priority(dairy > poultry > swine > cattle > sheep > goats > unknown)` over its tags (so a
`"dairy, cattle"` facility is **dairy_cattle**, a plain `"cattle"` facility is beef/feedlot
**cattle**). Ids are assigned in descending facility frequency.

| id | name | available facilities | selected (center) | tiles containing class |
|----|------|----------------------|-------------------|------------------------|
| 0 | cattle | 1578 | 1000 | 1055 |
| 1 | poultry | 317 | 317 | 337 |
| 2 | dairy_cattle | 184 | 184 | 220 |
| 3 | swine | 29 | 29 | 31 |
| 4 | unknown | 9 | 9 | 11 |
| 5 | sheep | 3 | 3 | 4 |
| 6 | goats | 1 | 1 | 1 |

Nodata/ignore = **255** (uint8). Global pixel values across all tiles = {0,1,2,3,4,5,6,255}
exactly. "Tiles containing class" counts a tile toward **every** class present in it (tiles
often catch a neighbouring facility of another type), so it exceeds "selected (center)".

The CAFO building footprints **are** the "infrastructure footprints" mentioned in the
manifest; there is no separate per-feature infrastructure attribute in the release, so each
footprint is labeled solely by its facility animal type.

## Encoding, tiling, sampling

- **Recipe**: polygon rasterization (spec §4). Each tile is **64×64** (640 m) at **10 m** in
  the sample's local UTM zone (EPSG:326xx), centered on a **guaranteed-interior
  representative point** of the facility footprint (the recorded facility lat/lon is
  occasionally offset a few hundred metres from the digitized geometry, which would center a
  tile off the footprint — using the geometry's own interior point removed the 1 empty tile
  this caused).
- **All intersecting footprints burned in**: every facility footprint intersecting a tile is
  rasterized to **its own** animal-type class id (`all_touched=True` so small barns survive);
  the rest of the tile is **255 (nodata)**.
- **Positive-only / no background** (spec §5): non-footprint pixels are left as nodata; we do
  **not** fabricate synthetic negatives. The pretraining assembly step supplies negatives by
  sampling locations from other datasets.
- **Sampling** (spec §5): tiles-per-class balanced via `sampling.balance_by_class(key=center
  facility class, per_class=1000)` (25k total cap, never approached). The dominant **cattle**
  class is truncated **1578 → 1000**; all other classes kept in full. Rare classes
  (`sheep`=3, `goats`=1) are **retained** per spec §5 (downstream assembly drops too-small
  classes, not this agent).
- **Time range**: static 1-year window anchored on **2017**; `change_time=null`.

## Verification (spec §9)

- 1543 `.tif` + 1543 `.json`, perfectly paired; all **single-band uint8**, UTM CRS
  (EPSG:32610/32611) at **10 m**, **64×64**, nodata **255**. No oversize tiles.
- Global pixel values = {0–6, 255}; all covered by the class map. No empty (all-nodata) tiles.
- Every `.json` has a **365-day** `time_range` (the shared `io.year_range` 1-calendar-year
  window used by all sibling datasets) and `change_time=null`.
- **Georeferencing round-trip**: each tile's foreground-pixel centroid lands within a few
  tens of metres of the source polygon (small offsets are expected because the tile averages
  all intersecting footprints); labels sit on real, human-validated California CAFO sites.
- All sample centers fall within the California bounding box (lon −124.28…−115.33,
  lat 32.60…41.93), i.e. in the Central Valley / SoCal dairy and poultry regions.
- Idempotent: re-running skips already-written `locations/{id}.tif`.

## Caveats

- **Cattle truncated to 1000** (of 1578 available) by the per-class default; raise
  `PER_CLASS` to include all. All other classes are complete.
- Large facilities (> 640 m, ~13 %) overflow the 64×64 tile and appear as a mostly-foreground
  patch — still a valid positive label.
- `cattle` vs `dairy_cattle` follows the tag priority above; a facility tagged only `cattle`
  is treated as beef/feedlot, one tagged `dairy` (usually `"dairy, cattle"`) as dairy. Some
  real dairies tagged only `cattle` may thus land in class 0.
- Very rare classes (`goats`=1, `sheep`=3) will likely be dropped by downstream rare-class
  filtering; they are kept here per spec §5.
- A full Sentinel-2 pixel overlay was not rendered; georeferencing is exact **by
  construction** (direct rasterization of WGS84 polygons through the validated
  `GeotiffRasterFormat` encode path, same as sibling polygon datasets) and was confirmed via
  the foreground-centroid round-trip above.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cal_ff_california_cafos
```

Idempotent: already-written `locations/{id}.tif` are skipped.
