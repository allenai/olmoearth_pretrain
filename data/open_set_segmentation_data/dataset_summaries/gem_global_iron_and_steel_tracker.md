# GEM Global Iron and Steel Tracker (GIST)

- **Slug:** `gem_global_iron_and_steel_tracker`
- **Status:** completed — classification (object-detection encoding), **1,986 samples**
  (986 positive plant tiles + 1,000 background negative tiles)
- **Source:** Global Iron and Steel Tracker (GIST), Global Energy Monitor (GEM),
  June 2026 (V1) release.
  <https://globalenergymonitor.org/projects/global-iron-and-steel-tracker/download-data/>
- **License:** CC-BY-4.0
- **Family / label_type:** industry / points (→ object-detection, positive-only)
- **Annotation method:** authoritative/expert asset-level inventory (company filings,
  government data, satellite imagery, news); coordinates carry an exact/approximate accuracy
  flag ("exact" = centroid satellite-confirmed).

## What the source is

An asset-level inventory of the world's crude iron and steel plants — every plant currently
operating at **≥ 500,000 t/yr (ttpa)** of crude iron or steel, plus plants proposed / under
construction since 2017 or retired / mothballed since 2020. The June-2026 release lists
**1,293 plants** (sheet "Plant data", one row per plant) with a geocoded `Coordinates`
point ("lat, lon"), a `Coordinate accuracy` flag (exact 1,141 / approximate 152), a
`Start date` (commissioning year or "unknown"), `Main production equipment` (the furnace mix:
BF, BOF, EAF, DRI, IF, sinter/coking/pelletizing …), capacities, and, in the "Plant
capacities and status" sheet, a per-unit operating status (operating, operating
pre-retirement, mothballed, construction, announced, cancelled, retired).

## Access — ACCEPT (no credential needed)

GIST is distributed behind a lightweight web **download form** (name/email/use-case), **not**
an authenticated credential gate. The download page ships a *public* Supabase "publishable"
key and an unverified flow: POST the form fields to `mint_submission` (with the public key) →
receive a short-lived `capability_token` → POST it to the `presign` function → receive a
presigned DigitalOcean Spaces URL → GET the `.xlsx` (~0.72 MB). The script reproduces this
flow automatically via `download.download_gem_tracker(["iron-steel-plant-tracker"], raw, contact)`
(the slug is read from the `<gem-download-form slugs=…>` element on the download page; the
`contact` name/email are supplied via the `--contact-name`/`--contact-email` CLI args). No
credential from `.env` was required. Only the label spreadsheet is
downloaded; no imagery (pretraining supplies imagery). See `raw/{slug}/SOURCE.txt`.

This matches the sibling `gem_global_cement_and_concrete_tracker` access recipe.

## Triage decision — ACCEPT (object-detection, positive-only)

Iron/steel plants are very large, clearly-discernible industrial complexes (blast furnaces,
stoves, sinter/coking plants, rolling mills, stockyards) — strongly observable at 10–30 m
from Sentinel-2/Sentinel-1/Landsat. Per the task's manifest note the presence points are
encoded with the **tunable detection recipe** (spec §4), not the sparse point-table path.

- **Encoding:** for each included plant, a 64×64 (640 m @ 10 m) local-UTM context tile
  centered on the plant pixel; the plant is a **`positive_size = 21` px (~210 m)** square of
  class `1` (steel/iron plant) sized to the discernible core of a large complex, ringed by a
  **`buffer_size = 15` px (~150 m)** nodata (255) band (≥ 10 px per spec; the coordinate is a
  point, not a footprint), all remaining pixels **background `0`**. `sampling.encode_detection_tile`.
- **Completeness → true negatives:** GIST is a complete inventory of ≥ 500 ktpa plants, so
  within-tile background is approximately a *true* negative, and every included plant falling
  inside a tile is marked positive (STRtree over all included plants; industrial clusters can
  put several plants in one tile). Sub-500 ktpa mini-mills are untracked (minor
  false-negative risk).

## Inclusion filter

Kept a plant iff **(a)** it has ≥ 1 unit with status ∈ {operating, operating pre-retirement,
mothballed, mothballed pre-retirement} — physically standing, visible structures — **and
(b)** its coordinate accuracy is **`exact`** (GEM satellite-confirmed). Result: **986** of
1,293 plants.

Dropped: announced / cancelled (not built), construction-only (not yet a plant), retired-only
(may be demolished and no dated retirement to time-anchor), and `approximate` coordinates
(city/subnational/country estimates, potentially many km off → beyond the detection buffer,
would place the positive box on empty land). Included plants span 88 countries (China, India,
United States, Iran, Japan, Russia, Türkiye … lead the counts).

## Time / change handling

A built steel mill is a **persistent** structure, not a dated change event, and `Start date`
resolves only to a calendar year (coarser than the ~1–2 month change-timing rule). So
**`change_time = null`** (presence/state, not change) and each positive tile gets a 1-year
window at/after the start year in the Sentinel era: window ∈ `[clamp(start, 2017, 2025), 2025]`
spread deterministically per plant (md5 of GEM plant id) for imagery diversity; `start` < 2017
or "unknown" → `[2017, 2025]`. This keeps pre-2016 plants (still standing post-2016) while
honoring the post-2016 rule. Negatives use a static representative window (2021). Observed
anchor-year spread: 2017:86, 2018:105, 2019:115, 2020:117, 2021:1102 (incl. the 1,000
negatives), 2022:105, 2023:117, 2024:128, 2025:111. All windows are ≤ 1 year and post-2016.

## Negatives

1,000 background-only tiles sampled globally as 15–80 km offsets from random plants (staying
near industrial/populated land) and kept ≥ 5 km from any plant (vectorized haversine).

## Class scheme

`0 = background`, `1 = steel/iron plant`. (Detection encoded as per-pixel classes →
`task_type = classification`; nodata/ignore = 255.)

## Verification (spec §9)

- **1,986** `.tif` + **1,986** `.json`. Sampled tifs: single-band, `uint8`, 64×64, local-UTM
  at 10 m (e.g. EPSG:32610/32611), nodata 255, pixel values ⊆ {0, 1, 255}.
- Every tif has a matching json; all `time_range`s are 1-year (0 exceed 1 yr); all
  `change_time = null`. 986 tiles contain class 1, 1,000 are background-only.
- `metadata.json` classes {0,1} cover all values in the tifs.
- **Spatial sanity:** parsed coordinates (lat, lon) place well-known plants exactly right —
  U.S. Steel Gary Works (41.62, −87.35), Tata Steel Jamshedpur (22.79, 86.20), Baosteel
  Baoshan/Shanghai (31.42, 121.44), İsdemir Payas/Türkiye (36.74, 36.21), Tata Steel Port
  Talbot (51.58, −3.78), Magnitogorsk (53.43, 59.05), POSCO/Pohang (36.00, 129.40). A
  country-bounding-box check on 910 plants in 11 major countries put 99.8 % inside their
  country (lat/lon order confirmed). Label placement is exact by construction (GEM lon/lat →
  same UTM pixel); a full S2 raster overlay was not rendered (consistent with other
  presence/detection datasets and GEM's own satellite confirmation of "exact" coordinates).
- Re-running is idempotent (`{'skip': 1986}`).

## Caveats

- Many steel complexes are **larger** than the 210 m positive core, so pixels in the
  background ring of a positive tile can still be plant; the 150 m nodata buffer and the
  dedicated negative tiles mitigate this.
- Background is only *approximately* a true negative: GIST omits < 500 ktpa mini-mills.
- 152 "approximate" plants and all retired/construction/announced plants are excluded — this
  drops some potentially-visible sites in favor of clean, precisely-located labels.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_iron_and_steel_tracker \
  --contact-name "<your name>" --contact-email "<your email>" \
  --contact-organization "<your org>"
```

The `--contact-name`/`--contact-email` fields are submitted to GEM's public download form
(name/email/use-case); supply your own.

Idempotent: re-downloads the xlsx only if missing (via the GEM `mint_submission`→`presign`
flow), skips already-written tiles, and rewrites `metadata.json`. Outputs on weka under
`datasets/gem_global_iron_and_steel_tracker/` (`locations/{id}.tif|.json`, `metadata.json`,
`registry_entry.json`); raw source under `raw/gem_global_iron_and_steel_tracker/`.
