# tick_tick_bloom_hab_severity

**Status:** completed · classification · 4,072 samples (point table)

## Source
DrivenData / NASA **"Tick Tick Bloom: Harmful Algal Bloom Detection"** competition
(https://www.drivendata.org/competitions/143/tick-tick-bloom/). The competition data has
been released for open reuse (with attribution) as the **Cyanobacteria Aggregated Manual
Labels (CAML)** dataset via NASA **SeaBASS / OB.DAAC** (DOI `10.5067/SeaBASS/CAML/DATA001`).

- Access: one SeaBASS `.sb` file,
  `CAML_cyanobacteria_abundance_20211229_R1.sb` (~2.7 MB), pulled from the OB.DAAC
  `getfile` endpoint. That endpoint requires **NASA Earthdata (URS) auth** — credentials
  come from `.env`
  (`NASA_EARTHDATA_USERNAME`/`NASA_EARTHDATA_PASSWORD`), written to `~/.netrc` so the URS
  OAuth redirect authenticates (spec §8). The script scrapes the SeaBASS archive dir to
  resolve the current content-hashed `getfile` URL, so it stays valid if the archive
  re-hashes.
- Content: **23,570** in-situ cyanobacteria cell-count measurements at points on US inland
  water bodies over **2013–2021**, aggregated from ~40 state/federal environmental
  agencies. Fields: `uid, data_provider, region, lat, lon, date, time, abun, severity,
  distance_to_water_m`.

## Triage decision — ACCEPT (classification)
- Sparse **single-pixel water-column point** measurements with precise lon/lat and a
  specific sample **date** → point-table dataset (spec §2a), `points.geojson`.
- **Label encoding = severity CATEGORY classification** (the competition's target, cleaner
  than continuous density), **5 ordinal classes** severity 1–5 → class ids 0–4. Raw
  density is carried as an **auxiliary** per-point field (see below).
- **Post-2016 rule:** dataset spans 2013–2021 (mix); kept only samples with a **sample
  date on/after 2016-01-01** (dropped 6,815 pre-2016; 16,755 remain before balancing).
- **change_time = null** — this is a *state at a time* (bloom severity on the sample date),
  not a dated change event.
- **Time window:** blooms are transient, so each point gets a **tight ±15-day window
  (30 days) centered on its sample date** via `io.centered_time_range` (well under the
  360-day cap).

## Processing
- Parse `.sb` body (comma-delimited, `/missing=-9999`). Drop rows with unparseable
  date/coords or severity ∉ {1..5} (0 dropped as "bad"; all rows well-formed).
- Filter to sample year ≥ 2016.
- Severity → class: `1→0, 2→1, 3→2, 4→3, 5→4`.
- Density band per severity (competition WHO thresholds, cells/mL):
  0 `<20k` · 1 `20k–100k` · 2 `100k–1M` · 3 `1M–10M` · 4 `≥10M`.
- Balance to **≤1000 per class** (`balance_by_class`, seeded, 25k total cap; spec §5). Rare
  severity-5 class kept in full (only 72 post-2016).
- Point ids assigned in `uid`-sorted order (deterministic / idempotent).

## Output
- `datasets/tick_tick_bloom_hab_severity/points.geojson` — 4,072 `Point` features. Per
  feature `properties`: `id, label (0–4), time_range (±15d), change_time=null,
  source_id=uid, region, density`.
- `datasets/tick_tick_bloom_hab_severity/metadata.json` — class map + descriptions,
  `class_counts`, `auxiliary_fields`.
- `raw/tick_tick_bloom_hab_severity/{CAML_...R1.sb, SOURCE.txt}`.

## Auxiliary field: density
`density` = SeaBASS `abun` (units **cells/L** in the archive; **multiply by 1000** for the
competition **cells/mL**). Per-severity `abun` bands (cells/L) are internally consistent:
sev1 `<20`, sev2 `20–100`, sev3 `100–1000`, sev4 `1000–10 000`, sev5 `≥10 000`, i.e. the
cells/mL thresholds ÷1000. Provided verbatim so a downstream user can recompute severity or
use a (log-scale) regression target instead.

## Class counts (selected)
severity_1_low 1000 · severity_2_moderate 1000 · severity_3_high 1000 ·
severity_4_very_high 1000 · severity_5_extreme 72.

Region distribution of the post-2016 pool: south 8,108 · west 4,152 · midwest 2,818 ·
northeast 1,677.

## Verification (spec §9)
- Valid GeoJSON `FeatureCollection`; 4,072 features; labels ∈ {0..4}; class counts match
  metadata; `count`=4,072.
- All coordinates inside the CONUS bbox; lon ∈ [-124.18, -67.70], lat ∈ [26.39, 48.97].
- All windows are 30-day spans (≤360d); `change_time` null on every feature.
- All **sample dates** are ≥ 2016 (10 features have a window *start* in late-Dec-2015
  purely because their early-Jan-2016 sample date's ±15d window reaches back a few days —
  expected, sample dates themselves are post-2016).
- **Water-proximity sanity check** using the dataset's own `distance_to_water_m` over the
  selected points: median **177 m**, **90.9%** within 1 km of mapped water (45% ≤100 m,
  69% ≤500 m); max 6.5 km. Used in lieu of an S2 image overlay since the labels are
  agency water-sampling coordinates.

## Caveats
- **Weak label for 10–30 m optical.** Bloom severity at a lake point is a plausible weak
  label for S2/Landsat water color, but each label is a **single-pixel water-column
  measurement**, not a full water-body segmentation. A minority of sampling coordinates
  sit tens–hundreds of metres from the mapped water pixel (recorded at access
  points/addresses), so a 1×1 label may occasionally fall off-water at 10 m — pretraining
  pairs by lon/lat + time overlap and should treat these as noisy point labels.
- Severity 5 is very rare (72 post-2016); downstream assembly may drop it under the
  min-per-class filter (spec §5). Kept regardless.
- License: competition data released for reuse **with attribution** — cite Gupta, Gelbart,
  Gupta, Wetstone, Dorne (2024), CAML, SeaBASS, DOI 10.5067/SeaBASS/CAML/DATA001.

## Reproduce
`python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.tick_tick_bloom_hab_severity`
(idempotent; skips the raw download if present, rewrites `points.geojson`/`metadata.json`).
Requires NASA Earthdata creds in `.env`.
