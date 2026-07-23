# Ukraine War Damage (UNOSAT-derived)

- **Slug:** `ukraine_war_damage_unosat_derived`
- **Status:** completed
- **Task type:** classification (sparse change points)
- **num_samples:** 4000 (1000 per class × 4 classes)

## Source & access

ETH Zurich / prs-eth **ukraine-damage-mapping-tool**
(<https://github.com/prs-eth/ukraine-damage-mapping-tool>, MIT-licensed tool code;
labels derived from UNOSAT VHR damage assessments — Dietrich et al., *Comms Earth &
Environment* 2025). The repo ships pre-processed UNOSAT Comprehensive Damage Assessment
(CDA) points in `data/unosat_labels.geojson`. No credentials required — the two label
GeoJSONs are fetched directly from the repo's `main` branch (raw.githubusercontent.com)
into `raw/ukraine_war_damage_unosat_derived/` (`unosat_labels.geojson` ~7 MB,
`unosat_aois.geojson`). Only the thin label layer is downloaded; the paper's Sentinel-1
imagery/GEE assets are not needed (pretraining supplies imagery).

## Source data

18,686 `Point` features over 18 Ukrainian AOIs (`UKR1`–`UKR18`; Mariupol, Makariv, Sumy,
Kharkiv area, etc.). Each feature carries: `damage` (UNOSAT numeric grade), `date` (the
**post-event VHR image / analysis date**, day-precise), `aoi`, `city`, `ep` (analysis
epoch), and `unosat_id`. All labels are **2022** (2022-03-14 … 2022-10-17). All 18,686
coordinates are distinct (no per-epoch coordinate duplicates), so each feature is treated
as one distinct building-damage location.

Grade distribution (all 18,686): `{1: 2471, 2: 8463, 3: 5614, 4: 1661, 5: 51, 6: 59,
7: 366, 15: 1}`.

## Class scheme (unified 4-class damage grades)

| id | name | UNOSAT grade | kept pts (grades 1–4: 18,209) |
|----|------|--------------|-------------------------------|
| 0 | destroyed | 1 | 2,471 |
| 1 | severe_damage | 2 | 8,463 |
| 2 | moderate_damage | 3 | 5,614 |
| 3 | possible_damage | 4 | 1,661 |

Grades **5/6/7/15** (477 pts) are ambiguous / non-building-damage categories
(no-visible-damage, other) and are dropped. Positive-only dataset (no intact/background
class): non-object pixels stay nodata (255); the assembly step supplies negatives from
other datasets (spec §5).

## Time-range & change handling (spec §5)

This **is** a change dataset. Each point's `date` is a day-precise **post-event** VHR image
date in 2022, and the damage it records occurred during the war (after 2022-02-24) within
weeks of that image — so the change date is known to well within ~1–2 months. Therefore
`change_time` = the assessment/image date, and instead of one centered window we emit **two
independent six-month windows** via `io.pre_post_time_ranges(change_time, pre_offset_days=45)`:
a **`post_time_range`** that starts at `change_time` and runs ~6 months (≤183 days) forward,
and a **`pre_time_range`** that **ends 45 days before `change_time`** (a guard offset, since
the imagery follows the destruction by weeks) and spans ~6 months (≤183 days) backward from
there, placing the pre window before the event. `time_range` = `null`. Pretraining pairs a
"before" stack with an "after" stack and probes on their difference. Verified: for all 4000
features the pre and post windows are each ≤183 days.

(Contrast with the sibling `unosat_conflict_damage_assessments`, sourced from HDX, which
recast to static presence/state with `change_time=null` because those comprehensive
products compare against baselines 1–3 years earlier so the event was not resolvable in
time. This ETH dataset's per-point dated 2022 assessments are resolvable, hence a real
change label.)

## Encoding

Sparse point segmentation → one dataset-wide GeoJSON point table
`datasets/ukraine_war_damage_unosat_derived/points.geojson` (spec §2a). One `Point`
feature per building-damage location; `properties.label` = class id, plus per-feature
`change_time` and a `pre_time_range` / `post_time_range` pair (`time_range` null). No
per-point GeoTIFFs. Balanced to **1000 per
class** (spec §5 classification cap); all four classes have ≥1000 source points so the
result is 1000×4 = 4000.

## Caveats

- **10 m observability:** an individual building is ~1 pixel at 10 m, so a single damaged
  structure is near the resolution limit. The observable signal is destroyed/severe damage
  of larger structures and the **dense clusters** of damage points in besieged cities
  (Mariupol/UKR1 has 6,093 pts, Kharkiv-area UKR3 5,650). Finer grades (moderate/possible)
  of isolated buildings likely are not resolvable at 10 m — grades are kept as a unified
  scheme so downstream training can select/merge; the limitation is flagged here.
- **Change-window edge:** `change_time` is the *assessment* date; the physical destruction
  happened somewhat earlier. The 45-day pre-window guard offset pushes the "before" window
  back so it precedes the event for most AOIs, where the assessment follows the destruction
  by weeks; for a few late-2022 assessments of areas damaged much earlier the destruction
  could still fall inside the pre window. Anchoring on the assessment date follows the task
  instruction.
- Balancing caps drop most of the abundant severe/moderate points (8,463/5,614 → 1,000
  each); full label set remains in `raw/` for any re-scope.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ukraine_war_damage_unosat_derived
```

Idempotent: the raw GeoJSONs are skipped if already downloaded; the point table is
regenerated deterministically (seeded balancing) and written atomically.
