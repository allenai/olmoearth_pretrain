# GEM Global Cement and Concrete Tracker (GCCT)

- **Slug:** `gem_global_cement_and_concrete_tracker`
- **Status:** completed — classification (presence points), 3,078 samples
- **Source:** Global Cement and Concrete Tracker (GCCT), Global Energy Monitor (GEM),
  July 2025 (V1) release. <https://globalenergymonitor.org/projects/global-cement-and-concrete-tracker>
- **License:** CC-BY-4.0
- **Family / label_type:** industry / points
- **Annotation method:** authoritative expert curation; plant locations satellite-confirmed
  (GEM "exact" coordinate accuracy = "plant location confirmed via satellite imagery").

## What the source is

An asset-level inventory of the world's cement/clinker plants (3,515 plants in the
July-2025 release), distributed as a single `.xlsx` (sheet "Plant Data"). Each plant record
carries a geocoded `Coordinates` point ("lat, lon"), a `Coordinate accuracy` flag
(exact 3,294 / approximate 221), an `Operating status` (operating 3,172, retired 105,
announced 85, mothballed 68, unknown 38, construction 37, operating pre-retirement 8,
cancelled 2), a `Plant type` (integrated 2,421 / grinding 947 / unknown 118 / clinker only
29), a `Start date` (commissioning year or "unknown"), and capacity / production-type /
ownership attributes.

## Access

No credentials required. The GEM download sits behind an email form that mints a
short-lived capability token from a public Supabase backend and returns a presigned
DigitalOcean Spaces URL. The script reproduces this flow automatically (see
`raw/{slug}/SOURCE.txt` for the exact recipe and the public key). Tracker slug:
`cement-concrete-tracker`. Only the label spreadsheet is downloaded (~0.95 MB); no imagery.

## Triage decision — ACCEPT (presence classification)

Cement plants are large industrial complexes clearly observable at 10 m: integrated plants
have rotary kilns, tall preheater towers, clinker/silo storage and an adjacent limestone
quarry; grinding plants have grinding mills and silos. GEM's "exact" accuracy means the
centroid was confirmed against satellite imagery, so those points land on the plant.

- **Encoding:** the source gives POINTS (centroids), not footprints → single-foreground-class
  presence dataset emitted as `points.geojson` (spec §2a), **not** per-point GeoTIFFs.
  Positive-only (spec §5): non-plant negatives are supplied downstream at assembly time; no
  synthetic negatives are fabricated.
- **Class scheme:** the manifest lists "cement plant (integrated/grinding)". Plant type is
  well-populated (integrated vs grinding), and integrated plants are generally larger, but
  reliably separating integrated vs grinding from a single 10 m POINT (no footprint) is only
  weakly observable, and "unknown"/"clinker only" do not map cleanly. Per the spec's default
  we use a **single presence class** `0 = cement_plant`. Plant type, capacity, and
  production type are retained only as documented auxiliary attributes (not class targets;
  none reliably observable at 10 m from a point).

## Status / year / accuracy filter

Kept only plants that are physically built and standing, with a usable point:

- `Operating status ∈ {operating, operating pre-retirement}` — dropped announced (85),
  construction (37), mothballed (68), retired (105), cancelled (2), unknown (38). Retirements
  have no dated retired-year in the release, so they cannot be time-bounded to a Sentinel-era
  window and are excluded.
- `Coordinate accuracy == exact` — dropped 99 "approximate" points (among kept statuses):
  these are city/subnational/country-level estimates that can be many km off the plant,
  unusable as a 1×1 point label.
- Valid parseable coordinates, in range, non-(0,0); de-duplicated at 5 dp (3 dup dropped).
- Operating plants with a start year after 2025: dropped (0 in practice).

Result: **3,078** presence points (integrated 2,192, grinding 826, unknown 32,
clinker-only 28 — plant type kept as auxiliary only). Well under the 25k per-dataset cap.

## Time / change handling

A built cement plant is a **persistent** structure, not a dated change event, and `Start
date` resolves only to a calendar year (coarser than the ~1–2 month change-timing rule), so
`change_time = null` (no dated change labels). Each plant gets a 1-year window sampled
(seeded, deterministic) from `[max(start_year, 2016), 2025]`; start "unknown" → assume a
pre-existing plant → `[2016, 2025]`. Windows spread evenly across 2016–2025 (242–346 per
year), all post-2016.

## Verification (spec §9)

- `points.geojson`: valid `FeatureCollection`, 3,078 `Point` features, `task_type =
  classification`, single label id `0`, unique ids `000000…003077`, all coordinates in
  range, every `time_range` a 1-year post-2016 window, all `change_time = null`. 0 invalid
  features.
- Spatial sanity: random samples resolve to named real cement plants at plausible worldwide
  locations, all with GEM "exact" (satellite-confirmed) accuracy — e.g. Ghori/Pul-i-Khumri
  (Afghanistan), CIMAF Bonaberi (Cameroon), San Clemente (Italy), plus many Chinese plants.
  The observability basis is GEM's own satellite confirmation of the "exact" coordinates; a
  full S2 overlay was not rendered (consistent with other presence-point datasets).

## Caveats

- Point centroids only (no footprints); presence label is a 1×1 pixel. Even "exact"
  centroids may sit tens of metres off the exact structure, but a cement plant footprint is
  hundreds of metres across, so the point lands on/beside the facility.
- 32 kept plants have "unknown" plant type and 28 are "clinker only"; these are all still
  cement/clinker plants and are labeled presence-only, so the ambiguity does not affect the
  class.
- Retired/mothballed plants are excluded (no dated retirement → cannot confirm they were
  standing in the chosen window); this drops potentially-still-visible sites but keeps labels
  reliable.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_cement_and_concrete_tracker
```

Idempotent: re-downloads the xlsx only if missing (via the GEM presign flow), then rewrites
`metadata.json` + `points.geojson`. Outputs on weka under
`datasets/gem_global_cement_and_concrete_tracker/`.
