# Reef Check Global Reef Tracker

- **slug**: `reef_check_global_reef_tracker`
- **status**: **rejected** — not observable at 10–30 m (fundamental; not a retry candidate)
- **task_type** (intended, had it been usable): classification (benthic substrate) or
  regression (hard-coral % cover)
- **num_samples**: 0

## Source

- Manifest name: `Reef Check Global Reef Tracker`
- Source: Reef Check Foundation; portal <https://www.reefcheck.org/global-reef-tracker/>
- Description: community/expert dive-survey records of coral-reef benthic cover and
  indicator organisms at georeferenced sites worldwide.
- Family: coral; region: Global (tropical + California); label_type: `points (transect
  sites)`; annotation_method: field survey (trained divers); license: "portal terms";
  manifest time_range: 2016–2024; have_locally: false.
- Manifest classes (9): hard coral, soft coral, recently killed coral,
  nutrient-indicator algae, sponge, rock, rubble, sand, silt.

## What the labels actually are

The Reef Check protocol records **substrate via point-intercept sampling every 0.5 m**
along the same line used for the fish/invertebrate belt transects, broken into four
20 m × 5 m segments (~100 m total transect). Each of the manifest's 9 classes is a
**benthic point-intercept substrate category** — i.e. the fractional composition of the
seafloor along a submerged dive transect. The headline products are per-site
**percent substrate cover** (notably % hard coral cover).

## Why rejected — observability at 10–30 m (SOP §8.2)

The phenomenon is **not resolvable from Sentinel-2 / Sentinel-1 / Landsat**:

- **Submerged benthos.** The substrate lies on the reef floor under the water column.
  S1 (C-band radar) and Landsat/S2 SWIR–NIR do not penetrate water; only S2/Landsat
  blue–green bands see shallow, clear-water bottom, and even then benthic composition
  retrieval is unreliable. Individual substrate types cannot be distinguished at 10 m.
- **Sub-pixel zonation.** Distinguishing hard coral vs. soft coral vs. rubble vs. sand
  vs. silt vs. rock at 0.5 m point intervals is far below a 10 m pixel; an entire reef
  patch is often only one or a few 10 m pixels. The spec explicitly flags "fine
  coral/seagrass zonation" as a class set that "may be unresolvable at 10 m" (§4).
- **Site-level coordinates.** Coordinates are per-site/transect (transect sites), not a
  per-pixel benthic map, so there is no polygon/mask footprint to rasterize.
- **No salvageable aggregate.** The only label expressible at 10–30 m would be a weak
  binary **"reef present here"** presence point. That degrades the rich 9-class substrate
  survey to a single presence class, and it is **redundant with a preferred reference
  alternative already in the manifest**: `UNEP-WCMC Global Warm-Water Coral Reefs`
  (label_type `polygons + points`), which directly provides coral-reef extent. Per SOP
  §8.2 ("defer to the reference") that pairing further argues against ingesting this as a
  presence point.

Because even with the data in hand the labels cannot be expressed as a meaningful
per-pixel classification or regression at 10–30 m, this is a **fundamental `rejected`**
(SOP §8.2: "phenomenon not observable at 10–30 m … and no aggregate/mask representation
salvages it"), **not** `temporary_failure` and **not** `needs-credential` (the access
gate below is secondary and moot given the observability failure).

## Secondary note — access gate

The Global Reef Tracker exposes per-site "Survey History / VIEW DETAILS" pages, but bulk
survey data (coordinates + substrate + coral cover) is obtained via a **Data Download
Request Form** (registration/approval), not an open API or direct download. The portal is
reachable (HTTP 200 on 2026-07-11), so this is **not** a transient outage. A partial open
mirror exists for the Australian subset only (Reef Check Australia on data.gov.au), which
does not change the observability verdict for the global substrate labels.

## Judgment calls

- **Reject on observability**, the fundamental and non-retryable ground, rather than
  `needs-credential`: obtaining the data would not make the substrate labels usable at
  Sentinel/Landsat resolution.
- Did **not** attempt to salvage a "reef here" presence point — it discards the survey's
  information content and duplicates the preferred UNEP-WCMC coral-reef reference.
- Did **not** attempt a hard-coral-% regression: coral cover along a submerged sub-pixel
  transect is not reliably retrievable from S2/S1/Landsat, and coordinates are site-level.
- Status `rejected` (not `temporary_failure`): source portal is up; the block is
  fundamental, not a transient error.

## Reproduce

No outputs were written to weka `datasets/` beyond `registry_entry.json`. To revisit,
one would submit the Data Download Request Form at
<https://www.reefcheck.org/global-reef-tracker/>, but the observability rejection stands
regardless of access.
