# World Database on Protected Areas (WDPA) — REJECTED

- **slug**: `world_database_on_protected_areas_wdpa`
- **status**: **rejected** — **label semantics not expressible as per-pixel
  classification; phenomenon not observable at 10–30 m** (SOP §8.2 "label semantics can't
  be expressed as per-pixel classification/regression" / "phenomenon not observable at
  10–30 m from S2/S1/Landsat"). WDPA polygons are **legal/administrative boundaries, not
  land-cover boundaries** (as the manifest note itself flags).
- **task_type**: n/a (rejected; would nominally have been polygon classification of IUCN
  category / designation)
- **num_samples**: 0
- **Not** `temporary_failure` and **not** `needs-credential`: the source is free and
  reachable (see access note below); the block is fundamental and permanent.

## Source

- Manifest name: `World Database on Protected Areas (WDPA)`; source **UNEP-WCMC & IUCN**,
  <https://www.protectedplanet.net/en/thematic-areas/wdpa>. Family `protected_area`,
  label_type `polygons`, region Global, license **free, non-commercial, attribution**,
  have_locally: false, time_range `[2016, 2026]`.
- Content: the most comprehensive global inventory of **marine and terrestrial protected
  areas** — legally designated boundaries (national parks, wildlife sanctuaries, nature
  reserves, Ramsar sites, marine protected areas, etc.) contributed by governments/NGOs.
  Attributes per polygon include **IUCN management category** (Ia strict nature reserve,
  Ib wilderness, II national park, III natural monument, IV habitat/species management, V
  protected landscape/seascape, VI sustainable-use protected area, plus "Not
  Reported/Assigned/Applicable"), **designation type**, and a **marine/terrestrial** flag.
- Access is **not** the blocker: WDPA ships as monthly geodatabase/shapefile releases from
  Protected Planet (free download after accepting non-commercial terms; also mirrored),
  and `.env` is irrelevant here because no credential is required.
  Access was therefore not investigated further — the rejection is on label semantics, and
  the SOP directs cheap semantic triage **before** downloading large archives (WDPA is a
  multi-GB global polygon DB with ~300k+ features).

## Why rejected — label is a legal boundary, not an imagery-observable phenomenon

The core requirement (SOP §2/§4/§8) is a **per-pixel** label that a model can learn to
predict **from S2/S1/Landsat imagery at 10–30 m**. WDPA fails this on the meaning of the
label itself, independent of resolution or access:

1. **Protected-area boundaries do not coincide with land-cover boundaries.** "Is this
   pixel inside a legally protected area?" is an administrative/legal fact, not a spectral
   or structural property of the surface. A boundary routinely runs straight through
   spatially continuous, homogeneous cover — the forest/grassland/reef one pixel *inside* a
   park is spectrally identical to the same cover one pixel *outside* it. Rasterizing the
   polygon (spec §4 polygons) would train the model to hallucinate an invisible edge. This
   is exactly the "legal boundaries (not land-cover boundaries)" caveat in the manifest.

2. **The class attributes are governance designations, not surface attributes.**
   - **IUCN category** encodes a *management regime*, not a land cover. A single category
     spans wildly different surfaces: Category V ("protected landscape/seascape") is
     explicitly a lived-in cultural/working landscape (farmland, villages, managed forest);
     Category VI is sustainable-use land; Category Ia/II can be forest, desert, tundra,
     coral reef, or open ocean. Conversely, one land cover (e.g. temperate forest) appears
     under every category. There is no per-pixel imagery signal that maps to IUCN category,
     so it is not a learnable per-pixel class.
   - **Designation type** (national park vs Ramsar site vs wildlife sanctuary …) is purely
     legal/jurisdictional — even less observable than IUCN category.
   - **Marine/terrestrial** *is* observable, but it is trivial land/water masking already
     covered far better by dedicated products (JRC surface water, land-cover datasets). It
     needs nothing from WDPA — and a marine-protected-area boundary sits in open water with
     no visible edge at all — so it is not a reason to keep WDPA.

3. **Polygons are heterogeneous internally.** A single protected area (often
   10²–10⁴ km²) contains forest + water + rock + grassland + settlements. Assigning the
   whole polygon one class id gives a per-tile label that disagrees with most pixels in the
   tile — the same "too coarse / not per-pixel truth" failure recorded for aggregated-unit
   datasets (cf. `optis_operational_tillage_information_system`, `eyes_on_the_ground_kenya`).

### Why the "defensible target" path was considered and declined

The task prompt explicitly allowed acceptance *if* a defensible target existed (e.g. "IUCN
category as a weak areal class over genuinely distinguishable protected land"). I
considered it and rejected it:

- The only imagery-visible correlate of protection is the occasional **"green island"
  effect** — a well-enforced reserve retaining intact vegetation amid a deforested/
  developed surround. But this is (a) **inconsistent** globally (paper parks, marine PAs,
  category V/VI working landscapes, and reserves inside already-intact wilderness show no
  contrast), (b) **confounded** (the visible signal is *land cover / disturbance*, already
  captured by intact-forest, land-cover, and deforestation datasets in this bank — e.g.
  `intact_forest_landscapes_ifl`), and (c) **not what the WDPA label encodes**: the label
  is the legal polygon, whose edge generally is *not* the vegetation edge. Training on it
  would teach the invisible-boundary hallucination in (1).
- A weak areal/regression "prior" painted over each polygon would attach one governance
  code to a region tens of km wide spanning many covers — the coarse-aggregation case the
  SOP flags for rejection, not a per-pixel truth.

So no defensible per-pixel target survives. The honest call is to **reject** and document,
rather than manufacture a label the model cannot learn from imagery.

## Judgment calls

- **`rejected`, not `temporary_failure`.** The block is intrinsic to what WDPA measures
  (legal protection status / management category), not a transient source/infra error.
  Re-running later cannot fix it.
- **`rejected`, not `needs-credential`.** WDPA is freely downloadable (non-commercial
  terms). Access is not why it is unusable.
- **Not the pre-2016 rule.** WDPA is continuously maintained (2016–2026 window is fine);
  timing is not the issue.
- **No large download performed.** Per SOP §8.2, georeferencing/semantics were triaged
  cheaply first; because the label semantics fail, pulling the multi-GB global polygon DB
  was unwarranted. Only `datasets/world_database_on_protected_areas_wdpa/registry_entry.json`
  (status `rejected`) is written to weka; no `raw/`, `metadata.json`, or `locations/`.

## Reproduce / revisit

No processing script was written. This dataset would become usable only if the target were
redefined away from legal protection status toward an imagery-observable phenomenon — but
that phenomenon (land cover, forest intactness, disturbance) is already better served by
dedicated reference/map datasets in this bank, so WDPA has no distinct per-pixel signal to
contribute. The manifest note ("Legal boundaries (not land-cover boundaries)") is the
crux: WDPA is authoritative and valuable for conservation analysis, but it is not
per-pixel segmentation truth for S2/S1/Landsat.
