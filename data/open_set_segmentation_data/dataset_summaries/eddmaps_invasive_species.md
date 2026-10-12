# EDDMapS Invasive Species

- **Slug:** `eddmaps_invasive_species`
- **Status:** completed (classification, 24,892 samples)
- **Family / label_type:** invasive_species / points -> 1x1 sparse point segmentation
- **Source:** GBIF open mirror of invasive-plant occurrences (EDDMapS bulk is gated)
- **License:** GBIF occurrences under source terms (mostly CC-BY / CC0); EDDMapS bulk
  itself is "viewable; bulk by request"

## Source and access (triage)

The manifest dataset is **EDDMapS Invasive Species** (University of Georgia / Bugwood), a
georeferenced database of invasive-plant occurrences across the US & Canada. Its bulk data
is licensed "viewable; bulk by request" — bulk download requires an account/request
approval we do not have, and there is no EDDMapS credential in `.env`.

Per the task spec (§8), rather than reject on the credential gate, we fall back to the
**open GBIF mirror** of the same signal. We take the universe of introduced/invasive plant
taxa registered for the US & Canada in the GBIF **GRIIS** checklists (Global Register of
Introduced and Invasive Species — the registry EDDMapS-tracked species belong to) and pull
their georeferenced occurrences (lon/lat + species + date) from the open GBIF Occurrence
API. GBIF ingests several EDDMapS-network / US invasive datasets (IPAMS, iNaturalist, USGS,
etc.), so there is real overlap, but this is a GBIF-sourced **proxy**, not the raw EDDMapS
export — documented here and in `metadata.json`.

- GRIIS checklists used (GBIF datasetKeys): US contiguous `32ad19ed-…`, Alaska
  `7b091962-…`, Hawaii `6baf6a53-…`, Canada `b95e74e0-…`.
- Occurrence filter: `country in {US, CA}`, `hasCoordinate=true`,
  `hasGeospatialIssue=false`, `year 2016–2026` (Sentinel era; manifest time_range).
- Everything is cached under `raw/eddmaps_invasive_species/` (`SOURCE.txt`,
  `checklist_species.json`, `top_species.json`, `occurrences.json`) so the run is idempotent
  and re-uses network results.

## Suitability decision (accepted as a weak / contextual label)

A single invasive plant is usually sub-10 m and not resolvable from Sentinel/Landsat, but
dense infestations (kudzu mats, water-hyacinth rafts, Spartina/cordgrass meadows,
cheatgrass-dominated range, tamarisk stands) can be. Matching the spec's support for weak /
contextual species-presence labels and huge taxonomies, we accept it: access is fully open,
coordinates are point observations that fit the sparse-point → 1×1 recipe exactly, and the
class = species. The caveat is recorded; downstream assembly supplies negatives and drops
too-rare classes.

## Processing

- **Class universe:** invasive/introduced Plantae species from the four GRIIS checklists,
  resolved to GBIF backbone `nubKey`s.
- **Class selection (254-class uint8 cap):** the GBIF `speciesKey` occurrence facet returns
  species in strictly descending US+CA (2016+) occurrence count; we scan it top-down,
  keeping invasive species, and stop at the top **254 by frequency** (ids 0..253 in
  descending frequency).
- **Flagship injection:** the four manifest-named flagship invasives — *Pueraria montana*
  (kudzu), *Tamarix ramosissima* (tamarisk/saltcedar), *Pontederia crassipes* (water
  hyacinth), *Sporobolus alterniflorus* (smooth cordgrass) — are force-included (ids
  250–253) even where their raw GBIF frequency falls just below the strict cut, displacing
  the lowest-frequency otherwise-selected classes to hold the 254-class cap. These are the
  classic dense-infestation species the task highlights as most observable at 10 m.
- **Occurrences:** up to 200 georeferenced occurrences fetched per selected species (parallel,
  rate-limit-aware GBIF client with 429 back-off).
- **Balancing:** `balance_by_class(per_class=1000, total_cap=25000)`. The binding constraint
  is the 25k total cap over 254 classes → ~98 samples/class; every class landed at exactly
  **98**, for **24,892** total points (perfectly balanced).
- **Tiles:** none — pure sparse points, so one dataset-wide `points.geojson` (spec §2a).
- **Time range:** 1-year window anchored on each occurrence's observation year (2016–2026),
  via `io.year_range`; `change_time=null` (presence, not change).
- **Provenance:** each feature carries `source_id = gbif:{key}`, `species_key`,
  `gbif_dataset_key`.

## Outputs

- `datasets/eddmaps_invasive_species/points.geojson` — FeatureCollection, 24,892 Point
  features, `task_type=classification`.
- `datasets/eddmaps_invasive_species/metadata.json` — 254 classes; each class description
  carries the scientific name, common name (flagships), GRIIS listing, backbone speciesKey,
  and source occurrence count; plus `class_counts`.
- `datasets/eddmaps_invasive_species/registry_entry.json` — `completed`.

## Verification

- `points.geojson` is a valid FeatureCollection with 24,892 features; `count` field matches.
- Labels span ids 0..253, all covered by the class map; every class has 98 samples.
- Sample feature geometry is a WGS84 `[lon, lat]` Point (e.g. `[-123.066, 49.228]`,
  Vancouver area); properties include `id`, `label`, `time_range`, `change_time`,
  `source_id`, `species_key`, `gbif_dataset_key`.
- All `time_range`s are 1-year windows within 2016–2026.

## Caveats

- **Proxy source:** GBIF occurrences of GRIIS-listed invasive/introduced plants, not the raw
  EDDMapS bulk export (which is gated). Same signal, overlapping datasets, but not identical
  provenance.
- **Weak label:** a single citizen-science plant observation is not directly inferable from a
  10–30 m pixel; treat as habitat/biogeographic context. Dense infestations of the flagship
  species are the observable end of the spectrum.
- **Truncated taxonomy:** GRIIS lists far more than 254 species for the region; only the top
  254 by GBIF occurrence frequency (plus forced flagships) are kept, per the uint8 cap.
- Points-only positive labels; no background/negative class (assembly supplies negatives).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eddmaps_invasive_species --workers 6
```

Idempotent: network results are cached under `raw/`; re-running re-uses them and rewrites the
same `points.geojson`.
