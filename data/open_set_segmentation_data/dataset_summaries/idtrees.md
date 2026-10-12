# IDTReeS

- **Slug**: `idtrees`
- **Task type**: classification (sparse-point tree-species segmentation)
- **Status**: completed — 1,148 samples, 33 taxon classes
- **Source**: IDTReeS 2018/2020 competition data (Weinstein et al.; idtrees.org). Zenodo
  record https://zenodo.org/records/3700197 ("IDTReeS 2020 Competition Data"). License
  CC-BY-4.0.
- **Family / label_type**: tree_species / polygons (crowns) + points.

## What the source is

IDTReeS is an individual-tree-crown (ITC) delineation + species-classification benchmark
built from co-registered NEON RGB / LiDAR / hyperspectral imagery over NEON field sites.
The **train** release (single 44 MB zip) contains:

- `Field/train_data.csv` — 1,166 field-mapped stems with per-stem `taxonID` species code,
  `scientificName`, `taxonRank`, site, structural attributes, keyed by `indvdID`.
- `Field/taxonID_ScientificName.csv` — the 33-code taxon → scientific-name lookup.
- `ITC/train_{MLBS,OSBS}.shp` — 1,215 individual tree-crown polygons (UTM EPSG:32617),
  each linked to a stem by `indvdID`. Two sites are present in train: **MLBS** (Mountain
  Lake Biological Station, VA — deciduous forest) and **OSBS** (Ordway-Swisher Biological
  Station, FL — longleaf-pine flatwoods). (The manifest mentions a third site/AL; only
  MLBS + OSBS crowns ship in the train ITC set.)

Only the crown geometries + field species labels are used; the multi-GB `RemoteSensing/`
imagery is not downloaded (pretraining supplies its own imagery).

## Access method

`download.download_zenodo("3700197", raw_dir)` — public, no credential. The train zip is
extracted (Field/ + ITC/ only) under `raw/idtrees/`. Idempotent (download + extract skip
existing files).

## Triage — why accepted (observability judgment call)

Individual tree crowns are **small**: median crown footprint here is **~4.2 m × 4.2 m**
(range 0.3–14.8 m), i.e. well under a single 10 m Sentinel-2 pixel. Per-crown *species*
identity is therefore **not directly resolvable at 10–30 m**, and the spec (§8) lists
"individual small trees" as an observability rejection ground.

It is nonetheless **accepted as a weak sparse-point label**, following the exact posture
that admitted `globalgeotree` / `geolifeclef_geoplant`: the crowns sit in **natural NEON
forest**, so a point acts as a weak habitat/species label because the surrounding canopy
correlates with the target. This is the key distinction from `auto_arborist` (rejected):
those were **urban street trees** in pavement-dominated pixels with no habitat proxy. Here
the context is contiguous natural forest, so the weak-label rationale transfers. The
downstream assembly step decides how much weight to give / which sparse classes to drop.

- **Coarsening not needed**: the source is already a manageable 33-class taxonomy (a few
  entries are genus-level, e.g. `BETUL`=Betula sp., `MAGNO`, `PINUS`, `QUERC`, `OXYDE`),
  far under the 254-class uint8 cap, so we keep the native taxon labels rather than forcing
  a genus/functional-type collapse. Note: the label is weak regardless of level.
- **Not `rejected`/`temporary_failure`**: fully accessible, no credential, real
  georeferenced field data in the Sentinel era.

## Processing decisions

- **Sparse points → GeoJSON point table** (spec §2a/§4): each crown polygon → its centroid
  → one `Point` feature, written to one dataset-wide `datasets/idtrees/points.geojson`
  (NOT per-crown GeoTIFFs — a crown is a sub-pixel 1×1 label). Centroids computed in the
  site's UTM CRS then reprojected to WGS84.
- **Class level = `taxonID`** (species code; 33 classes). Ids **0..32 by descending crown
  frequency** (id 0 = `PIPA2`/Pinus palustris, 328 crowns; tail includes several
  single-crown classes). `metadata.json` classes carry `scientific_name`, `taxon_rank`,
  `n_source_crowns`, `n_samples`.
- **Dropped**: 67 of 1,215 crowns had no matching field `taxonID` (unlabeled) → dropped;
  1,148 labeled crowns kept.
- **254-class cap**: not binding (33 « 254) — all classes kept.
- **Rare classes kept** (spec §5): the distribution is long-tailed (many ≤5-crown classes,
  several singletons). Per spec these are retained; the assembly step, not this script,
  filters classes that end up too small.
- **Balancing**: `balance_by_class(per_class=1000, total_cap=25000)` — every class has
  <1000 crowns and the total (1,148) is far under 25k, so all points are kept.
- **Time range**: a tree's species is effectively static; the competition field/flight
  campaign is 2018 (manifest 2018–2019). Per spec §5 (static labels) every sample is
  anchored on a single 1-year Sentinel-era window **2018-01-01 → 2019-01-01** (post-2016).
  `change_time=null` (not a change label).

## Outputs

- `datasets/idtrees/metadata.json` — 33 classes, `nodata_value=255`,
  `task_type=classification`, `num_samples=1148`.
- `datasets/idtrees/points.geojson` — FeatureCollection, `count=1148`, labels 0–32, all
  time ranges the single 1-year 2018 window, all coordinates valid.
- `datasets/idtrees/registry_entry.json` — status `completed`.
- `raw/idtrees/` — Zenodo train zip + extracted Field/ + ITC/ + `SOURCE.txt`.

Top classes by count: PIPA2 328, QURU 181, ACRU 146, QUAL 111, QULA2 74, QUCO2 60,
AMLA 51, NYSY 47, … (tail: ACSA3, CATO6, QUERC, QULA3, LYLU3, GOLA = 1 each).

## Verification (spec §9)

- `points.geojson`: 1,148 features; `label` ∈ [0, 32], 33 distinct; every label present in
  the `metadata.json` class map.
- All `time_range`s are the single 1-year window (≤ 360 days), post-2016; `change_time`
  null.
- Coordinates fall exactly on the two NEON sites — lon −82.0…−80.5, lat 29.68 (OSBS, FL)
  to 37.43 (MLBS, VA). Sample id `000000` = `OSBS/OSBS00029`, label 0 (`PIPA2`, longleaf
  pine) at (−81.997, 29.688) — Ordway-Swisher longleaf-pine flatwoods, an ecologically
  consistent placement.
- An S2 water/land overlay check is not meaningful for weak sub-pixel single-tree points
  (as with `globalgeotree`); coordinates were validated as real NEON forest-site locations
  instead.
- Idempotent: re-running skips the download/extract and rewrites the deterministic
  `points.geojson`/`metadata.json`.

## Caveats

- **Weak/contextual label** — per-crown species is not observable at 10–30 m; treat as a
  weak habitat/species signal (same posture as `globalgeotree`).
- Only train-split crowns are labeled with species; the competition test crowns' species
  are not in this release, so only train (1,148 crowns) is used — still all fair game as
  pretraining labels (spec §5, use all splits).
- Long-tailed taxonomy with several single-crown classes; genus-level codes (Betula sp.,
  etc.) are retained as their own classes.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.idtrees
```
