# Open-Set Segmentation Data — Agent Task Specification

## Context (why this exists)

OlmoEarth pretraining consumes Sentinel-2 / Sentinel-1 / Landsat imagery over 2560 m
UTM tiles with a 360-day time range. We want to enrich pretraining with a large,
diverse bank of **georeferenced ground-truth label rasters** ("open-set segmentation"
data) that can be paired with that imagery at training time. The labels come from ~300
candidate datasets catalogued in
`data/open_set_segmentation_datasets.json`
(32 already on local disk as rslearn datasets, 268 external). Targets are a mix of
**per-pixel classification** (crop type, land cover, tree species, …) and **per-pixel
regression** (canopy height, biomass, soil properties, fractional cover, population,
nightlights, bathymetry, …).

The immediate goal is to assemble, per dataset, **up to 1000 locations per class**
(classification) or **up to 5000 locations** (regression), each stored as a small
single-band GeoTIFF label patch plus sidecar metadata, so labels can later be co-located
with pretraining imagery by geography and time overlap.

**This document is the task spec.** It is written to be loaded into a fresh agent (with no
memory of the planning session) together with a single dataset's manifest entry. That
agent processes that one dataset end-to-end. The sections below are the instructions
agents follow.

Design decisions already made (do not re-litigate):
- Prefer manual/in-situ **reference** data; use derived-product **maps** only as a
  fallback (and then sample only homogeneous/high-confidence areas).
- Object detections split by negative source (§4): **global point inventories** (isolated
  object coordinates, no real scene) are **presence-only points** — no fabricated
  buffer/background/negatives; only **exhaustively-searched real scenes** (vessels/xView3/
  annotated infra windows) use the tunable detection-tile encoding with genuine in-scene
  negatives.
- Dense multi-class rasters use **tiles-per-class balanced** sampling (a tile counts
  toward every class present in it; prioritize rare classes to reach the target).
- Shared code lives in the repo module (not per-dataset `code/` dirs); summaries, the
  central `registry.json`, and this task spec live in the repo; only raw files + label
  outputs (and each dataset's own `registry_entry.json`) live on weka.

---

# AGENT TASK: process one open-set-segmentation dataset

You are given **one entry** from
`olmoearth_pretrain/data/open_set_segmentation_datasets.json` (name, description, source,
url, classes, time_range, family, region, label_type, annotation_method, license,
have_locally, notes). Take that dataset from raw source to finished label outputs, or
reject it with a documented reason. Work autonomously; make and record judgment calls.

## 0. Environment

- Repo: `.` (module `olmoearth_pretrain`). Python env via
  **uv** (`uv run ...`). rslearn is an installed dependency; core lib at
  `rslearn`, downstream examples at `rslearn_projects`.
- Compute/storage: run under a **Beaker** session with the weka mount
  (`--mount src=weka,ref=dfive-default,dst=/weka/dfive-default`). Outputs go to weka.
- Parallelize downloads/conversions with `multiprocessing.Pool` +
  `rslearn.utils.mp.star_imap_unordered` (see reusable refs).
- **Disk precondition (check before and periodically during any download/write):** run
  `df -B1 --output=avail /weka/dfive-default | tail -1` and confirm **≥ 5 TB
  (5e12 bytes)** free. If less, **stop immediately**, do not write further, and alert —
  do not delete anything to make room. Re-check every so often during large downloads.

## 1. Output contract (exact paths)

Let `{DATASET}` be a snake_case slug of the dataset name (lowercase, non-alphanumeric →
`_`, collapse repeats), e.g. `OlmoEarth Kenya Nandi crop type` →
`olmoearth_kenya_nandi_crop_type`. Record the slug in `metadata.json`.

On weka (`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/`) — bulk
label outputs only:
- `datasets/{DATASET}/registry_entry.json` — your dataset's own status entry (this is how
  you report status; see §1a). Written even for rejected datasets.
- `raw/{DATASET}/` — raw downloaded source files. For `have_locally: true` datasets, do
  **not** copy; write `raw/{DATASET}/SOURCE.txt` pointing at the source rslearn path.
- `datasets/{DATASET}/metadata.json` — dataset-level metadata (schema in §3).
- `datasets/{DATASET}/locations/{SAMPLE_ID}.tif` — one single-band label patch.
- `datasets/{DATASET}/locations/{SAMPLE_ID}.json` — per-sample metadata (schema in §3).

In the repo (`.`) — version-controlled:
- `data/open_set_segmentation_data/registry.json` — the shared **name → slug + status**
  registry (see §1a). **Orchestrator-owned; dataset scripts must NOT write it.**
- `data/open_set_segmentation_data/AGENT_SUMMARY.md` — this task spec.
- `olmoearth_pretrain/open_set_segmentation_data/` — shared code + per-dataset scripts
  (layout in §6). **Check for existing shared utilities before writing new code; add
  reusable pieces to the shared modules, not the per-dataset script.**
- `data/open_set_segmentation_data/dataset_summaries/{DATASET}.md` — processing summary
  (what the source is, decisions, how labels/classes map, sample counts, caveats, and how
  to reproduce). If rejected, this file records the rejection reason and nothing is
  written to weka `datasets/`.

`SAMPLE_ID`: zero-padded running index within the dataset (`000000`, `000001`, …).

## 1a. Registry (`registry.json`)

`data/open_set_segmentation_data/registry.json` (in the
repo, version-controlled) is the canonical map from manifest `name` → on-disk `slug`, plus
per-dataset status. It is
generated from the manifest with slugs already assigned (uniqueness enforced) — **use the
slug the registry gives you; do not re-derive it.** Each entry:
```json
{"name": "VIIRS Nightfire Gas Flaring", "slug": "viirs_nightfire_gas_flaring",
 "source": "...", "family": "...", "label_type": "...", "have_locally": false,
 "status": "pending", "task_type": null, "num_samples": null, "notes": ""}
```
`status` lifecycle: `pending` → `selected` (queued for work) → `in_progress` →
`completed` | `rejected` | `temporary_failure`. The registry — not the manifest — is the
source of truth for what slug a dataset uses.

**`rejected` vs `temporary_failure` — pick the right terminal status:**
- `rejected` = the dataset **cannot be used as-is** for a fundamental reason (no
  recoverable geocoordinates, all labels pre-2016, phenomenon not observable at 10–30 m,
  label semantics not expressible as per-pixel class/regression, license forbids use). Do
  not expect to retry these without new information.
- `temporary_failure` = the dataset **is a good fit and would process, but an external
  transient problem blocked it right now** — source server outage / HTTP 5xx, rate-limit,
  temporary infra failure, or a flaky mirror. These are **retry candidates**: re-running
  the same script later (once the source recovers) should succeed. Put the concrete failure
  and retry steps in `notes` and in the summary. (Datasets blocked only on missing
  credentials use `rejected` with `notes: "needs-credential: <what>"` as before — the user
  supplies creds/pre-downloaded copies out of band; use `temporary_failure` specifically
  for transient source/infra errors, not for permanent access gates.)

**CRITICAL — do NOT write the central `registry.json`.** It is owned solely by the
orchestrator. Many dataset scripts run concurrently, and direct writes to the shared file
corrupt it. Instead, record your status in your OWN per-dataset file
`datasets/{slug}/registry_entry.json` by calling
`manifest.write_registry_entry(slug, status, task_type=, num_samples=, notes=)` (the older
name `manifest.update_status` is an alias and is also safe — both write only the
per-dataset entry, never the central registry). Write it when you start (`in_progress`)
and when you finish (`completed` with `task_type`/`num_samples`, or `rejected` with the
reason in `notes`). The orchestrator periodically runs `manifest.aggregate_registry()` to
merge every per-dataset `registry_entry.json` (read from weka `datasets/{slug}/`) into the
central `registry.json` (in the repo), and backs the central file up to `registry.json.bak`
before launching any batch of agents. Rejected datasets still get a
`datasets/{slug}/registry_entry.json` (the one file they write to weka).

## 2. GeoTIFF spec (the label patch)

**When to write a GeoTIFF vs a point table:** GeoTIFFs are for labels **larger than a
single pixel** — dense rasters, rasterized polygons, and detection tiles (≥ a few px).
**Pure sparse-point datasets (1×1 labels) are NOT written as per-sample GeoTIFFs** — a 1×1
label is just a (location, class-or-value) pair, and writing millions of tiny files
cripples weka. Point-only datasets use one dataset-wide GeoJSON table instead (see §2a). The
rest of this section (§2) applies to the GeoTIFF case.

- **Single band**, **local UTM** projection, **10 m/pixel**, north-up (positive
  `x_resolution`, negative `y_resolution`). Pick the UTM zone from the sample's lon/lat via
  `rslearn.utils.get_utm_ups_crs.get_utm_ups_projection(lon, lat, 10, -10)`, or reuse the
  source window's CRS if already UTM at 10 m.
- **Size**: target **64×64**, hard cap 64×64. Size down to the label's real footprint —
  natively-small dense products at their native extent (e.g. WorldCover ≈ 10×10), sparse
  point labels → **1×1**. Never exceed 64 on either axis.
- **dtype + nodata**:
  - Classification → **uint8**. Class IDs start at **0**; value **255 = nodata/ignore**
    (unobserved pixels, detection buffers). If a dataset has an explicit
    background/negative class, it is a normal class ID (commonly 0).
  - Regression → dtype matching the source (**float32** default; int16/uint16 when the
    source is integer-valued and range fits). Nodata sentinel default **-99999** (repo's
    `MISSING_VALUE`); may override per dataset — record the chosen value in
    `metadata.json`.
- Write with rslearn so georeferencing is exact:
  `GeotiffRasterFormat().encode_raster(path_parent, Projection(crs, 10, -10), pixel_bounds,
  RasterArray(chw_array=arr[np.newaxis]), fname=...)` where `pixel_bounds` is
  `(col*W... )` integer pixel coords under that projection. Write atomically (`.tmp` then
  rename), as in the download template.

## 2a. Point-table format (sparse-point datasets only) — GeoJSON

For pure sparse-point datasets (each label is a single 10 m pixel with a class id or a
regression value), do **NOT** create `locations/{id}.tif` or per-point JSONs. Instead
write **one** dataset-wide **GeoJSON** table `datasets/{DATASET}/points.geojson`: a
`FeatureCollection` with one `Point` `Feature` per location. Coordinates are WGS84
`[lon, lat]` (GeoJSON's native CRS); pretraining projects them onto the S2 grid. Per-point
fields (`id`, `label`, `time_range`, `change_time`, `pre_time_range`, `post_time_range`,
`source_id`) live in each feature's
`properties`; `dataset`/`task_type`/`count` are FeatureCollection-level foreign members.
Change/event point datasets set `pre_time_range`/`post_time_range` and leave `time_range`
null, exactly as for GeoTIFF samples (§3, §5).
```json
{
  "type": "FeatureCollection",
  "dataset": "olmoearth_lcmap_land_use",
  "task_type": "classification",              // or "regression"
  "count": 5643,
  "features": [
    {"type": "Feature",
     "geometry": {"type": "Point", "coordinates": [-106.3927, 43.3233]},
     "properties": {
       "id": "000000",
       "label": 2,                            // class id (classification) OR value (regression)
       "time_range": ["2017-01-01T00:00:00+00:00", "2018-01-01T00:00:00+00:00"],
       "change_time": null, "pre_time_range": null, "post_time_range": null,
       "source_id": "test/sample_10000"}}
  ]
}
```
`label` is the class id for classification or the numeric value for regression. The
dataset-level `metadata.json` (class map / regression block, §3) is still written. Use
`io.write_points_table(slug, task_type, points)` — it takes the same list of point dicts
(`id`/`lon`/`lat`/`label`/`time_range`/`change_time`/`source_id`, plus optional
`pre_time_range`/`post_time_range` for change datasets) and writes the GeoJSON.
A single `points.geojson` handles even large sets (e.g. 50k features) fine; there is no
JSON-lines variant.

## 3. Metadata schemas

`datasets/{DATASET}/metadata.json` (dataset-level):
```json
{
  "dataset": "olmoearth_kenya_nandi_crop_type",
  "name": "OlmoEarth Kenya Nandi crop type",
  "task_type": "classification",            // or "regression"
  "source": "olmoearth", "license": "internal",
  "provenance": {"url": "...", "have_locally": true, "annotation_method": "..."},
  "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],

  // classification only: id <-> name, plus optional per-class description from the source
  "classes": [
    {"id": 0, "name": "Coffee",
     "description": "Perennial Coffea shrub plots, incl. shade-grown smallholder coffee."},
    {"id": 1, "name": "Trees", "description": null}
  ],
  "nodata_value": 255,

  // regression only: what we are regressing (name + optional description), plus range/dtype
  "regression": {
    "name": "canopy_height",
    "description": "Top-of-canopy height from fused GEDI lidar + Sentinel-2 (Lang et al. 2023).",
    "unit": "meters", "dtype": "float32",
    "value_range": [0.0, 61.2], "nodata_value": -99999,
    "buckets": [0, 5, 10, 20, 40]          // optional, if bucket-balanced
  },

  "num_samples": 4231,
  "notes": "..."
}
```
Use `classes` for classification and `regression` for regression (never both). In
`classes`, `description` is optional — populate it with a detailed class definition when
the source dataset provides one (legend notes, codebook, taxonomy description); set to
`null`/omit when unavailable. In `regression`, `name` is required (a short identifier for
the regressed quantity, e.g. `canopy_height`, `soil_organic_carbon`, `population_density`)
and `description` is optional (a fuller definition of the quantity and how it was measured).

`datasets/{DATASET}/locations/{SAMPLE_ID}.json` (per-sample; **GeoTIFF datasets only** —
point-only datasets put this info in the `points.geojson` feature `properties` instead, §2a):
```json
{
  "crs": "EPSG:32737",
  "pixel_bounds": [29696, -964608, 29760, -964544],   // integer px in crs, matches tif
  "time_range": ["2024-01-01T00:00:00+00:00", "2024-12-31T00:00:00+00:00"],
  "change_time": null,                 // ISO time if a change/event label (see §5)
  "pre_time_range": null,              // change datasets only: 6-mo "before" window (see §5)
  "post_time_range": null,             // change datasets only: 6-mo "after" window (see §5)
  "source_id": "sample_951",           // provenance back to source record, optional
  "classes_present": [0, 3]            // optional convenience for classification
}
```
Non-change labels set a single `time_range` (must be ≤ 1 year / 360 days) with
`pre_time_range`/`post_time_range` null. **Change/event labels (§5) instead set
`pre_time_range` and `post_time_range` (each ≤ 183 days) and leave `time_range` null** — the
two windows can be far apart, so no single ≤360-day span represents the sample; `change_time`
is retained as the reference event time. The GeoTIFF carries its own georeferencing;
`crs`/`pixel_bounds` are duplicated in JSON for convenience.

## 4. Processing recipes by `label_type`

- **points — sparse point segmentation** (the point carries a class, possibly a
  background/negative class; e.g. crop-type points, land-cover reference points, tree
  genus): **write to the GeoJSON point table (§2a), not GeoTIFFs.** One `Point` feature per
  point: geometry `[lon, lat]`, `properties.label=class_id`, `time_range`, … Subject to §5
  balancing. Same for **regression points** (canopy height / soil / biomass at a point): one
  feature per point with `properties.label=<value>`.
- **points — object detection, positive-only** (the point marks presence, absence is
  everywhere else; e.g. vessels, turbines, dams-as-points, mines-as-points). **Two cases —
  pick by where the negatives come from:**
  - **(a) Global point inventory** (an isolated list of object coordinates — turbine/dam/
    platform/mine/volcano databases — with NO real annotated scene): emit each detection as a
    **presence-only point** in the dataset-wide `points.geojson` (§2a): one `Point` feature,
    `label` = object class id (multi-class where the source distinguishes types), a static
    1-year `time_range`, `change_time` = null. Do **NOT** fabricate a background/buffer or
    synthetic negative tiles — there is no real observed absence to encode; the assembly step
    supplies negatives by sampling other datasets (§5). This is the **default** for object
    point inventories.
  - **(b) Exhaustively-searched real scene** (the detections were annotated within an actual
    image/window that was searched end-to-end, so object-free area is a *genuine observed*
    negative — e.g. SAR/optical vessel scenes, xView3, annotated marine-infrastructure or
    wind-turbine windows): keep the **tunable detection encoding** — a positive square
    (default 1×1, or object-sized) at the detection, a **nodata (255) buffer ring** around it,
    **background (0)** filling the rest of a context tile (default 32×32, ≤64). **Buffer ≥10 px**
    (default `buffer_size=10`): coordinates are rarely pixel-exact, so a thick ignore ring
    avoids penalizing a few-pixel offset (positive_size=1, buffer_size=10 → a 21×21 ignore
    region, still ample background in a 32×32/64×64 tile). Expose `positive_size`,
    `buffer_size`, `tile_size`; also emit background-only negative tiles drawn from the same
    exhaustively-searched scenes so the class has real negatives. Use `encode_detection_tile`.
- **polygons**: rasterize each polygon (or sampled sub-windows for large/dense coverage)
  into a ≤64×64 UTM tile at 10 m via `rasterio.features.rasterize` (transform built from
  pixel bounds + resolution; see seagrass ref). Value = class ID; outside-polygon =
  background or nodata per dataset semantics. Balance per §5.
- **dense_raster**: crop ≤64×64 windows from the source raster (reproject to UTM 10 m if
  needed). Use **tiles-per-class balanced** sampling. For **derived-product maps**, prefer
  spatially-homogeneous / high-confidence windows (e.g. windows where the class occupies a
  strong majority, or that pass the product's confidence layer).
  - **VHR-native labels** (sub-metre / aerial, e.g. Maldives at 0.35 m, OpenEarthMap,
    LoveDA, FLAIR): **resample the label to 10 m** and **tile** the (often >640 m) source
    into ≤64×64 patches. Use **mode/nearest** resampling for categorical labels (never
    bilinear) — `read_label_raster` with a 10 m projection reprojects via WarpedVRT, so
    pass a nearest/mode resampling. Before committing effort, judge suitability: some fine
    VHR classes (individual buildings, narrow roads, fine coral/seagrass zonation) may be
    unresolvable at 10 m — reject or coarsen the class set and note it.
- **bboxes / oriented boxes**: treat like detection (rasterize the box footprint as
  positive, buffer, background).
- **lines** (roads, glacier fronts, faults, forest roads): rasterize the line to a mask
  (small dilation so it's visible at 10 m); reject if the feature is not observable at
  10–30 m.
- **scene-level** (e.g. EuroSAT): emit a small uniform-class tile only if it is genuinely
  a coherent land-cover patch; otherwise reject as patch-classification, not segmentation.

## 5. Sampling, time range, and change labels

**Handled at pretraining-assembly time — do NOT special-case per dataset.** Two concerns
are resolved downstream when the single combined pretraining dataset is assembled from all
these labels, so per-dataset agents should not work around them:
- **Positive-only / no-background datasets.** Some reference datasets have only foreground
  classes and no background/negative class (presence points like hillforts/species; or
  foreground-type masks like rock-glacier active/transitional/relict, glacier zones). Do
  **not** fabricate synthetic negatives for these — leave non-object pixels as nodata/ignore
  (255) and record every real class. The assembly step gives them negatives by sampling an
  equal number of locations from *other* datasets. This now includes **object point
  inventories** (turbines/dams/platforms/mines/volcanoes/…), which are emitted as
  presence-only points with no fabricated negatives (§4 case (a)). The **only exception** is
  **exhaustively-searched real-scene detection** (§4 case (b): vessel/xView3/annotated
  infrastructure windows), which still emits its own `background`(0) + negative tiles because
  those negatives are *genuinely observed* within the searched scene.
  - **Assembly-time grouping (`assemble_classes.py`).** A dataset joins the shared
    presence-only training group **only if** it is foreground-only (no negative/background
    class) **and** has few classes (`PRESENCE_ONLY_MAX_CLASSES`, currently ≤3). Everything
    else — many-class rich classifications (crop-type, land-cover, species, ecosystem, vessel-
    type, commodity, mine-marker, …) and any dataset carrying its own negative class — becomes
    its **own standalone multiclass training group** (self-contained softmax; no background is
    fine). This keeps the small foreground detectors (dams, turbines, platforms, roads, …)
    pooled while letting rich maps train by themselves, and it stops crop/water/land-cover
    datasets from wrongly negating each other in the pool.
  - **Concept merge/conflict for the pool
    (`data/open_set_segmentation_data/presence_only_concepts.json`).** Pooled classes that
    denote the *same* real-world thing across datasets are **merged** into one global class
    (e.g. the wind-turbine detectors; the two road datasets). Classes whose concepts **overlap
    / subsume** but aren't identical are flagged as mutual **conflicts** (e.g. individual
    wind_turbine ↔ wind_farm; offshore oil/gas platform ↔ gas flare; mining ↔ tailings; a
    generic rock_glacier ↔ its active/transitional/relict states) and excluded as each other's
    negatives; disjoint concepts (wind turbine vs oil platform; maize vs wheat) stay normal
    negatives. `assemble_classes` emits, in the `__presence_only__` group of
    `class_mapping.json`, a `concepts` map and a `conflicts` adjacency per pooled global id.
    Curate the concept file to add clusters; unmatched entries are surfaced by the assembler.
- **Rare classes.** The assembly step discards classes with fewer than a minimum sample
  count when building the final dataset. So do **not** reject a dataset, or drop a class,
  merely because some classes are sparse (even single-sample classes) — keep every class you
  can, still honoring the 254-class uint8 cap and the top-254-by-frequency rule below. Note
  sparse classes in the summary; downstream filtering removes the too-small ones.

- **Per-dataset total cap: 25,000 samples (hard).** No dataset may exceed 25k label
  samples. `sampling.MAX_SAMPLES_PER_DATASET = 25000`. (`geolifeclef_geoplant`, 50,800,
  predates this cap and is grandfathered — do not use it as a precedent.)
- **Classification counts**: up to **1000 locations per class**, tiles-per-class balanced,
  **subject to the 25k total** — when `n_classes × 1000 > 25000`, lower the per-class limit
  to `25000 // n_classes` (this is what `balance_by_class(..., total_cap=25000)` does by
  default). Prioritize rare classes so they reach the (possibly reduced) target; log any
  truncation.
- **Large taxonomies & the 254-class cap**: classification labels are **uint8** (ids
  0–254, 255=nodata), so a dataset can have at most **254 classes**. When a source has more
  (tree species, GlobalGeoTree ~21k, EuroCrops HCAT ~175 is fine, GeoLifeCLEF ~10k),
  **keep the top 254 classes by frequency** (ids 0–253 in descending frequency), drop the
  rest, and record the dropped count in the summary. (Do not switch to uint16.)
- **Multi-target / mixed-modality sources** (e.g. solar polygons + wind-turbine points;
  buildings + damage levels): **combine into ONE dataset with a unified class scheme**
  (e.g. background / solar_pv / wind_turbine), not separate per-target datasets. Segmentation
  and detection targets can coexist in one class map; document the scheme in the summary.
- **Regression counts**: up to **5000 locations per dataset** (well under the 25k cap).
  Bucket-balance across the value range only when the raw distribution is very skewed
  (per-dataset judgment; record buckets in `metadata.json` if used).
- **Source splits**: if the source dataset has train/val/test splits, **use all of them**
  (all windows are fair game as pretraining labels; no split filtering required). You may
  record the source split in the sample JSON `source_id`/notes if convenient, but it is
  not required.
- **Large global derived-product rasters** (WorldPop, JRC TMF, global crop/plantation
  maps, etc.) with no in-situ reference alternative: **sample a bounded set of tiles** —
  download only enough of the product to draw the target count (≤1000/class or ≤5000
  regression) from representative regions. Do not attempt global coverage. Document the
  regions/sampling used in the summary.
- **Time range assignment**:
  - Specific-image / specific-date labels (vessel positions, dated detections): **~1 hour**
    (or the source image's acquisition time) — the label describes one image.
  - Seasonal/annual labels (crop type, land cover, most maps): **1 year**, anchored on the
    labeled year/season. If the valid period is longer than a year, **uniformly sample a
    1-year window** within it.
  - Static labels (geology, lithology, persistent sites): pick a representative 1-year
    window in the Sentinel era (2016+).
- **Change labels** (deforestation events, urban expansion, burn scars with a date,
  bitemporal change-detection pairs, etc.): the label is a **mask of where** a change
  occurred. Emit **two independent six-month observation windows**, `pre_time_range`
  (a "before" window) and `post_time_range` (an "after" window), and **leave `time_range`
  null**; keep `change_time` as the reference event time (may be null for cumulative masks).
  Pretraining pairs a "before" image stack (sampled from `pre_time_range`) with an "after"
  stack (from `post_time_range`) and probes on their difference. Build the windows with
  `io.pre_post_time_ranges(change_time, gap_days=, pre_offset_days=)` and pass
  `pre_time_range`/`post_time_range` to `io.write_sample_json` / the point dict:
  - **The two windows need NOT be adjacent.** Place them to match what the source actually
    compares, so the change reliably falls *between* them:
    - **Single dated event** (fire ignition, quake, flood, dated alert): default **adjacent**
      split at `change_time` (`gap_days=0`) — a 6-mo "before" and 6-mo "after".
    - **`change_time` is a post-event acquisition** (the event precedes it, e.g. the
      cloud-free post-fire scene): use `pre_offset_days` so the pre window ends before the
      true event (burn scars ~90 d; rapid post-disaster imagery ~45 d).
    - **Two-epoch / multi-year comparison** (bitemporal image pairs, pre-mosaic vs
      post-mosaic years apart): center each window on its own acquisition period
      (season-aligned), e.g. pre ≈ 6 mo around the earlier image, post ≈ 6 mo around the
      later one — they may be several years apart.
    - **Year-resolved or cumulative-span events** (annual GFC loss year; a cumulative
      multi-year disturbance mask): put the ambiguous span **in the gap** — pre in a year
      safely before it, post in a year safely after — so the exact timing no longer matters.
  - Each window must be **≤ 183 days**. Anchor windows in the Sentinel era (post ≥ 2016);
    drop or Landsat-note samples whose windows would fall entirely before it.
  - **This replaces the old "reject if the change date is not resolvable to ~1-2 months"
    rule.** Because the ambiguous span can sit in the gap between the two windows, coarsely-
    timed changes (year-resolved, multi-year pre/post comparisons) are now **usable** — do
    not reject them on timing grounds. (`oscd`, `olmoearth_land_cover_change`, `cam_forestnet`,
    `olmoearth_forest_loss_driver`, and `bark_beetle` were previously rejected on that ground
    and are now reprocessed under this scheme.) Still reject a change dataset only for the
    usual independent reasons (no geocoordinates, all labels pre-2016, not observable at
    10-30 m, etc.).
  - A **persistent post-change state** that stays visible long after the event (a burn scar,
    a completed clear-cut, a filled reservoir) may alternatively be encoded as
    **presence/state classification** with `change_time=null` and a single static-label
    1-year `time_range` (no pre/post) — use this when there is no meaningful "before" to pair,
    and note the reasoning in the summary.

## 6. Shared code module layout

Under `olmoearth_pretrain/open_set_segmentation_data/` (in the repo). **These already
exist and are validated — import and reuse them; extend rather than duplicate.** Run
per-dataset scripts as `python3 -m
olmoearth_pretrain.open_set_segmentation_data.datasets.{slug}` from the repo root (plain
`python3` has rslearn available; `uv run` also works).
- `manifest.py` — `load_manifest()`, `load_registry()`, `slugify(name)`, `find_slug(name)`,
  `get_entry(slug)`, `write_registry_entry(slug, status, task_type=, num_samples=, notes=)`
  (writes per-dataset `registry_entry.json`; `update_status` is a back-compat alias),
  `aggregate_registry()` (orchestrator-only: merge entries into central `registry.json`).
- `io.py` — `check_disk()`, `dataset_dir/locations_dir/raw_dir(slug)`,
  `utm_projection_for_lonlat`, `lonlat_to_utm_pixel`, `centered_bounds`, `year_range`,
  `write_label_geotiff(...)`, `write_sample_json(...)`, `write_dataset_metadata(...)`.
  Constants: `RESOLUTION=10`, `CLASS_NODATA=255`, `REGRESSION_NODATA=-99999`, `MAX_TILE=64`.
- `rslearn_read.py` — `iter_windows_metadata(ds_path)` (fast direct read of window
  metadata.json), `read_label_vector(...)`, `read_label_raster(...)`.
- `rasterize.py` — `geom_to_pixels(geom, src_proj, dst_proj)`,
  `rasterize_shapes(shapes_in_pixel_coords, bounds, fill, dtype, all_touched)`.
- `sampling.py` — `balance_by_class(records, key, per_class, total_cap=25000)` (enforces
  the 25k per-dataset cap by default), `bucket_balance_regression(records, value_key,
  total, n_buckets)`, `encode_detection_tile(positives, tile_size, positive_size=1,
  buffer_size=10, ...)`, constant `MAX_SAMPLES_PER_DATASET=25000`.
- `download.py` — `download_http`, `download_s3_unsigned`, `download_zenodo`, `hf_download`
  (all atomic).
- `datasets/{slug}.py` — per-dataset script: a `main()` that produces all outputs and is
  runnable/idempotent (skip already-written `{sample_id}.tif`).
  `datasets/olmoearth_lcmap_land_use.py` is a complete worked example (sparse points →
  `points.geojson` via `io.write_points_table`).

**Performance — this matters:** weka small-file I/O is ~70 ms/file cold. Reading tens of
thousands of source windows or writing thousands of patches **serially is unacceptably
slow** (30+ min). Always use a `multiprocessing.Pool(64)` (with
`rslearn.utils.mp.star_imap_unordered`) for both the scan and the write phases, as the
lcmap example does (26k windows scanned in ~18 s; 5.6k patches written in ~60 s).

## 7. Reusable references (read these; do not reinvent)

- **Download → 10 m GeoTIFF** (S3 + rasterio + `GeotiffRasterFormat`, mp.Pool, atomic
  rename): `olmoearth_pretrain/dataset_creation/wri_canopy_height_map/download_wri_canopy_height_map.py`.
- **Static single-band label conversion + metadata pattern**:
  `olmoearth_pretrain/dataset_creation/rslearn_to_olmoearth/cdl.py`.
- **Polygon rasterization aligned to a UTM window** (`rasterize_polygon`,
  `create_polygon_window`): `rslearn_projects/rslp/seagrass/create_test_windows.py`;
  also `rslearn_to_olmoearth/eurocrops.py` (`draw_polygon`).
- **rslearn read API + on-disk format**: `Dataset(UPath(path))`, `ds.load_windows(...)`,
  `Window` (`.projection`, `.bounds`, `.time_range`, `.options`), read via
  `GeotiffRasterFormat().decode_raster(dir, projection, bounds)` and
  `GeojsonVectorFormat().decode_vector(path, projection, bounds)` → `list[Feature]`
  (`feature.geometry.shp`, `feature.properties`). Core lib: `rslearn/rslearn/`
  (`dataset/`, `utils/geometry.py`, `utils/feature.py`, `utils/vector_format.py`,
  `utils/raster_format.py`). Local datasets: on-disk
  `windows/{group}/{name}/{metadata.json, items.json, layers/{label|label_raster}/...}`.
- **Class-extraction semantics**: `rslearn/train/tasks/classification.py` and
  `detection.py` (`property_name`, `classes`, `filters`, `read_class_id`).

## 8. Per-dataset agent workflow (SOP)

0. **Preconditions.** Check disk (§0: ≥5 TB free on weka, else stop). Record
   `in_progress` via `manifest.write_registry_entry(slug, "in_progress")` (writes your
   `datasets/{slug}/registry_entry.json`; do NOT touch the central registry).
1. **Read the manifest entry** and, if `have_locally`, inspect the source rslearn dataset
   (`config.json`, a few windows, label layer property names). If external, identify the
   download mechanism from `source`/`url`.
2. **Triage (accept/reject).** Reject and write only `dataset_summaries/{DATASET}.md`
   (with reason) if any of:
   - Not accessible: needs an account/credential or interactive auth we don't have
     (Kaggle, DrivenData, Earth Engine, registration portals), dead link, or license
     forbids use. **FIRST check `.env` for a usable credential before
     rejecting** — it holds authorized project credentials the user has approved for this
     work: `NASA_EARTHDATA_USERNAME`/`NASA_EARTHDATA_PASSWORD` (NASA URS → ORNL DAAC, OB.DAAC,
     LP DAAC, etc.), `COPERNICUS_USERNAME`/`PASSWORD` (Copernicus Data Space), `CDSAPI_KEY`
     (Climate Data Store), `M2M_USERNAME`/`M2M_TOKEN` and `TEST_USGS_LANDSAT_*` (USGS
     EarthExplorer/M2M), `PL_API_KEY` (Planet), and GEE service-account creds. For NASA
     Earthdata, source those two vars and write a `~/.netrc` (`machine urs.earthdata.nasa.gov
     login <user> password <pass>`, chmod 600) so URS OAuth redirects authenticate. **Do NOT
     use credentials found elsewhere** (e.g. other users' `~/.netrc` or tokens under other
     home dirs on weka) — only `.env`. If `.env` has no matching
     credential and unauthenticated/mirror/alternate access fails, try briefly then **reject
     with `notes: "needs-credential: <what>"`** so it collects in the registry for the user to
     act on later (provide creds or a pre-downloaded copy). Credentials NOT in `.env` (so
     still rejections): NEON API token, ISMN portal login, HuggingFace gated-repo access,
     Zenodo restricted-access approvals, and per-dataset registration portals (xView3, etc.).
     **Exception — transient source/infra errors** (server HTTP 5xx, timeout, rate-limit,
     temporarily-empty GeoServer) on an otherwise-usable, no-credential source: do NOT mark
     `rejected`. Record **`temporary_failure`** (see §1a) with the concrete error and retry
     steps in `notes`, so it is retried later rather than treated as a permanent drop.
   - A **preferred reference alternative is in the manifest** and this is the paired map
     product → defer to the reference (note the pairing).
   - Phenomenon not observable at 10–30 m from S2/S1/Landsat (individual small trees,
     VHR-only wildlife, coordinate-fuzzed points like FIA ~1 mi), and no aggregate/mask
     representation salvages it.
   - **Pre-2016 labels: reject if ALL labels fall before 2016** (outside the Sentinel era,
     with no usable post-2016 window). If the dataset is a **mix** of pre- and post-2016
     labels, keep only the post-2016 subset and process normally (filter the rest out); only
     a dataset whose labels are *entirely* pre-2016 is rejected on this ground. Note the
     cutoff/filtering in the summary. (Landsat-era-only labels are still rejected under this
     rule — we anchor to the Sentinel era.)
   - Label semantics can't be expressed as per-pixel classification or regression.
   - **No recoverable geocoordinates.** Many "ML-ready" patch/tensor releases (PNG tiles,
     `.npy` stacks, anonymized CSVs) strip lon/lat, so labels can't be placed on the S2
     grid — this is a common, fast rejection. **Check georeferencing cheaply first**
     (inspect the archive file listing / datasheet / a sample file's CRS) **before**
     downloading multi-GB archives. A per-sample tile/region id alone (e.g. an MGRS tile
     without within-tile pixel index) is not sufficient. If unrecoverable, reject.
   - ~~Change/event label whose change date is not known to within ~1-2 months~~ — **NO
     LONGER a rejection reason.** The pre/post two-window scheme (§5) places coarsely-timed
     changes (year-resolved, multi-year pre/post comparisons) in the gap between a "before"
     and an "after" window, so they are now usable. Process such datasets under §5 (do not
     reject on timing). Only reject a change dataset for an independent reason (no
     geocoordinates, all pre-2016, not observable at 10-30 m, etc.).
   - **Impractical download volume for the label signal.** Only the *labels* are needed —
     pretraining supplies its own imagery. If the labels are only distributed inside very
     large bulk archives (e.g. whole-region/full-planet OSM extracts, multi-TB SAR scene
     sets) and you'd have to pull tens of GB (or many such archives) to extract a thin label
     layer, do NOT bulk-download. Try: (a) a lighter label-only/metadata endpoint, (b) a
     smaller bounded set of regions/tiles (§5), or (c) range-request/selective extraction of
     just the needed files. If none is feasible, **reject** with `notes: "impractical-download:
     <what>"` (retain any partial download for a future re-scope) rather than spending hours
     pulling data. (openstreetmap_leisure_tourism_extracts was rejected on this ground after a
     ~14 GB whole-region OSM download loop produced no labels.)
   Otherwise **accept**; classify as classification vs regression.
3. **Download** raw to `raw/{DATASET}/` (or write `SOURCE.txt` for local). Download only what
   the labels require; prefer selective/range extraction over pulling whole bulk archives.
4. **Analyze**: enumerate classes / value range, CRS/resolution, per-record time, class
   distribution; capture any per-class descriptions from the source (for `metadata.json`);
   decide tile size, detection parameters, buckets, time-range rule.
5. **Write the per-dataset script** using shared utils; produce `metadata.json`,
   `locations/{SAMPLE_ID}.tif` + `.json`, applying §2–§5.
6. **Verify** (§9).
7. **Write** `dataset_summaries/{DATASET}.md`: source, access method, class/label mapping,
   time-range and change handling, tile size, sample counts per class (or regression
   stats), rejections/caveats, and the exact command to reproduce.
8. **Record final status** via `manifest.write_registry_entry(slug, "completed",
   task_type=..., num_samples=...)` or `(slug, "rejected", notes="...")` — writes your
   `datasets/{slug}/registry_entry.json`. Never write the central `registry.json`; the
   orchestrator aggregates.

## 9. Verification

- Open 3–5 output `.tif`s: confirm single band, correct dtype, UTM CRS at 10 m, size ≤64,
  and that pixel values are valid class IDs (or in the regression range) with the declared
  nodata.
- Confirm every `.tif` has a matching `.json` with a ≤1-year `time_range` (and
  `change_time` set for change datasets), and that `metadata.json` class IDs cover all
  values appearing in the tifs.
- Report class-balance counts (≤1000/class) or regression histogram (≤5000 samples).
- **Spatial/temporal sanity check**: for 1–2 samples, load a Sentinel-2 image at the tile's
  CRS/bounds/time (via rslearn) and eyeball that the label overlays sensibly (e.g. a
  "water" label sits on water). Note any misalignment in the summary.
- Re-running the script must be idempotent (skip existing outputs).

---

## Critical files (for the implementer of the shared module)

- Manifest: `olmoearth_pretrain/data/open_set_segmentation_datasets.json`.
- New shared code: `olmoearth_pretrain/open_set_segmentation_data/` (§6).
- New summaries: `data/open_set_segmentation_data/dataset_summaries/`.
- This task spec: `data/open_set_segmentation_data/AGENT_SUMMARY.md`.
- Templates to mirror: `dataset_creation/wri_canopy_height_map/download_wri_canopy_height_map.py`,
  `dataset_creation/rslearn_to_olmoearth/{cdl.py,eurocrops.py}`,
  `rslearn_projects/rslp/seagrass/create_test_windows.py`.

## Suggested build order

1. Land this spec + the shared module skeleton (`io.py`, `rasterize.py`, `sampling.py`,
   `rslearn_read.py`, `download.py`, `manifest.py`) with a first easy dataset (e.g. a local
   points dataset like `ethiopia_crops`) to exercise the classification path end-to-end.
2. Add one regression dataset (e.g. ETH Global Canopy Height) to exercise the regression
   path and float dtype.
3. Add one detection dataset (e.g. Sentinel-2 vessels) to exercise detection encoding.
4. Then fan out one agent per remaining dataset, each loading this doc + its manifest entry.
