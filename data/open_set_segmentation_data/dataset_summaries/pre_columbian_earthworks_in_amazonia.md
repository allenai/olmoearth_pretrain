# Pre-Columbian Earthworks in Amazonia

- **Slug**: `pre_columbian_earthworks_in_amazonia`
- **Status**: completed
- **Task type**: classification (sparse presence points, spec §2a)
- **Num samples**: 961 (single class)

## Source

Peripato et al., "More than 10,000 pre-Columbian earthworks are still hidden throughout
Amazonia", *Science* **382**, 103-109 (2023), doi:10.1126/science.ade2541. Data/code
release on Zenodo record **10214943** (doi:10.5281/zenodo.7750985), repo `Vperipato/ade2541`.

Access: unauthenticated HTTPS download of the release zip
`Vperipato/ade2541-v1.0.0.zip` (~20 MB) via the Zenodo files API. Ground truth is
`Database/Earthworks.rds` inside the zip: **961 confirmed, georeferenced pre-Columbian
earthwork sites** with columns `Longitude`, `Latitude`, `Database`. Read with `pyreadr`.
License: Zenodo "other-open" (free to use with citation).

Region: Amazonia (Brazil/Peru/Bolivia/Guianas), lon -76.79..-51.60, lat -13.88..5.41.

## Class mapping

The released ground truth carries **only** the site location and its source `Database`
(provenance: Amazon Arch, PAST, CNSA, INRAP & DAC, TREES/INPE, and multi-source combos) --
it does **not** carry an earthwork sub-type. The manifest's aspirational sub-classes
(geoglyphs / ring ditches / mound villages / ponds-wells / fortifications) are therefore
not recoverable from the data. All sites are collapsed into one presence class:

| id | name | count |
|----|------|-------|
| 0 | pre-Columbian earthwork | 961 |

Presence-only: no background/negative class is fabricated (spec §5); the assembly step
supplies negatives from other datasets. `source_id` records `"<Database> #<row>"` for
provenance back to the source archive.

## Observability decision (spec §8) — MIXED, accepted with caveat

Amazonian earthworks span a wide observability range. Many are small and/or lie under
closed forest canopy and are only detectable in LiDAR/VHR (e.g. the TREES/INPE
LiDAR-newly-detected sites). However, a large fraction of the confirmed set are the big
**deforested ditched enclosures ("geoglyphs")/ring ditches** of Acre, Bolivia and Peru
that are 100-300 m across and plainly resolvable in Sentinel-2/Landsat — the manifest
itself notes "Geoglyphs/ring ditches 100-300 m; discernible at 10-30 m", and the sampled
points fall in the well-documented Acre/Bolivia geoglyph belt (~-67 lon, -10 lat).

Per the task's explicit guidance ("if some large, cleared earthworks are resolvable,
treat as presence detection with judgment") the dataset is **kept as a weak
single-phenomenon presence label**. Caveat: some positives (under-canopy / LiDAR-only
sites) will not be visible in 10-30 m imagery, so this is a noisy-positive set. Recorded
in `metadata.json` notes.

## Time range

Persistent/static heritage sites -> a fixed representative 1-year Sentinel-era window
(2020-01-01 .. 2021-01-01). No change labels.

## What was NOT used

The record also ships IPP-model probability rasters (`IPPModel_EarthworkProb-linear.tif`,
`-log10.tif`) at **1 km** resolution. These are a *model prediction* (a derived product),
not reference ground truth, and 1 km is far coarser than the 10 m label grid, so they are
not used. Reference points are preferred over derived maps (spec design decisions).

## Sampling

Single class, 961 sites < the 1000/class cap, so all sites are kept (`balance_by_class`,
per_class=1000). 6 exact-duplicate coordinates exist in the source and are retained
(harmless for point labels). No pre-2016 filtering needed (labels are static heritage).

## Verification

- `points.geojson`: 961 `Point` features, task_type=classification, all labels = 0, all
  coordinates within the Amazonia bounding box, uniform 1-year `time_range`, `change_time`
  null.
- `metadata.json`: single class 0 covering all label values; nodata 255.
- Spatial sanity: sampled coordinates (e.g. -67.5, -10.0) land in the Acre/Bolivia
  geoglyph belt, consistent with the source.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.pre_columbian_earthworks_in_amazonia
```
Idempotent: the zip download and RDS extraction skip if present; `points.geojson` is
rewritten deterministically.
