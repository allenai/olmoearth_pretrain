# Coast Train — processing summary

- **Slug**: `coast_train`
- **Status**: **completed**
- **Task type**: classification (dense per-pixel)
- **Samples written**: **1772** label patches (64×64, single-band uint8, local UTM @ 10 m)
- **Source**: Coast Train, USGS Pacific Coastal and Marine Science Center.
  DOI [10.5066/P91NP87I](https://doi.org/10.5066/P91NP87I); paper
  [Buscombe et al. 2023, *Scientific Data*](https://www.nature.com/articles/s41597-023-01929-2).
- **License**: public domain (U.S. Government work).
- **Region / time**: U.S. Pacific, Gulf, Atlantic and Great-Lakes coasts; scenes 2016–2021.

## What the source is

A 1.2-billion-pixel human-labeled library of coastal imagery + dense per-pixel
land-cover labels, produced with the **Doodler** human-in-the-loop tool. The
release is 10 `{source}_{nclasses}_{version}.zip` archives (five imagery
sources) plus a release-wide `CoastTrain_imagery_details.csv`. Each archive is a
set of NPZ files, one per labeled image. An NPZ holds `label` (one-hot
`H×W×C` uint8; **channel k == class k of its `classes` list**), `classes`
(class-name list), the RGB `image`, doodles, etc. Multi-labeler images store
extra annotations under `00`/`0` key prefixes; we use the no-prefix (primary)
`label`/`classes`.

**Georeferencing** comes from the CSV, not the NPZ: per-image footprint
`XMin/XMax/YMin/YMax` (easting/northing), `epsg` (a projected UTM CRS),
acquisition Y/M/D, and `acc_georef` (~8 m). The doodled raster is a resampled
version of the native scene (e.g. a native ~302×432 px Sentinel-2 chip stretched
to 600×600 for doodling), so the CSV footprint is the authoritative extent; we
build the source affine from it over the actual label-array shape.

## Access method

Public HTTP, no credentials. Files hosted on `cmgds.marine.usgs.gov`
(`.../media/2022/10.5066-P91NP87I/<hash>/<file>`). Downloaded the CSV + six
NPZ archives to `raw/coast_train/`. (The initial `NAIP_11_001.zip` pull was
truncated and failed `unzip -t`; a fresh download to the full 4,295,601,922
bytes passed integrity — a re-run guard worth keeping since the release page
itself warns of occasional file-corruption issues.)

## Records processed vs. skipped

Processed (satellite + aerial at/near 10–30 m):

| record | native res | post-2016 images used |
|---|---|---|
| Sentinel2_11 | 10 m | 340 |
| Sentinel2_4  | 10 m | 103 |
| Landsat8_11  | ~15–30 m | 247 |
| Landsat8_12  | ~15–30 m | 39 |
| NAIP_11      | 1 m → 10 m | 229 |
| NAIP_6       | 1 m → 10 m | 46 |
| **total** | | **1004 images** |

Skipped:
- **Orthophoto_8 / 9 / 12** (UAS orthomosaics ~0.05 m): footprints are only
  ~50–100 m (5–10 px at 10 m); the fine coral/sediment/anthropogenic zonation
  they capture is unresolvable at 10 m — not useful as 10 m tiles.
- **Quadrangles_7** (USGS aerial ~6.8 m): all images are 2008/2012 (pre-Sentinel
  era) → excluded by the ≥2016 filter anyway.
- **72 `Landsat8_11` NPZs** (`klamathregion_*`, incl. Landsat-5 scenes) had **no
  matching CSV row** → no footprint coordinates → cannot be georeferenced, so
  skipped. Most are pre-2016 (L5) and would have been filtered regardless.
- **Per-image ≥2016 filter**: pre-2016 NAIP/Landsat scenes dropped (Sentinel-2 is
  entirely 2017–2021).

## Class scheme (unified)

Coast Train uses many per-record class sets. They are reconciled to the paper's
physical **superclasses**, keeping the six coherent land-cover classes and
folding the non-physical categories into ignore:

| id | name | source classes mapped in |
|---|---|---|
| 0 | water | water, sediment_plume |
| 1 | whitewater | whitewater, surf |
| 2 | sediment | sediment, sand, gravel, cobble_boulder, non-vegetated-wet |
| 3 | development | development, dev, developed, buildings, pavement_road, vehicles, people, coastal_defense, other_anthro |
| 4 | bare_natural_terrain | other_natural_terrain, bare_ground, non-vegetated-dry |
| 5 | vegetation | vegetated_surface, vegetated, vegtated_ground, agricultural, marsh_vegetation, terrestrial_vegetation, herbaceous/woody vegetation |
| 255 | nodata / ignore | nodata, cloud, unknown, unusual, generic "other" |

### Judgment calls (please review)
- **`other`/`cloud`/`unknown`/`unusual` → 255 ignore**, not a real "other" class.
  These are obscuration/uncertainty grab-bags, not a coherent land-cover class,
  so an explicit noise class would hurt pretraining. This differs from the
  manifest's 7-class list (which named a generic `other`).
- **`sediment_plume` → water** (follows the paper: suspended sediment is within
  the Water superclass).
- **`agricultural` → vegetation** (cropland is vegetated; no separate ag class).
- **`non-vegetated-wet` → sediment** (wet intertidal flat = sediment) and
  **`non-vegetated-dry` → bare_natural_terrain**.
- **Live Sentinel-2 pixel overlay not run.** Instead georeferencing was validated
  against the release's own coordinates: tile centers fall inside the CSV
  lon/lat footprints across three sensors/UTM zones, and ~pure-water tiles
  geolocate to open ocean (e.g. off Cape Hatteras, NC). The labels were
  hand-drawn directly on georeferenced satellite imagery, so coordinate
  agreement is a strong spatial check.

## Processing

`label_type = dense_raster`. Each image's unified label raster is reprojected
once to its **local UTM zone at 10 m** with **nearest** resampling (categorical),
then cut into non-overlapping **64×64** tiles. A tile is a candidate if ≤50 %
nodata and it contains ≥32 px of at least one class. Selection is
**tiles-per-class balanced** (spec §5): a tile counts toward every class present
(≥32 px); rarest classes are filled first, up to **1000 tiles/class**
(seed 42). 91,494 candidate tiles → 1772 selected (each tile carries ~4 classes,
so all six classes reach their cap from few tiles). Candidate tiles are sorted by
`(source_id, ti, tj)` before the seeded shuffle so selection and sample-id
assignment are **deterministic/reproducible** across runs (the scan pool returns
tiles unordered).

**Time range**: land-cover labels tied to a dated scene → **1-year window
centered on the acquisition date** (`change_time` = null; these are state, not
change, labels). Caveat: coastal `water`/`whitewater`/`sediment_plume` are
ephemeral relative to a yearly window; the stable classes (development,
vegetation, beach sediment, bare terrain) dominate and are unaffected.

## Class balance (selected tiles)

Tiles containing each class (≥32 px): water 1288, whitewater 1128, sediment
1135, development 1000, bare_natural_terrain 1189, vegetation 1345. All six
classes are at/near the 1000-tile cap — well balanced.

## Verification (spec §9)

- 1772 `.tif` each with a matching `.json`; 0 orphans.
- Spot-checked tiles: single band, uint8, UTM CRS (EPSG:326xx), 64×64, 10 m,
  nodata 255. Dataset-wide pixel values = {0,1,2,3,4,5,255}, exactly the
  declared class ids + nodata.
- All `time_range` spans ≤ 365 days; `change_time` null throughout.
- Georef/spatial sanity as above.
- Idempotent: re-running skips any `locations/{id}.tif` already present.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coast_train --workers 64
```

Raw archives + CSV: `/weka/.../open_set_segmentation/raw/coast_train/`.
Outputs: `/weka/.../open_set_segmentation/datasets/coast_train/`
(`metadata.json`, `locations/{id}.tif|json`).
