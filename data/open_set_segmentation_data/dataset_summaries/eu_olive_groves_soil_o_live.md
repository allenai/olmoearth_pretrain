# EU Olive Groves (Soil O-live) — `eu_olive_groves_soil_o_live`

**Status: REJECTED — needs-credential (access-restricted Zenodo record).**

## What the source is

- Manifest name: `EU Olive Groves (Soil O-live)`
- Source: Zenodo record **14748127** — "LPIS/GSAA of olive groves in the EU"
  (https://zenodo.org/records/14748127, DOI 10.5281/zenodo.14748127).
- Description: The Horizon Europe **Soil O-live** project (https://soilolive.eu/)
  harmonized national LPIS/GSAA georeferenced agricultural-parcel data into a pan-European
  olive-grove parcel dataset, provided as **shapefiles** for seven countries
  (Croatia, France, Greece, Italy, Portugal, Slovenia, Spain), for years **2021, 2022, 2023**.
- Label type: `polygons`; single class `olive grove`; family `plantation`.
- License (metadata): **CC-BY-4.0**.

This dataset would have been a good fit: a large single-class polygon (tree-crop) dataset,
post-2016, clearly observable at 10 m, expressible as per-pixel classification. Intended
processing would have mirrored `eurocrops.py` — rasterize each parcel polygon into a
≤64×64 UTM 10 m tile (class id 0 = olive grove inside the polygon, 255 = nodata/ignore
outside), tiles-per-class balanced under the 25k cap, 1-year time window anchored on each
country/year snapshot. It could not be executed because the files are not accessible.

## Why rejected (access gate, not a data problem)

The Zenodo record is **access-restricted**:

- `metadata.access_right == "restricted"` on the record and on its parent (14748125);
  only one version exists, also restricted.
- The record's `files` array is **empty** in the public API response.
- `GET /api/records/14748127/files` → **HTTP 403 `{"status":403,"message":"Permission denied."}`**
- `GET /api/records/14748127/files-archive` → **HTTP 403 FORBIDDEN**
- The record HTML page renders as "Restricted" and exposes an `access_request` link
  (`/api/records/14748127/access/request`).

Obtaining the files requires a **Zenodo login and owner approval** through the access-request
workflow — a permanent credential/authorization gate we do not have. This is not a transient
server error (the endpoints deterministically return 403 Permission denied), so it is
`rejected` with `needs-credential`, not `temporary_failure`.

No open alternative version or mirror was found on Zenodo. (Note: although the stated license
is CC-BY-4.0, the files themselves are gated behind restricted access.)

## How to recover / reproduce

1. Log in to Zenodo and use the "Request access" button on
   https://zenodo.org/records/14748127 ; wait for the record owner (Soil O-live project) to
   grant access, or obtain a shared access link / pre-downloaded copy out of band.
2. Place the country shapefiles under
   `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/eu_olive_groves_soil_o_live/`.
3. Then a per-dataset script modeled on `datasets/eurocrops.py` (single class, `all_touched`
   polygon rasterization into ≤64×64 UTM 10 m tiles, per-country year for the 1-year time
   range, 25k cap) would complete it. Because there is only one class, tiles-per-class
   balancing reduces to a random ≤1000-tile draw (or up to the 25k cap); note per the spec no
   synthetic negatives should be fabricated — outside-polygon pixels stay nodata (255).

## Outputs written

- `datasets/eu_olive_groves_soil_o_live/registry_entry.json` — status `rejected`,
  notes `needs-credential: ...` (the only file written to weka `datasets/`).
- This summary.
No `metadata.json`, label tiles, or `points.geojson` were written (nothing accessible to process).

## Verification / checks performed

- Zenodo API record fetch: `access_right = restricted`, empty `files`.
- `/files` and `/files-archive` endpoints: HTTP 403 Permission denied.
- Versions list: single restricted version; parent record also restricted.
- Disk precondition at start: ~34.7 TB free on `/weka/dfive-default` (≥ 5 TB OK).
