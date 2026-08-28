# Oil Spill Detection Dataset (Krestenitis / M4D) — REJECTED

- **Slug**: `oil_spill_detection_dataset_krestenitis_m4d`
- **Name**: Oil Spill Detection Dataset (Krestenitis / M4D)
- **Source**: M4D-ITI / MDPI — https://m4d.iti.gr/oil-spill-detection-dataset/
- **Reference**: Krestenitis et al. 2019, "Oil Spill Identification from Satellite
  Images Using Deep Neural Networks", *Remote Sensing* 11(15):1762.
- **Family / region**: marine / Global (EMSA CleanSeaNet events, Sep 2015 – Oct 2017)
- **Label type (manifest)**: dense_raster, 5 classes
  (oil spill, look-alike, ship, land, sea surface)
- **Status**: **REJECTED**
- **Reason**: No recoverable geocoordinates (coordinate-free JPG/PNG image tiles);
  secondary blocker: access gated behind an institutional request/agreement.

## What the dataset is

A Sentinel-1 SAR oil-spill segmentation benchmark: ~1000 training + ~110 test images at
10 m nominal spatial resolution, each with an expert-annotated pixel mask over five
semantic classes. The underlying SAR scenes were selected from ESA/Copernicus using EMSA
CleanSeaNet confirmed-spill coordinates and timing, processed in SNAP, then **exported as
JPG image chips with color-coded PNG ground-truth masks** (total ~400 MB).

## Why it is rejected (checked cheaply, before any download — per spec §8.2)

1. **No recoverable geocoordinates (fundamental).** The public release ships the labels as
   **coordinate-free JPG images + PNG masks** — the extracted chips carry no CRS, no
   geotransform, and no per-image lon/lat mapping. The EMSA/CleanSeaNet coordinates that
   drove scene selection are *not* published alongside the chips in any usable per-image
   index. Without recoverable lon/lat the pixel masks cannot be placed on the S2/UTM grid,
   so they cannot be co-located with pretraining imagery. This matches the spec's flagged
   common caveat for ML-ready SAR oil-spill sets ("many are coordinate-free PNG/tensor
   releases; if no recoverable lon/lat, reject fast"). This is a permanent blocker: even if
   the archive were obtained, the labels remain unplaceable.
2. **Access gate (secondary).** Download is not open — it requires reading a Terms-of-Use
   document, preparing a project title/abstract, and submitting a request via an official
   institutional email template (student requests must come from a supervisor). No
   unauthenticated / mirror path exposes the georeferenced originals; the Kaggle re-uploads
   found are the same coordinate-free chips (or unrelated tabular oil-spill sets).

Because the primary blocker (no recoverable georeferencing) is permanent and would remain
even after the access request were granted, this is a `rejected` (fundamental) rather than
a `temporary_failure` or a pure `needs-credential` case. No raw data was downloaded and no
`datasets/` outputs were written (only this summary and the `registry_entry.json`).

## If this is ever revisited

The only path to salvage would be to obtain, from the M4D authors, the **original
georeferenced Sentinel-1 GRD scene id + subset window (lon/lat or UTM bounds)** for each
chip — i.e. a per-image coordinate/footprint table that the current release omits. With
that mapping the masks could be reprojected to UTM 10 m and tiled as a normal
`dense_raster` classification dataset (background/sea, oil spill, look-alike, ship, land).
Absent that table, the dataset is unusable for geo-co-located pretraining.

## Reproduce

No processing script. Determination was made from the source dataset page
(https://m4d.iti.gr/oil-spill-detection-dataset/) and the dataset publication, which
document the JPG-chip + PNG-mask, coordinate-free format and the request-based access.
