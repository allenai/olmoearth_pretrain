# Statoil/C-CORE Iceberg Classifier

- **slug**: `statoil_c_core_iceberg_classifier`
- **status**: **rejected** — two independent, fundamental grounds:
  1. **No recoverable geocoordinates** (SOP §8.2). The released `train.json`/`test.json`
     are anonymized SAR patches: each record has only `id`, `band_1`/`band_2`
     (75×75 flattened dB arrays), `inc_angle`, and `is_iceberg`. There is **no lon/lat**,
     no acquisition timestamp, and no scene identifier that maps to a geolocation, so the
     ship/iceberg labels cannot be placed on the Sentinel-2 grid.
  2. **Needs-credential** (SOP §8.2). Kaggle competition data requires accepting the
     competition rules with a Kaggle account + API credentials. No `KAGGLE_*` credential
     exists in `.env`.
  Both are permanent, not retry candidates. The georeferencing failure is primary: even
  with Kaggle access granted, the anonymized patches remain ungeoreferenceable.
- **task_type** (intended, had it been usable): detection/points — ship vs iceberg point
  targets on Sentinel-1 SAR off Canada's east coast.
- **num_samples**: 0

## Source

- Manifest name: `Statoil/C-CORE Iceberg Classifier` (source Kaggle, family `snow_ice`,
  label_type `points/patches`, region "East coast of Canada", classes `[ship, iceberg]`,
  time_range `[2016, 2017]`, license "Kaggle competition", have_locally: false).
- Competition: <https://www.kaggle.com/c/statoil-iceberg-classifier-challenge>.
- Content: a balanced set of 75×75 pixel Sentinel-1 SAR patches (HH + HV polarizations)
  with per-patch incidence angle, hand-labeled by C-CORE GIS specialists as ship (0) vs
  iceberg (1). Training set is 1,604 patches (753 iceberg / 851 ship).

## What the released data actually is

Verified cheaply from the public competition data description (mirrored in numerous public
repos, e.g. github.com/HankyuJang/Statoil-C-CORE-Iceberg-Classifier-Challenge) **without
downloading via Kaggle**. Each JSON record contains exactly:

- `id` — an **anonymized** image id (not a coordinate or resolvable scene reference).
- `band_1`, `band_2` — 5,625 floats each = a 75×75 grid of radar backscatter in dB, for HH
  and HV polarizations.
- `inc_angle` — incidence angle of the acquisition (some marked `"na"`).
- `is_iceberg` — target, 1 = iceberg, 0 = ship (train only).

That is the entire schema. No latitude/longitude, no CRS/geotransform, no date, and no
tile/scene id from which a location could be recovered.

## Why rejected

### Georeferencing (SOP §8.2 — primary, fundamental)

The patches are deliberately anonymized for the competition. A per-patch `id` with no
within-scene pixel index and no scene geolocation is not a sufficient geolocation (§8.2:
"A per-sample tile/region id alone ... is not sufficient"). Because the labels cannot be
co-located with pretraining imagery by geography, this is a **fundamental `rejected`**, not
`temporary_failure`. The 75×75 patch could in principle be tiled/resampled, but with no
coordinates there is nowhere to place it on the S2 grid.

### Access gate (SOP §8.2 — secondary, needs-credential)

Kaggle competition data is gated behind accepting the competition rules with a Kaggle
account and API token. Kaggle is explicitly listed in §8.2 as an account/credential gate we
do not have, and `.env` holds no Kaggle credential (only AWS,
Copernicus, CDS, USGS M2M, NASA Earthdata, Planet, and GEE creds). Under §1a, missing
credentials use `rejected` with `notes: "needs-credential: ..."`. This ground alone would
justify rejection, but is moot given the georeferencing failure.

## Judgment calls

- **Rejected, not `temporary_failure`.** Neither block is a transient source/infra error;
  both are permanent (anonymization is intrinsic to the release; the Kaggle gate is a
  standing access requirement).
- **Did not download.** Per §8.2, georeferencing is checked cheaply first; the public data
  description already establishes there are no coordinates, so pulling the archive (which
  would also require Kaggle auth) would add nothing.
- If a **coordinate-bearing** version of these detections were obtained (original C-CORE
  SAR scenes with the ship/iceberg pick locations and acquisition times), this would be a
  good Sentinel-1 detection dataset (ship vs iceberg point targets, ~1-hour time range per
  acquisition, detection encoding per §4) and should be reconsidered. Absent coordinates it
  is unusable here. Note: the manifest already carries related, georeferenced iceberg
  sources (e.g. `circum_antarctic_icebergs_sentinel_1`) that cover the iceberg class.

## Reproduce

No outputs were written to weka `datasets/statoil_c_core_iceberg_classifier/` beyond
`registry_entry.json`. To revisit: obtain a georeferenced source of these ship/iceberg
detections (lon/lat + acquisition date per detection) — the anonymized Kaggle
`train.json`/`test.json` cannot be geolocated regardless of Kaggle access.
