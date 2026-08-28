# ISMN (International Soil Moisture Network)

- **slug**: `ismn_international_soil_moisture_network`
- **status**: **rejected** — `needs-credential`
- **task_type** (intended): regression (continuous in-situ soil moisture)
- **num_samples**: 0 (data not obtainable)

## Source

- Manifest name: `ISMN (International Soil Moisture Network)`
- Source: TU Wien; homepage <https://ismn.earth/en/>
- Description: Harmonized, QC'd in-situ soil moisture from 2,500+ stations with fixed
  coordinates and time series.
- Family: soil; region: Global; label_type: points; license: "free (registration)";
  manifest time_range: 2016–2026; have_locally: false.

## Fit assessment (would be a good dataset)

The dataset is a genuinely good fit for open-set-segmentation pretraining and would be
processed as a **sparse-point regression** target (spec §2a / §4 regression points),
analogous to the existing `wosis_soil_profiles` script:

- ~2,500 stations with **fixed lon/lat** (recoverable geocoordinates — good).
- Continuous surface/near-surface **soil moisture** value per station → regression;
  ≤5,000 locations cap easily satisfied (one representative post-2016 1-year window per
  station, or a few windows per station across years, up to 5,000 points).
- Manifest time range 2016–2026 is squarely in the Sentinel era, so the pre-2016
  rejection rule does not apply (post-2016 records exist; any pre-2016 subset would be
  filtered out).
- Station metadata (land cover, climate, soil texture) is available for provenance.

Planned processing (for the retry): read the downloaded ISMN archive with the
`TUW-GEO/ismn` reader (`ISMN_Interface`), select the surface soil-moisture variable at
each station, restrict to post-2016 timestamps, compute a representative value per
station over a 1-year window (e.g. annual mean surface soil moisture), and write one
`points.geojson` regression feature per (station, window) via `io.write_points_table`,
bucket-balancing across the value range to ≤5,000 samples. Time range = the labeled
1-year window (soil moisture is dynamic, so anchor the value on the same window used for
the label, not an arbitrary static year).

## Why rejected (access gate)

ISMN in-situ data is **only downloadable through the web Data Portal after creating a
free account and logging in**. Verified on 2026-07-11:

- Site is reachable (`GET https://ismn.earth/en/` → HTTP 200), so this is **not** a
  transient/infra outage (would be `temporary_failure`) — it is a permanent access gate.
- Download requires registration (`/accounts/signup/`) + login (`/accounts/login/`),
  then filter/select in the Data Portal (`/dataviewer/`). Large requests are delivered
  as an emailed zip link.
- **No** unauthenticated REST/FTP/token/OPeNDAP endpoint or direct download URL exists
  (official docs describe only the web-portal flow).
- The `TUW-GEO/ismn` Python package **only reads already-downloaded archives** — it has
  no fetch/download capability ("Data used in the tutorials is not provided… create an
  account at ismn.earth to download the required files").
- No open **Zenodo / PANGAEA / mirror** of the full network was found (brief search;
  only QA4SM validation *results* reference Zenodo, not the raw ISMN in-situ data, which
  carries network-specific redistribution terms).

No credential is available in this environment, so the data cannot be obtained.
Per the task spec (§8.2 / §1a), a missing-credential access gate is recorded as
**`rejected`** with `notes: "needs-credential: …"` (not `temporary_failure`).

## How to retry / reproduce

1. Register a free ISMN account and log in at <https://ismn.earth/>.
2. In the Data Portal (`/dataviewer/`), select all networks, soil-moisture variable,
   time range 2016-01-01 onward, and download in the default **"Header+values"** format
   (or CEOP). Await the emailed zip if the request is large.
3. Place the extracted archive at
   `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/ismn_international_soil_moisture_network/`.
4. Implement `datasets/ismn_international_soil_moisture_network.py` mirroring
   `datasets/wosis_soil_profiles.py` (point-table regression via
   `io.write_points_table`), using the `ismn` reader to load per-station surface
   soil-moisture time series; then run
   `python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ismn_international_soil_moisture_network`.

## Judgment calls

- Classified as **regression** (continuous soil moisture), not classification, per the
  manifest description and spec guidance.
- Treated as a **permanent credential gate → `rejected` (needs-credential)**, not
  `temporary_failure`, because the site is up and the block is an account requirement,
  not a transient error.
- Would use **post-2016** records only and assign each label a **1-year time range
  matching the window used to derive the soil-moisture value** (soil moisture is
  time-varying, unlike the quasi-static soil pH in `wosis_soil_profiles`).
