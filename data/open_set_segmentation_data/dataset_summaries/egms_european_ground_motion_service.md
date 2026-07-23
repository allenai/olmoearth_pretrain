# EGMS (European Ground Motion Service) — REJECTED (needs-credential: EGMS Copernicus Land registration)

- **Slug**: `egms_european_ground_motion_service`
- **Name**: EGMS (European Ground Motion Service)
- **Source**: Copernicus Land Monitoring Service — <https://egms.land.copernicus.eu/>
  (product pages under <https://land.copernicus.eu/en/products/european-ground-motion-service>)
- **Family / region**: subsidence / Europe + Norway, UK, Iceland; 2016–2024.
- **Label type (manifest)**: `points + gridded raster`; InSAR-derived (PSI/DS) from Sentinel-1.
- **License**: "free (registration)" — free to use but download is gated behind an account.
- **Status**: **rejected** — `needs-credential: EGMS Copernicus Land registration`.
- **task_type** (intended, had access been available): **regression** — vertical
  displacement velocity (mm/yr) from the L3 Ortho product. See "Intended recipe" below.
- **num_samples**: 0 (no data downloaded).

## What EGMS is

The European Ground Motion Service is the strongest continental ground-motion product: a
Persistent-Scatterer / Distributed-Scatterer InSAR analysis of the full Sentinel-1 archive
over the EEA39 territory (EU + Norway, UK, Switzerland, Iceland, etc.). It measures
millimetre-scale surface displacement over time. Three product levels are distributed:

- **L2a Basic** — per-satellite-track line-of-sight (LOS) velocity + displacement time
  series at native PS/DS point density (~20×5 m footprint), one value per measurement point.
- **L2b Calibrated** — the same, GNSS-calibrated to an absolute reference (2016–present,
  updated yearly; "Calibrated 2016-present (vector)").
- **L3 Ortho** — LOS velocities decomposed into **vertical (up–down)** and **east–west
  horizontal** velocity components, resampled onto a regular **100 m grid**
  ("Ortho, 100 m"). This is the gridded-raster form referenced in the manifest.

Each point/pixel carries a **multi-year average velocity (mm/yr)** plus a full displacement
time series. Negative vertical velocity = subsidence (sinking), positive = uplift, near-zero
= stable. This is an excellent, in-scope signal for this pipeline: velocity is a per-pixel
continuous quantity (regression) that is directly derivable, and the 100 m L3 grid is
coarser than the 10 m target but resamples cleanly.

## Why it is rejected (needs-credential)

EGMS data is downloadable **only** through the EGMS Explorer web application, which is gated
behind **EU Login** (`ecas.ec.europa.eu`) and an interactively-generated, time-limited
session token:

1. The download endpoint is
   `https://egms.land.copernicus.eu/insar-api/archive/download/{FILE}.zip?id={TOKEN}`.
   The `{TOKEN}` is a ~32-character time-limited user-session id **generated only by
   authenticating on the EGMS Explorer in a browser** and extracted from the tail of any
   download link the Explorer produces (this is exactly how the community `EGMStoolkit`
   works — you paste the manually-copied token via `-t`; the toolkit has **no**
   username/password login path). There is no documented REST/OAuth token exchange.
2. Tested here: `GET .../insar-api/archive/download/EGMS_L3_..._2018_2022_1.zip` **without a
   token returns HTTP 401**. There is no unauthenticated/anonymous download and no open
   mirror of the bulk tiles.
3. `egms.land.copernicus.eu` states plainly: *"You are not logged in. You must log in before
   performing a search,"* and the only login is the **EU Login** button.

**The credential in `.env` does not apply.** `.env` holds
`COPERNICUS_USERNAME`/`COPERNICUS_PASSWORD` (favyenb@allenai.org), but these are **Copernicus
Data Space Ecosystem (CDSE)** credentials, a *different* identity system from EU Login.
Verified during triage: those credentials successfully obtained a valid `access_token` from
the CDSE Keycloak endpoint (`identity.dataspace.copernicus.eu/.../CDSE/.../token`),
confirming they are CDSE-realm credentials — **not** EU Login (ECAS) credentials, and not an
EGMS session token. EGMS / Copernicus **Land** (`land.copernicus.eu`) authenticates against
EU Login, for which `.env` has no username/password, and even a valid EU Login would still
require replicating the Explorer's interactive session to mint the time-limited download
token.

Per SOP §8 this is a **persistent access gate / registration portal we do not have
credentials for**, so it is `rejected` with `notes: "needs-credential: EGMS Copernicus Land
registration"` — **not** `temporary_failure` (the 401 is an intended auth gate, not a
transient outage). No bulk archives were downloaded; only the small auth checks above were
performed.

## Intended recipe (if a token / EU Login is provided later)

Chosen framing: **regression on vertical displacement velocity (mm/yr)** from the **L3 Ortho
(100 m)** product, over a **bounded set of representative European regions** (do NOT pull the
whole continent — §5). Rationale for regression over the manifest's suggested
subsidence/uplift/stable classification: velocity is intrinsically continuous, the class
boundaries (e.g. |v| < 2 mm/yr = stable, v ≤ −2 = subsidence, v ≥ +2 = uplift) are arbitrary
thresholds that discard signal, and a regression target preserves magnitude. (If a
classification variant is ever wanted, record the thresholds explicitly in `metadata.json`;
±1.5–2 mm/yr is the commonly-cited stability band.)

Concrete plan:
- **Regions**: sample tiles from a handful of well-known, high-signal ground-motion areas to
  span the value range — e.g. the Netherlands / Po Valley / Venice (strong subsidence),
  Groningen gas field, London/Thames, Mexico-City-analogue European basins, plus large
  spatially-stable bedrock regions (Scandinavian/Alpine hardrock) for near-zero velocities.
  Draw only enough 100 m tiles to reach the target count.
- **Target count**: ≤ **5000** regression samples (well under the 25k cap), optionally
  bucket-balanced across the velocity range if the raw distribution is strongly peaked near 0
  (`sampling.bucket_balance_regression`), since most of Europe is stable.
- **GeoTIFF spec (§2)**: crop ≤64×64 windows from the L3 vertical-velocity band, reproject
  100 m → local-UTM **10 m** (bilinear is acceptable for a continuous velocity field; note
  it in the summary), single-band **float32**, `nodata = -99999`.
  `regression` block: `{"name": "vertical_displacement_velocity", "unit": "mm/yr",
  "dtype": "float32", "nodata_value": -99999, "value_range": [observed]}`.
- **Time / change (§5)**: velocities are **multi-year averages**, so treat as a **static
  label** with a representative **1-year window** in the Sentinel era (e.g. 2020, within the
  product's 2018–2022 / 2019–2023 span), and **`change_time = null`** — ground motion is a
  rate, not a dated event, so this is not a change-detection label.
- **Alternative (points)**: if the L2b Calibrated **point** vectors are used instead of the
  L3 grid, each PS/DS point is a single-pixel label → use the **point-table GeoJSON**
  (`points.geojson`, §2a) with `properties.label = velocity`, again ≤5000 points across the
  bounded regions.

## Reproduce / recover

No outputs were written to weka `datasets/egms_european_ground_motion_service/` beyond
`registry_entry.json`. To re-confirm the gate (no credential needed):

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.egms_european_ground_motion_service
```

This re-verifies the 401 auth gate and re-writes the `registry_entry.json` rejection
status; it produces no dataset outputs (and does not overwrite this hand-authored summary).

To process once access exists: (1) register / log in to the EGMS Explorer at
<https://egms.land.copernicus.eu/> via **EU Login**; (2) generate any download link and copy
the `?id=<TOKEN>` value (or supply an EU Login credential in a form the download step can use
— note this is a *separate* account from the CDSE `COPERNICUS_*` creds in `.env`);
(3) implement the regression recipe above against
`https://egms.land.copernicus.eu/insar-api/archive/download/{FILE}.zip?id={TOKEN}`, pulling
only the bounded set of L3 Ortho tiles for the chosen regions (the community `EGMStoolkit`,
<https://github.com/alexisInSAR/EGMStoolkit>, can enumerate/download tiles given the token).

## Sources

- EGMS Explorer / portal: <https://egms.land.copernicus.eu/>
- Product family: <https://land.copernicus.eu/en/products/european-ground-motion-service>
- EGMStoolkit (token-based downloader): <https://github.com/alexisInSAR/EGMStoolkit>,
  docs <https://alexisinsar.github.io/EGMStoolkit/>
