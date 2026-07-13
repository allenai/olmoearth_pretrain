# SeaBASS (NASA ocean color in-situ)

- **slug**: `seabass_nasa_ocean_color_in_situ`
- **status**: **rejected** — **observability** (SOP §8): in-situ ocean-color chlorophyll-a is
  a heavily time-varying, instantaneous point measurement that is not a meaningful per-pixel
  regression target for S2/S1/Landsat pretraining pairing. This is a fundamental reject, not
  `needs-credential` (access now works — see below) and not `temporary_failure`.
- **task_type** (would-have-been): regression (surface chlorophyll-a, mg/m^3).
- **num_samples**: 0

## Access — RESOLVED (credentials supplied and verified)

The user authorized NASA Earthdata credentials (`.env`:
`NASA_EARTHDATA_USERNAME`, `NASA_EARTHDATA_PASSWORD`). I wrote `~/.netrc`
(`machine urs.earthdata.nasa.gov`, chmod 600) and **confirmed the full authenticated
download route works**:
- Public archive browse: `GET https://seabass.gsfc.nasa.gov/archive/<AFFIL>/<EXPERIMENT>/
  <CRUISE>/archive/` lists `.sb` files and their ODPS download URLs.
- Authenticated file fetch: `GET https://oceandata.sci.gsfc.nasa.gov/ob/getfile/<hash>_<name>.sb`
  with `curl --netrc --location-trusted` returned **HTTP 200 and real SeaBASS data** (not a
  login page). Verified by downloading `NASA_GSFC/ChesBay/CB2015_August/.../Pigment_CB2015_August.sb`.
- The File Search UI POSTs to `/search_results/bio` (endpoint confirmed working, HTTP 200)
  with the full serialized bio form (global bbox `north/south/east/west`, `startdate`/
  `enddate`, `bio_measurement_type`, etc.); the results table is rendered client-side.
  Programmatic bulk retrieval is feasible via archive-browse crawling of the ODPS getfile
  URLs. So earlier `needs-credential` no longer applies — the block is now purely scientific.

## What SeaBASS actually is (from an inspected file)

SeaBASS = SeaWiFS Bio-optical Archive and Storage System (NASA OBPG / OB.DAAC), the public
archive of in-situ bio-optical oceanography data. The manifest's four "classes"
(chlorophyll-a, pigments, remote-sensing reflectance Rrs, absorption/backscatter IOPs) are
heterogeneous **measurement families with different units**, not a label set; only chl-a is
even a candidate per-pixel target.

Representative real file `Pigment_CB2015_August.sb` (Chesapeake Bay, HPLC pigments):
- **Geolocation is per-file/station, not per-row.** Header carries a single tight station
  bbox (`north/south_latitude=39.000/38.997`, `east/west_longitude=-76.360`); data rows have
  only `time,depth,Tot_Chl_a,...` — one geographic point per cast.
- **Instantaneous.** `start/end_date=20150820`, `start/end_time=11:59-14:07[GMT]` — a
  ~2-hour cast on one day.
- **Depth profile, not a surface value.** Rows span depth 0-10 m; `Tot_Chl_a` (mg/m^3) falls
  from ~19 at the surface to ~3.4 at 10 m. Satellites see only the optically-weighted
  near-surface; we would have to take the shallowest row.
- **`cloud_percent=100`**, "collected samples in between thunderstorms" — i.e. **no
  coincident optical satellite scene existed that day**. This is typical, not incidental.

The archive contains post-2016 data (e.g. `EXPORTS`, `OCEANX_2025`, PACE-era cruises), so
this is **not** a pre-2016 rejection — the ground is observability.

## Why rejected — observability judgment (SOP §8)

After inspecting real data, chl-a is **not a defensible per-pixel regression target** for
this S2/S1/Landsat pretraining bank, for two compounding reasons:

1. **Temporal coincidence is essentially never satisfied.** The measurement is an
   instantaneous cast; per SOP §5 it gets a ~1-hour `time_range`. Pretraining only uses a
   label when an input image window spans that time at that location. Over open water, an S2
   (~5-day revisit) or Landsat 8/9 (~8-day combined) scene that is both within ~1 h of the
   cast **and** cloud-free is vanishingly rare — the inspected station was 100% cloudy. The
   entire SeaBASS *validation* subsystem exists precisely because coincident in-situ/satellite
   ocean-color match-ups are scarce; for a generic (non-ocean-color, non-daily) sensor stack
   the realized pairing yield is effectively negligible. The labels would almost never be
   usable.
2. **Weak per-pixel observability even when a scene exists.** Chl-a retrieval is the domain of
   dedicated ocean-color sensors (SeaWiFS/MODIS/VIIRS/OLCI/PACE) with narrow high-SNR blue
   bands (412/443/490 nm) and specialized over-water atmospheric correction. S2/Landsat are
   land sensors: fewer/wider bands, low SNR over dark water, coarse aerosol correction over
   water, plus sun-glint/adjacency effects. In open-ocean Case-1 waters (the bulk of SeaBASS,
   chl ~0.01-1 mg/m^3) the reflectance signal is below what S2/Landsat resolve reliably;
   **S1 (SAR) has zero chl sensitivity**. Coastal Case-2 waters (like the 19 mg/m^3 example)
   carry more signal and have published S2 chl algorithms, but are a minority, are confounded
   by CDOM/turbidity, and still suffer problem (1). There is no aggregate/mask representation
   that salvages an instantaneous point tracer.

This is exactly the §8 case: "Phenomenon not observable at 10-30 m from S2/S1/Landsat ...
and no aggregate/mask representation salvages it," reinforced by the heavily time-varying
nature the coordinator flagged. → fundamental **`rejected` on observability**.

## Judgment calls

- **Rejected on observability, not needs-credential.** Credentials were supplied and the
  authenticated download route was verified working, so the access gate is gone; the
  remaining, decisive problem is scientific.
- **Considered and declined a coastal-only build** (chl-a only, Case-2 stations, 2016+,
  ~1h time_range, cap 5000). Rejected because the temporal-coincidence/cloud constraint makes
  even coastal samples almost never pairable in this pretraining setup, so the per-pixel
  target would be non-actionable — not worth adding low-yield, high-noise labels to the bank.
- **Not `temporary_failure`.** Source is fully reachable; nothing transient. Re-running would
  not change the observability conclusion.

## If reconsidered later

This dataset would only make sense paired with a **dedicated ocean-color sensor** (MODIS/
VIIRS/OLCI/PACE) with same-day global coverage and proper over-water atmospheric correction —
i.e. a different imagery stack than OlmoEarth's S2/S1/Landsat. If that stack is ever added,
revisit: crawl the SeaBASS archive (auth via `~/.netrc`), keep `Tot_Chl_a`/`chl` files 2016+,
take the shallowest depth row per station as one point, set ~1h `time_range` from the cast,
`nodata=-99999`, cap 5000. Under the current S2/S1/Landsat setup it is not usable.

## Reproduce

No outputs written to weka `datasets/seabass_nasa_ocean_color_in_situ/` beyond
`registry_entry.json`. Access verification: write `~/.netrc` for `urs.earthdata.nasa.gov`
from the supplied `.env`; browse `https://seabass.gsfc.nasa.gov/archive/NASA_GSFC/` and fetch
any listed `https://oceandata.sci.gsfc.nasa.gov/ob/getfile/<hash>_<name>.sb` with
`curl --netrc --location-trusted` (returns real `.sb` text). Inspected file:
`NASA_GSFC/ChesBay/CB2015_August/.../Pigment_CB2015_August.sb`.
