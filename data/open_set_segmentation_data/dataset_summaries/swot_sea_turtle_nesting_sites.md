# SWOT Sea Turtle Nesting Sites — REJECTED

- **Slug**: `swot_sea_turtle_nesting_sites`
- **Manifest name**: SWOT Sea Turtle Nesting Sites
- **Source**: SWOT / Duke MGEL (OBIS-SEAMAP) — https://seamap.env.duke.edu/swot
- **Family / label_type**: wildlife / points
- **License**: free + attribution (OBIS-SEAMAP / SWOT Terms of Use)
- **Final status**: **rejected** — reason: **needs-credential** (interactive
  registration portal: ToU agreement + "Who and for What" intended-use form + emailed
  passcode; no matching credential in `.env`)
- **task_type intended**: classification (per-point species class over nesting beaches)

## What the source actually is

The State of the World's Sea Turtles (SWOT) global database, hosted/maintained by Duke
University's Marine Geospatial Ecology Lab as the OBIS-SEAMAP project (dataset id 545). It
compiles >6,000 nesting data records across >3,000 monitored beaches globally for all sea
turtle species (green, leatherback, loggerhead, hawksbill, flatback, olive ridley, Kemp's
ridley). The label we want is the **SWOT Site Locations** layer: georeferenced nesting
**beach** points, each tagged with a species. Distributed as CSV or ESRI Shapefile from
the SWOT Nesting Sites mapping application.

## Observability triage — PASSES (rejection is NOT on observability)

Per the task guidance, we distinguish an unobservable animal-presence point from an
observable habitat/land-cover signal. A SWOT "site" marks a **nesting beach** (a sandy
coastal habitat), which is a coherent land-cover/habitat feature discernible at 10–30 m
from S2/S1/Landsat — matching the manifest note "Nesting beaches discernible at 10-30 m;
individuals not." Encoded as sparse 1×1 point labels (spec §2a `points.geojson`) with the
species as the class id, this would be a legitimate **weak-habitat** point-segmentation
dataset (species class is a beach attribute, not a per-pixel-observable trait; that is
acceptable, as with other presence/wildlife point sets). Time range: static/persistent
nesting beaches → a representative post-2016 1-year window; the manifest time_range
(2016–2024) is fully in the Sentinel era, so the post-2016 rule is satisfied. So the
dataset is a good conceptual fit — the blocker is purely **access**.

## Why rejected (access gate — evidence)

The SWOT nesting-site locations are distributed **only** through the OBIS-SEAMAP SWOT
mapping application, behind an interactive registration/authorization gate that cannot be
automated and for which we hold no credential:

1. **Registration portal.** Per the official download instructions
   (https://seamap.env.duke.edu/html/help/download_swot.html and the mapper's download
   form): every user must (a) agree to both the SWOT and OBIS-SEAMAP Terms of Use, (b)
   fill in a "Who and for What" form (first/last name, affiliation, e-mail, intended use
   >20 words), and (c) enter a **passcode e-mailed** to verify the address before any
   download begins. This is exactly the kind of interactive registration portal the SOP
   (§8.2) lists as a reject-worthy access gate (like xView3 / DrivenData / Kaggle), and it
   is a **permanent** gate, not a transient error.
2. **No credential in `.env`.** Checked `.env` (spec §8): it holds
   only `NASA_EARTHDATA_*`, `COPERNICUS_USERNAME/PASSWORD` (Copernicus **Data Space** =
   Sentinel hub, a different account system from this source), and `CDSAPI_KEY`. None
   applies to SWOT / OBIS-SEAMAP / Duke MGEL. Only credentials from that file may be used.
3. **No open alternate mirror for the SITE-locations layer.**
   - OBIS-SEAMAP is "a publisher to OBIS and GBIF", but the discoverable OBIS/GBIF
     sea-turtle datasets are third-party **telemetry/survey occurrence** sets, not the
     curated SWOT nesting-**site** GIS layer. Targeted searches
     (`api.obis.org/v3/dataset?q=State of the World's Sea Turtles`,
     `api.gbif.org/v1/dataset/search?q="State of the World's Sea Turtles"`, `q=SWOT`) did
     not surface the SWOT nesting-sites compilation as a downloadable open dataset.
   - The Copernicus Marine catalog entry **EXT_SWOT_TURTLES** ("Sea Turtle Nesting Sites
     and Regional Management Units") is an **External** product: its description states the
     data "are hosted on the OBIS-SEAMAP/SWOT website … available for viewing under the
     OBIS-SEAMAP Terms of Use." It is a catalog reference that funnels back to the same
     gated OBIS-SEAMAP portal, not a directly downloadable CMEMS product.
4. **Additional (transient) observation — not the deciding factor.** At triage time the
   entire `seamap.env.duke.edu` web app returned a backend database outage on every page
   (`Database connection failed: SQLSTATE[08006] … connection to server at
   "seamapsql.env.duke.edu" … No route to host`); the Apache front end is up (clean 404s
   for bad paths) but the Postgres backend is unreachable. Even if this outage clears, the
   registration-portal gate remains, so the terminal status is **rejected (needs-credential)**
   rather than `temporary_failure` (the SOP reserves `temporary_failure` for transient
   errors on an *otherwise-usable, no-credential* source; this source requires
   registration).

## What would unblock it (for anyone revisiting)

- A maintainer completes the OBIS-SEAMAP/SWOT ToU + intended-use form + email-passcode
  once and downloads **SWOT Site Locations** as CSV or Shapefile (species + lon/lat per
  beach), then drops it under `raw/swot_sea_turtle_nesting_sites/`. From there the dataset
  is straightforward: map the 6 species → class ids, dedupe to one point per beach, filter
  to post-2016 (already true), assign a static 1-year window, and write
  `datasets/swot_sea_turtle_nesting_sites/points.geojson` via `io.write_points_table`
  (`task_type="classification"`), balanced ≤1000/class (well under the 25k cap). This is
  "needs-credential" (user supplies a pre-downloaded copy out of band), not a code problem.

## Reproduce the triage

```bash
# Canonical source — every page returns the backend-DB error page (218 bytes):
curl -s -A "Mozilla/5.0" https://seamap.env.duke.edu/swot | head
# -> Database connection failed: SQLSTATE[08006] ... seamapsql.env.duke.edu ... No route to host
# Front end is up (clean 404 for unknown paths):
curl -s -A "Mozilla/5.0" https://seamap.env.duke.edu/api/ | head
# Download requires ToU + "Who and for What" form + emailed passcode:
#   https://seamap.env.duke.edu/html/help/download_swot.html
# Copernicus Marine EXT_SWOT_TURTLES is an External reference back to OBIS-SEAMAP:
#   https://data.marine.copernicus.eu/product/EXT_SWOT_TURTLES/description
```
