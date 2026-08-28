# ICP Forests Crown Condition — REJECTED

- **Slug:** `icp_forests_crown_condition`
- **Name:** ICP Forests Crown Condition
- **Source:** ICP Forests (UNECE ICP on Assessment and Monitoring of Air Pollution Effects
  on Forests), Programme Co-ordinating Centre (PCC) at Thünen Institute.
  Portal <https://www.icp-forests.org/data> (redirects to the data/maps site
  <https://www.icp-forests.net/data-maps/data-requests>).
- **Family / region:** forest / Europe (~40 countries). **Manifest label_type:** points
  (tree-level plots on a 16×16 km Level I grid). **Time range (manifest):** 2016–2026.
- **License (manifest):** "open (registration)".
- **Status:** **REJECTED**
- **Reason:** `needs-credential: ICP Forests data request/registration`

## What the dataset is

Pan-European field survey of tree crown condition on the ICP Forests Level I systematic
16×16 km plot grid: annual visual crown assessments recording **defoliation** (in 5 %
classes), **discoloration**, **crown dieback**, and **damage cause** (insects, drought,
wind, etc.), collected via the standardized TRE/TRC forms (Manual Part IV, Visual
Assessment of Crown Condition). It is a genuinely relevant forest-health signal
(defoliation influences canopy reflectance and is at least partly observable at 10–30 m),
so the rejection is purely an access issue, not a fit issue.

## Triage — access (spec §8.2)

- **No credential in `.env`.** Checked: the env holds NASA Earthdata,
  Copernicus, CDS, USGS/M2M, Planet, GEE, AWS, and an internal datasets-API token — none of
  which authorize ICP Forests. There is no ICP Forests login/token available.
- **No direct open download exists.** I checked the ICP Forests data portal and the
  "Data Requests" page. Access to crown-condition data is gated behind a **formal data
  request / registration workflow**, not a bulk CSV or open endpoint:
  1. Complete and sign a "data request" form plus a ~1-page project description and email it
     to `pcc-icpforests@thuenen.de`.
  2. The PCC forwards the request to all participating countries' National Focal Centres.
  3. After **2–4 weeks** the PCC returns a decision; access is granted only **if no member
     state objects** (any country can veto).
  4. The requester must accept the ICP Forests Intellectual Property and Publication Policy.
  This is a per-dataset registration/approval portal with a manual, multi-week,
  member-state-vetoable approval — exactly the class the SOP (§8) says to reject as a
  credential/registration gate we cannot satisfy autonomously. No unauthenticated mirror or
  alternate bulk source was found (the Eionet reporting-obligation record is a
  reporting/metadata pointer, not an open data package).

## Coordinate-precision caveat (secondary — not reached)

Even if access were granted, ICP Forests plot coordinates are typically fuzzed/coarsened for
privacy (often to ~grid level rather than true plot lon/lat). Per the task's coordinate-
precision guidance, coarse 16×16 km grid-cell ids without precise plot lon/lat would fall
under "no recoverable geocoordinates" and could be a second, independent blocker. This was
not verified because access could not be obtained; it is noted so a future retry (with data
in hand) checks coordinate precision before processing.

## Disposition

Rejected on access grounds. To revisit: the user submits an ICP Forests data request and,
once the delivered data is on disk, an agent re-triages — first confirming that plot
coordinates are precise enough (true plot lon/lat, not just a fuzzed grid cell) to place
labels on the Sentinel-2 grid, then processing the post-2016 subset as a sparse point table
(§2a) with defoliation as either % regression or 5 %-class classification.

## Reproduce

No script written. `raw/icp_forests_crown_condition/` and `datasets/.../locations/` were not
created; only `datasets/icp_forests_crown_condition/registry_entry.json` (status `rejected`)
and this summary were written.
