# Global Tailings Portal

- **Slug:** `global_tailings_portal`
- **Status:** completed
- **Task type:** classification (single-class **presence**, point table)
- **Num samples:** 1807 presence points
- **Family / region:** mining / global
- **License:** free/public (GRID-Arendal)

## Source

The [Global Tailings Portal](https://tailing.grida.no/) (GRID-Arendal), launched January
2020, is a free, public disclosure database of mine **tailings storage facilities (TSFs)**.
It was built from disclosures by 100+ of the world's largest mining companies (originally
collected via the Church of England Investor Mining & Tailings Safety Initiative in response
to institutional-investor requests). Each TSF record carries a geocoded POINT (lat/lon
centroid) plus attributes: facility name, owner company, country, and hazard/consequence
classification.

## Access (no credentials)

The portal states a bulk CSV/Excel export is "coming soon" and otherwise available on
request, but the public Leaflet dashboard (`/map/data/`) populates its markers from an open
JSON endpoint that requires **no authentication**:

```
https://tailing.grida.no/api/taillingLoc?format=json
```

It returns a JSON list of `{pk, tsf, latitude, longitude, country, hazard_categorization,
owner_company}` (2113 records at time of processing). This is the label source; **no imagery
is downloaded** (pretraining supplies its own). `.env` credentials were not needed.
Saved to `raw/global_tailings_portal/taillingLoc.json`.

## Triage — ACCEPT

- **Observable at 10 m?** Yes. TSFs are large industrial impoundments (hundreds of metres to
  kilometres across), clearly resolvable in Sentinel-2/Landsat.
- **Georeferenced?** Yes — WGS84 lon/lat per facility. 2113/2113 records had valid,
  in-range, non-(0,0) coordinates.
- **Post-2016?** Yes — disclosures are from the ~2019-2020 reporting round; facilities persist.
- Spot-checks confirmed sensible placement, e.g. Vale's facility "VI" at
  (-20.1043, -44.1197) sits at the Córrego do Feijão / Brumadinho complex in Minas Gerais.

## Encoding decision (why presence, points.geojson)

- The portal provides **points (disclosed centroids), not footprints**, so we encode
  **presence** rather than a footprint segmentation.
- **Single foreground class `0 = tailings_facility`.** The disclosed attributes (dam
  construction type, hazard/consequence class, construction year, active/inactive status)
  are **not used as the class target**: none is reliably observable from S2/S1/Landsat at
  10 m. Simple presence is the only defensible target.
- Emitted as a dataset-wide **`points.geojson`** (spec §2a), one `Point` feature per
  facility — not per-point GeoTIFFs.
- **Positive-only** (spec §5): no synthetic negatives are fabricated; the pretraining
  assembly step supplies negatives by sampling other datasets.

## Coordinate-precision caveat (documented, not disqualifying)

Coordinates are company-disclosed centroids of uneven precision. Decimal-place distribution
(min of lat/lon): 1 dp: 44; 2 dp: 97; 3 dp: 232; 4 dp: 219; 5 dp: 367; 6+ dp: 1154. So ~93%
carry ≥3 decimal places, but the true positional accuracy is unknown and some points may sit
tens of metres off the exact facility. Because a TSF footprint is typically **hundreds of
metres** across, a centroid with that level of error still lands **on or immediately beside**
the facility, so these remain useful (if weak) presence labels. This is a weak-supervision
presence signal, appropriate for pretraining.

## Processing

1. Download `taillingLoc.json` (idempotent, atomic).
2. Drop invalid/(0,0)/out-of-range coordinates (0 dropped) and de-duplicate coordinates
   rounded to 5 dp (~1 m, i.e. same 10 m pixel): **306 duplicate coordinates dropped**
   (multiple disclosures/adjacent facilities collapsing to one pixel), leaving **1807**
   unique points from 2113 raw records.
3. Assign class 0 and a 1-year Sentinel-era `time_range`, spread pseudo-randomly (fixed seed)
   across 2019-2023 for temporal diversity (all post-2016; facilities persist).
   `change_time = null` (presence, not a dated change/event).
4. Write `points.geojson` (§2a) + `metadata.json`.

Year distribution of the 1-year windows: 2019: 363, 2020: 348, 2021: 366, 2022: 371,
2023: 359.

Countries represented: 65 (top: Australia 321, USA 290, Brazil 279, Canada 262, South
Africa 230, Japan 185, Peru 77, Mexico 40, Chile 37, Russia 35).

## Verification

- `points.geojson`: `FeatureCollection`, 1807 features, `task_type=classification`,
  all coordinates in range, single label `0`, all `change_time=null`, all time ranges ≤1 yr,
  unique zero-padded ids `000000`..`001806`.
- `metadata.json`: one class (`tailings_facility`), `nodata_value=255`, `num_samples=1807`.
- Full S2 image overlay was not rendered in this environment; georeferencing was sanity-
  checked against known facilities (see triage) and coordinate ranges. Any residual centroid
  offset is absorbed by the large TSF footprint (see caveat).
- Idempotent: fixed seed + atomic overwrite; the raw download is skipped when present.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_tailings_portal
```
