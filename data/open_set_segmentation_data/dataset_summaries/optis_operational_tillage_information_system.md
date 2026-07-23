# OpTIS (Operational Tillage Information System)

- **slug**: `optis_operational_tillage_information_system`
- **status**: **rejected** — **label semantics too coarse for per-pixel segmentation**
  (SOP §8.2 / §5 aggregation caveat), independently reinforced by **no downloadable data
  product** (SOP §8.2 access). Not a `temporary_failure`: both blocks are fundamental and
  permanent, not transient.
- **task_type** (intended, had usable per-pixel labels existed): regression (adoption
  fractions: no-till %, reduced-till %, conventional-till %, cover-crop %, residue cover %)
  or classification of the dominant tillage class.
- **num_samples**: 0

## Source

- Manifest name: `OpTIS (Operational Tillage Information System)`; source **USDA Ag Data
  Commons**, <https://data.nal.usda.gov/dataset/operational-tillage-information-system-optis-tillage-residue-and-soil-health-practice-dataset>
  (redirects to figshare-backed Ag Data Commons article **25212500**); family `tillage`,
  label_type `aggregated polygons`, region CONUS/Corn Belt, license **U.S. Public Domain**
  (open — license is not the blocker), have_locally: false.
- Product: remote-sensing-derived annual conservation-tillage / residue-cover / winter-
  cover-crop adoption, developed by Applied GeoSolutions + CTIC + The Nature Conservancy.
  Manifest time_range `[2016, 2022]`; the Ag Data Commons description states 2005–2018 and
  the current CTIC portal states 2015–2021 (so post-2016 data conceptually exists — the
  Sentinel-era cutoff is **not** the rejection reason).

## Why rejected — reason 1 (primary): aggregation is far too coarse; no high-purity sub-units

OpTIS **by design distributes only spatially-aggregated results** to protect individual-
producer privacy. Verbatim from the dataset description: *"While the OpTIS calculations are
performed and validated at the farm-field scale, the privacy of individual producers is
fully protected by distributing only spatially-aggregated results — at the county and
watershed (8-digit HUC) scale."* The current portal reports at **HUC8 watershed** and
**Crop Reporting District (CRD)** levels.

Each reported value is an **areal fraction per aggregation unit** (e.g. "No-Till =
area/percentage of indicated crops in the unit that were not tilled"; likewise Reduced
Tillage, Conventional Tillage, residue cover, cover crop). The aggregation units are
enormous relative to a pretraining tile:

- **HUC8 watershed**: median ~4,400 km² (tens of km across).
- **County** (Corn Belt): ~1,500 km².
- **CRD**: groups of counties — larger still.

A pretraining label tile is 64×64 px at 10 m = **640 m × 640 m ≈ 0.41 km²**. A single
HUC8 contains on the order of **10,000** such tiles, and OpTIS assigns them all one and the
same watershed-wide fraction. Because field-level (or any sub-unit) data is deliberately
never released, there are **no high-purity polygons** to salvage as confident
classification tiles, and the fraction cannot be localized within the unit at all. Even the
weakest allowed framing — a regression *prior* painted over the polygon (§5 aggregation
caveat) — would attach one number (an average over thousands of heterogeneous fields, most
of which do not match it) to a region tens of km wide. That is not per-pixel truth and is
not recoverable as high-purity classification, so per §5/§8.2 the label is **too
weak/misaligned** to be useful and the dataset is rejected. This mirrors the coarse-region
rejections already on file (e.g. `eyes_on_the_ground_kenya`, GADM village boxes).

## Why rejected — reason 2 (independent): no downloadable data product

Even setting aside the coarseness, there is no obtainable data file:

- The Ag Data Commons / figshare article (id 25212500) hosts **zero data files**. Its only
  "resource" is a **single zero-byte link** named `OpTIS` pointing at `https://ctic.org/OpTIS`
  (confirmed via the figshare API: `num_files: 1`, `size: 0`, `download_url:
  https://ctic.org/OpTIS`). No CSV, shapefile, GeoTIFF, or GeoJSON is served.
- `https://ctic.org/OpTIS/` (HTTP 200) is an overview page with no download links, no API
  endpoint, and no embedded data. The `.../optis/croplands/` subpage is now a
  **partnership-inquiry contact form** ("Are you interested in learning more about OpTIS or
  partnering with CTIC or its data partners?") routed via Connector.ag / Regrow — access is
  gated behind a business partnership, not a self-serve download.
- **Not `needs-credential`**: `.env` holds no CTIC/OpTIS/Regrow/TNC
  credential, and this is a partnership gate rather than a standard API credential the
  project could supply. **Not `temporary_failure`**: the servers respond 200 (no 5xx / rate
  limit); the data simply is not published for download.

Because reason 1 is fundamental and permanent (the aggregation coarseness is intrinsic to
OpTIS's privacy design), the dataset would remain unusable for 10 m per-pixel segmentation
**even if** the raw HUC8/CRD tables were obtained. Hence `rejected`, not
`temporary_failure`.

## Judgment calls

- **Rejected, not the regression-prior path.** The manifest/task explicitly permitted a
  documented regression-prior encoding as an alternative, but only when the aggregation
  unit is not "very large/coarse." HUC8 (~4,400 km²) / county / CRD is exactly the
  very-large/coarse case the caveat flags for rejection; no high-purity sub-polygons exist
  to switch to confident classification either.
- **Rejected, not `temporary_failure`.** Sources return HTTP 200; the block is a
  permanent absence of a downloadable, sufficiently-fine product, not a transient outage.
- **License is open** (U.S. Public Domain) — not a factor.
- **Pre-2016 is not the reason** — post-2016 years exist; the blocker is granularity +
  access.

## Reproduce / revisit

No outputs written to weka `datasets/optis_operational_tillage_information_system/` beyond
`registry_entry.json`. To reconfirm (no credential needed):

```bash
# 1. Ag Data Commons article hosts only a zero-byte link, no data file:
curl -s "https://api.figshare.com/v2/articles/25212500" -H "User-Agent: Mozilla/5.0" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print([(f['name'],f['size'],f['download_url']) for f in d['files']])"
# -> [('OpTIS', 0, 'https://ctic.org/OpTIS')]

# 2. CTIC croplands page is a partnership contact form, no download/API.
curl -s -L "https://ctic.org/optis/croplands/" | grep -ic download   # -> 0
```

This dataset would become usable only if OpTIS released **field-scale (or sub-km,
high-purity)** tillage/cover-crop rasters or polygons with recoverable geocoordinates — at
which point it is an attractive 2016+ CONUS conservation-practice signal (tillage class +
residue % + cover-crop % as classification/regression, 1-year annual windows,
`change_time=null`). As distributed today (HUC8/CRD areal fractions, download-gated behind
a CTIC partnership), it cannot be placed on the 10 m S2 grid.
