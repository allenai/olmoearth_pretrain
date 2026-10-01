# HABSOS (Harmful Algal BloomS Observing System)

- **Slug**: `habsos_harmful_algal_blooms_observing_system`
- **Status**: completed
- **Task type**: classification (5 classes)
- **Num samples**: 5000 (1000 / class, balanced)
- **Family / region**: ocean_color / Gulf of Mexico + SE US coast
- **License**: public domain (NOAA)

## Source & access

NOAA NCEI HABSOS is an in-situ georeferenced marine harmful-algal-bloom observation
database (product page:
https://www.ncei.noaa.gov/products/harmful-algal-blooms-observing-system). Each record is a
dated water sample at a lon/lat point with a Karenia brevis (red tide) cell count (cells/L),
an NCEI-assigned bloom-abundance `CATEGORY`, and ancillary fields (salinity, water temp,
wind, depth, state).

No CSV export link is published on the product page, but NCEI serves the full "Cell Counts"
layer through a **public ArcGIS MapServer with no credential**:
`https://gis.ncdc.noaa.gov/arcgis/rest/services/ms/HABSOS_CellCounts/MapServer/0`
(Query capability, maxRecordCount 350k). We page it via `download.download_arcgis_layer`
with `where="SAMPLE_DATE >= date '2016-01-01'"`, `outSR=4326` -> one GeoJSON in
`raw/{slug}/habsos_cellcounts_2016plus.geojson` (83,056 features).

## Sentinel-era filtering

HABSOS spans ~1953-present. We requested only `SAMPLE_DATE >= 2016-01-01` at the server;
all pre-2016 observations are dropped. The retained set spans 2016-2026. Every post-2016
record has GENUS=`Karenia`, SPECIES=`brevis`, CELLCOUNT_UNIT=`cells/L`.

## Label decision: classification into K. brevis bloom categories

Cell abundance could be regression (log10 cells/L) or classification into the standard NOAA
K. brevis bloom bins. We chose **classification** using NCEI's precomputed `CATEGORY` field
(a natural, well-defined, authoritative labeling). Categories are ordered by increasing
abundance. Observed HABSOS cell-count thresholds (cells/L), matching the standard NOAA
red-tide bins, are recorded in `metadata.json`:

| id | class        | CATEGORY       | cells/L threshold          | selected |
|----|--------------|----------------|----------------------------|----------|
| 0  | not_present  | not observed   | 0                          | 1000     |
| 1  | very_low     | very low       | 1 - <10,000                | 1000     |
| 2  | low          | low            | 10,000 - <100,000          | 1000     |
| 3  | medium       | medium         | 100,000 - <1,000,000       | 1000     |
| 4  | high         | high           | >=1,000,000                | 1000     |

Available post-2016 counts before balancing: not_present 69,230; very_low 6,089; low 3,414;
medium 2,975; high 1,136; uncategorized (None) 212 (dropped). All 5 real classes have
>=1000 samples, so `balance_by_class(per_class=1000)` yields exactly 5000 balanced points
(well under the 25k cap). `not_present` is a genuine absence class (K. brevis not detected),
kept as class 0.

## Output format

Sparse in-situ POINT dataset -> one dataset-wide GeoJSON point table (spec 2a),
`datasets/{slug}/points.geojson`, **not** per-point GeoTIFFs. Each `Point` feature carries
`label` (class id), `time_range`, `change_time`, `source_id` (`OBJECTID_*`), plus auxiliary
`category`, `cellcount_cells_per_l`, `observation_date`, `state`.

## Time-range decision

A red-tide bloom is a specific-date, rapidly-varying phenomenon (match-up truth), not a
static annual state. Each point gets a **SHORT window of +/-7 days (14 days total) centered
on the observation date** via `io.centered_time_range(date, 7)` — tight enough to bracket
the observed bloom state (K. brevis blooms evolve over days-weeks and are visible as surface
discoloration) while giving pretraining a realistic chance of finding a near-cloud-free
S2/S1/Landsat acquisition. Well under the 360-day cap. `change_time = null` — this is a
state/condition label at an instant, not a dated change event (precedent: live-fuel-moisture
/ snow-presence condition labels).

## Verification (spec 9)

- `points.geojson`: FeatureCollection, task=classification, count=5000, 5000 unique ids,
  label counts {0..4}=1000 each, all time windows = 14 days, all change_time=null.
- Coordinates fall in lon [-97.39, -79.96], lat [24.47, 30.71] — Gulf of Mexico + SW/E
  Florida coastal marine waters, the expected domain (in-situ water samples are inherently
  coastal). Category<->cellcount<->label are mutually consistent (e.g. a `high` point has
  cellcount 3.2M >= 1,000,000).
- `metadata.json`: 5 classes with descriptions + thresholds, nodata=255, num_samples=5000.
- Idempotent: raw download uses `skip_existing`; re-running re-derives the same seeded
  selection.

## Caveats

- Observable phenomenon at 10-30 m is the red-tide surface discoloration; the label is the
  in-situ cell measurement, which need not perfectly co-locate with a visible surface signal
  (subsurface blooms, patchiness). Treated as match-up truth.
- Uncategorized (None) records (212) dropped.
- Points can repeat at the same station across dates; each date is its own sample.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.habsos_harmful_algal_blooms_observing_system
```
