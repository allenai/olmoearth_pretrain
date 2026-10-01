# Collapse Caldera Database (CCDB)

- **Slug:** `collapse_caldera_database_ccdb`
- **Status:** completed (ACCEPTED as a weak single-phenomenon presence classification)
- **Task type:** classification (single presence class)
- **Num samples:** 417 points
- **Family / region:** volcano / global
- **License:** CC-BY-NC-4.0

## Source & access

CCDB — "The collapse caldera worldwide database" (Geyer & Martí, GVB-CSIC), version 4.0
(2019), published on Zenodo (record **10636011**, concept DOI 10.5281/zenodo.10636010).
A single Excel workbook `CCDB4_zenodo.xls` (~576 KB, no credentials needed) downloaded via
`download.download_zenodo("10636011", ...)`. The `Calderas` sheet holds **477** collapse-
caldera records with WGS84 `Latitude`/`Longitude`, caldera `Max/Min diameter (km)`, `Area`,
geological `Age`/`Age epoch`, and many structural/petrological attributes (magma
composition, rock suite, collapse/subsidence type, chamber depth, tectonic setting,
preservation, GVP links). Reproduce:

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.collapse_caldera_database_ccdb
```

(`xlrd>=2.0.1` is required to read the legacy `.xls`; install if missing.)

## Triage decision (why ACCEPT, not reject)

The task flagged this as a strong reject candidate. Applying spec §2/§5/§8 rigorously:

- **Change framing rejected, static presence accepted.** Every record's `Age`/`Age epoch`
  is geological (Precambrian, Ordovician, …, Miocene, Pleistocene, Holocene — thousands to
  millions of years old). The caldera *collapse event* is therefore **not** an observable
  Sentinel-era change, so a change encoding is invalid. But a collapse caldera is a
  **persistent landform** still plainly visible today, so it is valid as a present-day
  static observation: `change_time=null`, a representative 1-year Sentinel-era window
  (2020). Documented so downstream never treats the geological age as a change signal.
- **Observability at 10 m — YES.** Calderas are large: `Max_diameter` median **10 km**,
  343/386 with a diameter are ≥5 km, only 2 <1 km, up to 110 km (Toba, Island Park). These
  are unambiguous topographic collapse depressions at 10–30 m (manifest note: "Calderas
  km-scale; clearly discernible"). Coordinate precision is mixed (~129 records rounded to
  ≤2 decimal places ≈ 1 km; the rest finer), but ~1 km positional error is small relative
  to a km-scale landform, so the point still lands on the caldera.
- **Label meaningfulness — single presence class only.** The only coherent imagery-
  observable per-location label is "a collapse caldera is present here". Subsurface /
  geological attributes (magma composition, collapse type, chamber depth, rock suite) are
  **not** inferable from optical/SAR at 10 m, and preservation/state is recorded for only
  35/477 records — so none of them is used as a class. This mirrors the accepted
  `atlas_of_hillforts_of_britain_and_ireland` weak-presence precedent.

Had the database provided only imprecise points with a geologically-defined class and no
extent, the correct call would have been reject (observability/label-validity). It clears
the bar because it supplies km-scale extents and coordinates that resolve the landform.

## Classes

Presence-only (no background/negative class — the assembly step supplies negatives from
other datasets, spec §5):

| id | name | count |
|----|------|-------|
| 0  | `collapse_caldera` | 417 |

## Processing

- Kept the 462/477 records with valid WGS84 lon/lat (dropped 15 with no coordinates).
- De-duplicated coincident records at 6 dp (45 dropped — e.g. Phlegrean Fields I/II/III and
  the Colli Albani entries share one volcano centre), leaving **417** unique-location
  points. All are one class, well under the 1000/class and 25k caps → no truncation.
- Written as a dataset-wide **`points.geojson`** point table (spec §2a), one `Point`
  feature per caldera: `properties.label=0`, `time_range=[2020-01-01, 2021-01-01)`,
  `change_time=null`, `source_id=IDCaldera` (GVP-style id, e.g. `0803-A`). No per-point
  GeoTIFFs (1×1 sparse points).
- Static-label 1-year window fixed at **2020** (representative Sentinel-era year for these
  persistent landforms).

## Verification (§9)

- `points.geojson`: valid `FeatureCollection`, count=417, all `label∈{0}`, all
  `change_time=null`, all coordinates within valid lon/lat, `time_range` = a 1-year window.
- `metadata.json`: task_type=classification, class id 0 covers all feature labels.
- **Spatial sanity check:** famous calderas land exactly on their real locations —
  Yellowstone (~44.4, −110.67), Crater Lake (42.93, −122.12), Aira/Kagoshima (31.64,
  130.75), Santorini (36.40, 25.40), Toba (2.58, 98.83), Campi Flegrei (40.83, 14.14),
  Long Valley (37.70, −118.87), Taupō (−38.78, 176.12), Deception Island/Antarctica
  (−62.93, −60.57, explaining the far-south latitudes). Georeferencing confirmed correct.
- Idempotent: Zenodo download is skip-existing; selection is seeded/deterministic.

## Caveats

- Weak presence label: a caldera centre is one 10 m point, while the landform spans km;
  pretraining projects the point onto the S2 grid and pairs imagery by geography/time.
- ~1 km coordinate rounding on a minority of records; acceptable given landform scale.
- Presence-only, single class — depends on assembly-time negatives from other datasets.
