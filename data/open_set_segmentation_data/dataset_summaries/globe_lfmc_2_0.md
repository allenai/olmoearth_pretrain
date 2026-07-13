# Globe-LFMC 2.0 — `globe_lfmc_2_0`

**Status:** completed · **Task:** regression · **Samples:** 5000 (point table)

## Source
Globe-LFMC 2.0 (Yebra et al. 2024, *Scientific Data* 11:332), a global compilation of
in-situ **Live Fuel Moisture Content (LFMC)** field measurements.
- figshare dataset: DOI [10.6084/m9.figshare.25413790](https://doi.org/10.6084/m9.figshare.25413790)
  (article 25413790, single file `Globe-LFMC-2.0 final.xlsx`, ~72 MB).
- paper: <https://doi.org/10.1038/s41597-024-03159-6>
- license: **CC-BY-4.0**. No credential required (open figshare download).

The Excel `LFMC data` sheet has **293,796 rows**, each a dated, georeferenced destructive
field sample (>2,000 sites, 15 countries, 1977–2023): WGS84 lat/lon (EPSG:4326, ~5 decimals
≈ 1 m), sampling date (YYYYMMDD), species + functional type, and the LFMC value
`LFMC[%] = 100·(Wf − Wd)/Wd` (fresh vs oven-dry weight).

## Access / download
`download.download_http("https://ndownloader.figshare.com/files/45049786", raw/{slug}/Globe-LFMC-2.0_final.xlsx)`.
The needed columns of the `LFMC data` sheet are cached once to `lfmc_core.parquet`
(reading the 72 MB xlsx takes ~80 s; parquet reload is instant → fast idempotent reruns).
Only the label table is pulled; pretraining supplies its own imagery.

## Label mapping (regression)
- **Target:** `live_fuel_moisture_content`, unit **percent**, dtype **float32**,
  nodata **-99999**.
- **Value QC:** kept LFMC in **[10, 400] %** (>99% of post-2016 rows). The raw column
  contains error sentinels (values up to 599999) and an implausible heavy tail above 500%;
  values <10% are non-physical for live fuel. This removes those.
- **Sample unit = (site, date).** A site+date frequently has several per-species rows at
  identical coordinates; these are aggregated to the **mean LFMC** for that (location, day),
  so each sample is one (pixel, time, value). Post-2016 QC'd rows → **51,602** site+date
  samples.
- **Bucket-balanced to the 5000 regression cap** (spec §5) across 10 quantile buckets
  (seed 42), because the site+date-mean distribution is right-skewed (median ~101%, tail to
  ~400%). Selected value range **[10.0, 397.4] %**, mean ~107%.

## Time-range handling (important)
LFMC is a **rapidly-varying condition**, valid only near its sampling date — **not** a
static annual property. Each sample therefore gets a **short ±15-day (~1-month) window
centered on its sampling date** (`io.centered_time_range`), so pretraining pairs it with
imagery within ~2 weeks of the field measurement. `change_time = null` (this is a condition,
not a dated change event / where-mask). All windows are 30 days (≤ 360-day cap). The
earliest window starts 2015-12-17 (for a 2016-01-01 measurement, whose −15-day edge dips a
few days into late 2015; Sentinel-2 was already operational).

## Post-2016 filtering (spec §8.2)
Globe-LFMC spans 1977–2023. ~118k of ~294k rows are dated **2016-01-01 or later**; only
these Sentinel-era measurements are kept, the pre-2016 majority is dropped. (Not an
all-pre-2016 dataset, so not rejected on that ground.)

## Caveats
- Coordinates are field-site GPS (~1 m precision) but a destructive sample represents
  plot-scale vegetation; treat as approximately placed on the 10 m grid.
- A full Sentinel-2 overlay eyeball is not meaningful for a scalar moisture point (LFMC is
  not directly visually identifiable), so the §9 imagery sanity check is limited to
  confirming valid on-land WGS84 site coordinates from the authoritative field database.
- Species heterogeneity within a site+date is collapsed to the mean; the per-species spread
  is discarded (documented here rather than emitting conflicting labels at one pixel+time).

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.globe_lfmc_2_0
```
Idempotent and deterministic (parquet cache + seeded bucket balancing); outputs
`datasets/globe_lfmc_2_0/{metadata.json, points.geojson}` and `registry_entry.json` on weka.
