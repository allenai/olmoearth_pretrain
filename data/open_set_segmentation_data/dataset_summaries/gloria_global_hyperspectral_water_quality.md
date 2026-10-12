# GLORIA (Global hyperspectral water quality)

- **Slug:** `gloria_global_hyperspectral_water_quality`
- **Task type:** regression (point table, spec 2a)
- **Primary target:** chlorophyll-a concentration (Chla), unit `mg m-3`, dtype float32,
  nodata -99999
- **num_samples:** 1,635
- **Status:** completed

## Source

GLORIA — "A global dataset of remote sensing reflectance and water quality from inland and
coastal waters" (Lehmann et al. 2023, *Scientific Data* 10:100,
https://doi.org/10.1038/s41597-023-01973-y). Data hosted on PANGAEA,
DOI 10.1594/PANGAEA.948492, single archive `GLORIA-2022.zip` (~59 MB, **CC-BY-4.0**).
GLORIA is 7,572 curated in-situ hyperspectral remote-sensing-reflectance measurements from
450 water bodies worldwide, each with at least one co-located laboratory water-quality
measurement: chlorophyll-a (Chla), total suspended solids (TSS), CDOM absorption at 440 nm
(aCDOM440), and Secchi disk depth.

## Access method

Plain HTTP from PANGAEA. No credential required, but a **Firefox User-Agent** must be sent
(PANGAEA returns 403 for generic urllib UAs). The zip is downloaded to
`raw/{slug}/GLORIA-2022.zip`; only the label table member `GLORIA_2022/GLORIA_meta_and_lab.csv`
is extracted (the bulky per-wavelength radiometry CSVs — Es/Lsky/Lt/Lu/Lw/Rrs, ~250 MB
uncompressed — are not needed for the label signal and are left inside the zip).

## Label mapping / target choice

- **Primary regression target = Chla (chlorophyll-a, mg m-3).** Chosen because it is the
  most standard water-color / trophic-state variable retrievable from S2/S1/Landsat AND the
  best-populated post-2016 column: 1,635 usable Chla points vs 1,568 Secchi / 1,530 TSS /
  980 aCDOM440. Stored as the point `label`.
- **Auxiliary per-point fields** (carried in each feature's `properties` where measured):
  `tss` (g m-3, n=1140), `secchi_depth` (m, n=1295), `acdom440` (m-1, n=675),
  `turbidity` (NTU, n=549), plus `water_body_type` (GLORIA code: 1=lake/reservoir, 3=river,
  4=estuary, 5=coastal/ocean) and `country`. Documented in `metadata.json.auxiliary_fields`.

## Filtering

From the 7,572 rows: keep only rows that are **post-2016** (`Date_Time_UTC >= 2016-01-01`,
Sentinel era, spec 8.2 — GLORIA spans 1990–2022, 2,411 rows are post-2016), have valid WGS84
lat/lon, and a non-null positive Chla value → **1,635 samples**. No additional value QC was
required: the post-2016 Chla column is clean (curated), with no zeros, negatives, or sentinel
values; range 0.052–659.08 mg m-3.

## Distribution / bucketing

Chla is strongly right-skewed (roughly log-distributed): median ~6.8, 95th pct ~98, max
659 mg m-3. At 1,635 points the set is already well under the 5,000-sample regression cap, so
**all points are kept (no bucket downsampling)**. Decile bucket edges of the value
distribution are recorded in `metadata.json.regression.buckets` for reference:
`[0.052, 1.21, 1.79, 2.62, 3.96, 6.83, 11.82, 18.78, 33.87, 62.91, 659.08]`.

## Time-range handling

Chla is an **instantaneous match-up quantity** tied to the acquisition date and varies on
days-to-weeks timescales (algal blooms). Each sample gets a **short ±15-day (~1-month) window
centered on its measurement date** (`io.centered_time_range`), not a static year. All 1,635
features have a uniform 30-day span (well under the 360-day cap). `change_time = null` (this
is a water-column state at a time, not a dated change / where-mask event).

## Caveats

- **Weak / point-scale label for 10–30 m optical.** Each label is a single-pixel
  surface-water point measurement, not a full-water-body segmentation. Pretraining projects
  the lon/lat onto the S2 grid (1×1 point, spec 2a).
- Coordinates are inherently over water (in-situ water samples; `water_body_type` recorded).
  No S2 overlay was pulled for a spatial eyeball check, since by construction the points lie
  on the sampled water bodies; the point-scale nature is the main caveat, not geolocation.
- Contributor methods for Chla differ (spectrophotometric / HPLC / fluorometric); GLORIA
  harmonizes and QCs them but some inter-lab variance remains.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gloria_global_hyperspectral_water_quality
```

Idempotent: the zip download and CSV extraction skip if present, and selection is
deterministic (sorted by GLORIA_ID).

## Outputs

- `datasets/gloria_global_hyperspectral_water_quality/points.geojson` — 1,635 Point features
- `datasets/gloria_global_hyperspectral_water_quality/metadata.json` — regression block + aux fields
- `raw/gloria_global_hyperspectral_water_quality/` — `GLORIA-2022.zip`, extracted
  `GLORIA_meta_and_lab.csv`, `SOURCE.txt`
