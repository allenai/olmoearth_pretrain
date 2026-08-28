# WorldPop Global Population Density

- **Slug:** `worldpop_global_population_density`
- **Task type:** regression (dense_raster → per-pixel continuous value)
- **Source:** WorldPop, "Global per-country 2000-2020" unconstrained population product
  (hub listing [id=76](https://hub.worldpop.org/geodata/listing?id=76)); GeoTIFFs served
  from `https://data.worldpop.org/GIS/Population/Global_2000_2020/{year}/{ISO3}/{iso3}_ppp_{year}.tif`.
- **License:** CC-BY-4.0
- **Family / region:** population / Global (bounded sample of 18 countries)
- **Samples written:** 5000

## Regressed quantity & units

**`population_density`, in persons per square kilometre**, float32, nodata `-99999`
(`io.REGRESSION_NODATA`).

The native WorldPop raster stores **persons-per-pixel counts** ("ppp"): EPSG:4326,
~3 arc-second (~100 m) pixels, float32, nodata `-99999`, produced by a random-forest
dasymetric model. Per-pixel *counts* are **not** resolution-invariant, so resampling them
is meaningless. We therefore convert to **density (persons/km²)** — a resolution-invariant
intensity — as the regressed quantity:

```
area_km2(lat) = (dx_deg * 111320 * cos(lat)) * (dy_deg * 110574) / 1e6
density        = ppp / area_km2(lat)
```

`dx_deg`, `dy_deg` are the source pixel size in degrees (0.000833° here). The `111320`/
`110574` m-per-degree constants are WGS84 means (≈0.5 % accurate over a tile), documented
as an approximation.

## Resampling choice

Because the source pixel area is ~constant over a 640 m tile, **bilinearly reprojecting the
count field to a 10 m UTM grid and then dividing by the (fixed) source pixel area is
equivalent to reprojecting the density field.** Bilinear is appropriate for a smooth,
continuous intensity like density (never for categorical data). Reprojection is done with
rslearn `GeotiffRasterFormat.decode_raster(..., resampling=Resampling.bilinear)` over a
`WarpedVRT`, which honours the source `-99999` nodata (excluded from the bilinear kernel);
resulting nodata / out-of-coverage pixels (e.g. tile edges over water) are written as
`-99999`. Note: upsampling ~100 m → 10 m does not add real sub-100 m detail; it produces a
smooth interpolated density surface at the target grid.

## Sampling (bounded-tile, bucket-balanced)

Global product with no in-situ reference alternative → **bounded-tile sampling** from a
**fixed diverse set of 18 countries** (no global coverage), one product year each spread
across the manifest range 2016–2020 for temporal diversity:

| Continent | Countries (ISO3 → year) |
|-----------|-------------------------|
| Africa | KEN 2016, NGA 2017, EGY 2018, ETH 2019, ZAF 2020 |
| Asia | IDN 2016, VNM 2017, PHL 2018, JPN 2019, BGD 2020 |
| Europe | DEU 2016, FRA 2017, GBR 2018, POL 2019 |
| Americas | MEX 2016, PER 2017, COL 2018 |
| Oceania | NZL 2020 |

Chosen to span continents, development levels, climate zones, and settlement patterns
(arid Nile concentration, tropical archipelagos, dense deltas, temperate developed,
Andean/Amazon). The >1.5 GB giants (USA, BRA, CHN, IND, AUS) were deliberately excluded to
keep total download moderate (~7.6 GB); medium/small proxies were kept for each region.

Candidate generation: each country raster is read decimated (every 7th pixel ≈ one
candidate per 640 m tile footprint), valid pixels converted to density, and up to 30 000
candidates kept per country → **540 000 candidates**. Population is extremely right-skewed,
so candidates are **bucket-balanced with `sampling.bucket_balance_regression`** across
**log10(density+1) deciles** (10 buckets), yielding 5000 tiles (≈500/bucket). The density
bucket edges (persons/km²) are recorded in `metadata.json.regression.buckets`:
`[0, 0.03, 0.64, 2.73, 7.73, 16.28, 29.64, 55.0, 120.9, 447.0, 272029]`.

Per-country tile counts are near-uniform (261–300 each) — a natural consequence of
balancing across value buckets rather than by country.

## Value distribution

Per-pixel value range across written tiles: **[0.0, 163 253.9] persons/km²**
(`metadata.json.regression.value_range`). Selected-tile candidate-density percentiles:
p50 ≈ 16.3, p90 ≈ 444, p99 ≈ 3239, max ≈ 45 891 persons/km².

Histogram of selected-tile centre density (persons/km²):

| bucket | count |
|--------|-------|
| [0, 1) | 1134 |
| [1, 10) | 1017 |
| [10, 50) | 1277 |
| [50, 100) | 454 |
| [100, 500) | 647 |
| [500, 1 000) | 213 |
| [1 000, 5 000) | 229 |
| [5 000, 10 000) | 21 |
| [10 000, 50 000) | 8 |
| ≥ 50 000 | 0 |

The heavy low-density mass is expected (most inhabited land is rural); balancing keeps the
dense-urban tail well represented relative to a plain random sample.

## Time range

Each tile is assigned a **1-year window = its WorldPop product year** (`io.year_range`,
Jan 1 → Jan 1). Years are spread across 2016–2020 (see table). Population is a
slowly-varying annual quantity, so a 1-year window is appropriate; `change_time` is null.
`source_id` records `{ISO3}_{year}`.

## GeoTIFF spec (verified)

Single band, **float32**, local UTM, **10 m/pixel**, **64×64** (~640 m), nodata **-99999**.
Verified on random samples: correct band/dtype/shape, UTM CRS (EPSG:326xx), 10 m resolution,
values in the density range with `-99999` nodata, and every `.tif` has a matching `.json`
whose `crs` and `pixel_bounds` agree with the GeoTIFF and whose `time_range` is one calendar
year. A Nairobi test tile gave 134–6315 persons/km² (realistic dense-urban), an unpopulated
tile gave all-zero, matching expectations.

## Caveats

- **Model-derived product**, not in-situ reference: WorldPop is a random-forest dasymetric
  redistribution of census counts using covariates; absolute density is uncertain,
  especially in data-poor regions.
- **Upsampling 100 m → 10 m** adds no true sub-100 m structure; the 10 m grid is a smooth
  interpolation of the ~100 m density field.
- Pixel-area conversion uses spherical m-per-degree constants (~0.5 % error); negligible vs.
  the model's own uncertainty.
- Bounded 18-country sample is **not** globally representative by design (task spec §5 for
  large global products); it is a diverse but finite slice.
- A full Sentinel-2 overlay eyeball check was not performed; georeferencing was validated via
  CRS/resolution/pixel-bounds consistency and spot density sanity checks against known
  urban/rural locations.

## Reproduce

```
# Idempotent: skips already-downloaded country rasters and already-written .tif tiles.
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.worldpop_global_population_density --workers 64
# (add --skip-download if raw/ is already populated)
```

Outputs on weka under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/`:
`raw/worldpop_global_population_density/` (18 `{iso3}_ppp_{year}.tif` country rasters) and
`datasets/worldpop_global_population_density/` (`metadata.json`, `locations/{id}.tif` + `.json`).
