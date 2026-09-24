# LUCAS Topsoil — `lucas_topsoil`

**Status:** completed · **Task:** regression · **Samples:** 5000 · **Format:** point table (`points.geojson`, spec §2a)

## Source

The **LUCAS (Land Use/Cover Area frame Survey) Soil Module** — the only harmonised,
EU-wide in-situ topsoil survey (single 2018 sampling period, all major land-cover types).

- **Authoritative raw dataset (NOT used): EC JRC / ESDAC "LUCAS 2018 TOPSOIL data"** —
  18,984 in-situ samples with pH (H2O & CaCl2), organic carbon content (g/kg), CaCO3, N,
  P, K, EC, oxalate Fe/Al, bulk density, coarse fragments. CC-BY-4.0 but **registration-
  gated** (https://esdac.jrc.ec.europa.eu/content/lucas-2018-topsoil-data, DOI
  10.2905/JRC.J2EXD50). No ESDAC credential is present in `.env`, so
  this CSV is not directly reachable. (Eurostat hosts the general LUCAS 2018 land-cover +
  point coordinates, but the lab soil properties themselves are the ESDAC-gated part.)
- **Open mirror USED (CC-BY-4.0): Chen et al. (2024)**, *"European soil bulk density and
  organic carbon stock database using LUCAS Soil 2018"* — Zenodo record **10211884** (DOI
  10.5281/zenodo.10211884; paper ESSD 16:2367–2383, doi:10.5194/essd-16-2367-2024). It
  republishes LUCAS Soil 2018 topsoil (0–20 cm) **at the in-situ GPS locations** with SOC
  stock, fine-earth bulk density, and coarse-fragment volume.
- Raw stored at `raw/lucas_topsoil/`:
  `LUCAS SOIL 2018 BD SOCS Local-RFFRFS.csv` (used) and `... T-PTF4.csv` (alternate PTF),
  plus `SOURCE.txt`.

CSV columns (primary file): `POINTID, Bdfine (g cm-3), SOCS (kg cm-2)*, coarse_vol,
GPS_LAT, GPS_LONG, BDfine method`.
*The header unit `kg cm-2` is a typo; the values (0.4–62) are **kg m⁻²** for the 0–20 cm
layer (= 10× Mg ha⁻¹).

## Regression target: topsoil SOC stock (0–20 cm)

The task recommends **soil organic carbon** as the primary target. The open mirror does
not carry the raw OC **content** in g/kg (that stays behind ESDAC registration); it
carries the closely-related, more policy-relevant **SOC stock (SOCS)** for the 0–20 cm
layer, which is emitted here as the single regression quantity:

`SOCS = measured LUCAS topsoil OC content × fine-earth bulk density × 0.2 m ×
(1 − coarse-fragment volume fraction)`.

Bulk density is **measured** for 5,163 points and **locally-predicted** (random-forest
pedotransfer function) for the remaining 10,226 — recorded per point in the `bd_method`
property. SOC stock is thus in-situ-anchored (OC content, coarse fragments and locations
are measured) with a modelled bulk-density component for most points.

- `regression.name = soil_organic_carbon_stock_topsoil`, `unit = kg m-2 (0-20 cm)`,
  `dtype = float32`, `nodata = -99999`.
- Auxiliary per-point properties: `bulk_density_g_cm3`, `coarse_fragment_vol_frac`,
  `bd_method`.

## Processing

1. `download_zenodo("10211884", raw_dir)` → both CSVs (open, no credential).
2. Read `LUCAS SOIL 2018 BD SOCS Local-RFFRFS.csv`; parse SOCS, BD, coarse, GPS_LAT/LONG.
3. Filters: drop rows missing SOCS/coords; keep `SOCS > 0`; coordinates within an EU study
   box (lon −32…45, lat 27…72) and not (0,0) → **15,389** points passed (all rows valid).
4. **Bucket-balance** across the SOC-stock range (`bucket_balance_regression`, 10 quantile
   buckets, seed 42) down to the **5000**-sample regression cap. The distribution is
   strongly right-skewed (organic/peat-soil tail), so bucketing gives even coverage of the
   full carbon range instead of over-weighting common mineral soils.
5. Write `points.geojson` (spec §2a) via `io.write_points_table` with
   `{id, lon, lat, label=<SOCS>, time_range, change_time=null, source_id=lucas_pointid_<id>}`
   plus the auxiliary soil properties.

## Time range

LUCAS Soil 2018 was surveyed **Apr–Oct 2018** (Sentinel-2 era). Topsoil SOC stock is a
**quasi-static** property, so per spec §5 (static labels) every point is anchored to a
representative **1-year window on the survey year (2018-01-01 → 2019-01-01)**.
`change_time` is null.

## Value distribution (5000 selected samples)

SOC stock range **0.47–55.5 kg m⁻²**, mean 5.38. Histogram (5 kg m⁻² bins):

| kg m⁻² | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 | 40–45 | 45–50 | 50–55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| count | 3037 | 1490 | 314 | 84 | 24 | 16 | 10 | 7 | 10 | 6 | 1 |

Even after bucket-balancing the histogram stays skewed because the upper buckets are
supply-limited (few very-high-carbon points exist); bucketing still lifts tail coverage
well above a plain random sample.

## Caveats

- **Target is SOC *stock*, not raw OC content (g/kg).** The g/kg content and the full
  multi-property LUCAS suite (pH, N/P/K, CaCO3, CEC, texture) require **ESDAC free
  registration** — a `needs-credential` item the user can supply later to enable those
  additional targets from the same in-situ points.
- **Modelled bulk-density component:** SOC stock uses locally-predicted bulk density for
  ~66% of points (RF pedotransfer function; `bd_method=Local-Prediction`), so it is a
  hybrid in-situ/derived quantity rather than a pure lab measurement.
- **Georeferencing:** `GPS_LAT/GPS_LONG` are the true field GPS sampling locations
  (better than the LUCAS theoretical grid points); lon/lat→UTM spot-checked as sensible EU
  placements (e.g. EPSG:32633 Austria, 32629 Portugal, 32631 France).
- Point-only dataset → no per-sample GeoTIFFs (spec §2a); per-tif verification N/A. Sample
  count within the 5000 regression cap and 25k hard cap.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lucas_topsoil
```

Idempotent: re-downloading skips the existing CSVs; the script recomputes
`points.geojson` deterministically (seed 42).
