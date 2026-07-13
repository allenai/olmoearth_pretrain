# HydroWASTE (Global Wastewater Treatment Plants)

- **Slug:** `hydrowaste_global_wastewater_treatment_plants`
- **Status:** completed
- **Task type:** classification (positive-only object detection, single foreground class)
- **Num samples:** 1500 (1000 WWTP-positive tiles + 500 background-only negative tiles)
- **Family / region:** industry / Global
- **License:** CC-BY-4.0

## Source

HydroWASTE version 1.0 — a spatially-explicit global database of **58,502 wastewater
treatment plants (WWTPs)** with capacity, population served, treatment level and estimated
river outfall. HydroSHEDS product.

- Paper: Ehalt Macedo, H., Lehner, B., Nicell, J. A., Grill, G., Li, J., Limtong, A.,
  Shakya, R. "Distribution and characteristics of wastewater treatment plants within the
  global river network." *Earth Syst. Sci. Data* 14, 559–577 (2022).
  https://doi.org/10.5194/essd-14-559-2022
- Data: figshare https://doi.org/10.6084/m9.figshare.14847786.v1 (one 2.4 MB zip →
  `HydroWASTE_v10.csv` (58,502 rows) + `README.txt`).
- Product page: https://www.hydrosheds.org/products/hydrowaste

**Access:** openly downloadable, no credentials. Direct file URL
`https://ndownloader.figshare.com/files/31910714`.

## Access method

Downloaded the figshare zip via `download.download_http` into
`raw/hydrowaste_global_wastewater_treatment_plants/`, extracted `HydroWASTE_v10.csv`, and
parsed it with `csv.DictReader` (latin-1 encoding). No imagery pulled — only the label
point table. A `SOURCE.txt` provenance file is written alongside the raw data.

## Triage / suitability

**Accepted.** WWTP aeration/settling ponds and clarifier tanks are readily discernible at
10–30 m from Sentinel-2/Sentinel-1/Landsat, so this is a good fit for **positive-only
object detection** (spec §4). Openly licensed (CC-BY-4.0), no-credential download.

### Coordinate precision (the key concern for a geocoded point registry)

HydroWASTE gives a **reported plant location** (`LAT_WWTP`/`LON_WWTP`) plus a per-record
location-quality flag `QUAL_LOC`:

| QUAL_LOC | meaning | count |
|---|---|---|
| 1 | high (>80% of a country/region's points tested accurate) | 7,521 |
| 2 | medium (50–80% accurate) | 44,991 |
| 3 | low (<50% accurate) | 2,540 |
| 4 | quality not analysed | 3,450 |

Most coordinates are 3-decimal (~110 m) precision; all 58,502 rows have valid, non-zero
coordinates. Rather than reject or trust every point, mitigations were applied (below).
The distinct **outfall** location (`LAT_OUT`/`LON_OUT`) is the modeled river discharge
point, **not** the physical plant — we use the plant location.

## Encoding / label mapping

Tunable detection encoding (`sampling.encode_detection_tile`):

- **Class scheme** (uint8): `0 = background`, `1 = wastewater_treatment_plant`,
  `255 = nodata/ignore` (buffer rings). `nodata_value = 255`.
- **Tile:** 48×48 at 10 m (480 m context), local UTM, north-up.
- **positive_size = 1** (1 px positive at the plant point).
- **buffer_size = 12** → a 25×25 (~250 m) nodata ignore ring around each positive. This is
  deliberately more generous than the default 10 to absorb geocoding imprecision
  (QUAL_LOC-medium points, ~110 m coordinate rounding) **and** the plant's real footprint,
  which we don't know precisely. A 48×48 tile keeps ~1,679 background px per single-plant
  positive tile — ample negatives within the tile.
- Every other well-located plant that falls inside a tile is also marked positive (12 of
  the 1000 positive tiles contain 2 plants; the rest 1).

### Reliability filters (positive tile centers)

Positive tile centers are drawn only from the **well-located, built** subset:
`QUAL_LOC ∈ {1,2}` **and** `STATUS` not in {Projected, Proposed, Under Construction,
Construction Completed} — i.e. plants likely both correctly located and physically present
during 2016–2022. This yields **52,078** eligible plants; 1000 are sampled (seeded,
spec §5 per-class cap). QUAL_LOC 3/4 (potentially mislocated) points are excluded as
positive centers to avoid stamping a positive on empty land.

### Negatives

500 background-only tiles are placed 3–20 km from a random plant and required to be
**≥1 km from *every* one of the 58,502 reported plants** (KDTree over the full set, incl.
low-quality points, so negatives stay clear even where a point may be mislocated). Clamped
to lat ∈ [−58, 74].

## Time range

WWTPs are persistent, undated structures → **static-label** handling (spec §5): each sample
gets a **1-year window** at a representative Sentinel-era year, spread pseudo-randomly over
**2016–2022** (the manifest time range) for temporal diversity. `change_time = null`.
Distribution: ~210–220 samples per year. All windows are exactly 1 year (verified ≤360 d).

## Sample counts

- WWTP-positive tiles: **1000**
- Background-negative tiles: **500**
- Total: **1500** (well under the 25k cap)

## Verification (spec §9)

- Opened multiple `.tif`s: single-band **uint8**, **UTM @ 10 m**, **48×48**, nodata 255.
  Positive tiles = {1 px class 1, 624 px nodata ring, 1679 px background}; negatives = 2304
  px background.
- Every `.tif` has a matching `.json`; all `time_range`s ≤ 1 year; `metadata.json` class
  IDs {0,1} cover all values in the tifs.
- **Georeferencing check:** reprojecting each positive pixel back to WGS84 reproduces the
  source `LON_WWTP`/`LAT_WWTP` to 4 decimals (e.g. 000000 → (13.927, 48.476)).
- Idempotent: a second run skips all 1500 outputs.
- Spatial/temporal note: a full Sentinel-2 overlay was not rendered here, but the exact
  point→pixel round-trip confirms placement; residual per-point offset (geocoding) is
  absorbed by the 250 m ignore ring, which is the intended design for this registry.

## Caveats

- Positives mark **presence** at a geocoded point, not a segmented plant footprint; the
  250 m ignore ring is essential and downstream training must respect nodata=255.
- QUAL_LOC 3/4 plants are used only for negative-exclusion, not as positive centers.
- Only 1000 of ~52k well-located plants are used (per-class cap); the script can be re-run
  with a larger `PER_CLASS` if more positives are wanted.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.hydrowaste_global_wastewater_treatment_plants
```
