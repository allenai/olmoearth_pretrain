# Kelpwatch (Landsat/Sentinel-2 kelp canopy) — COMPLETED (classification)

- **Slug**: `kelpwatch_landsat_sentinel_2_kelp_canopy`
- **Source**: SBC LTER / kelpwatch.org — *"Time series of quarterly NetCDF files of kelp
  biomass in the canopy from Landsat 5, 7 and 8, since 1984 (ongoing)"*, Bell, Cavanaugh &
  Siegel. EDI data package **`knb-lter-sbc.74`** (revision 33, published 2026-05-26).
- **Access**: fully open, **no credential** required. Single NetCDF (~2.6 GB) downloaded
  from the EDI PASTA data endpoint:
  `https://pasta.lternet.edu/package/data/eml/knb-lter-sbc/74/33/c2bea785267fa434c40a22e2239bb337`
  (file `LandsatKelpBiomass_2026_Q1_withmetadata.nc`). **License: CC-BY-4.0.**
- **Label type**: `dense_raster` (derived-product map). **Task: classification.**
- **Region/time**: US West Coast + Baja California (lat 27–48.4°N, lon −124.8 to −114°W);
  quarterly, 1984 Q1 – 2026 Q1 (we use the **Sentinel-era 2016 Q1 – 2026 Q1** subset).

## Why accepted (and not deferred to Floating Forests)

The manifest notes Floating Forests citizen-science kelp outlines as a manual alternative.
That dataset (`floating_forests_global_kelp_canopy`) was **rejected** as entirely pre-2016
(latest Landsat scene 2013), so it provides no usable Sentinel-era reference — there is no
preferred alternative to defer to. KelpWatch is a **validated derived product** with dense
Sentinel-era coverage (2016–2026), so it is processed here.

## Source structure

The NetCDF is a **point cloud** of **593,426 fixed 30 m Landsat pixels** (WGS84 lat/lon;
UTM zones 10N and 11N), each carrying a 169-quarter time series of:
- `area` — surface kelp canopy area (m²) within the 900 m² pixel (0–900; canopy fraction =
  area/900),
- `biomass` — giant-kelp wet biomass (kg),
- `passes` — number of Landsat scenes averaged that quarter (0 / NaN `area` = unobserved).

Essentially every station is a "kelp-capable" reef pixel (has kelp in *some* quarter), so a
given quarter cleanly partitions observed stations into **surface-canopy-present** (area>0)
and **bare-reef/water** (area==0).

## Task decision: classification (presence/absence)

Two classes, matching the manifest (`["kelp canopy", "water"]`):

| id | name        | definition |
|----|-------------|------------|
| 0  | water       | observed kelp-capable reef pixel with **no** surface canopy this quarter (`area == 0`) |
| 1  | kelp canopy | surface kelp canopy detected this quarter (`area > 0`) |
| 255| nodata      | unobserved this quarter, or non-reef pixel (no station) |

Classification was chosen over **canopy-fraction regression** (`area/900`, also derivable
from this same file) because presence/absence is robust to the per-pixel area noise
(especially at low fractions), matches the manifest's two classes, produces interpretable
dense kelp-forest masks, and allows tiles-per-class balancing up to the 25k cap. A future
regression variant (float32 canopy fraction, ≤5000 samples, bucket-balanced) is feasible
from the raw file if desired.

## Time-range handling (seasonal, quarter-specific)

Kelp canopy is highly seasonal (summer/autumn peak, winter storm loss) and interannually
dynamic, so **a label is valid only for its quarter**. Each tile therefore gets a **~3-month
`time_range` matching its labeled quarter** (Q1=Jan–Mar, …, Q4=Oct–Dec), **not** a static
year, and `change_time = null` (a recurring seasonal state, not a dated change event). This
lets pretraining pair each tile with imagery from the correct season/year.

## Sampling & reconstruction (bounded-tile dense_raster)

Large derived product → **bounded-tile** sampling (spec §5) with **tiles-per-class
balancing** (spec §4):
1. Snap every station to a **64 px (640 m) tile grid** in its local UTM zone (pixel math
   verified identical to `io.lonlat_to_utm_pixel`). → 6,079 unique spatial tiles.
2. For each Sentinel-era quarter, emit candidate tiles: **kelp tiles** with `≥ MIN_KELP=15`
   kelp pixels (high-confidence kelp forests, ≥~13,500 m² canopy) and **water tiles** with
   `≥ MIN_OBS=150` observed pixels and zero kelp (bare-reef negatives). → 66,537 candidates
   (38,746 kelp, 27,791 water).
3. `sampling.balance_tiles_by_class(per_class=1000, total_cap=25000)` selects a balanced set;
   kelp is the rarer class and drives selection, yielding **1000 tiles, every one of which
   contains kelp (class 1) plus surrounding water (class 0)** — an ideal segmentation signal.
4. Reconstruct each 64×64 UTM 10 m tile by painting each 30 m station as a **3×3 block** of
   10 m pixels (nearest upsample; categorical), water first then kelp on top. Unpainted
   pixels remain nodata (255).

## Output

- 1000 × `locations/{id}.tif` (uint8, single band, 64×64, UTM 10 m, nodata 255) + `.json`.
- Class values present: 1000 tiles contain kelp (1), 997 contain water (0); pixel totals
  ≈ **water 1.09M, kelp 608k, nodata 2.40M**.
- Per-year tile spread (2016→2026): 94, 107, 129, 87, 103, 114, 121, 51, 83, 93, 18 — both
  UTM zones (10N/11N), whole coast Baja→WA.
- `metadata.json` records the two classes with descriptions, `nodata_value=255`,
  `task_type=classification`.

## Verification (spec §9)

- 3–5 tifs inspected: single band, uint8, EPSG:32610/32611 at 10 m, 64×64, nodata 255,
  values ⊆ {0,1,255} (0 bad tiles over all 1000).
- Every tif has a matching json; all `time_range`s are ≤ 92 days (quarterly); `change_time`
  null; metadata class ids cover all observed pixel values.
- **Spatial sanity**: all 1000 tile centers fall inside the KelpWatch coastal bbox
  (27.5–48.4°N, −124.7 to −114.8°W); kelp pixels form a few coherent nearshore components
  per tile (kelp-forest patches), and the same location recurs across quarters showing the
  expected seasonal variation. A full Sentinel-2 pixel overlay was not run (offshore kelp is
  hard to eyeball and needs S2 data-source setup); georeferencing was validated instead by
  exact agreement of the reconstruction with `io.lonlat_to_utm_pixel` and coastline placement.

## Caveats

- 30 m native product upsampled to 10 m (3×3 nearest) — labels are blocky at 30 m grain.
- Only kelp-capable reef pixels are observed; open ocean and land are nodata (255), so
  "water" (0) means *bare reef within the kelp mask*, not all sea surface. Downstream
  negative sampling (spec §5) supplies additional negatives.
- Low canopy fractions are the noisiest part of the source; presence/absence at `area>0`
  inherits some of that noise near the detection threshold.

## Reproduce

    # 1) download (idempotent): the script's download step / or manually:
    #    curl -L -o raw/<slug>/kelp_biomass_canopy_landsat.nc \
    #      https://pasta.lternet.edu/package/data/eml/knb-lter-sbc/74/33/c2bea785267fa434c40a22e2239bb337
    # 2) build labels:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.kelpwatch_landsat_sentinel_2_kelp_canopy
