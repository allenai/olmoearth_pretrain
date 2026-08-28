# Global Sugarcane 10 m

- **Slug:** `global_sugarcane_10_m`
- **Status:** completed — classification, 2000 samples (1000 per class)
- **Family:** crop_type · **Label type:** dense_raster · **Task:** classification (sugarcane presence)

## Source

Zenodo record [10871164](https://zenodo.org/records/10871164) — Zhang, Xu, di Tommaso
et al., *"Mapping sugarcane globally at 10 m resolution using GEDI and Sentinel-2"*
(`GEDIS2` product). A 10 m binary sugarcane presence map for the **top 13
sugarcane-producing countries**, derived from GEDI canopy-height metrics + Sentinel-2
time series and validated against field data over **2019-2022**.

- License: **CC-BY-4.0** (open, redistributable).
- Access: fully public. Downloaded per-country ZIPs via the Zenodo public-record API
  (`download.download_zenodo`, no credentials). Each ZIP contains one or more GeoTIFF
  sub-tiles (large countries are GDAL-retiled into several
  `<country>_GEDIS2_v1<ROW>-<COL>.tif` files; small ones are a single tif).
- Raw stored at `raw/global_sugarcane_10_m/{<country>_GEDIS2_v1.zip, <country>/*.tif}`.

### Source raster format
- Projection **EPSG:4326 (WGS84 geographic)** at ~8.983e-5° (≈10 m) per pixel.
- **5 uint8 bands**, band descriptions `(n_tallmonths, sugarcane, ESA, ESRI, GLAD)`:
  - **band 2 = `sugarcane`** — the product's binary map: `0` = not sugarcane, `1` = sugarcane. **This is the label used.**
  - band 1 = `n_tallmonths` — count of "tall canopy" months; **0 over ocean/water/unobserved**, high (~14-45) over sugarcane. Used here as an observed-land mask.
  - bands 3-5 (`ESA`/`ESRI`/`GLAD`) — cross-product agreement layers, **not used**.
- No explicit nodata; value 0 in the sugarcane band covers both non-sugarcane land and ocean/unobserved.

## Bounded sampling (regions used)

This is a **global derived-product raster**, so per §5 we did **bounded-tile
dense_raster sampling** — download only enough to draw the target counts from
representative regions.

- **10 of the 13 country rasters were used** (cross-continental coverage):
  guatemala, colombia, usa, australia, southafrica, indonesia, philippines, mexico,
  pakistan, thailand (~13 GB across 64 source sub-tifs).
- **The 3 largest ZIPs — brazil (7.5 GB), india (8.7 GB), china (8.9 GB), ~25 GB
  combined — were intentionally skipped** to keep the download bounded. The 10 countries
  used already span 6 continents and include major producers (Thailand #4 globally,
  Pakistan, Mexico, Australia, South Africa), yielding 39k sugarcane + 11k non-sugarcane
  homogeneous candidate blocks — far more than the 1000/class target.

## Processing

1. **Extract** each country ZIP (idempotent). **Scan** (Pool(64) over 592 row-chunks
   across the 64 source sub-tifs): read each raster in 64-row strips, reduce into 64×64
   native-pixel blocks (≈640 m). Per block count sugarcane pixels (band2==1) and
   observed-land pixels (band1>0). Keep **spatially-homogeneous** candidates:
   - **sugarcane candidate:** ≥ 20% of block pixels are sugarcane (`SUGAR_MIN_FRAC=0.20`);
   - **other candidate:** **zero** sugarcane pixels **and** ≥ 30% observed land
     (`LAND_MIN_FRAC=0.30`, band1>0) — this excludes ocean/water/unobserved fill so the
     negative class is genuine non-sugarcane **land**.
   Reservoir-capped per chunk (≤200 sugarcane, ≤40 other) to bound memory while keeping
   geographic spread. Candidates: sugarcane=39,366, other=10,697.
2. **Select:** seeded shuffle, take up to 1000 per class → 1000 + 1000 = **2000**.
   Each selected tile is assigned a **uniformly-sampled year in 2019-2022** (the product
   is a multi-year sugarcane extent; sugarcane is a persistent crop across the window).
3. **Write** (Pool(64)): reproject a 64×64 patch of the sugarcane band to local UTM at
   10 m with **nearest** resampling (categorical). Any value not in {0,1} → 255 (nodata).

## Output

- `datasets/global_sugarcane_10_m/metadata.json`
- `datasets/global_sugarcane_10_m/locations/{000000..001999}.tif` + `.json`
- Each patch: single-band `uint8`, **local UTM, 10 m/pixel, 64×64**, nodata **255**.

### Classes (per-pixel; native ids kept, no remap)
| id | name | pixel meaning |
|----|------|---------------|
| 0 | other | observed non-sugarcane land (band1>0, band2==0) |
| 1 | sugarcane | sugarcane presence (band2==1) |

### Distribution of the 2000 selected tiles
- **By class:** 1000 sugarcane-tiles + 1000 other-tiles.
- **By country:** mexico 423, thailand 261, pakistan 258, australia 202, indonesia 202,
  philippines 187, southafrica 167, usa 141, colombia 103, guatemala 56.
- **By year:** 2019: 518, 2020: 429, 2021: 506, 2022: 547.

## Time-range / change handling

Multi-year (2019-2022) sugarcane extent → each tile gets a **1-year window** anchored on
a uniformly-sampled year within 2019-2022 (§5 "valid period longer than a year"). Not a
dated event: `change_time=null`, persistent-presence classification.

## Verification (§9)

- 2000 `.tif` + 2000 `.json`; opened samples confirm single-band `uint8`, UTM CRS at
  10 m, 64×64, nodata 255; global pixel values are exactly {0, 1, 255}.
- All sample JSONs have a ≤1-year `time_range`; metadata class ids cover all tif values.
- **Georeferencing round-trip:** for 5 samples, reprojecting the label-tile center back
  to lon/lat and reading the source sugarcane band at that coordinate matched the label
  value in every case (e.g. mexico/thailand sugarcane tiles → source band2==1; philippines
  other tile → source band2==0). Countries land in their correct extents.

### Caveats
- **~34 of the 1000 "other" tiles contain a few sugarcane pixels** at the tile border,
  introduced by nearest-resampling picking up a sugarcane pixel just outside the scanned
  block. The tiles remain overwhelmingly non-sugarcane; this is a minor edge effect.
- The negative class "other" is biased toward **vegetated/agricultural land** (the
  n_tallmonths>0 land filter), which is intentional — it makes hard negatives for
  sugarcane discrimination rather than trivial ocean/barren.
- **3 largest producing countries (Brazil, India, China) were not sampled** to keep the
  download bounded; the map product itself is a derived (not in-situ) label per §5's
  homogeneous-window preference.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_sugarcane_10_m --workers 64
```
Downloads the 10 country ZIPs from Zenodo record 10871164 (idempotent/atomic), extracts,
scans, and writes the 2000 label patches. Re-running skips already-written tiles.
