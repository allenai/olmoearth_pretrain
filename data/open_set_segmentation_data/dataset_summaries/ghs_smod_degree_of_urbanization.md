# GHS-SMOD (Degree of Urbanization)

- **Slug:** `ghs_smod_degree_of_urbanization`
- **Source:** EC JRC / GHSL — GHS Settlement Model grid, release R2023A
  (<https://human-settlement.emergency.copernicus.eu/ghs_smod2023.php>)
- **Task type:** classification
- **Label type:** dense_raster (global derived-product map)
- **License:** open + attribution (CC BY 4.0)
- **num_samples:** 7000 (1000 / class, all 7 classes)

## Source

GHS-SMOD encodes the UN **Degree of Urbanisation (DEGURBA)** level-2 rural–urban
classification per grid cell. It is distributed as a single-band global raster in
**Mollweide (ESRI:54009) at 1000 m** native resolution
(`GHS_SMOD_E<epoch>_GLOBE_R2023A_54009_1000`). We used epoch **E2020** (a recent,
Sentinel-era epoch within the manifest range 2016–2025). The whole global file is only
~29 MB (zip), downloaded directly from the JRC open-data FTP without credentials:

```
https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_SMOD_GLOBE_R2023A/
  GHS_SMOD_E2020_GLOBE_R2023A_54009_1000/V1-0/
  GHS_SMOD_E2020_GLOBE_R2023A_54009_1000_V1_0.zip
```

## Class mapping (source code → id)

The 8 source codes are collapsed to the 7 manifest classes by merging very-low (11) and
low (12) density rural into one class.

| id | name | source code(s) |
|----|------|----------------|
| 0 | water | 10 |
| 1 | very-low/low-density rural | 11, 12 |
| 2 | rural cluster | 13 |
| 3 | suburban | 21 |
| 4 | semi-dense urban cluster | 22 |
| 5 | dense urban cluster | 23 |
| 6 | urban centre | 30 |

Source `-200` (no data) → label nodata `255`.

## Processing

Global derived-product map → **bounded-tile sampling** per the spec. The single global
1 km file was downloaded (no global tiling needed at this size). Grid-cell counts per
class in E2020 range from ~310k (dense urban cluster) to ~365M (water); all classes have
far more than 1000 cells, so the dataset is trivially class-balanced at the 1000/class
target (7×1000 = 7000 ≪ 25k cap).

For each class, up to 1000 grid cells were sampled **globally** (seeded uniform random
over all cells of that class). Around each selected cell centre a **64×64** label tile in
a **local UTM** projection at **10 m** was cut and reprojected from the 1 km Mollweide
source with **nearest** resampling (categorical). Written single-band `uint8`, nodata 255,
atomically via the shared `io.write_label_geotiff`. Multiprocessing `Pool(64)`.

**Resampling note (important):** this is a heavy **1 km → 10 m upsampling (100×)**. A
64×64 @10 m tile is 640 m — smaller than one native 1 km cell — so each tile is
essentially the homogeneous DEGURBA class at that location (some tiles straddle a cell
boundary and contain 2–3 classes). This is intentional: the DEGURBA class is *defined* on
the 1 km grid, so it cannot be resolved more finely. The manifest note mentioning a 100 m
native product does not apply to R2023A SMOD, which is a 1 km product.

## Time range

Static/epoch label → 1-year window anchored on the E2020 epoch:
`[2020-01-01, 2021-01-01)`. No `change_time`.

## Sample counts per class

All classes: **1000** (water, very-low/low-density rural, rural cluster, suburban,
semi-dense urban cluster, dense urban cluster, urban centre).

## Verification

- 7000 `.tif` + 7000 matching `.json`; every tif is single-band `uint8`, UTM CRS,
  10 m resolution, 64×64, nodata 255; all pixel values are valid class ids 0–6 or 255.
- All `time_range`s are the 1-year 2020 window.
- Geo-sanity: urban-centre samples land on Philadelphia, Dubai, Osaka, and the Fergana
  Valley; a water sample sits in the Pacific; a mixed suburban+dense tile near Shenyang.
  Coordinates are taken directly from the authoritative 1 km grid so georeferencing is
  exact (no S2 overlay needed to confirm placement).
- Idempotent: re-running skips existing `{sample_id}.tif`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ghs_smod_degree_of_urbanization
```
