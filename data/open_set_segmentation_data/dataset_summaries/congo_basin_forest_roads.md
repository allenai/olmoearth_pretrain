# Congo Basin Forest Roads

- **Slug:** `congo_basin_forest_roads`
- **Status:** completed
- **Task type:** classification (positive-only line segmentation)
- **Samples:** 22,047 label tiles (single foreground class)
- **Label type:** lines → rasterized dilated masks
- **Family / region:** deforestation / Congo Basin (Central Africa)

## Source

"Forest roads (Congo Basin)" — Zenodo record
[13739812](https://doi.org/10.5281/zenodo.13739812) (doi 10.5281/zenodo.13739812),
CC-BY-4.0. Companion paper: Slagter B., Fesenmyer K., Hethcoat M., Belair E., Ellis P.,
Kleinschroth F., Peña-Claros M., Herold M., Reiche J. (2024). *Monitoring road
development in Congo Basin forests with multi-sensor satellite imagery and deep
learning.* Remote Sensing of Environment, doi:10.1016/j.rse.2024.114380.

A deep-learning model is applied to 10 m Sentinel-1 + Sentinel-2 imagery to detect forest
roads across the Congo Basin monthly from 2019 onward. This release covers 2019–2023
(46,311 km of roads). Forest roads are a logging / selective-logging access indicator and
a **forest-degradation proxy**.

### Access method

Single 30 MB archive `forestroads_afr_2019-01_2023-12.zip`, downloaded via
`download.download_http` from the Zenodo file API and unzipped into `raw/{slug}/`. It
contains the road lines in both `.shp` and `.geojson`; we read the shapefile. No
credentials required; no imagery pulled (pretraining supplies its own).

### Source data structure

- **355,995 `LineString` segments**, CRS **ESRI:54009** (World Mollweide, metres).
- Segment lengths: min 4.5 m, median ~42 m, mean ~130 m, 95th pct ~549 m, max ~10.5 km.
- Attributes: `NetworkID` (connected-network id), `SegLenM`, `NetLenM`, `Month`, `Year`
  (segment **opening** month/year), `MonthNum` (months since 2019-01).
- Opening-year distribution of source segments: 2019: 55,653; 2020: 65,336; 2021: 81,009;
  2022: 86,958; 2023: 67,039.

## Suitability (accept)

ACCEPTED. The product is *itself* derived from 10 m Sentinel-1+2 imagery with a
deep-learning detector, i.e. these are exactly the linear disturbance features resolvable
at 10 m in Sentinel imagery. A centerline dilated to ~2–3 px is a meaningful 10 m label.
(Spec §4 "lines": rasterize to a mask with small dilation; reject only if not observable
at 10–30 m — not the case here.)

## Label / class mapping

Single foreground class, **positive-only** (spec §5):

| id | name | meaning |
|----|------|---------|
| 0 | `forest_road` | mapped forest-road segment (dilated to ~20–30 m) |
| 255 | (nodata) | all non-road pixels — ignore |

We do **not** fabricate a background class or negative tiles; the assembly step supplies
negatives from other datasets. `NetworkID` / `SegLenM` / `NetLenM` / `MonthNum` are
retained as source provenance but collapsed to the single road class per the task spec.

## Processing recipe

1. Partition all road segments onto a fixed **640 m grid in the source (Mollweide) CRS**.
   A segment is assigned to every grid cell its bounding box overlaps (segments are short,
   so 1–4 cells for almost all). **94,284 cells** are occupied.
2. Each occupied cell → one **64×64 (640 m) tile** in the local **UTM** projection at
   **10 m/pixel**, centered on the cell center. Every segment overlapping the cell is
   reprojected to UTM pixel space, buffered by ~1 px (`all_touched`) → ~20–30 m (2–3 px)
   wide line, and rasterized (value 0), clipped to the tile; non-road = nodata (255).
3. Tiles whose road mask has `< 3` road pixels are dropped (trivial slivers from
   bbox-overlap membership). 2,953 of the sampled cells dropped this way.
4. **Sampling:** 94,284 > 25,000 cap, so a deterministic seeded (seed 42) random subsample
   of 25,000 cells is taken before rasterization; 22,047 survive the min-pixel filter.
5. Written with `multiprocessing.Pool(64)`; idempotent (skips already-written `.tif`s).

Output dtype uint8, single band, north-up UTM at 10 m, nodata 255 — verified.

## Time range & change handling

Each segment carries an **opening month/year**, which could support a change-label
framing (timing is resolved to ~1 month, within the spec §5 ≤1–2 month precision).
However, a road is a **persistent** feature once built, so per the task instruction and
spec §5's persistent-post-change-state clause we treat this as a **presence/state**
label: `change_time = null`, static **1-year window**. The window for each tile is
anchored on the **latest opening year** among the tile's segments, so imagery in that
window post-dates construction of every mapped road in the mask (earlier roads persist).
All anchor years fall in the manifest range [2019, 2024].

Anchor-year distribution of the **written** tiles: 2019: 3,220; 2020: 3,746; 2021: 4,405;
2022: 5,052; 2023: 5,624.

## Sample counts

- Total: **22,047** positive tiles, single class `forest_road` (id 0).
- Road-pixel fraction per tile (sampled): median ~4%, max ~12% — expected for thin
  dilated lines in a 640 m tile.

## Verification (spec §9)

- Opened 5 output `.tif`s: single band, uint8, UTM (e.g. EPSG:32632/32633) at 10 m,
  64×64, values ∈ {0, 255}, nodata 255. ✔
- All 22,047 `.tif`s have a matching `.json`; all `time_range`s ≤ 1 year;
  `change_time = null`. ✔
- **Spatial/temporal sanity:** for 5 high-road-fraction tiles, fetched a low-cloud
  Sentinel-2 L2A scene (public earth-search STAC) in the tile's CRS/bounds/time and
  compared road-labeled vs non-road pixels. 3/5 clearly show road pixels **brighter**
  (RED +150–200) and **lower NDVI** (Δ ~0.08–0.11) than surrounding forest — the expected
  bare-track / logging-road signature, i.e. labels overlay sensibly. 1 scene was
  cloud/haze-contaminated and 1 was neutral (canopy-closed / narrow single-track). ✔
- Re-running the script is idempotent (all writes report `skip`). ✔

## Caveats

- The source is a **derived deep-learning product**, not in-situ reference; omitted or
  spuriously-detected roads are possible (sampled homogeneously across the network per the
  spec's derived-product guidance).
- Narrow single-track roads sit near the 10 m resolution limit and can be canopy-obscured,
  so not every labeled pixel is spectrally obvious in a single S2 date; the yearly window
  and thin-line dilation partly mitigate this.
- Bbox-overlap cell membership plus the min-pixel filter mean a few road slivers near tile
  edges are dropped; this only trims trivial content, never mislabels.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.congo_basin_forest_roads
```

Outputs to
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/congo_basin_forest_roads/`
(`metadata.json`, `locations/{id}.tif` + `{id}.json`, `registry_entry.json`); raw source
in `.../raw/congo_basin_forest_roads/`.
