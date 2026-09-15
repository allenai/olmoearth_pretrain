# OlmoEarth WorldCereal cropland

- **Slug:** `olmoearth_worldcereal_cropland`
- **Status:** completed
- **Task type:** classification (binary)
- **Samples:** 2000 (1000 Cropland + 1000 Non-Cropland)
- **Output:** `datasets/olmoearth_worldcereal_cropland/points.geojson` (spec 2a point table) + `metadata.json`

## Source

Local rslearn dataset (`have_locally: true`):
`/weka/dfive-default/rslearn-eai/datasets/crop/worldcereal_cropland/20250422`.
This is an existing OlmoEarth eval derived from the ESA WorldCereal Reference Data Module
(RDM) harmonized in-situ reference. One window group (`h3_sample100_66K`) with **65,759
windows**, each a single labeled 10 m pixel (H3-cell sampled).

Each window:
- `metadata.json`: UTM `projection` (52 distinct UTM zones, global), 1×1 `bounds`, a
  ~1-month observation `time_range`, and `options` with WorldCereal `sample_id`,
  `ewoc_code`, `level_123`, H3 cell, `quality_score_lc`/`quality_score_ct` (93–100 in the
  sampled set), and source `split` (train/val).
- vector `label` layer `data.geojson`: one 1×1 polygon feature with
  `properties.category` = `"Cropland"` or `"Non-Cropland"`.

## Access

No download — read the local rslearn dataset directly. `raw/olmoearth_worldcereal_cropland/SOURCE.txt`
points at the source path. All 65,759 window `metadata.json` + `label/data.geojson` pairs
were scanned in parallel (`multiprocessing.Pool(64)`, ~1m49s); none were dropped.

## Label mapping

Manifest class order → id: `Cropland`→0, `Non-Cropland`→1. Class id read directly from
each label feature's `category`. This is a **sparse-point** dataset (each label is a single
10 m pixel), so it is written as one dataset-wide `points.geojson` (spec §2a), **not**
per-sample GeoTIFFs. Point lon/lat computed from the window's UTM pixel-center via
`io.pixel_center_lonlat` (verified to match the coords encoded in each window name).

## Sampling & time range

- **Balancing:** `balance_by_class(per_class=1000)` → 1000 per class, 2000 total (well under
  the 25k cap). Source raw distribution is near-balanced (~55% Cropland / ~45%
  Non-Cropland), so no rare-class concerns.
- **Splits:** all source splits (train + val) used as candidates (spec §5).
- **Time range:** cropland is a seasonal/annual label, so each point gets a **1-year window
  anchored on its labeled year** (`io.year_range`). Selected-sample year distribution:
  2017:188, 2018:746, 2019:291, 2020:195, 2021:529, 2022:30, 2023:21 — **all post-2016**
  (Sentinel era). A guard drops any pre-2016 sample (none present).
- No change labels.

## Caveats

- Global coverage but not spatially uniform (H3-sampled reference; source labels are
  concentrated where WorldCereal RDM has in-situ data — e.g. EuroCrops/LPIS in Europe).
- 64k+ source points available; capped at 1000/class per the classification target. Raising
  the cap later is trivial (re-run with a larger `--per_class`, subject to the 25k cap).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_worldcereal_cropland
```
Idempotent: re-running rewrites `points.geojson`/`metadata.json` from the source scan.
