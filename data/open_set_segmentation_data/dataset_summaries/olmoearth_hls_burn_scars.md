# OlmoEarth HLS Burn Scars

- **Slug**: `olmoearth_hls_burn_scars`
- **Status**: completed
- **Task type**: classification (dense per-pixel, binary)
- **Num samples**: 1505 tiles (64×64 @ 10 m, local UTM)
- **Family / region**: fire / CONUS (United States)
- **License**: CC-BY-4.0

## Source

NASA/IBM **HLS Burn Scars** — binary burn-scar segmentation over 512×512 Harmonized
Landsat-Sentinel (HLS) 30 m scenes across the CONUS, 2018–2021, with per-pixel masks
derived from **MTBS** (Monitoring Trends in Burn Severity). Public source:
HuggingFace `ibm-nasa-geospatial/hls_burn_scars`.

We consumed the **internally-staged rslearn copy** (`have_locally: true`) at
`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/hls_burn_scars/` — nothing was
downloaded; `raw/olmoearth_hls_burn_scars/SOURCE.txt` points at it. Only the
`label_raster` layer is used (the co-located HLS imagery bands are ignored — pretraining
supplies its own imagery).

**Georeferencing verified real.** The staged copy has genuine per-window UTM projections
at 10 m (not the dummy EPSG:3857 (0,0) bounds seen in the sibling `olmoearth_pastis`
copy). Sample centroids land across CONUS fire regions (California, Sierra Nevada,
Arizona, Texas, Oklahoma, the Southeast, etc.); UTM zones span EPSG:32610–32618, matching
each scene's MGRS tile. A per-tile array check confirmed every written label byte-matches
the source subtile.

## Staged layout

The staged copy already **resampled the native 30 m masks to 10 m (nearest)** in local
UTM and split each 512×512 (30 m) scene into a **6×6 grid of 256×256 (10 m) windows**
named `HLS_S30_{MGRS}_{YEAR}{DOY}_r{r}_c{c}`, under `windows/{train,val}/` (28,834 windows
= 19,378 train + 9,456 val, over 540+ unique scenes). Each window's `label_raster`
(`layers/label_raster/label/geotiff.tif`) is **int16** with values `0` = not burned,
`1` = burned, `-1` = nodata, and its `metadata.json` carries the UTM projection and a tight
~2-day acquisition `time_range` around the HLS scene date.

## Class mapping (nodata = 255)

| id | name | meaning |
|----|------|---------|
| 0  | unburned | HLS pixel outside the burn-scar mask (observed, non-burnt) |
| 1  | burned   | HLS pixel inside the MTBS-derived burn-scar mask |
| 255 | nodata  | source `-1` (unobserved / outside-scene fill) |

## Processing

`label_type = dense_raster`. Each staged 256×256 (10 m) window is cut into a **4×4 grid of
64×64 (10 m) tiles** (source is already UTM 10 m, so **no further resampling** here — the
30 m→10 m nearest resample happened upstream in the staged copy). Source `int16` values are
mapped to `uint8` (0/1, `-1`→255). Tiles more than half nodata are skipped; a tile counts
toward a class only with ≥ 32 px of it.

Sampling is **tiles-per-class balanced** (spec §5) via
`sampling.select_tiles_per_class`, rarest class (burned) filled first, up to
`PER_CLASS = 1000` tiles/class under the 25k cap (same convention as `cabuar`/`floga`).
From 460,247 candidate tiles, **1505** were selected.

Tiles containing each class (a tile may contain both):

| class | tiles |
|-------|------:|
| unburned | 1000 |
| burned   | 1044 |

The CRS is re-derived to a **canonical EPSG UTM** projection from each window centroid; the
staged CRS is a numerically-identical non-EPSG WGS84-UTM WKT, so pixel bounds are preserved
exactly.

## Time range & change-label decision

A burn scar is a **change/event label** (forest → burned). The HLS scene is acquired
shortly after the fire (MTBS-derived scenes are chosen to capture the burn scar), so:

- **`change_time` = the HLS acquisition date** (midpoint of the window's ~2-day acquisition
  `time_range`), e.g. `HLS_S30_T10SDH_2020248` → 2020-09-04.
- **`time_range` = a 360-day window centered on `change_time`** (±180 days, ≤ 1 year).

Rationale (spec §5): the fire ignition falls a few weeks-to-months before the HLS
acquisition, comfortably inside the ±180-day window, so imagery the pretraining pairs with
this label brackets the forest→burned transition and the where-mask stays aligned. This is
the same treatment as the sibling burn datasets `cabuar_california_burned_areas`
(change_time = post-fire S2 acquisition) and `floga` (change_time = ignition). All scene
dates are 2018–2021, i.e. post-2016 (Sentinel era) — no pre-2016 filtering needed.

We chose the **change-label** framing (change_time set) over the persistent-state framing
(change_time = null) because HLS burn-scar scenes are deliberately acquired close to the
fire, so the acquisition date reliably brackets the event to within ~1-2 months.

## Verification (spec §9)

- 1505 `.tif` + 1505 matching `.json`; all `.tif` single-band **uint8, 64×64, 10 m**, UTM
  EPSG 32610–32618; pixel values ⊆ {0, 1, 255}.
- All sidecars have `change_time` set, a **360-day** `time_range` with
  `t0 ≤ change_time ≤ t1`, change years 2018–2021.
- 8 random tiles byte-match their source subtiles (label placement exact).
- Centroids land in CONUS fire regions.
- Re-running is **idempotent** (existing `{sample_id}.tif` skipped).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_hls_burn_scars
```

## Caveats

- Binary dataset, so `1000/class` yields ~1.5k tiles (well under the 25k cap), consistent
  with the other burn datasets; downstream assembly adds negatives from other datasets.
- Only the `label_raster` layer is consumed; the staged HLS imagery is ignored.
- The upstream 30 m→10 m nearest resample means burn-scar boundaries are at 30 m native
  precision (block-replicated to 10 m), not true 10 m detail.
