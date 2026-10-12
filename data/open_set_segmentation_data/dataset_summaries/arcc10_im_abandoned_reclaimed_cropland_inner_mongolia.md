# ARCC10-IM (Abandoned & Reclaimed Cropland, Inner Mongolia)

- **Slug:** `arcc10_im_abandoned_reclaimed_cropland_inner_mongolia`
- **Registry family / label_type:** cropland / `dense_raster + sample points`
- **Task type:** classification (dense_raster)
- **Status:** completed — **1037 label patches**
- **Source:** figshare article 25687278 (Wuyun, Sun, Han, Li, Shi, Duan; *Scientific Data*),
  "A 10-meter annual cropland activity map and dataset of abandonment and reclaimed
  cropland." License **CC-BY-4.0** (per figshare record; manifest note of CC-BY-NC-ND is
  superseded by the record's stated CC BY 4.0).
- **Region:** Inner Mongolia, China. **Years:** 2016–2023 (all post-2016).

## Source contents

The article bundles four groups (README.txt), all GeoTIFF, EPSG:4326, ~10 m:

- **ARCC10-IM-ACA** — 8 annual cropland-**activity** maps (one raster per year 2016–2023),
  values `{1: inactive cropland, 2: active cropland}`, `0 = nodata / non-cropland`. Each
  raster is 321788 × 177294 px, uint8, LZW, 128×128 tiled. Also per-year reference
  **sample-point** shapefiles (`{year}_samples_merge.shp`, `type` field `{0: inactive,
  1: active}`; ~22.7k points in 2016).
- **ARCC10-IM_AC** — abandoned-cropland mask (`1 = abandoned`).
- **ARCC10-IM_RC** — reclaimed-cropland mask (`1 = reclaimed`).
- **ARCC10-IM-CLU** — cumulative 2016–2023 land use (`{1: continuously abandoned,
  2: unstable, 3: continuously active}`).

Only **ARCC10-IM-ACA** (1.9 GB `.7z`) was downloaded and used; the 8 annual `.tif`s were
extracted (huge `.ovr` overviews skipped). Raw at
`raw/arcc10_im_abandoned_reclaimed_cropland_inner_mongolia/`.

## Key decision — change-timing (spec §5/§8)

"Abandoned" / "reclaimed" / "unstable fallowing" are cropland **transition** classes that
the authors derive with a **multi-year sliding-window temporal segmentation** over the
2016–2023 annual maps. Those events are only resolved to a **year-of-change / multi-year
span**, far coarser than the spec's **~1-2 month** change-timing requirement, so per §5 they
are **not usable as dated change labels** and a `change_time` must **not** be forced. The
AC / RC / CLU transition layers were therefore intentionally **not** used.

Instead, following the spec's "recast as a persistent per-year state" allowance, this
dataset uses the **annual cropland-activity maps**: each pixel's **per-year** state
(active vs inactive cropland) is a persistent static class over that year's 1-year window.
This yields a clean, post-2016, valid **2-class classification** with `change_time = null`:

| id | name              | source value | meaning |
|----|-------------------|--------------|---------|
| 0  | inactive cropland | 1            | cropland not actively cultivated that year (fallow/abandoned/bare) |
| 1  | active cropland   | 2            | cropland under active cultivation that year |

`0` (non-cropland / nodata) → `255` ignore. `nodata_value = 255`.

## Processing (bounded-tile dense_raster sampling, spec §5)

Because the annual maps are large regional derived-product rasters, we do bounded-tile
sampling rather than global coverage:

1. **Scan** all 8 annual rasters in block-aligned 3840×3840 2D chunks (31,584 chunks) with
   `multiprocessing.Pool(64)` + `rslearn.utils.mp.star_imap_unordered`, splitting each into
   64×64 native blocks. A block is a candidate if ≥ **40%** of its pixels are mapped cropland
   (`MIN_VALID_FRAC=0.40`, so tiles are not dominated by ignore). A class counts as present at
   ≥ **5%** coverage (`MIN_CLASS_FRAC`). Per-chunk reservoir caps bound memory (inactive is
   rarer → higher cap). 54,227 candidates found (38,425 containing inactive, 53,100 active).
2. **Balance** with `sampling.balance_tiles_by_class(per_class=1000)` — tiles-per-class,
   rarest-first (inactive prioritized), 25k cap enforced. **1037 tiles** selected.
3. **Write** each 64×64 block reprojected to its **local UTM** zone at 10 m (`nearest`
   resampling, categorical) as a single-band uint8 GeoTIFF via `io.write_label_geotiff`, with
   a per-sample JSON (`io.write_sample_json`): 1-year `time_range` on the labeled year,
   `change_time=null`. Idempotent (skips existing `{id}.tif`; deterministic sample ids).

## Counts

- **num_samples: 1037** (each 64×64 px = 640 m tile).
- Class tile counts (a tile counts toward every class present): **inactive 0 ≈ 1034**,
  **active 1 ≈ 1029** (well-balanced; both near the 1000/class target).
- Per-year tiles: 2016:118, 2017:163, 2018:113, 2019:114, 2020:100, 2021:101, 2022:164,
  2023:164 (all 8 years represented).

## Verification (spec §9)

- All 1037 `.tif` are single-band uint8, 64×64, local UTM (EPSG:326xx/327xx) at 10 m; all
  have a matching `.json`. Distinct pixel values across the corpus = `{0, 1, 255}`, exactly
  the two class ids + nodata.
- All `time_range`s are 1-year and `change_time` is null everywhere.
- **Georeferencing:** 5 random tiles independently re-reprojected from the source raster
  matched the stored patches at **100%** pixel agreement; all sampled tile centers fall
  inside the Inner Mongolia bounding box and the correct year's raster. (No Sentinel-2
  overlay was rendered; alignment was validated against the source product, which is itself
  S1/S2-derived at 10 m.)

## Caveats

- Labels are an **ML-derived product** (S1/2 multi-feature stacking + ML), not in-situ truth;
  we require ≥40% cropland coverage per tile for higher confidence but do not filter by a
  per-pixel confidence layer (none provided).
- The dataset's headline abandonment/reclamation transition maps are deliberately **omitted**
  (change-timing rule). The per-year sample-point shapefiles were also not emitted as a
  separate point table — the dense per-year activity tiles already encode the same
  active/inactive state with richer spatial context.

## Reproduce

```bash
# raw already downloaded+extracted under raw/<slug>/ACA/ ; re-download if absent:
#   figshare files 49476444 (ARCC10-IM-ACA.7z) -> extract the 8 classified_cropland_*.tif
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.arcc10_im_abandoned_reclaimed_cropland_inner_mongolia
```
