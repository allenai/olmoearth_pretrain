# Kuro Siwo

- **Slug**: `kuro_siwo`
- **Status**: **completed** — task_type = **classification** (dense per-pixel, 3-class), **1576 samples**
- **Family / region**: flood (change) / global (43 Copernicus EMS flood activations, 40 post-2016)
- **Source**: Kuro Siwo — Bountos et al. 2024, *NeurIPS* (Orion-AI-Lab). Repo:
  https://github.com/Orion-AI-Lab/KuroSiwo . Labels from the companion label-only release
  **`Orion-AI-Lab/KuroSiwo-annotations`** (https://github.com/Orion-AI-Lab/KuroSiwo-annotations).
- **License**: CC-BY.
- **Access**: public GitHub, no credentials.

## What the dataset is

Kuro Siwo is a global, manually annotated (by SAR experts) multi-temporal **Sentinel-1**
flood-mapping benchmark built on **Copernicus EMS Rapid Mapping** flood activations
(`EMSR<act_id>_<region>`). The full GRD/SLC products bundle the SAR imagery with the masks
in large Dropbox / Hugging Face archives, but pretraining supplies its own S1/S2 imagery —
**only the labels are needed**. Kuro Siwo publishes its annotation polygons separately in
the small companion repo `KuroSiwo-annotations` (git-cloned, a few hundred MB, no SAR), and
the per-event acquisition/reference dates live in the main repo's `catalogue/catalogue.yaml`.
We use only those two sources (spec §3/§8 label-only extraction; no imagery downloaded).

Per activation, one or more AOIs, each a mapped revision folder with three EPSG:3857
shapefiles:
- `aoi/aoi.shp` — the mapped AOI extent (defines the observed region);
- `event/event.shp` — the observed flood-water extent polygons for the event;
- `hydro/hydroA.shp` — reference permanent-water bodies (rivers, lakes, reservoirs).

The catalogue lists 43 activations. Three are entirely pre-2016 and dropped on the
Sentinel-era rule (EMSR118 Spain 2015, EMSR130 Myanmar 2015, EMSR147 Cumbria 2015),
leaving **66 processable AOI revisions across 40 activations**; selected-sample event years
span **2016 → 2022**.

## Class scheme (dense per-pixel classification)

| id  | name            | definition |
|-----|-----------------|------------|
| 0   | no_water        | inside the mapped AOI but neither flood-water nor permanent water at the event acquisition (dry land / non-water observed by the annotation) |
| 1   | permanent_water | reference permanent open water (rivers, lakes, reservoirs) from the Copernicus EMS hydrography layer (`hydroA`); painted last so it **wins** flood/no-water overlaps (a flooded permanent channel stays permanent water) |
| 2   | flood           | observed flood-water extent at the event's Sentinel-1 acquisition (`event` delineation), excluding pixels reclassified as permanent water |
| 255 | nodata/ignore   | outside the mapped AOI (unobserved) |

This is Kuro Siwo's native MLU scheme; "permanent wins over flood" matches sen1floods11's
convention. Invalid / outside-AOI pixels become nodata (255).

## Processing (label_type = dense_raster, from polygons)

- Each AOI is reprojected to its **local UTM zone at 10 m** (UTM picked from the AOI
  centroid; 22 distinct zones across the corpus).
- The AOI is rasterized **once** across its whole pixel grid (no_water=0 inside AOI /
  255 outside), then flood (`event`) and permanent (`hydroA`) polygons are burned in
  (permanent last, wins). Even the largest AOI — **Pakistan 2022, ~25k × 41k px, ~48k
  flood polygons** — is ~1 GB uint8, so a single whole-AOI rasterization is memory-safe and
  fast (~70 s), versus re-rasterizing tens of thousands of giant polygons per window.
- The full label array is sliced into **64×64** tiles; only tiles containing ≥32 px of a
  water class (flood or permanent) **and** ≥50 % inside the AOI are kept. no_water
  co-occurs inside those tiles as the surrounding land.
- Per-AOI candidate caps (**400 flood-bearing, 200 permanent-only** tiles, seeded
  subsample) keep a few enormous AOIs from dominating and preserve geographic diversity.
- **Sampling**: tiles-per-class balanced (spec §5), ≤ **1000 tiles/class**, rarer class
  filled first, under the 25k cap. From **20,641 candidate tiles**, **1576** were selected
  (tiles containing: **no_water 1501, permanent_water 1164, flood 1000** — flood is the
  rare/priority class and hits its cap; a tile counts toward every class it contains), from
  **40 distinct activations**. Selected-sample change-years: 2016:1, 2017:125, 2018:536,
  2019:90, 2020:359, 2021:339, 2022:126.

## Time-range / change handling

Flood extent is a transient **change/event** label with a **day-precise acquisition date**,
so it is processed as a change label (spec §5), not a static presence class:
- `change_time` = the activation's `ref_date` from `catalogue.yaml`, resolved to the day
  (≪ the ~1–2-month timing-precision requirement).
- `time_range` = a **360-day window centered on `change_time`** (≤ 1 year). The Sentinel-1
  event acquisition lands on the reference date, comfortably inside the window, so imagery
  sampled from the window spans the flood and the where-mask stays aligned.

## Output

- `datasets/kuro_siwo/metadata.json` (3 classes, nodata 255).
- `datasets/kuro_siwo/locations/{id}.tif` — single-band uint8, local UTM, 10 m, 64×64.
- `datasets/kuro_siwo/locations/{id}.json` — crs / pixel_bounds / time_range / change_time /
  source_id (`{EMSR..._region}/aoi{aoi_id}/{rev}/r{row}_c{col}`) / classes_present.
- `raw/kuro_siwo/` — cloned `KuroSiwo-annotations` + `catalogue.yaml` + `SOURCE.txt`
  (label-only; no SAR imagery).

## Verification (spec §9)

- All 1576 `.tif` are single-band **uint8, 64×64, UTM at 10 m** (22 zones); pixel values ∈
  {0, 1, 2, 255}; every `.tif` has a matching `.json`.
- All `time_range`s are exactly **360 days**; `change_time` set on every sample and the
  window is centered on it; all change-years ≥ 2016.
- `metadata.json` class ids {0,1,2} cover all non-nodata values in the tifs.
- Spatial sanity: tile centers reproject to plausible flood locations — for every
  activation the centers fall inside the named country's bbox, with the sole "mismatch"
  being **EMSR1111007_Nepal**, whose AOI is *Patna* at ~25.6 °N, 85 °E, i.e. Bihar, India
  just across the border (the flood spanned the Nepal/India border) — a correct location,
  not a georeferencing error. Labels are rasterized directly from georeferenced expert
  polygons (not derived from imagery), so georeferencing is exact by construction; a full
  Sentinel-2 visual overlay was not rendered in this headless run (and live S2 ingestion was
  avoided to sidestep the transient network cancels that interrupted earlier attempts).
- Re-running the script is idempotent (existing `{id}.tif` are skipped).

## Caveats

- `no_water` (0) is the observed non-water land inside each AOI; for coastal AOIs a few
  tiles may include sea labeled `no_water` (the EMS AOI mask, not a separate sea mask,
  defines the observed region). Impact is minor (tiles are centered on flood/permanent
  water) and handled downstream.
- Per-AOI caps (400/200) mean a handful of very large activations (e.g. Pakistan 2022) are
  subsampled rather than exhaustively tiled — intentional, to keep geographic diversity and
  bound scan cost; the dataset only needs ~1000 flood tiles overall.
- Reference `hydroA` permanent water is co-registered to the event date; where flood and
  permanent overlap, permanent wins (a flooded river channel stays `permanent_water`).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.kuro_siwo --workers 64
```
Clones the annotation polygons into `raw/kuro_siwo/KuroSiwo-annotations/` and fetches
`catalogue.yaml` (label-only; SAR imagery not downloaded), then writes all outputs. Use
`--probe` to scan and report class balance without writing tiles.
