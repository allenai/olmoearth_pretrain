# UNEP-WCMC Global Warm-Water Coral Reefs

- **Slug:** `unep_wcmc_global_warm_water_coral_reefs`
- **Status:** completed
- **Task type:** classification (positive-only, single foreground class)
- **Samples written:** 1000 label GeoTIFFs (64×64, UTM, 10 m)

## Source

UNEP-WCMC, WorldFish Centre, WRI, TNC (2021). *Global distribution of warm-water coral
reefs*, version 4.1 (product WCMC-008), the most comprehensive global baseline map of
tropical/subtropical coral reefs. Compiled from the Millennium Coral Reef Mapping Project
(IMaRS-USF and IRD 2005; IMaRS-USF 2005), the World Atlas of Coral Reefs (Spalding et al.
2001) and other sources. Data DOI [10.34892/t2wk-5t34](https://doi.org/10.34892/t2wk-5t34);
landing page https://data.unep-wcmc.org/datasets/1.

- **License:** UNEP-WCMC General Data License (excluding WDPA) — free use with attribution,
  **non-commercial**, and no redistribution of the *source* data. We derive internal label
  rasters for pretraining only (not a redistributable copy of the source), and record the
  required citation below. Manifest listed this as "free + attribution".
- **Citation:** UNEP-WCMC, WorldFish Centre, WRI, TNC (2021). Global distribution of
  warm-water coral reefs, v4.1. Cambridge (UK): UN Environment Programme World Conservation
  Monitoring Centre.

## Access (no credential)

Single public S3 download (~208 MB), no login/credential needed:
`https://datadownload-production.s3.us-east-1.amazonaws.com/WCMC008_CoralReefs2021_v4_1.zip`

The zip contains two EPSG:4326 shapefile layers:
- `WCMC008_CoralReef2021_Py_v4_1` — **17,504 reef-presence polygons** (real footprints; used).
- `WCMC008_CoralReef2021_Pt_v4_1` — 925 point-only reefs (`GIS_AREA_K = 0`, no footprint; excluded).

## Label mapping / class scheme

Single foreground class, **positive-only / no-background** (spec §5):
- `id 0` = **warm-water coral reef** presence.
- `255` = nodata/ignore (all non-reef / unmapped pixels).

No background class is written; the assembly step supplies negatives from other datasets.
Reef footprints are rasterized from the presence polygons with `all_touched=True` so thin
reef lines are not dropped.

## 10 m suitability & minimum reef size

Larger reef complexes are clearly resolvable in shallow-water optical at 10 m. To keep only
resolvable footprints:
- Polygons with `GIS_AREA_K >= 0.01 km²` (~≥100 px @ 10 m) are kept (11,510 of 17,504);
  sub-0.01 km² slivers are dropped.
- Each written tile must carry **≥ 16 reef pixels** (all 1000 selected tiles passed).
- The **925 point-only reefs are excluded** — single sub-pixel locations with no footprint.

Written reef pixels per tile: min 77, median 2619, mean 2553, max 4096 (full tile). 239
tiles are fully reef; the rest mix reef + nodata.

## Sampling (bounded global; spec §5)

The product is global, so we take a bounded, geographically diverse sample rather than global
coverage:
1. One candidate 640 m (64 px @ 10 m) UTM tile per reef polygon, snapped to a per-UTM-zone
   tile grid and deduplicated → **11,159 candidate tiles** across **3,210 distinct 0.25°
   cells**.
2. Select **round-robin across 0.25° cells** (seeded) so the 1000-tile budget spreads across
   the world's reef provinces instead of over-representing dense regions (e.g. the Great
   Barrier Reef). Result: nearly one tile per distinct cell.

Selected-tile spread (by longitude basin): Coral Triangle / W-Pacific 608, W-Indian
Ocean / Red Sea 212, Atlantic / Caribbean 112, Central/East Pacific 63, other 5. All centers
fall within ±34° latitude (the tropical reef band), matching the product extent (−34.3° to
32.5°).

## Time range

Coral reefs are persistent geological/biological structures. Although the source compiles
surveys mostly dated 1989–2002, the reefs remain in place, so (spec §5, static labels) each
sample gets a **representative 1-year Sentinel-era window (2020-01-01 … 2021-01-01)** with
`change_time = null`. This matches the sibling `allen_coral_atlas` handling. (The old survey
dates do **not** trigger the pre-2016 rejection rule: that rule targets time-bound
observations, not persistent-feature presence.)

## Output

- `datasets/unep_wcmc_global_warm_water_coral_reefs/metadata.json`
- `datasets/unep_wcmc_global_warm_water_coral_reefs/locations/{000000..000999}.tif` — single
  band, uint8, local UTM @ 10 m, 64×64, values {0 = reef, 255 = nodata}.
- `datasets/unep_wcmc_global_warm_water_coral_reefs/locations/{000000..000999}.json` — CRS,
  pixel_bounds, 1-year `time_range`, `change_time=null`, `source_id`, `classes_present=[0]`.

## Verification (spec §9)

- 1000 `.tif` + 1000 matching `.json`. All tiles: single band, uint8, UTM CRS at 10 m,
  64×64, nodata 255. Global unique pixel values across all tiles = {0, 255} only; every tile
  contains class 0.
- All `time_range`s are exactly 1 year (2020); `change_time` null.
- `metadata.json` class ids ({0}) cover all non-nodata values present in the tiles.
- **Spatial sanity:** georeferencing round-trips correctly; all 1000 sample centers land in
  the tropical reef band (±34° lat) at plausible reef locations (e.g. Hawaii ~21.2°N,
  −157°; Line Islands; Red Sea; Caribbean). A full Sentinel-2 pixel overlay was not run in
  the agent; the coordinate/CRS round-trip and reef-band placement were used as the sanity
  check.

## Caveats

- Point-only reefs (925) and sub-0.01 km² polygon slivers are excluded as sub-pixel.
- Presence-only: tiles are reef-or-nodata; there is no explicit non-reef class.
- Source is a compilation of variable-vintage surveys; reef footprints are approximate
  (e.g. some entries are "coral line buffered to 300 m").

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.unep_wcmc_global_warm_water_coral_reefs
```

Idempotent: re-running skips already-written `{sample_id}.tif`. Downloads/extracts the zip
into `raw/unep_wcmc_global_warm_water_coral_reefs/` if not present.
