# SpaceNet 7 (multi-temporal buildings)

- **Slug**: `spacenet_7_multi_temporal_buildings`
- **Status**: completed
- **Task type**: classification (building presence; `change_time=null`)
- **Samples**: 1578 label patches (64×64, UTM, 10 m) — 1000 building-present + 578 background-only

## Source & access

SpaceNet 7 Multi-Temporal Urban Development Challenge, hosted on the public AWS Open Data
bucket `s3://spacenet-dataset/spacenet/SN7_buildings/` — anonymous/unsigned access, no
credentials required (License **CC-BY-SA-4.0**).

- `train/` — **60 AOIs worldwide**, each ~4 km × 4 km, imaged as monthly PlanetScope
  mosaics (~4 m/px, EPSG:3857) for ~25 months (2018-01 … 2020-01). Per AOI/month:
  `labels/…_Buildings.geojson` (manually digitized building footprints, CRS84 polygons) and
  `labels/…_UDM.geojson` (unusable-data mask).
- `test_public/` — 20 AOIs, **imagery only (labels withheld)** → **excluded**.

**Only the small label GeoJSONs are downloaded** (one representative month per AOI), plus
each AOI image's *header* (read remotely via `/vsis3/` to get the mosaic extent/CRS). No
PlanetScope mosaic rasters are pulled — pretraining supplies its own S2/S1/Landsat imagery.
Raw labels land under `raw/spacenet_7_multi_temporal_buildings/{AOI}/`.

## Encoding choice — building PRESENCE (not change)

Native imagery is 4 m and an individual SpaceNet-7 building is sub-10 m, so single
buildings are not resolvable at 10 m. However, building **footprint presence** aggregates
into a **built-up-vs-not** signal that *is* observable at 10 m. So (matching the manifest
note "building density/settlement footprints discernible at 10 m") each AOI is encoded as a
dense 2-class raster:

| id | name | definition |
|----|------|------------|
| 0 | background | AOI-extent pixel not covered by any building footprint (real observed "not built") |
| 1 | building | 10 m pixel touched by any building polygon in the chosen month |

nodata/ignore = 255 (used only for UDM unusable-data polygons, when present).

**One representative month per AOI** is used: the *latest available* month (most-complete
built-up state; typically 2020-01, else late-2019 where 2020 is absent). The union of that
month's footprints is rasterized (`all_touched=True`) from CRS84 into a local-UTM 10 m grid
(UTM zone from the AOI centroid), and the **full AOI extent** (from the image header,
reprojected 3857→UTM) is tiled into ≤64×64 patches.

This is a **genuine dense fully-annotated scene**, so background (0) is a *real observed*
class (the whole ~4 km AOI is annotated), **not** a fabricated negative — hence a proper
background class is emitted rather than the positive-only/nodata convention.

**Why not the change/tracking task?** SpaceNet 7's headline task is building
change/tracking, and construction is monthly-resolved (within the ~1–2-month change-timing
tolerance, so a change encoding would be *permissible*). We deliberately chose the simpler,
robust **presence** encoding: it captures the salvageable 10 m signal (built-up extent) as
a dense static label without the complexity/edge-cases of per-building construction-event
timing. `change_time` is therefore `null`.

## Balancing (spec §5)

Building-present tiles vs background-only tiles are balanced via `balance_by_class` on a
per-tile presence category, up to **1000 each**. Of 3148 candidate tiles (2570 building,
578 background-only), all 578 background-only + a shuffled 1000 building tiles were selected
= **1578** samples, spread across all 60 AOIs (well under the 25k cap). Most SN7 AOIs are
dense urban, so background-only tiles are the scarcer category (all retained). Pixel-class
tile counts (a building tile also contains background pixels): `background=1578,
building=1000`.

## Time range (spec §5 seasonal/annual rule)

Static presence label → `time_range` is a **360-day window centered on the 15th of the
chosen month** (e.g. 2020-01 → 2019-07-19 … 2020-07-13), ≤ the 360-day cap. All labels are
2018–2020 (post-2016). `change_time=null`.

## Verification (spec §9)

- Scanned all 1578 tifs: every one is single-band `uint8`, **UTM at 10 m** (EPSG:326xx/327xx,
  0 non-UTM), **64×64** (max size 64×64), nodata=255; union of pixel values = `{0, 1, 255}`
  (all valid class ids + nodata).
- Every `.tif` has a matching `.json` (1578/1578); all `time_range`s ≤ 1 year and
  `change_time=null` (0 violations); `metadata.json` classes cover all values in the tifs.
- **Spatial sanity**: tile centroids span lon −121.7…+145.0, lat −37.6…+52.5 (US, Europe,
  China, SE Asia, Australia, S. America) across 60 AOIs — squarely the SpaceNet-7 worldwide
  AOIs. Labels come from authoritative CRS84 source geometries reprojected exactly, so S2/S1
  grid placement is exact; sampled building tiles show plausible built-up fractions.
- Re-running is idempotent (existing `{id}.tif`+`.json` skipped).

## Caveats

- Individual buildings are sub-10 m and under-resolved; the reliable signal is **built-up
  extent** (footprint presence aggregated to 10 m), consistent with the manifest note.
- Padded AOI extent: tiling uses the reprojected image bbox, so a thin border of edge tiles
  may extend slightly beyond the exact imaged footprint; those areas carry no footprints and
  read as background (correct for "not built").
- UDM unusable-data polygons are burned as nodata (255); they were empty for the sampled
  months, so nodata is rare in practice.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spacenet_7_multi_temporal_buildings
```
Outputs: `datasets/spacenet_7_multi_temporal_buildings/{metadata.json, locations/{id}.tif,.json}`
on weka; raw labels under `raw/spacenet_7_multi_temporal_buildings/{AOI}/`.
