# World Settlement Footprint (WSF) 2019

- **Slug**: `world_settlement_footprint_2019`
- **Status**: completed
- **Task**: classification (binary: non_settlement / settlement)
- **Samples**: 2000 label patches (1000 settlement windows + 1000 pure non-settlement windows)
- **Source**: DLR EOC Geoservice — WSF 2019 (CC-BY-4.0)
  - Landing: https://geoservice.dlr.de/web/maps/eoc:wsf2019
  - Download: https://download.geoservice.dlr.de/WSF2019/files/

## What the source is

WSF 2019 is a **global 10 m binary human-settlement mask** derived from 2019 multitemporal
Sentinel-1 + Sentinel-2 imagery. It is distributed as 5138 GeoTIFF tiles in **EPSG:4326**,
each covering a 2°×2° area (~222×222 km, ~22488×22488 px at ~8.98e-5°/px ≈ 10 m) with a 0.1°
overlap buffer. Tiles are named by their lower-left corner (e.g. `WSF2019_v1_12_18.tif` covers
12–14 °E, 18–20 °N). **Pixel values: 255 = settlement, 0 = non-settlement.** There is no
source nodata value.

## Label choice: dense_raster tiling (validation points NOT usable)

The manifest flags ~1M crowdsourced photointerpreted **validation points** (collected with
Google / MapSwipe support) and suggests seeding samples from them. Those validation points are
**not publicly released as a downloadable file** — the WSF2019 download directory exposes only
the raster tiles, a 19.3 GiB global COG, thumbnails, and STAC sidecars (verified 2026-07; no
point/CSV/shapefile product exists there or on the linked geoservice). So per the task spec we
use the intended **fallback: bounded dense_raster tiling of the WSF mask itself** (spec §4
dense_raster, §5 bounded regional sampling). WSF is already 10 m, so **no resampling of
resolution** is needed — only reprojection from EPSG:4326 to local UTM.

## Class mapping

| class id | name | source value |
|---|---|---|
| 0 | non_settlement | 0 |
| 1 | settlement | 255 |

Both are meaningful classes, so neither is treated as nodata. `nodata_value = 255` is used only
for pixels with no source coverage (reproject fill at a 2° tile boundary); 75 / 2000 tiles
(~3.8 %) carry a few such nodata pixels near tile edges.

## Sampling (bounded, regional)

WSF is a global derived-product raster with no in-situ reference alternative, so we download
only **34 representative 2°×2° tiles** (~319 MB total) and tile them:

- **28 major-city tiles** on every inhabited continent, for diverse settlement morphology:
  London, Paris, Berlin, Moscow, Istanbul, Cairo, Lagos, Nairobi, Johannesburg, Kinshasa,
  New York, Los Angeles, Mexico City, Chicago, São Paulo, Bogotá, Lima, Buenos Aires, Delhi,
  Mumbai, Dhaka, Beijing, Shanghai, Tokyo, Jakarta, Bangkok, Sydney, Dubai.
- **6 rural / arid / boreal / forest tiles** for clean non-settlement landscapes: US Great
  Plains, Amazon, Sahel, Australia outback, Siberia, Canada prairie.

For each tile we cut **64×64 @ 10 m windows** in local UTM, reprojecting the EPSG:4326 source
with **NEAREST** resampling (categorical — never bilinear). Two window pools:

- **settlement windows** — centred on settlement pixels, keeping windows with **≥ 5 %
  settlement** (they carry the settlement footprint and its boundary; nearly all also contain
  non-settlement pixels, so they count toward both classes);
- **non_settlement windows** — pure background (0 % settlement), drawn across all tiles for
  clean, homogeneous, high-confidence negative-region labels.

Up to **1000 per class** (spec §5), balanced and seeded → 2000 samples (well under the 25k cap).
Final content: 1002 tiles contain settlement (mean settlement fraction **0.42**, median 0.42),
998 are pure background; class 0 is present in all 2000 tiles, class 1 in 1002.

**Latitude correction:** because WSF is in geographic degrees, a native pixel spans ~10 m in
latitude but only ~10·cos(lat) m in longitude, so a fixed 64-column native window is narrower on
the ground than the 640 m UTM output at high latitude. The candidate scan widens the native
column window by 1/cos(lat) (tile-centre latitude) so the settlement fraction it tags matches
the reprojected UTM label — otherwise high-latitude "settlement" windows (Moscow/Berlin/Siberia)
came out settlement-sparse. Each written patch's `classes_present` is recomputed from the actual
UTM label, so per-sample metadata is exact regardless.

## Time range

Static 2019 product → **1-year window `[2019-01-01, 2020-01-01)`**, `change_time = null`
(spec §5 static/annual). The STAC metadata confirms the product epoch is calendar-year 2019.

## Output format

- Single-band **uint8** GeoTIFFs, local **UTM @ 10 m**, north-up, ≤ 64×64 (all exactly 64×64).
- Values {0, 1}; 255 = nodata (edge fill only).
- Per-sample sidecar JSON with CRS, pixel bounds, 1-year time range, `change_time=null`,
  `source_id` (source WSF tile + native row/col), and recomputed `classes_present`.

## Verification (spec §9)

- 2000 `.tif` each with a matching `.json`; single band; dtype uint8; CRS UTM at 10 m res;
  size 64×64.
- Pixel values ∈ {0, 1, 255}; 255 confined to a small edge-fill minority (75 tiles).
- All `time_range` = 2019 one-year window; all `change_time` = null.
- `metadata.json` class ids {0,1} cover all non-nodata values in the tifs.
- Coordinate sanity check: settlement-window centres fall on the expected city footprints
  (e.g. Chicago −87.5,41.3; Cairo 30.8,30.1; NYC region −72.1,41.7) and background windows sit
  in rural areas. A full Sentinel-2 pixel overlay was not rendered, but the label is the
  authoritative WSF raster and georeferencing uses the shared, validated rslearn UTM utilities
  (`io.lonlat_to_utm_pixel` / `get_transform_from_projection_and_bounds`) used across all
  sibling datasets.
- Idempotent: existing `{sample_id}.tif` are skipped on re-run.

## Caveats

- Binary product: only settlement vs non-settlement (no building type/height/use). WSF's known
  limitations (some commission over bright bare soil / rocky terrain in arid regions; omission of
  very sparse rural huts) are inherited.
- ~3.8 % of tiles have a few edge-fill nodata pixels near 2° tile boundaries (ignored downstream).
- The non_settlement class is genuinely represented here (pure rural/natural windows); downstream
  assembly additionally supplies cross-dataset negatives (spec §5), so no synthetic negatives were
  fabricated.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.world_settlement_footprint_2019
```

Downloads the 34 tiles to `raw/world_settlement_footprint_2019/` (throttled to 4 concurrent
requests with retry/backoff — the DLR server returns HTTP 503 under heavy concurrency) and writes
label patches + metadata under
`datasets/world_settlement_footprint_2019/`.
