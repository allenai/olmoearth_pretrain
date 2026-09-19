# Global Mining Footprint (Tang & Werner)

- **slug**: `global_mining_footprint_tang_werner`
- **status**: completed
- **task_type**: classification (binary segmentation)
- **num_samples**: 25,000 tiles (20,177 with mine pixels, 4,823 background-only)

## Source

Tang, L. & Werner, T.T. (2023), "Global mining footprint mapped from high-resolution
satellite imagery", *Communications Earth & Environment*. Zenodo record **6806817**
(https://doi.org/10.5281/zenodo.6806817), license **CC-BY-4.0**.

Single archive `Supplementary 1：mine area polygons.rar` (RAR5, 103,532,548 bytes)
containing three vector layers:

- `74548 mine polygons/74548_projected.shp` — **the headline product: 74,548 mine-area
  polygons** finely contouring the surface footprint of mining across 135 countries,
  manually photointerpreted from high-resolution satellite imagery (~2019). CRS is
  *WGS 84 / NSIDC EASE-Grid Global* (cylindrical equal area, metres). Geometry: Polygon Z.
  Attribute fields: `OBJECTID`, `Name`, `Shape_Le_1`, `Shape_Area`.
- `Artisanal and small-scale mine/…shp` — 4,058 ASM polygons (**not used**, see caveat).
- `Larger scale mine/large scale mine area.gdb` — 761 large-scale mine features (**not
  used**).

### Access notes (reproducibility)
- The Zenodo **API** download path (`/api/records/6806817/files/…/content`) returned
  **HTTP 403** ("unusual traffic from your network") — Zenodo rate-limiting our shared
  network, not a permanent gate. The **records** download URL worked:
  `https://zenodo.org/records/6806817/files/Supplementary%201%EF%BC%9Amine%20area%20polygons.rar?download=1`
  (note the URL-encoded fullwidth colon `%EF%BC%9A`). Use `curl -C - --retry` — the first
  attempt truncated at 35 MB while still reporting HTTP 200, so **verify the on-disk size
  equals 103,532,548 bytes** before extracting.
- The archive is **RAR5**; `bsdtar`/libarchive mis-parses it (silently extracts only a
  stray `doc.kml`, or a corrupt partial `.shp`). Extract with a **modern 7-Zip**
  (conda-forge package `7zip` v26 provides `7z`): `7z x mine_area_polygons.rar -oextract`.

## Label mapping — binary mine footprint

    0 = background (outside any mapped mine)
    1 = mine (inside a Tang & Werner mine-area polygon)

**Why binary, not the manifest's 6 classes.** The manifest lists six fine mine-feature
classes (waste-rock dumps, pits, ponds, tailings dams, heap leach, processing), but the
**released polygons carry no per-polygon feature-type attribute**. The only descriptive
field, `Name`, is leftover KML placemark text — its values are dominated by
`多边形`/`未命名多边形` ("polygon"/"unnamed polygon"), `Placemark`, and stray digits/ore
symbols (`Cu`, `Au`, `Fe`, years), none of which encode the six feature types.
Per-feature-type classification is therefore **not expressible** from this data release, so
the dataset is mapped to the well-supported, undifferentiated **mine vs background**
footprint signal. Mines are large (median polygon `Shape_Area` ≈ 0.12 km², i.e. tens of
10 m pixels; max > 1,300 km²) and clearly observable at 10–30 m from S2/S1/Landsat.

## Processing

Follows the validated GLAKES template (bounded, geographically stratified polygon → tile
rasterization). 64×64 uint8 tiles, local UTM at 10 m/pixel, nodata sentinel 255 (unused —
this is a positive-only-style binary mask with a genuine background class).

- **Stratified sampling**: round-robin over 1° lon/lat cells (centroids reprojected
  EASE→WGS84) for global spread.
- **Positive tiles (20,000)**: centered on sampled mine-polygon centroids; every Tang &
  Werner polygon intersecting the 640 m tile is rasterized to class 1 (`all_touched=True`),
  rest is background 0.
- **Negative tiles (5,000)**: mine anchors offset by a random ~3–9 km vector so no mine
  falls in the tile. If an offset still clips a mapped mine it is rasterized correctly
  (that tile then counts as a mine tile — hence 20,177 mine / 4,823 background actual).
- **Total capped at 25,000** (spec §5 hard cap). Per-tile intersecting polygons are read
  with a pyogrio bbox spatial filter in the source EASE-Grid CRS (query box = the tile's
  lon/lat extent reprojected to EASE), so no giant in-memory STRtree is needed and the
  write phase parallelizes over a 64-worker `multiprocessing.Pool` (~2.5 min for 25k tiles).

### Time range
Mine footprints are quasi-static; imagery ~2019, manifest window 2016–2019. Each tile gets
a **1-year window with start year sampled uniformly from 2016–2019** (Sentinel era). No
change labels.

## Verification (spec §9)
- 25,000 `.tif` and 25,000 matching `.json`; 0 unmatched.
- Sampled tiles: single band, uint8, 64×64, projected UTM CRS at (10, 10) m, values ⊆
  {0, 1, 255}, `time_range` ≤ 1 year. UTM zone matches each tile's lon/lat (e.g. lon 101°E
  → EPSG:32647/32649/32651; a Southern-Hemisphere tile → EPSG:32736).
- `metadata.json` classes {0: background, 1: mine}, nodata 255, num_samples 25000 — covers
  all values present in the tifs.
- A standalone reprojection test on a known mine (lon 101.02, lat 28.40) produced a
  contiguous 642-pixel mine footprint in the correct UTM zone, confirming exact
  georeferencing. A **live Sentinel-2 overlay was not run** (requires configuring an
  external imagery source); georeferencing is exact by construction via rslearn's
  `GeotiffRasterFormat` + verified UTM-zone selection.

## Caveats / judgment calls
- **6 fine classes dropped → binary.** The manifest's per-feature-type classes are not in
  the released attributes; only the mine/background footprint is recoverable. Reviewer may
  wish to confirm no richer typed version exists elsewhere (the paper's SI describes the
  feature types conceptually but the polygons are undifferentiated).
- **ASM and large-scale layers unused.** The 74,548-polygon layer is the authoritative,
  most complete geometry and is the manifest's headline count. The separate ASM (4,058) and
  large-scale (761) layers have no shared join key to the 74,548 set and their overlap is
  unverified, so folding them in as a scale-based class scheme was not attempted; doing so
  would trade the full footprint for a much smaller, ambiguous subset. Left as a possible
  future enrichment.
- **Negatives may contain unmapped mines** or other bare-earth/industrial features labeled
  background (the product maps a curated global set, not exhaustive coverage) — standard
  for positive-derived negatives.

## Reproduce

```
# 1. Download (verify size == 103,532,548 bytes) into raw/<slug>/mine_area_polygons.rar
curl -sL -C - --retry 5 -A "Mozilla/5.0" -o mine_area_polygons.rar \
  "https://zenodo.org/records/6806817/files/Supplementary%201%EF%BC%9Amine%20area%20polygons.rar?download=1"
# 2. Extract RAR5 with modern 7z (conda-forge 7zip): 7z x mine_area_polygons.rar -oextract
# 3. Run:
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mining_footprint_tang_werner
```
Idempotent: existing `locations/{id}.tif` are skipped.
