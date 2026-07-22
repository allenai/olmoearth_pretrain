# GTPBD (Global Terraced Parcel and Boundary Dataset)

- **Slug**: `gtpbd_global_terraced_parcel_and_boundary_dataset`
- **Status**: completed
- **Task type**: classification (dense binary segmentation)
- **Num samples**: 1000 label patches
- **Classes**: `0 = background` (non-terraced land), `1 = terraced parcel`
- **License**: CC-BY-NC-4.0 (non-commercial; acceptable for this research pretraining use).
  Attribution: Zhang et al., "GTPBD: A Fine-Grained Global Terraced Parcel and Boundary
  Dataset", NeurIPS 2025 (arXiv 2507.14697).

## Source & access

Public, non-gated Hugging Face dataset `wxqzzw/GTD`, a single `GTPBD_enhenced_png.zip`
(16 GB). Downloaded with `download.hf_download` to `raw/{slug}/`. The archive holds 47,537
manually annotated 512×512 tiles cropped from high-resolution optical scenes (GF-2 + Google
Earth, 0.1–1 m GSD) over 7 Chinese agricultural zones plus transcontinental regions, with
three co-registered **binary** label rasters per tile (`mask_labels/`, `boundary_labels/`,
`parcel_labels/`) split train/val/test × region. Each label PNG carries a GDAL
`.png.aux.xml` PAM sidecar with a CRS + GeoTransform. Only `mask_labels/*.png` + their
`.aux.xml` are decoded (imagery not needed; pretraining supplies its own S2/S1/Landsat).

## Georeferencing — PARTIAL recovery (spec §8 gate)

The per-tile geotransforms fall into two regimes, distinguished by pixel size:

- **Case-B — KEPT (6150 mask tiles).** Pixel size is a genuine small WGS84 degree value
  (~4.5e-6–7.2e-6 deg/px ≈ 0.44–0.8 m GSD). Verified internally consistent: within a scene,
  neighbouring tiles' origins differ by exactly `tile_pixel_offset × px_deg` (checked
  Δlon/Δcol = px to 1e-9). These carry correct WGS84 georeferencing. Sample centers land at
  lon 104.5–112.0, lat 26.7–29.5 — the central/SW-China terrace belt (e.g. Hunan near the
  Ziquejie terraces), corroborating placement.
- **Case-A — REJECTED (9994 mask tiles, incl. ALL "Rest of the world"/global tiles).** Pixel
  size is stored as `0.3` in a WGS84 (degrees) CRS. 0.3 deg ≈ 33 km/px — impossible for a
  512-px VHR tile: this is a units bug (the ≈0.3 m GSD written into a degree CRS), and each
  sub-tile origin was computed as `parent_origin + pixel_offset × 0.3 (deg)`, giving
  off-the-earth origins (lon 194, lat −277, …). The parent-image origin is recoverable
  (subtract `offset×0.3`), but the **true per-image GSD (0.1–1 m, varying per scene per the
  paper) is not reliably recoverable**: single-block parents give no scale information, and
  multi-block parents' block-offset naming has ambiguous pixel units. An assumed GSD would
  mis-scale/misplace the label on the S2 grid (errors up to ~km for corner tiles), so these
  were dropped rather than emit misregistered labels.

**Caveat:** the processed subset is geographically narrower than full GTPBD — Southwest +
Central China only (final selection: 617 Southwest, 383 Central China), losing the global
"Rest of the world" tiles.

## Class scheme (spec §5 multi-target → one unified map)

GTPBD ships three co-registered binary label types. Decision on which to encode:

- **`mask_labels` → ENCODED** as the per-pixel signal. Terrace parcel areas are clearly
  resolvable at 10 m (manifest note: terraced hillslopes visible at 10–30 m). Value 1 →
  class `terraced parcel`, value 0 → class `background`.
- **`boundary_labels` → NOT encoded.** Parcel ridge/edge lines are sub-metre (~1–3 native px,
  <0.3 px at 10 m) and do not survive mode resampling to 10 m (spec §4 VHR unresolvable-fine-
  feature guidance).
- **`parcel_labels` → NOT encoded.** Instance-oriented (binary interior mask), not a fixed
  per-pixel class set.

Result: a 2-class dense binary segmentation (background / terraced parcel). Background here is
genuine non-terraced land within the scene (spatially meaningful), so it is a real class 0,
not fabricated negatives.

## VHR → 10 m handling (spec §4 VHR-native)

Each 512×512 binary mask (WGS84, ~0.44–0.8 m) is reprojected to a local UTM grid at 10 m with
**MODE** resampling (categorical majority; never bilinear), yielding one tile per source tile.
Output sizes 23–43 px (all ≤ 64). Augmented tiles (`_flip` / `_rot90`) are geometric copies of
originals and excluded. Tiles that contain no terrace after resampling are dropped.

## Time range

No per-tile acquisition date; imagery spans 2016–2025. Terraces are static agricultural
features (spec §5 static rule) → representative 1-year Sentinel-era window (2020-01-01 →
2021-01-01). All labels are ≥ 2016; no pre-2016 filtering needed.

## Sampling

All source splits (train/val/test) used. Tiles-per-class balanced (`select_tiles_per_class`,
per_class=1000, total_cap=25000). Both classes co-occur in nearly every tile, so selection
reaches 1000 tiles containing background and 1000 containing terrace → **1000 samples**
(background: 1000 tiles, terraced parcel: 1000 tiles).

## Verification (spec §9)

- 1000 `.tif` + 1000 `.json`. All single-band uint8, UTM CRS at 10 m/px, sizes 23–43 (≤64),
  pixel values ∈ {0,1}, nodata 255 (declared; unused — full coverage). Every `.tif` has a
  matching `.json` with a 1-year `time_range` and `classes_present`. `metadata.json` class ids
  {0,1} cover all values in the tifs.
- Georeferencing validated by Case-B internal consistency (Δlon/Δcol = px exactly) and by
  sample centers falling in known China terrace regions. A pixel-level Sentinel-2 overlay was
  not run (would require S2 data-source setup); georeferencing confidence rests on the
  internal-consistency and landmark checks above plus the exact rslearn UTM encoding.
- Re-running is idempotent: the scan is cached to `raw/{slug}/scan_cache.pkl` and `_write_one`
  skips existing `{sample_id}.tif`.

## Reproduce

```bash
# raw zip must be present at raw/{slug}/GTPBD_enhenced_png.zip (download.hf_download wxqzzw/GTD)
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gtpbd_global_terraced_parcel_and_boundary_dataset --workers 64
```
