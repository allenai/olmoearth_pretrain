# CV4A Kenya Crop Type

- **slug**: `cv4a_kenya_crop_type`
- **status**: completed — **classification** (per-pixel crop type)
- **num_samples**: 2824 label patches (7 classes)
- **source**: Source Cooperative `radiantearth/african-crops-kenya-02` (STAC id
  `ref_african_crops_kenya_02`), the ICLR-2020 CV4A workshop "Crop Type Detection"
  competition training data. License **CC-BY-SA-4.0**.
- **region / time**: Western Kenya (Bungoma area), 2019 growing season.

## Source & access
Public, no credentials. Downloaded via the S3-style HTTP endpoint
`https://data.source.coop/radiantearth/african-crops-kenya-02/` (urllib gets HTTP 403 —
a `User-Agent` header via curl works). We pulled only the label rasters:
`{0..3}_label.tif` (crop code 1-7, 0 = unlabeled), `{0..3}_field_id.tif` (integer field
id), plus `FieldIds.csv` and `Documentation.pdf`. The Sentinel-2 band time series (686
tifs) was **not** copied — we only need the labels. Raw files + `SOURCE.txt` in
`raw/cv4a_kenya_crop_type/`.

## Georeferencing recovery (the crux of this dataset)
**The Source Cooperative mirror strips all georeferencing.** Every tif is a plain
`tifffile.py` array: identity transform, no CRS, no GCPs. The original Radiant MLHub STAC
(which held the per-tile `proj:transform`) is defunct and not archived (the MLHub API
required auth, so the Wayback Machine has nothing). We reconstructed and **validated** the
grid:

1. **Tile layout** — edge cross-correlation of adjacent tile borders (B08, 2019-06-06)
   gives an unambiguous contiguous 2×2 mosaic (each source tile is 2016×3035):
   ```
   [tile1 | tile3]   (top row)
   [tile0 | tile2]   (bottom row)
   ```
   → mosaic 6070 rows × 4032 cols @ 10 m = 40.32 km × 60.70 km.
2. **Absolute placement** — the dataset's WGS84 bounding box from NASA CMR collection
   `C2781412688-MLHUB` (W 34.02206853, E 34.38442998, N 0.71604663, S 0.16702187),
   reprojected to UTM 36N (**EPSG:32636**), matches those dimensions to ~1 px. Near the
   equator UTM convergence is negligible, so the mosaic top-left snaps cleanly to
   **E = 613740, N = 79160** (10 m grid).
3. **Validation (pixel-exact)** — the reconstructed mosaic B08 cross-correlated against
   the real Sentinel-2 scene **S2B_36NXF_20190606** (same MGRS tile 36NXF, same date, from
   the open `sentinel-cogs` AWS bucket) peaks at **correlation 0.9999999 at pixel offset
   (0, 0)**. The recovered grid is exact and aligned to the native S2 grid (S2 origin
   600000/100020; our origin = 600000+1374·10 / 100020−2086·10). Output patch UTM bounds
   were confirmed to fall inside the mosaic extent.

## Classes
Taken from the dataset **Documentation.pdf, Appendix D** (the authoritative legend).
**The manifest's class list is wrong for this dataset** (it lists maize/cassava/common
bean/sugarcane/groundnut/sweet potato/sorghum; the real legend is the 7 FAO crop /
intercrop classes below). Crop codes 1-7 → class ids 0-6:

| id | name | fields (labeled) | samples written |
|----|------|------------------|-----------------|
| 0 | Maize | 1462 | 1000 (capped) |
| 1 | Cassava | 829 | 829 |
| 2 | Common Bean | 98 | 98 |
| 3 | Maize & Common Bean (intercropping) | 487 | 487 |
| 4 | Maize & Cassava (intercropping) | 172 | 172 |
| 5 | Maize & Soybean (intercropping) | 160 | 160 |
| 6 | Cassava & Common Bean (intercropping) | 78 | 78 |

All 7 classes kept (well under the 254 cap). Common Bean (98) and Cassava & Common Bean
(78) are sparse — kept per spec §5 (downstream assembly handles rare-class filtering).

## Label encoding
Per-field label patches (EuroCrops-style), one per **labeled** field. For each field:
its pixel bounding box in the mosaic defines a `<=64×64` UTM 10 m window centered on the
field; the crop class id is burned at **every labeled pixel** in that window (neighboring
labeled fields included), and **255 (nodata/ignore)** fills all unlabeled land — we only
have ground truth inside surveyed fields, so unlabeled land is ignore, not a background
class. Tiles are sized to the field footprint (median field ≈ 10 px; max output dim 23 px,
hard-capped 64). No multi-label fields exist (each field is a single clean crop).

- **Withheld test fields** (label 0 but a valid `field_id`; 1402 of 4688 fields) are
  **excluded** — their crop labels are hidden in the competition release. 3286 labeled
  (train) fields remain; 2824 selected after class balancing.
- **Time range**: 1-year window on 2019 (`[2019-01-01, 2020-01-01)`), the growing season.
- **Sampling**: tiles-per-class balanced, up to 1000 fields/class, 25k cap
  (`balance_by_class`). Only Maize is truncated (1462 → 1000).

## Judgment calls / caveats
- Recovered georeferencing from scratch (mirror stripped it); validated pixel-exact
  against Sentinel-2, so alignment is trustworthy despite the missing source metadata.
- Used the Documentation.pdf legend, not the (incorrect) manifest `classes`.
- Excluded competition test fields (labels withheld = 0).
- Classes 4-7 are **intercropping** mixes; kept as distinct classes per the source legend.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cv4a_kenya_crop_type
```
Idempotent (skips already-written `locations/{id}.tif`). Raw download via curl with a
`User-Agent` header (see the script header / `raw/.../SOURCE.txt`).
