# Murray Global Intertidal Change (tidal flats)

- **Slug:** `murray_global_intertidal_change_tidal_flats`
- **Status:** completed — **classification**, **2000 samples** (1000 tidal flat / 1000 other)
- **Family / region:** coastal / global coastline (60°S–60°N)
- **License:** CC-BY-4.0 (metadata says CC-BY 4.0; Figshare record tags CC0 — freely usable either way)

## Source

Murray Global Intertidal Change Classification **v1.2 (1999–2019)** — a supervised
per-pixel classification of the Landsat archive over the global coastline at 30 m,
distributed in seven 3-year epochs. Papers: Murray et al. 2019, *Nature* 565:222–225
(doi:10.1038/s41586-018-0805-8) and the accompanying *Scientific Data* descriptor.
Project site: https://www.intertidal.app/ .

## Access (how it was obtained, unauthenticated)

The primary catalog entry is Google Earth Engine (`UQ/murray/Intertidal/v1_1/...`), which
is credential-gated. However, a **fully unauthenticated direct download exists on
Figshare** (Springer Nature collection 5884598), so this dataset was **accepted, not
rejected**:

- Figshare article 19334465 → single file `Global_Intertidal_v1_2.zip` (56 GB), served
  from S3 with HTTP byte-range support:
  `https://ndownloader.figshare.com/files/34337744`.
- The outer zip contains one **DEFLATE** nested zip per epoch plus a metadata PDF. We only
  needed the latest epoch, so we **range-read the compressed bytes of the `2017-2019.zip`
  member and stream-decompressed it to disk (9.78 GB)** — bounded sampling that avoids
  downloading the full 56 GB / all 7 epochs.
- The epoch zip unpacks to `global_intertidal/` (108 EPSG:4326 GeoTIFFs, each
  **74213×74213 uint16, 20°×20° at 30 m, LZW**) and a `qa_pixel_count/` folder (ignored).
  Tiles were read in place via GDAL `/vsizip/` (no full extraction).

Raw location on weka: `raw/murray_global_intertidal_change_tidal_flats/2017-2019.zip`
(+ `SOURCE.txt`).

## KEY FINDING — binary product, not 3 classes

The manifest lists three classes `[tidal flat, permanent water, other]`, but the
**distributed v1.2 raster is a BINARY tidal-flat mask**. Verified empirically: across all
108 global tiles the only pixel values present are **{0, 1}**, where **1 = tidal flat** and
**0 = everything else**. Permanent water is **not** separated in the published GeoTIFF
(this matches the EE catalog, whose classification band is bit 0 = intertidal /
non-intertidal). The 3-class description in the abstract refers to the internal
classifier; the released extent product collapses to tidal-flat presence.

We therefore produced a **2-class classification**:

| id | name | meaning | source value |
|----|------|---------|--------------|
| 0 | tidal flat | mudflats / sand flats / tidal rock-platforms subject to regular tidal inundation (excludes mangroves & vegetated marsh) | 1 |
| 1 | other | non-tidal-flat: open/permanent water + all other cover (manifest's "permanent water" and "other" merged) | 0 |

nodata = 255 (unused in practice — the source covers the full tile extent, so output tiles
have no 255 pixels).

## Processing

- **Epoch:** 2017–2019 (latest; matches manifest range [2016, 2019]). Time range per
  sample = **2018-01-01 → 2019-01-01** (1-year window anchored on the epoch center).
  task_type = classification, no `change_time`.
- **Bounded, tiles-per-class-balanced sampling** across the **82** tiles that contain tidal
  flat (of 108). Each tile was scanned in native-resolution row strips; 660 m (22×22 native
  px ≈ a 64 px @ 10 m tile) blocks were classified:
  - **tidal-flat window** (label 0) if ≥ 5% of the block is tidal flat;
  - **coastal "other" window** (label 1) if the block has zero tidal flat but lies within 3
    blocks of a tidal-flat block (i.e. genuine coastline adjacent to flats — **not** open
    ocean or deep inland, which would be trivial negatives).
  - Per-tile cap of 40 candidates/class for geographic diversity.
- Candidates: 6574 (3254 tidal-flat, 3320 coastal-other) from 83 tiles → balanced to
  **1000 + 1000 = 2000**, drawn from **83 distinct global tiles**.
- Each 660 m native window is **reprojected to a local UTM projection at 10 m, 64×64, with
  nearest resampling** (categorical). Tidal-flat windows carry both classes (real
  tidal-flat/background boundaries); coastal windows are mostly `other`.

## Output

- `datasets/murray_global_intertidal_change_tidal_flats/metadata.json`
- `datasets/.../locations/{000000..001999}.tif` (single-band uint8, UTM @10 m, 64×64) + `.json`
- Class-pixel totals across the 2000 tiles: tidal flat 1,129,703 px; other 7,062,297 px; 0 nodata.

## Verification

- 2000 `.tif` + 2000 `.json`; every tif single-band **uint8**, **64×64**, **local UTM at
  10 m**, nodata 255, values ⊆ {0, 1}; metadata class ids cover all values.
- Every sample JSON has a matching tif, a ≤1-year `time_range` (2018), and correct
  `crs`/`pixel_bounds`.
- **Spatial sanity:** high-tidal-flat samples land squarely on world-renowned tidal-flat
  coastlines — Amazon delta mouth (−49.8, 0.0), Suriname/Guiana mud coast (−56.3, 5.9),
  Kimberley NW Australia (123.6, −17.4), Bohai Bay / Liaohe estuary China (121.9, 40.7),
  Gulf of Khambhat India (72.5, 22.2), Gulf of Mannar Sri Lanka (79.9, 9.0),
  Wadden-area Denmark (11.1, 57.3), Sudan Red Sea coast (37.7, 18.7). Label polarity
  (0 = tidal flat) confirmed correct.
- Idempotent: `_write_one` skips existing `{id}.tif`; `download_epoch` skips the present zip.

## Caveats / open question

- **QUESTION FOR USER:** the manifest expected 3 classes (tidal flat / permanent water /
  other) but the distributed v1.2 GeoTIFF is a binary tidal-flat mask, so this was built as
  a 2-class (tidal flat / other) segmentation. If a genuine 3-class label separating
  *permanent water* is required, it would have to come from a different source (e.g. JRC
  Global Surface Water as a water layer, or Earth-Engine-side reprocessing) — out of scope
  here. Confirm the 2-class product is acceptable.
- "Other" windows are coastal negatives adjacent to tidal flats; they intentionally exclude
  open ocean / deep inland to stay informative. A handful (~11) picked up a few tidal-flat
  pixels at tile edges after reprojection — harmless and label-accurate.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.murray_global_intertidal_change_tidal_flats
```
(Downloads the 2017–2019 epoch via HTTP range read on first run, then scans + writes.)
