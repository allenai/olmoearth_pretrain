# Sentinel-2 Water Edges Dataset (SWED)

- **Slug**: `sentinel_2_water_edges_dataset_swed`
- **Source**: UK Hydrographic Office (UKHO), <https://openmldata.ukho.gov.uk/>
- **Family / region**: coastal / global coastlines
- **label_type**: dense_raster → **binary per-pixel classification**
- **Annotation method**: photointerpretation (expert-checked) on the Sentinel-2 image
- **License**: Geospatial Commission Data Exploration licence (free, open download; no
  credential/registration required — data is on a public S3 bucket)
- **Status**: completed · task_type=classification · **num_samples=1770**

## Source

SWED contains pairs of Sentinel-2 L2A images and a **binary water/non-water** segmentation
mask (label value `1` = water, `0` = non-water), created by the UKHO Data Science team for
coastline / water-edge segmentation. It is distributed as **one ~19 GB zip** on a public S3
bucket, `https://ukho-openmldata.s3.eu-west-2.amazonaws.com/SWED.zip`, containing:

- `SWED/train/labels/*.npy` — **28,224** label chips (`int16`, `(1,256,256)`, values `{0,1}`),
  one per 256×256 chip cut from a **42×42 grid** off the top-left of the S2 granule. Spread
  over **16 distinct S2 scenes** (one product per MGRS tile), acquisition years **2017–2020**.
  Paired `train/images/*.npy` (256×256×12) were **not** used.
- `SWED/test/labels/*.tif` — **98** label GeoTIFFs (`uint16`, `{0,1}`, EPSG:4326 at ~10 m).
  Paired `test/images/*.tif` were not used.

Only the **label** files are needed (pretraining supplies its own imagery), so we
**selectively extract just the labels** via HTTP range requests (`remotezip`) — ~3.7 GB of
the 19 GB archive, never pulling the image bulk. Extracted labels are staged locally at
`swed_scratch/`; weka `raw/` holds only a `SOURCE.txt` pointer.

## Georeferencing

- **Train `.npy` chips carry no embedded georeferencing**, but the filename encodes the full
  S2 product id (hence the MGRS tile) and the within-granule chip index `chip_{row}_{col}`.
  The S2 granule origin (ULX/ULY) + UTM CRS is **deterministic per MGRS tile**, looked up
  once from the Planetary Computer STAC (`proj:transform` of the 10 m bands) and hardcoded in
  `TILE_GEO` (16 tiles). Chip `(r,c)` covers granule pixels `[r·256:(r+1)·256, c·256:(c+1)·256]`,
  giving each chip an exact UTM 10 m footprint (chips tile from the granule top-left; 42·256 =
  10752 < 10980, so they trim the granule edges).
  - **Validated**: reconstructing the UTM window for a train image chip and correlating it
    against the real S2 granule read at those bounds gave **r ≈ 0.996–0.997** (transposed
    orientation ≈ 0), confirming CRS, granule origin, chip index order, and axis orientation.
- **Test `.tif`** are already georeferenced (EPSG:4326 ~10 m); each is reprojected to its
  **local UTM at 10 m** with **nearest** resampling (categorical) before tiling.

## Processing

- **Classes** (ids follow the source encoding): `0 = non-water`, `1 = water`. 255 = nodata
  (only appears at test-set reprojection edges; train chips are fully valid).
- Each source scene is tiled into **64×64** UTM patches. A tile counts toward a class only if
  it holds ≥ `MIN_CLASS_PX = 64` px of it; tiles > 50% nodata are dropped.
- **All source splits used** (train + test). Candidate 64×64 tiles: 428,049.
- Selection: **tiles-per-class balanced** (`sampling.select_tiles_per_class`, ≤ 1000 tiles per
  class, 25k cap), rarer class (water) filled first.
- **Time range**: the water mask is a **per-image STATE** — water extent varies with tidal
  state (the dataset even ships a per-test-scene tidal CSV) — so per spec §5 (specific-image
  labels) `time_range` is a **~1-hour window at the S2 acquisition datetime** (parsed from the
  product id) and `change_time = null`. (Not treated as a change label.)

## Output stats

- **num_samples = 1770** (single-band `uint8` 64×64 GeoTIFFs, local UTM at 10 m).
- Tiles containing each class: `non-water = 1000`, `water = 1094`.
- By split: `train = 1764`, `test = 6` (the balancer draws mostly from the abundant native-UTM
  train chips; the 16 train scenes give global coastal spread).

## Verification (spec §9)

- Opened random outputs: single band, `uint8`, UTM CRS at 10 m, 64×64, values ⊆ {0,1,255},
  each `.tif` has a matching `.json` with a 1-hour `time_range` and `change_time=null`.
- `metadata.json` class ids {0,1} cover all values in the tifs.
- **Spatial sanity**: for water-containing samples, S2 NDWI at the tile bounds/time is clearly
  higher over water-labeled pixels than land (e.g. −0.13 vs −0.31; −0.04 vs −0.40) — labels
  overlay correctly on real water.

## Caveats

- Binary task → only 2 classes, so the per-class-1000 target yields ~1770 samples (spec-compliant).
- Test tiles are a small fraction of the selection (reprojected, and abundant train tiles win
  the class-balanced draw); the 16 train scenes still provide global coastal diversity.
- The 12-band `image` `.npy`/`.tif` are intentionally not downloaded (pretraining supplies imagery).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sentinel_2_water_edges_dataset_swed --workers 64
```

Idempotent: re-runs skip already-extracted labels and already-written `locations/*.tif`.
Requires `remotezip` (selective zip extraction) in the environment.
