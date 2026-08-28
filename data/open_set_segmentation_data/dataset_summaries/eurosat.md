# EuroSAT

- **Slug:** `eurosat`
- **Status:** completed
- **Task type:** classification (scene-level, spec §4)
- **Num samples:** 10,000 (1,000 per class × 10 classes)

## Source

EuroSAT (Helber et al., 2018/2019) — a Sentinel-2 patch-classification benchmark of 27,000
images (64×64 @ 10 m ≈ 640 m footprint) hand-labeled into 10 land-use / land-cover
categories. Reference: <https://github.com/phelber/EuroSAT> (georeferenced MS release,
Zenodo record 7711810). License: **MIT**.

`have_locally: true`. The source is the internal rslearn dataset (aka `small_eurosat`) at
`/weka/dfive-default/rslearn-eai/datasets/eurosat/rslearn_dataset` (27,000 windows in group
`default`, built from the georeferenced EuroSAT_MS GeoTIFFs). **Raw is not copied** — see
`raw/eurosat/SOURCE.txt` pointing at that path.

## Triage: ACCEPT

Per spec §4 (scene-level, EuroSAT is the named example): EuroSAT patches are **genuinely
coherent land-cover patches** — each 640 m tile was constructed to be a single homogeneous
LULC class — so we emit **one uniform-class 64×64 tile per patch** rather than rejecting it
as mere patch classification. Georeferencing is present and exact: each source window's
`metadata.json` carries the patch's real UTM projection (CRS + x/y_resolution ≈ 10 m) and
its 64×64 integer pixel bounds, which we reuse verbatim (spec §2: "reuse the source window's
CRS if already UTM at 10 m"). Labels are 2018 (post-2016). Sparse-point rules do not apply
(label footprint is 640 m ≫ 1 px).

## Label / class mapping

The class is read from each window's `options.category` (a folder-derived category that
matches the window's `label` vector feature). We do **not** read the imagery or the label
vector layer. Source category → class id (manifest order):

| id | name                  | source category        | count |
|----|-----------------------|------------------------|-------|
| 0  | Annual Crop           | `AnnualCrop`           | 1000  |
| 1  | Forest                | `Forest`               | 1000  |
| 2  | Herbaceous Vegetation | `HerbaceousVegetation` | 1000  |
| 3  | Highway               | `Highway`              | 1000  |
| 4  | Industrial            | `Industrial`           | 1000  |
| 5  | Pasture               | `Pasture`              | 1000  |
| 6  | Permanent Crop        | `PermanentCrop`        | 1000  |
| 7  | Residential           | `Residential`          | 1000  |
| 8  | River                 | `River`                | 1000  |
| 9  | Sea/Lake              | `SeaLake`              | 1000  |

Each output tile is filled uniformly with its single class id. `nodata_value = 255` (unused
here — tiles are fully labeled). Source distribution before balancing:
`{AnnualCrop 3000, Forest 3000, HerbaceousVegetation 3000, Residential 3000, SeaLake 3000,
Highway 2500, Industrial 2500, PermanentCrop 2500, River 2500, Pasture 2000}`.

## Tile spec

- Single-band **uint8**, **64×64** (native EuroSAT patch size, == MAX_TILE cap).
- **Local UTM** projection at ~10 m/pixel, reused exactly from the source window
  (verified CRS e.g. EPSG:32630/32631/32632; resolutions ≈ 9.96–10.01 m — EuroSAT's native
  sub-permille deviation from 10 m, retained rather than resampled).
- Georeferencing (crs + pixel_bounds) matches the source window byte-for-byte.

## Time range

1-year window `2018-01-01 .. 2019-01-01`, taken from the source window `time_range` (how the
rslearn dataset materialized its S2 imagery). The manifest lists 2017/2018 as the EuroSAT
acquisition years; either is a valid ≤1-year window, and we keep the source's 2018 anchor.
`change_time = null` (static land-cover label). All source splits (train + val) are used
(spec §5).

## Sampling

Tiles-per-class balanced to ≤ 1000/class (`balance_by_class`, seed 42), 10 classes ⇒ 10,000
tiles, well under the 25k per-dataset cap. Every class has ≥ 2000 source windows, so all 10
classes reach the full 1000 (no truncation of rare classes).

## Verification (spec §9)

- 10,000 `.tif` + 10,000 matching `.json`; `metadata.json` covers all 10 class ids.
- Sampled tiles (classes 0,3,5,7,8,9): single-band uint8, 64×64, UTM @ ~10 m, pixel values
  = the declared class id, 1-year `time_range`, and crs/pixel_bounds identical to the source
  window (`src_match=True`). All sample lon/lats fall in Europe.
- **Spectral sanity check** on the source S2 imagery: Sea/Lake patch mean NDWI +0.53 /
  NDVI −0.30 (water), Forest NDVI +0.72, Residential NDVI +0.32 — labels overlay the
  imagery sensibly.
- Script is idempotent (skips a sample when both its `.tif` and `.json` already exist).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurosat
```

## Caveats

- Uniform-class tiles: every pixel in a tile shares the patch label (EuroSAT provides no
  within-patch segmentation). This is the intended scene-level representation for a coherent
  land-cover patch, not dense segmentation.
- Native pixel resolution deviates from exactly 10 m by < 0.1% (inherited from EuroSAT);
  downstream pretraining reprojects onto the S2 grid, so this is immaterial.
