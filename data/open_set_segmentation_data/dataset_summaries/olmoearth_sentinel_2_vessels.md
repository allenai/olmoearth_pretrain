# OlmoEarth Sentinel-2 vessels

- **slug**: `olmoearth_sentinel_2_vessels`
- **task_type**: classification (object detection encoded as per-pixel classes)
- **status**: completed
- **num_samples**: 2000 (1000 vessel-positive tiles + 1000 background-negative tiles)
- **classes**: `0 = background` (open water), `1 = vessel`; `255 = nodata/ignore` (buffer ring)

## Source

Local rslearn dataset (`have_locally=true`, not copied — see
`raw/olmoearth_sentinel_2_vessels/SOURCE.txt`):

```
/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessels/dataset_v1/20250213
```

This is the existing OlmoEarth / Satlas **Sentinel-2 vessel-detection eval**: manually
annotated windows, each a specific-image crop already in a **local UTM projection at
10 m/pixel** (~780×780 px) with a ~10-minute Sentinel-2 acquisition `time_range`.

- Label layer `label` (vector GeoJSON): one `Point` feature per annotated vessel,
  `properties.category == "vessel"`. Coordinates are in the window's projection (pixel)
  coordinates, matching the window `bounds`, so **no reprojection is needed**.
- `metadata.options.has_objects` flags windows that contain ≥1 vessel.

Window inventory (excluding sargassum groups): **44,656 windows**, of which **8,821 are
positive** (contain ≥1 vessel; **37,929 vessels total**) and **35,835 are vessel-free**
(usable negatives, including the dedicated `train-bg` / `valid-bg` groups).

## Label type note

Manifest `label_type` is `bboxes`, but the on-disk annotations are **points** (vessel
centroids), so the dataset is processed with the tunable **detection** encoding (spec §4
`bboxes / points → detection`), not polygon rasterization.

## Encoding (detection, spec §4)

- One **32×32** (`DET_TILE`) context tile per vessel, centered on the vessel pixel, written
  in the window's **own UTM projection** (source already local UTM @ 10 m → georeferencing
  is exact).
- Vessel = **1×1 positive** (id 1), ringed by a **10 px nodata (255) buffer** (vessel
  centroids are not pixel-exact), all other pixels **background** (id 0). With
  `positive_size=1`, `buffer_size=10` the ignore ring is 21×21, leaving ample background in
  a 32×32 tile.
- Every other annotated vessel of the same source window that falls inside a tile is also
  marked positive (so multi-vessel tiles are labeled completely).
- **Negatives**: background-only tiles sampled from vessel-free windows
  (`has_objects == false`, incl. `train-bg`/`valid-bg`) so the background class has
  spatially-meaningful negatives (spec §5 detection exception).

Shared code reused: `sampling.encode_detection_tile`, `io.centered_bounds`,
`io.write_label_geotiff`, `io.write_sample_json`, `io.write_dataset_metadata`,
`manifest.write_registry_entry`, `rslearn.utils.mp.star_imap_unordered`.

## Sampling / counts (spec §5)

Single class → up to **1000 positive vessel tiles** (of 37,929 available; shuffled, seed 42)
+ **1000 background-negative tiles** (of 35,835 vessel-free windows). Total 2000, far under
the 25k cap. All vessel splits are used (splits are pretraining-agnostic).

## Time range (spec §5, specific-image)

Each sample uses its **source window's own ~10-minute S2 acquisition `time_range`** (verified
uniformly 600 s across all 2000 samples — well under the ~1 hour specific-image budget). No
`change_time`.

## Judgment calls

- **`sargassum_train` / `sargassum_val` groups excluded.** They belong to a different
  (sargassum) task where vessels were **not** annotated, so their scenes are unsafe as
  vessel negatives (unlabeled vessels could be present). All other groups used.
- **label_type bboxes → treated as point detection** because the actual annotations are
  centroids (documented above).
- Vessel/background are both spatially meaningful within a tile, so this dataset emits its
  own negatives (detection exception to the "no synthetic negatives" rule).

## Verification (spec §9)

- 2000 `.tif` each paired with a `.json`; all single-band **uint8**, **32×32**, local UTM
  CRS at **10 m**, nodata **255**. Positive tiles contain values {0, 1, 255}; negatives are
  all 0.
- All `time_range`s are exactly 10 min (≤ 1 yr); `metadata.json` class ids {0,1} cover all
  values in the tiles.
- **Spatial sanity**: over 40 positive samples, the brightest source-S2 pixel within a 7×7
  window of each annotation is a **median 1.71× / mean 2.17×** brighter than the surrounding
  water (90% > 1.15×), confirming vessel positives land on bright spots over dark water — no
  gross misalignment.
- Re-running the script is **idempotent** (second run: `{'skip': 2000}`).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_sentinel_2_vessels --workers 64
```
