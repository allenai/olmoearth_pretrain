# WorldFloods v2 — COMPLETED (classification, dense_raster)

- **Slug**: `worldfloods_v2`
- **Name**: WorldFloods v2
- **Source**: Hugging Face `isp-uv-es/WorldFloodsv2` (Image Signal Processing group,
  Universitat de València).
- **Citation**: Portalés-Julià, Mateo-García, Purcell, Gómez-Chova, "Global flood
  extent segmentation in optical satellite images", *Scientific Reports* 13:20316
  (2023). DOI 10.1038/s41598-023-47595-7.
- **Family / region**: flood / global (Copernicus EMS activations, 500+ events).
- **License**: CC-BY-NC-4.0 (non-commercial).
- **Label type**: dense_raster → per-pixel **classification**.
- **Task type**: classification. **num_samples**: 1968 tiles.

## Source

509 scenes (train/val/test = 475/16/18), each a Sentinel-2 image paired with a
flood segmentation mask curated from Copernicus EMS rapid-mapping /
photointerpretation. Every scene is already in its **local UTM projection at
10 m/pixel, north-up**. We use ONLY the label rasters — NOT the ~76 GB of S2
imagery:

- `{split}/gt/{name}.tif` (int16, **2 bands**):
  - band 1 = cloud layer: `0` invalid, `1` clear, `2` cloud.
  - band 2 = land/water layer: `0` invalid, `1` land, `2` water.
- `{split}/PERMANENTWATERJRC/{name}.tif` (int16, 1 band): JRC Global Surface
  Water permanent-water overlay co-registered to the scene. Value `3` = permanent
  water (verified: value-3 pixel count equals the per-scene meta
  `pixels permanent water S2`).
- `dataset_metadata.csv`: per-scene `split`, `s2_date` (the paired Sentinel-2
  acquisition timestamp), `crs`, `transform`, `bounds` — used for dating and
  split assignment (no per-scene meta JSON needed).

## Access method

Public, no credentials (CC-BY-NC). Pulled with `huggingface_hub.snapshot_download`
restricted to `allow_patterns=["*/gt/*", "*/PERMANENTWATERJRC/*",
"dataset_metadata.csv"]`, so only ~1019 small label files (~290 MB) download and
the S2 imagery is skipped. (HF returned transient HTTP 429s mid-download; the
snapshot retried and completed — no data lost.)

## Class mapping (4 classes)

The manifest lists a combined `land / water-flood / cloud` scheme. Following the
completed **sen1floods11** precedent (same flood family), we split the water class
into flood vs permanent using the provided JRC overlay, giving a richer 4-class
scheme that is directly comparable across the flood family:

| id | name | definition |
|----|------|-----------|
| 0 | flood water | observed water (land/water band == water) AND NOT JRC permanent — the flood inundation |
| 1 | permanent water | observed water that JRC marks permanent (value 3): rivers, lakes, reservoirs |
| 2 | land | land/water band == land, where not cloud-covered |
| 3 | cloud | cloud band == cloud; **overrides** land/water (the S2 image shows cloud there) |
| 255 | nodata/ignore | both bands invalid / unobserved |

Per-pixel fusion order: nodata default → land → water (split by JRC) → cloud
overrides. **Cloud takes priority over observed land/water** because the label is
paired with the S2 image at `s2_date`: where the cloud band flags cloud, the
optical surface is obscured, so `cloud` is the honest label for label↔image
alignment. (This is why observed-water counts shrink relative to the reference
flood delineation, which was drawn partly from cloud-penetrating SAR.)

Splitting flood vs permanent is a documented **judgment call**: the manifest's
"water/flood" is enriched, not contradicted, and the JRC layer is shipped
precisely to enable it. `flood water = observed water − JRC permanent` matches the
dataset authors' own `pixels flood water` definition.

## Processing

Each scene raster is already UTM 10 m north-up, so there is **no reprojection** —
the scene CRS is reused and the raster is tiled directly into 64×64 patches
(integer rslearn pixel bounds derived from the raster transform; origins are
S2-grid-aligned multiples of 10). Tiles >50% nodata are dropped; a tile counts
toward a class only with ≥32 px of it. **Tiles-per-class balanced** selection
(spec §5) via `sampling.select_tiles_per_class` (≤1000 tiles/class, 25k dataset
cap, rare classes filled first). All three source splits are used (spec §5).
1,185,899 candidate tiles → 1968 selected.

**Tiles containing each class** (a tile can count for several):

- flood water: 1106
- permanent water: 1157
- land: 1368
- cloud: 1000

Well under the 25k per-dataset cap.

## Time range & change handling

Flood water here is treated as a **per-image STATE** observed in one specific
Sentinel-2 acquisition (not a diffuse yearly change). Per spec §5 (specific-image
labels), `time_range` is a short **~1-hour window at the `s2_date` acquisition**
and `change_time` is **null**. This deliberately differs from sen1floods11, which
treated its mask as a change label with a year-centered window — for WorldFloods
the flood mask describes the water visible in that S2 image, so pretraining should
pair it with imagery at that acquisition. All 509 `s2_date`s fall 2016–2023, fully
within the Sentinel era (no pre-2016 filtering needed).

## Caveats

- Cloud-priority fusion means flooded pixels under cloud in the S2 image are
  labeled `cloud`, not water. This is intentional for optical label↔image
  alignment but reduces water-class area in cloudy scenes.
- Permanent water is restricted to pixels observed as water AND JRC-permanent; a
  JRC-permanent pixel not observed as water (e.g. under cloud, or read as land) is
  not forced to permanent water (more conservative than sen1floods11's JRC
  override of land).
- Only 344 of 509 scenes contribute selected tiles (balancing caps reached).
- Non-commercial license (CC-BY-NC-4.0) — flagged for downstream use.

## Verification

- 1968 `.tif` + 1968 matching `.json`, no unpaired; every tile single-band uint8,
  local UTM, 10 m, 64×64, values ⊆ {0,1,2,3,255} with nodata=255.
- `time_range` is a 1-hour window and `change_time` is null on every sample;
  `metadata.json` class ids {0,1,2,3} cover all values in the tifs.
- **Round-trip**: 4 random tiles rebuilt from the source `gt`+`PERMANENTWATERJRC`
  rasters matched the written arrays exactly (`array_match True`) with matching
  pixel bounds and CRS.
- **Coordinate sanity**: tile from `10132016_Ashley_River_near_North_Charleston_SC`
  centers at lon −80.15, lat 32.98 (North Charleston, SC) — as expected.
- Re-running skips already-written tiles (idempotent).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.worldfloods_v2
```
