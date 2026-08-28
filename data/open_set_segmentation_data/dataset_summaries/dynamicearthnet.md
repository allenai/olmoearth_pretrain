# DynamicEarthNet

- **Slug:** `dynamicearthnet`
- **Status:** completed
- **Task type:** classification (dense land cover, `dense_raster`)
- **Samples:** 2464 label tiles (≤64×64, uint8, 10 m UTM)

## Source

DynamicEarthNet (Toker et al., CVPR 2022), TU Munich. 75 global AOIs of daily 4-band
PlanetFusion imagery (3 m, 2018-01-01..2019-12-31) with **monthly pixel-wise 7-class
land-cover labels**. Paper: https://arxiv.org/abs/2203.12560. License **CC-BY-SA-4.0**.

- Landing page: https://mediatum.ub.tum.de/1650201
- Data server (mediaTUM / dataserv): rsync at
  `rsync://m1650201@dataserv.ub.tum.de/m1650201/`, password `m1650201`.

## Access method — labels only (no imagery)

Only the *labels* are needed; pretraining supplies its own S2/S1/Landsat imagery. The data
server exposes a dedicated **`labels.zip` (1.4 GB)** separate from the `planet.*.zip` image
cubes (~500 GB) and the `sentinel1.zip`/`sentinel2.zip` extras, so only `labels.zip` (+
`LICENSE`) was pulled. No huge imagery bundle was downloaded (task-spec impractical-download
guard satisfied — labels are cleanly separable).

Reproduce the download:
```
RSYNC_PASSWORD=m1650201 rsync -av \
  rsync://m1650201@dataserv.ub.tum.de/m1650201/labels.zip \
  rsync://m1650201@dataserv.ub.tum.de/m1650201/LICENSE \
  /weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/dynamicearthnet/
```

## Label format

Inside `labels.zip`: `labels/<AOI>/[Labels/]Raster/<sub>/<sub>-YYYY[-_]MM[-_]01.tif`.
Each is a **7-band, 1024×1024, 3 m, uint8 one-hot** land-cover mask in a local UTM CRS
(each band is 0 or 255; band *b* = 255 means the pixel is class *b*). **1320 raster labels =
55 AOIs × 24 months** (2018-01..2019-12). (The paper's 75 AOIs include a held-out test set
whose labels are not shipped in `labels.zip`; all 55 AOIs that carry raster masks are used —
all splits are fair game per task-spec §5.) A parallel `Vector/*.geojson` copy of the same
labels was ignored. Two filename quirks handled: date separators are either `-` or `_`, and
AOI `1417_3281_13_11N` omits the `Labels/` path level (uses `Raster/` directly).

## Class mapping (output id = source band index)

| id | class | source band |
|----|-------|-------------|
| 0 | impervious surface | 0 |
| 1 | agriculture | 1 |
| 2 | forest & other vegetation | 2 |
| 3 | wetlands | 3 |
| 4 | bare soil | 4 |
| 5 | water | 5 |
| 6 | snow & ice | 6 |

Pixels with no active band (unlabeled; only 6515 px total across ~1.38e9 label px) →
**nodata 255**. Verified against the official `dynnet` loader that bands 0–5 map to
impervious/agriculture/forest/wetland/soil/water. **Divergence from the official benchmark:**
the official DynamicEarthNet evaluation uses only 6 classes and maps the snow-&-ice band (6)
to *ignore*; here snow & ice is kept as a real class (it genuinely appears in 48 of 1320
AOI-months) per task-spec §5 "keep every class you can" — downstream assembly can filter it
if too sparse.

## Processing (dense_raster / VHR resample, task-spec §4)

1. Collapse the 7-band one-hot 3 m mask to a single-band class-id raster (argmax of active
   bands; unlabeled → 255).
2. Reproject 3 m → **10 m** within the AOI's native UTM CRS with **MODE** resampling
   (categorical majority; never bilinear), snapped to the 10 m UTM grid → ~308×308 per AOI.
3. Cut into non-overlapping **≤64×64** tiles (~25 per AOI-month); drop fully-nodata tiles.
   → **33,000 candidate tiles**.
4. **Tiles-per-class balanced** selection (`sampling.select_tiles_per_class`, ≤1000
   tiles/class, rarest-class-first, ≤25k total) → **2464 tiles**.

Georeferencing verified: written tile bounds fall inside their source AOI footprints and
pixel centers reproject to the correct regions (e.g. AOI `1417_3281_13_11N` → California,
UTM 11N; `6466_3380_13_48N` → Sichuan, China, UTM 48N).

## Time range / change handling (task-spec §5)

Each monthly label is the per-month **land-cover state**, not a dated change event.
Therefore `change_time = null` and `time_range` is a **~3-month window centered on the
label's month** (built as `centered_time_range(center = 15th of the label month,
half_window_days=45)`, i.e. ~90 days bracketing that calendar month). All labels are
2018–2019, well within the Sentinel era.

## Class tile counts (a tile counts toward every class present)

| id | class | tiles |
|----|-------|-------|
| 0 | impervious surface | 1281 |
| 1 | agriculture | 1000 |
| 2 | forest & other vegetation | 1966 |
| 3 | wetlands | 1039 |
| 4 | bare soil | 2242 |
| 5 | water | 1017 |
| 6 | snow & ice | 946 |

Snow & ice is the rarest class (946 tiles, all available snow tiles kept, under the 1000
cap). Common classes exceed 1000 because they co-occur in tiles selected to satisfy rarer
classes.

## Caveats

- Snow & ice kept as a real class despite being ignored by the official 6-class benchmark
  (see class mapping).
- Monthly labels of the same AOI are near-identical from month to month; the shuffled
  tiles-per-class selection spreads picks across AOIs/months, but some spatial/temporal
  redundancy remains (55 distinct AOIs). Each monthly label now gets its own ~3-month
  `time_range` centered on that month, so the up-to-12 monthly labels of the same footprint
  no longer collapse onto a single shared year-long window.
- The finest land-cover distinctions (individual buildings, narrow roads) are folded into
  neighbouring classes by mode resampling at 10 m; zone-level land cover is well preserved.

## Reproduce

```
# (after the rsync download above)
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.dynamicearthnet --workers 64
```
Idempotent (skips already-written `locations/{id}.tif`); scan cached to
`raw/dynamicearthnet/scan_cache.pkl`.
