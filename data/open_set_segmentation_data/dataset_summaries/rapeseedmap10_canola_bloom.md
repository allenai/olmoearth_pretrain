# RapeseedMap10 (Canola Bloom)

- **Slug:** `rapeseedmap10_canola_bloom`
- **Status:** completed — classification, 2000 samples (1000 per class)
- **Family:** phenology · **Label type:** dense_raster · **Task:** classification (rapeseed presence)

## Source

Mendeley Data DOI [10.17632/ydf3m7pd4j.3](http://dx.doi.org/10.17632/ydf3m7pd4j.3) —
Han, Zhang, Luo et al., *"Developing a phenology- and pixel-based algorithm for mapping
rapeseed at 10 m spatial resolution using multi-source data."* A regional 10 m annual
rapeseed/canola presence map that exploits the distinctive canola flowering (bloom)
signal in Sentinel-1/-2 time series. Version 3 (the version pinned by the manifest URL)
covers **18 regional tiles × 3 years (2017, 2018, 2019)** = 54 GeoTIFFs.

- License: **CC-BY-4.0** (open, redistributable).
- Access: fully public. Downloaded the single archive `rapeseed map.zip` (~995 MB) plus
  `README.txt` via the Mendeley public-files API (no credentials needed).
- Raw stored at
  `raw/rapeseedmap10_canola_bloom/{rapeseed_map.zip, README.txt, rapeseed map/*.TIF}`.

### Source raster format
- Projection **EPSG:4326 (WGS84 geographic)** at ~8.983e-5° (≈10 m) per pixel.
- Single band, `uint8`. Values: **0 = non-rapeseed (observed land)**, **1 = rapeseed**.
- **Nodata is inconsistent across tiles** — some tiles declare nodata `3`, others `255`.
  The script keys off the `{0, 1}` class set (treating every other value as nodata)
  rather than trusting a single sentinel; this was a real bug source (see caveats).
- Filenames encode year + corner lon/lat, e.g. `2018Y010E50N.TIF` = 2018, 10°E / 50°N.
  Individual tiles are large (up to ~154k × 175k px), spanning several degrees.

## Processing

Regional derived-product map → **bounded-tile dense_raster sampling** (§4/§5 of the spec).

1. **Scan** (Pool(64) over the 54 source tiles): read each tile in 64-row strips and
   reduce into 64×64 native-pixel blocks (64 px × 10 m ≈ 640 m). Per block, count valid
   pixels (`==0 or ==1`) and rapeseed pixels (`==1`). Keep only **spatially homogeneous /
   high-confidence** blocks:
   - **rapeseed candidate:** rapeseed fraction ≥ 0.25 over observed pixels;
   - **non-rapeseed candidate:** zero rapeseed pixels and ≥ 90% observed (pure observed
     non-rapeseed land — excludes nodata/ocean).
   Reservoir-capped per tile (≤2000 rapeseed, ≤150 non-rapeseed) to bound memory while
   preserving geographic spread. Yielded 77,880 rapeseed and 8,100 non-rapeseed candidates.
2. **Select:** seeded shuffle, take up to 1000 per class → 1000 + 1000 = 2000.
3. **Write** (Pool(64)): for each selected block, take its center lon/lat, compute the
   local UTM projection at 10 m, and reproject a 64×64 UTM patch from the source with
   **nearest** resampling (categorical). Any value that is not 0/1 becomes 255 (nodata).

## Output

- `datasets/rapeseedmap10_canola_bloom/metadata.json`
- `datasets/rapeseedmap10_canola_bloom/locations/{000000..001999}.tif` + `.json`
- Each patch: single-band `uint8`, **local UTM, 10 m/pixel, 64×64**, nodata **255**.

### Classes (per-pixel; native ids kept, no remap)
| id | name | pixel meaning |
|----|------|---------------|
| 0 | non-rapeseed | observed land, not rapeseed |
| 1 | rapeseed (bloom-based) | canola presence from bloom signal |

### Counts
- **Selection basis:** 1000 rapeseed-primary tiles, 1000 non-rapeseed tiles.
- **Per-pixel presence** across the 2000 dense tiles: class 0 appears in 1989 tiles
  (rapeseed tiles are mixed and contain non-rapeseed pixels too), class 1 in 1011 tiles.
- Samples are spread across 2017/2018/2019 and multiple UTM zones (Europe, N. America,
  China, S. America canola belts).

### Time range
Annual presence label → **1-year** `time_range` anchored on the file's labeled year
(`[year-01-01, year+1-01-01)`). `change_time` is null: this is yearly presence
classification, **not** a dated bloom-event change label. (Bloom is a phenological event,
but the product provides only per-year presence, so a precise bloom date is not available.)

## Caveats
- Reprojection from EPSG:4326 to UTM uses nearest resampling; sub-pixel class boundary
  shifts are possible but negligible at 10 m for this coarse binary product.
- Nodata inconsistency (3 vs 255) across tiles was handled by keying on the `{0,1}` class
  set; a naive single-sentinel approach produced all-nodata tiles (caught and fixed).
- Non-rapeseed negatives are drawn from within the same canola-relevant regions (observed
  land only), so they are meaningful negatives rather than trivial ocean/desert.
- The map is a derived product, not in-situ reference; rapeseed tiles were restricted to
  ≥25% rapeseed to favor confident, homogeneous positives.

## Reproduce
```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.rapeseedmap10_canola_bloom --workers 64
```
Idempotent: re-running skips any `locations/{id}.tif` already present (selection is
seeded/deterministic). Registry status is owned by the orchestrator; the script does not
write `registry.json`.
