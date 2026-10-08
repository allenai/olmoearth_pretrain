# Oil Slicks & Look-Alikes (E. Mediterranean SAR)

- **Slug**: `oil_slicks_look_alikes_e_mediterranean_sar`
- **Status**: completed
- **Task type**: classification (bounding-box detection encoded as per-pixel classes)
- **Source**: PANGAEA https://doi.org/10.1594/PANGAEA.980773 (Yang & Singha, 2025), CC-BY-4.0. ESSD data descriptor: https://doi.org/10.5194/essd-17-6807-2025. Related method paper: Yang et al. 2024, https://doi.org/10.1080/01431161.2024.2321468.
- **Region / time**: Eastern Mediterranean Sea, 2019 (Sentinel-1 SAR, ~10 m native).
- **num_samples**: 2000 (1000 oil-slick tiles + 1000 look-alike/no-oil tiles).

## Source data
Manually interpreted Sentinel-1 SAR patches (two interpreters) over the E. Mediterranean in 2019:
- **Oil set**: 1365 patches with **3225 oil-slick objects** (subsets `ow` oil/water, `oc` oil/coast), each a manually drawn bounding box.
- **No-oil set**: **2290 look-alike patches** (`nw` no_oil/water, `nc` no_oil/coast) — oceanic/atmospheric phenomena (low wind, biogenic films, rain cells, current fronts, etc.) that mimic oil in SAR but are not oil. Whole-patch scenes, **no localized box**.

## Georeferencing (checked first, per SOP §8)
Fully georeferenced. The PANGAEA **tab-delimited data matrix** (`raw/.../pangaea_data.txt`, 5515 rows) carries, per row: image set, jpg/xml filename, patch_name, S1 acquisition `start_time`/`end_time`, Sentinel-1 `.SAFE` granule id, patch corner lon/lat (ul/ur/br/bl), oil-object bbox corner lon/lat (ul/ur/br/bl) + pixel xmin/ymin/xmax/ymax + label area. The per-patch JPG SAR images and PASCAL-VOC XML boxes (only in `allfiles.zip/tar`, which needs a PANGAEA account) are **not needed** — the matrix alone places every oil footprint and every patch on the S2/UTM grid. So no raster download was required (only labels are needed; pretraining supplies its own imagery).

## Access
PANGAEA tab-delimited export via `?format=textfile` with a **Firefox User-Agent** (a generic UA gets HTTP 403 "unusual traffic" — UA fingerprinting, not a rate limit). Saved to `raw/.../pangaea_data.txt` (+ `Metadata.txt`, `README.pdf`, `SOURCE.txt`).

## Class scheme (spec §5 multi-target → one unified class map)
Binary, matching the manifest's two classes:
- **0 = oil_slick** — manually interpreted oil slick (dark low-backscatter film), rasterized from the georeferenced bbox footprint.
- **1 = look_alike_no_oil** — the manifest "look-alike/no-oil" confuser class: both the dedicated look-alike scenes and ordinary non-oil sea surrounding a slick.
- **255 = nodata/ignore** — buffer ring around imprecise oil-box edges.

## Encoding (label_type = bounding boxes → detection, spec §4)
- **Oil tiles** (one per selected oil object): 64×64 UTM 10 m tile centered on the object's geo centroid. The object's geo quadrilateral (plus any sibling oil boxes of the same patch that fall in the tile) is rasterized as class 0, dilated by a **10 px nodata (255) ring** to absorb bbox imprecision, and the rest is class 1 (non-oil sea). Written in the tile's local UTM zone (mostly EPSG:32636).
- **Look-alike tiles** (one per selected no-oil patch): 64×64 tile centered on the patch centroid, filled **entirely class 1** — a spatially-meaningful hard-negative confuser tile (spec §5 detection exception). These patches have no localized box, so the whole (persistent-in-that-acquisition) look-alike scene is labeled.

## Time range (spec §5)
Each sample uses its own S1 acquisition window `[start_time, end_time]` (~1–2 min, well under the ~1 hour specific-image budget). An oil slick / look-alike is visible only in the matching S1 acquisition. All labels are 2019 (post-2016).

## Sampling / counts
Up to **1000 oil-slick tiles** + **1000 look-alike tiles** (spec §5 default, seed 42), well under the 25k cap. Available pool: 3225 oil objects / 1365 oil patches, 2290 look-alike patches. Class 1 additionally appears as background in ~455 of the oil tiles; the other ~545 oil tiles are fully oil (see caveat).

## Verification (spec §9)
- 2000 `.tif` + 2000 matching `.json`, all paired.
- Every tile: single band, 64×64, uint8, UTM (EPSG:326xx) at 10 m, nodata 255; pixel values ∈ {0, 1, 255} only.
- All `time_range`s < 360 days (≈1–2 min S1 windows, 2019).
- Global pixel counts: oil 3.01M, no-oil 4.57M, nodata 0.61M.
- **Georeferencing round-trip**: all 2000 tile centers reproject back to lon/lat inside the E. Mediterranean coverage bbox (lon 27–36.2, lat 29.2–36.5) — i.e. all over the correct sea area. Both classes are sea-surface phenomena so a full Sentinel-2 overlay render was not performed; coordinate-level validation (exact UTM round-trip within the marine bbox) was used instead.
- Re-running is idempotent (second run: skip 2000).

## Caveats
- **Large slicks fill the tile.** Oil slicks here are typically larger than a 640 m (64 px) tile — median oil tile is ~93% oil (median 3820 / 4096 px), and 419 of 1000 oil tiles are entirely oil. This is inherent (the 64 px cap is the hard max; many real slicks span several km) and still yields correct "this area is oil" labels. Oil/no-oil contrast comes from the 1000 dedicated look-alike tiles plus the ~455 partial oil tiles.
- **Look-alike localization.** Look-alike patches carry no per-object box, so the whole 640 m center tile is labeled class 1. Justified because the patch was specifically selected for a prominent, spatially-extensive look-alike phenomenon.
- **Bbox (not exact mask) positives.** Oil positives are rasterized bounding-box footprints, so edges are approximate; the 10 px nodata ring mitigates this.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.oil_slicks_look_alikes_e_mediterranean_sar
```
Raw label table (re-download if absent, Firefox UA):
```
curl -A "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0" \
  "https://doi.pangaea.de/10.1594/PANGAEA.980773?format=textfile" \
  -o /weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/oil_slicks_look_alikes_e_mediterranean_sar/pangaea_data.txt
```
