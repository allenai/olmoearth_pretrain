# Rubber & Rubber-Related Deforestation (SE Asia)

- **Slug**: `rubber_rubber_related_deforestation_se_asia`
- **Status**: completed
- **Task type**: classification (rubber presence)
- **Samples**: 2000 (1000 rubber-rich tiles + 1000 non-rubber tiles)
- **Family / region**: plantation / Southeast Asia
- **License**: CC-BY-4.0

## Source

Wang et al. 2023, *"High-resolution maps show that rubber causes substantial
deforestation"*, Nature. Data on Zenodo record **8425153**
(https://zenodo.org/records/8425153), single archive `WangEtAl_Nature.zip` (801 MB),
downloaded via `download.download_zenodo` and extracted under
`raw/rubber_rubber_related_deforestation_se_asia/WangEtAl_Nature/`. The archive holds two
products:

- `Rubber_10m/` — a **10 m binary rubber-plantation map for 2021** (value `1` = rubber,
  `0` = non-rubber), delivered as 53 EPSG:4326 GeoTIFF tiles of up to 65536×65536 px
  (~10 m/px near the equator), LZW-compressed, no nodata. **This is the product used.**
- `Deforestation_30m/` — a 30 m float32 rubber-related deforestation layer. Its pixel
  values are small floats (~1e-6 to ~1e-3), not clean planting years, and the encoding is
  undocumented/ambiguous. The manifest classes are only rubber / non-rubber, so this layer
  was **not converted**. It remains in `raw/` if a future pass wants to decode it.

## Labels / class mapping

Binary classification derived directly from the 10 m map:

| id | name       | meaning |
|----|------------|---------|
| 0  | non-rubber | anything not mapped as rubber (forest, other crops, built-up, water, bare) |
| 1  | rubber     | rubber (Hevea) plantation mapped for 2021 |

Patches carry the **real per-pixel values** (0/1); nodata sentinel is 255 (none present in
practice, as the source is fully observed). This is a `dense_raster` derived product, so
per the spec we take a **bounded set of spatially-homogeneous / high-confidence windows**
(no full coverage).

## Sampling

- Each source tile is read at a **64×-decimated average** (Resampling.average) to get the
  rubber fraction of every non-overlapping 64×64 block — a cheap locator.
- **rubber tiles**: interior blocks with decimated rubber fraction ≥ 0.5 (rubber-rich).
  These windows are mostly rubber but also contain non-rubber pixels.
- **non-rubber tiles**: interior blocks with decimated rubber fraction == 0, restricted to
  each tile's rubber bounding box so they are on-land landscapes rather than open ocean.
- Up to 120 candidates per class per source tile (geographic diversity), then a seeded
  shuffle keeps **1000 per class**. Candidate pool: 3141 rubber, 3840 non-rubber blocks.
- Each selected block is reprojected from EPSG:4326 into a **local UTM 64×64 grid at 10 m**
  with **nearest** resampling (categorical), centered on the block center. Output GeoTIFFs
  are single-band uint8, 10 m, north-up.

Class presence across the 2000 patches: **class 0 in 1996 tiles, class 1 in 1177 tiles**
(some rubber-free windows contain a few rubber pixels because the decimated locator is
approximate; rubber-rich windows nearly always contain both classes).

## Time range

The rubber map is the 2021 (2021–22 composite) product, so every sample gets a **1-year
window anchored on 2021** (`[2021-01-01, 2022-01-01)`). No change labels (the deforestation
layer, which would carry event dates, was not used).

## Verification

- 2000 `.tif` + 2000 `.json`; all single-band, UTM (EPSG:326xx/327xx), 10 m, 64×64, uint8,
  nodata 255; only values {0,1}; all time ranges ≤ 1 year.
- Georeferencing round-trip: for 30 rubber-heavy patches, the patch rubber fraction matched
  the source map's rubber fraction at the patch center within 0.3 for **30/30**; centers
  span Vietnam, Borneo, Sulawesi, Halmahera, etc. (SE Asia), confirming correct
  reprojection/placement.

## Caveats

- Derived-product map (not in-situ reference); adds the **rubber** class to the label bank.
- `Deforestation_30m/` not converted (ambiguous float encoding, not planting years).
- Rubber-free windows are drawn from within the rubber bounding box of each tile, so
  non-rubber diversity is biased toward rubber-adjacent landscapes.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.rubber_rubber_related_deforestation_se_asia
```

Idempotent (skips existing `locations/{id}.tif`). Downloads + extracts the Zenodo archive
if absent; scans 53 tiles with a Pool(64) and writes 2000 patches.
