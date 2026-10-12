# Amazon Mining Watch

- **Slug**: `amazon_mining_watch`
- **Status**: completed
- **Task type**: classification (open-set / presence segmentation, single positive class)
- **Family / region**: mining / Amazon basin (nine Amazonian countries)
- **License**: CC-BY-4.0
- **Source**: Source Cooperative — Earth Genome,
  <https://source.coop/earthgenome/amazon-mining-watch>

## What the source is

Amazon Mining Watch publishes annual, machine-learning-detected polygons of artisanal /
illegal **gold-mine scars** in Sentinel-2 imagery across the Amazon basin. Detections are
produced on overlapping 480 m × 480 m Sentinel-2 patches, merged into polygons where they
overlap, and released as GeoJSON (CRS84 / EPSG:4326). Each annual file is **cumulative
from 2018** (the 2020 file contains every scar detected 2018-2020, etc.). Scars are large
areal bare-earth / tailings-pond footprints: median polygon extent ≈ 830 m, many multi-km,
so this is a presence/segmentation label, not a positive-only point detection.

Files used (`raw/amazon_mining_watch/`), one per year 2018-2024:
`{year}/amazon_basin_..._2018-{year}cumulative.geojson` (2024 file:
`amazon_basin-48px_v0.X-SSL4EO-MLPensemble_2018-2024cumulative-clean.geojson`). The 2025
quarterly files and 2025 raster masks in the bucket are **not** used (outside the requested
2018-2024 range).

## Access method

Public, unsigned S3: bucket `us-west-2.opendata.source.coop`, prefix
`earthgenome/amazon-mining-watch/`. Downloaded with an unsigned boto3 client (no
credentials). See `raw/amazon_mining_watch/SOURCE.txt`. ~65 MB total.

## Class / label mapping

Single positive class plus an explicit background/negative class:

| id  | name          | encoding |
|-----|---------------|----------|
| 0   | background    | non-mine surface (forest, water, other land cover) |
| 1   | mine_scar     | artisanal/illegal gold-mine scar polygon rasterized into the tile |
| 255 | nodata/ignore | declared, but not used (no ignore regions in this encoding) |

Encoding: a global 640 m (64 px @ 10 m) grid is laid over each local UTM zone. Every grid
cell a mine-scar polygon actually intersects becomes a **positive tile**: all polygons of
the relevant year intersecting the cell are reprojected to the cell's UTM pixel space and
rasterized (`rasterio.features.rasterize`, `all_touched=False`) as `mine_scar` (1), with
`background` (0) filling the rest of the 64×64 tile. Because scars are large, positive
tiles range from full-interior (all 1) to partial-coverage edge tiles — a natural mix for
segmentation. **Background-only negative tiles** (all 0) are placed 3.8-25.6 km from a
random positive, in the same UTM zone, and verified free of any mine polygon across all
years, giving the class genuine negatives.

## Time range and change handling

Files are cumulative, so each grid cell is assigned to the **earliest year it appears**
(greedy across 2018→2024); the tile's `time_range` is that ~1-year window
(`io.year_range`). This spreads tiles across years and gives each the year the scar first
became detectable. No `change_time` is set — these are presence labels of a persistent
areal feature, not dated instantaneous events. Negative tiles inherit the year of their
seed positive.

**Caveat (from the source README):** the 2024 data uses a newer, more sensitive model
(SSL4EO ViT-DINO ensemble) than the 2018-2023 legacy CNN ensemble, so the additional
mining detected in 2024 is partly a model-shift artefact; cross-year *trends* are not
reliable. This does not affect per-tile label validity.

## Sampling

- Bounded to **≤1000 tiles per class**. 113,843 distinct positive grid cells were found;
  positives were stratified by earliest year (~140/year × 7 years) for temporal diversity
  and capped at 1000 selected.
- Of the 1000 selected positive cells, **36 were dropped** at rasterization time as
  degenerate (polygon intersected the cell box but covered no pixel center at 10 m),
  leaving **964 mine_scar tiles**. Sample-id numbering therefore has 36 gaps in the
  `000000`-`000999` range; every written `.tif` has a matching `.json`.
- **1000 background negative tiles** were generated.
- **Total: 1964 samples** (964 mine_scar + 1000 background).

`year_counts` in `metadata.json` reflects the 1000 *selected* positives (≈140/year), not
the 964 actually written after degenerate drops.

## Output GeoTIFF spec

Single-band uint8, local UTM at 10 m/pixel, north-up, 64×64, nodata=255. Class IDs {0,1}.

## Verification performed

- 7 sample tifs (positives + negatives) inspected: single band, uint8, UTM CRS, 10 m
  resolution, 64×64, nodata=255. Positives contain values ⊆ {0,1} (edge tiles mix 0/1;
  scar-interior tiles are all 1); negatives are all-0.
- File pairing: 1964 `.tif` and 1964 `.json`; every tif has its sidecar.
- Time ranges are 365-366 days (≤1 year); `metadata.json` class IDs cover all tif values.
- **Georeferencing sanity check**: for 3 positive tiles, mine (value-1) pixels were
  reprojected back to lon/lat and tested against the source polygons — **100% (90/90
  sampled pixels) fall inside a source mine-scar polygon**, confirming exact alignment.
- Not done: full Sentinel-2 image overlay (georeferencing is exact via rslearn write and
  the label→polygon round-trip above).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.amazon_mining_watch
```

Idempotent: downloads and per-tile writes skip existing files; selection is seeded. All
logic lives in
`olmoearth_pretrain/open_set_segmentation_data/datasets/amazon_mining_watch.py`; no shared
module was modified, and the script does not write `registry.json`.

## Caveats

- Derived product (ML + human review), not in-situ reference; usable per the task spec.
- 2024 model shift (see above) makes year-over-year counts non-comparable.
- Very large scar polygons are capped at 64 sampled grid cells each so no single scar
  dominates the bounded sample.
- Near UTM zone boundaries, the local UTM zone is chosen from a 1° centroid cache, so a
  handful of tiles may sit in an adjacent zone; each tile remains correctly georeferenced
  (its stored CRS + pixel bounds match the written raster).
