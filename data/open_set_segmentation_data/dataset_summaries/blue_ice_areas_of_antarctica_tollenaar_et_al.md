# Blue Ice Areas of Antarctica (Tollenaar et al.)

- **Slug:** `blue_ice_areas_of_antarctica_tollenaar_et_al`
- **Status:** completed
- **Task type:** classification (binary dense segmentation)
- **Family / region:** glacier / Antarctica
- **Source:** Tollenaar, V. et al., *"Where the White Continent is blue: deep learning locates
  bare ice in Antarctica"*, Geophysical Research Letters (2024).
  Zenodo record **10539933** (concept DOI `10.5281/zenodo.8333864`), license **CC-BY-4.0**.
- **Num samples:** 1500 tiles (64×64, 10 m, local UTM/UPS)

## Source & access

Downloaded the polygon shapefiles from Zenodo (no credentials needed) with
`download.download_zenodo("10539933", ...)` and unzipped them under
`raw/{slug}/extracted/`. The record also contains two large rasters we deliberately did
**not** download:

- `BIA_map.nc` (9.2 GB) — per-pixel blue-ice presence raster.
- `merged_bands_composite*.tif` (5.8 GB) — the input imagery composite.

Only the **labels** are needed (pretraining supplies its own imagery), and the vectorized
polygon product carries the same information at a tiny fraction of the download volume, so we
used `smoothed_BIAs.shp` (6644 polygons, EPSG:3031 / Antarctic Polar Stereographic) — the
smoothed, published continent-wide Blue Ice Area product.

The record additionally ships **hand-labelled** blue-ice polygons for 5 training squares
(`handlabels_sq{246,264,265,278,409}.shp`) — a higher-quality *manual* reference, but they
cover only 5 regions. We chose the **validated continent-wide** `smoothed_BIAs` product
instead to obtain pan-Antarctic geographic diversity (blue ice is spectrally very distinct
and the product was validated against manual test squares, so polygon interiors are
high-confidence; spec §4/§5 derived-product handling). The hand-label and train/val/test
square shapefiles are retained in `raw/` for provenance / possible future higher-quality use.

## Label mapping (classes)

Binary dense segmentation, `uint8`, nodata = 255 (unused here):

| id | name | meaning |
|----|------|---------|
| 0 | background | non-blue-ice Antarctic surface within the tile (snow / firn / exposed rock / other or snow-covered ice) adjacent to a blue-ice area. Genuine, spatially-meaningful negatives — not fabricated. |
| 1 | blue_bare_ice | perennially wind-scoured, snow-free bare/blue glacial ice (spectrally distinct). |

Every tile is drawn around blue ice, so background is real surrounding terrain within the
tile; no separate far-away negative tiles were generated (spec §5).

## Processing recipe

1. Load `smoothed_BIAs.shp` (EPSG:3031); build a shapely `STRtree` for fast intersection.
2. **Candidate tiles:** from each polygon, rejection-sample points inside it (≈ one per 640 m
   tile of polygon area, capped at 6/polygon), convert each to lon/lat → local UTM/UPS
   projection (`get_utm_ups_projection(lon, lat, 10, -10)`), and snap to a 64-px grid. Dedup
   → 19,061 unique candidate tiles.
3. **Rasterize** each unique 64×64 (640 m @ 10 m) tile: query the STRtree with the tile
   footprint (in 3031 m), clip intersecting polygons to the tile, reproject to the tile's
   local projection pixel space, and burn value 1 (blue ice) with background fill 0
   (`rasterize.rasterize_shapes`, `all_touched=False`).
4. **Selection:** stratify candidates across blue-ice-fraction buckets
   {`interior` ≥ 0.85, `edge`, `sliver` ≤ 0.15} and take ≤ 500 each
   (`sampling.balance_by_class`) → **1500 tiles**, giving homogeneous, boundary, and
   background-dominant geometry. Well under the 25k cap.
5. Write each tile with rslearn `GeotiffRasterFormat` (exact georeferencing, atomic write)
   plus a sidecar JSON. Idempotent: existing `{id}.tif` are skipped.

**Key fix during development:** the source-projection helper must use an *identity*
resolution `Projection(EPSG:3031, 1, 1)` (mirroring rslearn's `WGS84_PROJECTION`); an initial
`(1, -1)` flipped the Y axis and mislocated tiles. Verified post-fix that rasterized blue-ice
pixels reproject back **inside** the source polygons (inside-fraction = 1.00).

## Time-range & change handling

Blue ice areas are **persistent** geomorphological features (perennial wind ablation keeps
them snow-free for years–decades), so they are treated as **static** labels: a single
representative 1-year Sentinel-era window **2019** (`[2019-01-01, 2020-01-01)`). The
underlying composite spans ~2016–2024 and the features are stable across it. `change_time` is
`null`. All windows are post-2016.

## Sample counts

- Total tiles: **1500** (frac buckets: interior 500, edge 500, sliver 500).
- Class **tile-appearance** counts: background (0) = 1203, blue_bare_ice (1) = 1500.
- 53 distinct UTM/UPS zones (incl. UPS South EPSG:5042); tile centres span ~ −68° to −82° S
  across all longitudes → pan-Antarctic coverage.

## Verification (spec §9)

- All 1500 `.tif` are single-band `uint8`, 10 m, 64×64, values ⊆ {0, 1}, nodata 255; each has
  a matching `.json` with a ≤1-year `time_range` and `change_time=null`.
- `metadata.json` class ids {0,1} cover all pixel values present.
- **Spatial sanity:** for random samples, rasterized blue-ice pixels reproject back inside the
  source `smoothed_BIAs` polygons (inside-fraction 1.00) and tile centres lie within
  Antarctica. A full Sentinel-2 image overlay was not run (Antarctic S2 ingestion is
  heavy/spotty here); georeferencing is exact (rslearn-encoded) and the mask↔polygon
  self-consistency check confirms correct placement.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.blue_ice_areas_of_antarctica_tollenaar_et_al --workers 64
```

## Caveats

- The `smoothed_BIAs` product is CNN-derived (though spectrally distinct blue ice is a
  relatively easy target and the product was manually validated). Higher-quality manual
  hand-label squares exist for 5 regions if a reference-only variant is ever wanted.
- Background is only sampled adjacent to blue ice (within tiles); the dataset does not include
  far-field pure-background tiles (no continent land/ocean mask was used, and per spec §5 the
  assembly step supplies additional negatives from other datasets).
