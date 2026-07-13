# USGS ASTER Hydrothermal Alteration Maps

- **Slug:** `usgs_aster_hydrothermal_alteration_maps`
- **Status:** completed
- **Task type:** classification (dense multi-class raster)
- **Num samples:** 4372 label tiles (64x64, 10 m, local UTM)
- **Source:** USGS Open-File Report 2013-1139, "Hydrothermal Alteration Maps of the
  Central and Southern Basin and Range Province of the United States Compiled From ASTER
  Data" (Mars, 2013). https://mrdata.usgs.gov/surficial-mineralogy/ofr-2013-1139/ ,
  doi:10.3133/ofr20131139.
- **License:** CC0 / public domain (USGS). No credentials required.
- **Region:** central & southern Basin and Range, western US (approx. lon -120.4..-107.4,
  lat 30.65..42.4; verified: 500 sampled tile centers all fell inside this bbox).

## What the source is

ASTER VNIR-SWIR reflectance + IDL logical-operator band-ratio algorithms were used to map
surficial minerals diagnostic of hydrothermal alteration (permissive of gold/copper
deposits). It is a **derived-product / automated spectral map**, native ASTER resolution
15-90 m (SWIR ~30 m). The manifest notes alteration minerals are discernible in
Sentinel-2 / Landsat SWIR, so labels were resampled to 10 m UTM.

## Access / format (important deviation from the manifest)

The manifest labels this `dense_raster`, but the product is **not distributed as a
GeoTIFF** — there is no raster layer to download. It is published only as **polygon
shapefiles**, one shapefile per alteration type, plus KMZ and OGC WMS/WMTS services. We
downloaded the five per-type zipped shapefiles (~588 MB total, no auth) to
`raw/usgs_aster_hydrothermal_alteration_maps/` and **rasterized** them. The alteration
type is a property of the whole layer (there is no per-feature type attribute; the only
attribute is `PARTS`). Feature counts: epi_chlor 1.72M, phyllic 1.09M, hydro_silica 0.94M,
argillic 0.93M, carbonate 0.50M polygons (each polygon is essentially a cluster of altered
~30 m pixels; median footprint ~60 m).

## Class scheme (unified, spec §5)

The five source layers were combined into one unified single-band class map:

| id | name | source layer | mineralogy |
|----|------|--------------|------------|
| 0 | argillic | argillic | advanced-argillic: alunite-pyrophyllite-kaolinite |
| 1 | phyllic | phyllic | phyllic/sericitic: sericite-muscovite (illite) |
| 2 | propylitic_epidote_chlorite | epi_chlor | propylitic: epidote-chlorite(-albite) |
| 3 | carbonate | carbonate | calcite-dolomite (propylitic carbonate group) |
| 4 | hydrothermal_silica | hydro_silica | hydrous quartz, chalcedony, opal, amorphous silica |

**Manifest class list mismatch (documented decision):** the manifest lists "advanced
argillic, phyllic, propylitic alteration, clays, carbonates, iron oxides." The actual
OFR 2013-1139 product has exactly the five mineral-group layers above — there is **no
distinct "clays" layer** (clay/kaolinite falls inside the argillic layer) and **no
iron-oxide layer at all**. We therefore use the five real data layers and drop the two
manifest classes that have no corresponding source layer.

## Nodata / background handling

This is a **foreground-only / positive-only** map (spec §5): polygons mark WHERE
alteration was detected. Unaltered / unmapped ground is written as **nodata 255**, not a
fabricated background class. The pretraining-assembly step supplies negatives from other
datasets. Every 64x64 tile therefore contains only real alteration classes + 255.

## Method

1. **Candidate windows:** every polygon centroid across all five layers was snapped to a
   ~640 m lon/lat grid cell (= one 64 px @ 10 m tile footprint; lon cell 0.00715 deg, lat
   cell 0.00575 deg at the ~36.5 deg mid-latitude). Per-cell per-class centroid counts were
   accumulated (767,568 occupied cells). A class counts as **present** in a cell at
   `>= MIN_POLYS = 10` centroids — a homogeneity / high-confidence proxy (>=~10% of the
   cell's native pixels). 126,976 candidate cells resulted; all five classes had well over
   1000 candidate cells (carbonate lowest at ~6100).
2. **Selection:** `select_tiles_per_class` (tiles-per-class balanced, rarest-first),
   `per_class=1000`, 25k total cap. 4372 tiles selected.
3. **Rasterization:** for each selected tile, the actual polygons of every layer
   intersecting the tile were queried via a shapely STRtree (per layer), reprojected from
   WGS84 to the tile's local UTM at 10 m, and burned (exact polygon burn, `all_touched`,
   never bilinear — categorical). Overlapping alteration (co-occurring minerals) is
   resolved **rarest-class-wins**: layers are burned most-common -> rarest (epi_chlor,
   phyllic, hydro_silica, argillic, carbonate) so rare classes survive overlaps.
4. **Write:** patches + sidecar JSON written with a 64-worker pool (idempotent — existing
   `{id}.tif` are skipped on re-run).

## Time range

Static geologic label -> a representative 1-year Sentinel-era window anchored on **2016**
(manifest time_range [2016, 2016]); `time_range = 2016-01-01 .. 2017-01-01`,
`change_time = null` for every sample. (Uses the shared `io.year_range` helper, same as CDL
/ lcmap.)

## Sample counts per class (tiles containing the class; a tile counts toward every class in it)

| class | tiles |
|-------|-------|
| argillic | 1711 |
| phyllic | 1813 |
| propylitic_epidote_chlorite | 1719 |
| carbonate | 1448 |
| hydrothermal_silica | 2133 |

Counts exceed 1000/class because tiles are multi-label (co-occurring alteration), so a
tile selected to fill one class often also contains others. Total 4372 tiles, well under
the 25k cap. No class is sparse.

## Verification (spec §9)

- Opened multiple output `.tif`s: single band, `uint8`, 64x64, local UTM (e.g. EPSG:32610,
  32612, 32613) at 10 m, nodata 255. Values over 300 random tiles = {0,1,2,3,4,255}, all
  valid class ids; metadata.json class ids (0-4) cover every pixel value present.
- All 4372 `.tif` have a matching `.json`; `time_range` span = 1 calendar year,
  `change_time = null`.
- 500 sampled tile centers all fall inside the source's stated Basin-and-Range bounding
  box (geolocation sanity). A per-tile spatial cross-check (STRtree polygon query) confirms
  each burned tile overlaps real source polygons; a dense phyllic cell rasterized to ~57%
  coverage. Note: an RGB Sentinel-2 overlay is not a meaningful visual check here — surface
  hydrothermal alteration is a subtle SWIR spectral signal, not visible in true color — so
  placement/coverage checks were used instead.
- Re-running is idempotent (existing outputs skipped).

## Caveats

- Derived spectral product (automated ASTER band-ratio), not in-situ reference; the
  MIN_POLYS=10 threshold biases tiles toward homogeneous / high-confidence alteration.
- Overlapping mineral groups are collapsed to a single label per pixel (rarest-wins); a
  pixel mapped as multiple alteration types keeps only the rarest class.
- Manifest classes "clays" and "iron oxides" are not represented (no corresponding source
  layer; see class-scheme note).
- Native ASTER 30 m upsampled to 10 m (nearest/exact polygon burn); do not over-interpret
  sub-30 m detail.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_aster_hydrothermal_alteration_maps
```
(Downloads the five per-type shapefile zips to
`raw/usgs_aster_hydrothermal_alteration_maps/` if not already present, then regenerates all
outputs. Script: `olmoearth_pretrain/open_set_segmentation_data/datasets/usgs_aster_hydrothermal_alteration_maps.py`.)
