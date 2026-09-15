# GMIE Central Pivot Irrigation

- **Slug**: `gmie_central_pivot_irrigation`
- **Source**: GMIE / GCPIS, Tian et al. (ESSD), Harvard Dataverse `doi:10.7910/DVN/HKBAQQ`
- **License**: CC0 1.0 (Dataverse-declared; manifest listed CC-BY-4.0, actual is CC0)
- **Task**: classification (dense per-pixel, 3 classes)
- **Samples**: 3000 (1000 per class, bounded-tile sampling)

## Source

Two derived products, both global, WGS84 (EPSG:4326):

1. **GMIE-100** — 67 GeoTIFF tiles of *maximum irrigation extent* at ~100 m. Single band;
   pixel value = irrigation proportion in [0, 1]; background = -99. Produced from dry
   months over 2017-2019 (regularly irrigated regions) and driest months 2010-2019
   (occasionally irrigated). Total ~7 GB.
2. **GCPIS** — `GCPIS.shp`, 179,942 polygons of machine-detected central-pivot irrigation
   systems (the distinctive irrigated circles). Property `area` (km^2), `value`=1.

Accessed unauthenticated via the Harvard Dataverse file-access REST API
(`https://dataverse.harvard.edu/api/access/datafile/{id}`). No credentials needed.

Raw files: `raw/gmie_central_pivot_irrigation/` (67 `GMIE-100_*.tif`, `GCPIS.zip` +
extracted shapefile, `GMIE-100_Description.docx`; see `SOURCE.txt`).

## Classes and label construction

This is a global derived-product *map*, so we used **bounded-tile sampling** (≤1000
tiles/class, no global coverage) and preferred spatially-homogeneous windows. Every label
patch is a **64×64 uint8** tile in **local UTM at 10 m**; GMIE (EPSG:4326 ~100 m) was
reprojected into the tile grid with **nearest** resampling (categorical/proportion label,
never bilinear). Unified 3-class segmentation:

| id | name | definition |
|----|------|------------|
| 0 | central pivot irrigation system | inside a GCPIS pivot polygon (overlaid on every tile; wins where present) |
| 1 | irrigated cropland | GMIE irrigation proportion ≥ 0.5 |
| 2 | non-irrigated | GMIE proportion ≤ 0.05 (observed land, not irrigated) |
| 255 | nodata/ignore | GMIE background (-99) or ambiguous mid proportion (0.05–0.5) |

The ambiguous 0.05–0.5 band is set to nodata to keep windows high-confidence/homogeneous.

## Sampling

- **central pivot (1000)**: centre a tile on each of a geographically-stratified set of
  GCPIS polygons (round-robin over 1° cells for global spread). Guarantees class 0.
- **irrigated cropland (1000)**: homogeneous GMIE windows — a coarse ~640 m cell (7×7
  GMIE px) whose *minimum* proportion ≥ 0.5.
- **non-irrigated (1000)**: homogeneous GMIE windows — a coarse ~640 m cell fully within
  [0, 0.05] and free of background.

GCPIS polygons intersecting a selected tile are overlaid on all tiles, so an irrigated or
non-irrigated window that contains a pivot still labels that pivot as class 0. Tiles are
therefore multi-class where classes co-occur; per-class *tile presence* over the 3000
samples: class 0 in 1062 tiles, class 1 in 1443, class 2 in 1602. `class_counts_primary`
in `metadata.json` records the class each tile was sampled *for* (1000/1000/1000).

## Time range

GMIE is a seasonal/annual product spanning the 2017–2019 production period. Each sample is
assigned a 1-year window uniformly chosen from {2017, 2018, 2019} (`time_range` in the
sample JSON; no change labels).

## Caveats

- GMIE and GCPIS are independent products: a pivot circle's surroundings inherit GMIE's
  proportion, which is sometimes low (class 2) even adjacent to a detected pivot.
- GMIE is a derived map (not in-situ reference) with manual VHR validation; proportion
  thresholds (0.5 / 0.05) were chosen for homogeneity, not calibrated to a legend.
- Coarse-cell homogeneity is enforced at ~640 m (7 GMIE px); after nearest reprojection to
  10 m a homogeneous cell yields a uniform class-1/2 tile.
- Georeferencing verified: pivot-tile centres round-trip to within ~5 m of the source
  GCPIS polygon centroid. Sentinel-2 imagery overlay was not performed; pivot circles are
  visually distinctive at 10 m per the source description.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gmie_central_pivot_irrigation
```

Downloads (idempotent) the 67 GMIE tiles + GCPIS shapefile to
`raw/gmie_central_pivot_irrigation/`, then writes `metadata.json` and
`locations/{id}.tif`+`.json`. Re-running skips already-written tiles.
