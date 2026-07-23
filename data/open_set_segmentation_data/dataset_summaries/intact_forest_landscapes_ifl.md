# Intact Forest Landscapes (IFL)

- **Slug**: `intact_forest_landscapes_ifl`
- **Status**: completed
- **Task type**: classification (binary presence)
- **Samples**: 2000 (1000 intact-forest-landscape tiles + 1000 background-only negatives)

## Source

intactforests.org / GLAD — The IFL Mapping Team (Potapov, Turubanova, Glushkov, et al.;
World Resources Institute / University of Maryland / Greenpeace International). Intact
Forest Landscapes are "forest wildlands": roadless forest landscapes ≥ 500 km² and ≥ 10 km
wide, within the current forest zone, showing no signs of significant human transformation
(no conversion, roads, settlements, or industrial resource extraction). Mapped globally by
manual photointerpretation of Landsat / high-resolution imagery. License **CC-BY-4.0**.

- Data page: https://intactforests.org/data.ifl.html
- Description PDF: https://intactforests.org/shp/IFL_2000-2025.pdf
- Access: direct unsigned HTTP download of per-epoch GeoPackages (no credentials).
- Epochs available: 2000, 2013, 2016, 2020, 2025. **We use the 2020 epoch** (a
  representative Sentinel-era layer; downloaded `IFL_2020.gpkg`, 345 MB).

## Source structure

Single layer `IFL_2020`, 2053 MultiPolygon features in **EPSG:4326**. Fields: `IFL_ID`
(e.g. `SAM_5`; the alphabetic prefix is a region code) and `Area2020` (polygon area in
**hectares**; total ≈ 1.126e9 ha ≈ 11.3 M km², consistent with reported global IFL area).
Polygons are enormous (median ≈ 126 k km², min ≈ 481 k ha). Region prefixes:
SAM (512), NEA (545), NAM (368), SEA (315), AFR (260), AUS (53).

## Label / class mapping

Binary presence, uint8:

| id | name | meaning |
|----|------|---------|
| 0 | background | non-IFL land/water outside an IFL 2020 polygon |
| 1 | intact_forest_landscape | inside an IFL 2020 polygon |
| 255 | nodata | declared for consistency; not emitted |

## Processing

- Each label is a **64×64 (640 m) tile at 10 m/pixel** in the local UTM zone.
- IFL polygons intersecting a tile are reprojected to the tile's UTM pixel grid and
  rasterized as class 1 (`all_touched=True`); everything else is class 0.
- Because IFL polygons are so large, most positive tiles fall entirely inside one polygon
  (951 of 1000 tiles are all-class-1; 49 boundary tiles are mixed 0/1).
- **Sampling (bounded / regionally-diverse — this is a large global derived product, not
  global coverage per spec §5):** positives are area-weighted interior points with an
  **even per-region quota** across the 6 IFL_ID regions (SAM, AFR, NAM, AUS, SEA, NEA),
  167 positives each (shuffled, capped to 1000). Polygon selection is weighted by reported
  IFL hectares; within a chosen multipolygon a part is picked by geometric area, then a
  random interior point is drawn. Near-duplicate centers deduplicated on a ~5 km grid.
- **Negatives:** 1000 background-only tiles offset 30–150 km from a random positive and
  verified IFL-free (no IFL polygon within a padded box). All labeled class 0. (Downstream
  assembly also supplies cross-dataset negatives; these in-scheme negatives give the
  background class spatially meaningful examples.)

## Time range / change handling

IFL reduction between epochs is a **multi-year** process, not a dated event, so per the
task spec presence is treated as a **static** label: `change_time = null`, and every sample
gets a **1-year window anchored on 2020** (`2020-01-01 → 2021-01-01`), inside the
Sentinel era.

## Verification

- 2000 `.tif` + 2000 `.json`; all single-band uint8, 64×64, UTM CRS @ 10 m; pixel values
  ∈ {0, 1}. Global class counts: 1000 tiles contain IFL, 1000 are background-only.
- Every `.json` has a ≤1-year `time_range` and `change_time = null`.
- **Georeferencing check**: reprojecting each tile's center back to WGS84 and testing
  against the source polygons, **1000/1000 positive centers fall inside an IFL polygon and
  1000/1000 negative centers fall outside all IFL polygons** — confirms the UTM/pixel-bounds
  round-trip is exact. (This point-in-polygon test against the source vectors was used in
  place of a Sentinel-2 overlay, which would require the heavier imagery pipeline; the
  vector check directly validates label placement.)

## Caveats

- Only the 2020 epoch is used; other epochs (2016, 2025) are available at the same source
  if a multi-epoch or change formulation is wanted later.
- Positive tiles are overwhelmingly homogeneous (whole-tile IFL) because IFL polygons dwarf
  a 640 m tile; boundary diversity comes from the ~5% mixed tiles plus the negatives.
- Area-weighting uses reported hectares at the polygon level and geometric (deg²) area
  within a polygon; the within-polygon deg² distortion is negligible since a polygon's parts
  share a latitude.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.intact_forest_landscapes_ifl --workers 64
```
Idempotent: existing `locations/{id}.tif` are skipped. Raw GeoPackage + PDF are downloaded
to `raw/intact_forest_landscapes_ifl/` on first run.
