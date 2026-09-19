# Global Debris-Covered Glaciers (Herreid & Pellicciotti)

- **Slug:** `global_debris_covered_glaciers_herreid_pellicciotti`
- **Task type:** classification (polygon → per-pixel mask)
- **Source:** Zenodo record [3866466](https://zenodo.org/records/3866466) — "Supplementary
  Information for Herreid and Pellicciotti, *Nature Geoscience*, 2020".
- **License:** CC-BY-4.0
- **Family / region:** glacier / Global (all 18 RGI first-order regions except Antarctica)
- **Samples written:** 3000 (1000 per class)

## Source

One Zenodo file, `SupplementaryInformation.zip` (1.37 GB), containing three nested zips:
- `S1.zip` — per-RGI-region shapefiles (the vector product used here).
- `S2.zip` — glacier footprints (not used).
- `S3.zip` — Scherler-2018 comparison layers (not used).

`S1/` has one folder per RGI first-order region (`01Alaska` … `18NewZealand`; RGI region
19 Antarctica is absent). The relevant polygon layers per region:
- `{region}_minGl1km2.shp` — glacier outlines ≥ 1 km² (RGI-derived; ice extent, with
  attributes RGIId, Zmin/Zmax, totDeb_km2, perc_deb, …).
- `{region}_minGl1km2_debrisCover.shp` — supraglacial debris-cover polygons (attribute
  `img_time` = decimal source-imagery year).
- `{region}_ablationZone.shp` — ablation-zone polygons.
- `{region}_debrisExpansionLine.shp`, `{region}_equilibriumLine.shp` — line layers, **not
  used**. `minGl2km2*` — coarser (≥2 km²) subset of the same glaciers, **not used**.

Source CRSs are regional equal-area projections (NAD83 Alaska Albers, North Pole LAEA,
Asia/Europe/South-America Albers, UTM 59S for New Zealand); each is reprojected per-polygon
to local UTM 10 m.

## Class mapping

Three classes emitted as **single-class 64×64 UTM 10 m positive masks** (class ID inside
the polygon, 255 = nodata everywhere else):

| id | name | source layer | notes |
|----|------|--------------|-------|
| 0 | debris-covered area | `minGl1km2_debrisCover` | supraglacial debris polygons |
| 1 | clean ice | `minGl1km2` | glacier outline with debris polygons **subtracted** (burned to nodata) so only debris-free ice = 1 |
| 2 | ablation zone | `ablationZone` | ablation-zone polygons |

Rationale: debris-covered area and clean ice are the two mutually-exclusive surface-cover
classes (debris ⊂ glacier outline), so clean ice is defined as glacier-minus-debris. The
ablation zone is an *elevation-based* partition orthogonal to surface cover, so it is
emitted as its own mask rather than composited with the other two (which would create
ambiguous overlaps). Each tile is therefore a clean positive mask for exactly one class,
which also makes the ≤1000-tiles-per-class balance exact.

## Tiling & sampling

- Each selected polygon → a 64×64 (640 m) window centered on a **representative interior
  point** of the polygon, in the local UTM zone at 10 m. Large glaciers exceed the window
  → homogeneous interior tiles; small polygons show their shape against nodata.
- `all_touched=True` so thin/small polygons register at least one pixel.
- Clean-ice windows: glacier burned as 1, then any debris polygons intersecting the window
  burned as 255. Windows that end up fully debris-covered (no valid ice pixel) are skipped
  by an `np.any(arr == class_id)` guard (none occurred in this run).
- **Round-robin across all 18 regions** for geographic diversity, up to 1000 tiles/class
  (≈56/region; ablation backfilled from larger regions where a region has < 56 ablation
  polygons). 39 distinct UTM zones appear in the output.

Candidate polygon counts: debris 175 223, clean ice 42 134, ablation 6179.

Per-region tile counts (deb / ice / abl): each region contributes 55–56 debris and
55–56 clean-ice tiles; ablation is 23–64 per region (regions with few ablation polygons —
Scandinavia 23, NorthAsia/LowLatitudes 33, NewZealand 31 — are backfilled by others).

## Time range

Supraglacial debris and glacier outlines are slowly-changing features; the manifest anchors
this product at 2016–2017 and per-feature imagery years (`img_time`) span ~1986–2016. A
uniform **1-year window (2016-01-01 → 2017-01-01)** in the Sentinel era is assigned to every
sample. The per-feature source-imagery year is preserved in the sample `source_id`
(e.g. `05Greenland/minGl1km2_debrisCover/10392@img_time=1998.65`). `change_time` is null.

## GeoTIFF spec

Single band, uint8, local UTM, 10 m/pixel, ≤64×64, nodata = 255. Verified on random samples:
correct band count/dtype/shape, UTM CRS (EPSG:326xx/327xx), 10 m resolution, values only in
{0, 1, 2, 255}, and every `.tif` has a matching `.json` with a 1-year `time_range` and
`classes_present`.

## Caveats

- Clean ice and debris come from glaciers ≥ 1 km²; smaller glaciers are excluded by the
  source product.
- `img_time` varies widely by region/feature (some pre-Sentinel); labels are treated as
  persistent glacier features rather than tied to a specific acquisition, so all tiles use
  the same 2016 window. Debris extent can shift over decades, so a small fraction of tiles
  may not exactly match 2016 imagery.
- A full Sentinel-2 overlay eyeball check was not performed; georeferencing was validated
  via CRS/resolution/pixel-bounds checks and the debris-subtraction test, and coordinates
  derive directly from the authoritative RGI-based source polygons.
- Single-class masks (positive class + nodata background) rather than multi-class composite
  tiles, to avoid the surface-cover vs. ablation-zone overlap ambiguity.

## Reproduce

```
# 1. Download + unzip (idempotent; download_zenodo skips existing)
python3 -c "from olmoearth_pretrain.open_set_segmentation_data import io, download; \
  raw=io.raw_dir('global_debris_covered_glaciers_herreid_pellicciotti'); raw.mkdir(parents=True,exist_ok=True); \
  download.download_zenodo('3866466', raw)"
cd <raw>/global_debris_covered_glaciers_herreid_pellicciotti && \
  unzip -o SupplementaryInformation.zip && \
  unzip -o SupplementaryInformation/S1.zip -d SupplementaryInformation/S1_extracted

# 2. Build label patches (idempotent; skips existing .tif)
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_debris_covered_glaciers_herreid_pellicciotti --workers 64
```

Outputs on weka under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/`:
`raw/global_debris_covered_glaciers_herreid_pellicciotti/` (source + `SOURCE.txt`) and
`datasets/global_debris_covered_glaciers_herreid_pellicciotti/` (`metadata.json`,
`locations/{id}.tif` + `.json`).
