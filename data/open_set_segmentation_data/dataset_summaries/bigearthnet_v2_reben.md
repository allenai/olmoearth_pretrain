# BigEarthNet v2 (reBEN) — COMPLETED (classification, dense_raster)

- **Slug**: `bigearthnet_v2_reben`
- **Name**: BigEarthNet v2 (reBEN)
- **Source**: Zenodo record 10891137 (https://zenodo.org/records/10891137);
  Clasen et al. 2024, "reBEN: Refined BigEarthNet Dataset for Remote Sensing"
- **License**: CDLA-Permissive-1.0 (public, no credentials)
- **Family / region**: land_cover / Europe (Finland, Portugal, Serbia, Lithuania,
  Ireland, Austria, Belgium, Switzerland, Luxembourg, Kosovo)
- **Label type**: dense_raster (per-pixel CORINE Land Cover 2018 reference maps)
- **Task type**: **classification** (19-class BigEarthNet-19 nomenclature)
- **Status**: **completed**
- **num_samples**: **6795** label tiles (64×64 @ 10 m)

## What reBEN is

BigEarthNet v2 ("reBEN") is a large multi-modal Sentinel-1 + Sentinel-2 patch archive
(549,488 patches; 480,038 in the cleaned recommended set) over 10 European countries,
imagery 2017–2018, annotated from CORINE Land Cover 2018. Alongside the S1/S2 imagery,
each patch ships a per-pixel **reference map** GeoTIFF carrying the underlying CORINE
CLC Level-3 codes.

## Access / download (bounded)

The archive is ~120 GB across two patch tarballs (`BigEarthNet-S1.tar.zst` 54 GB,
`BigEarthNet-S2.tar.zst` 63 GB). We downloaded **only** the label component:
- `Reference_Maps.tar.zst` — **0.28 GB** (549,488 per-patch `*_reference_map.tif`)
- `metadata.parquet` — 4 MB (patch_id, 19-class labels, split, country, S1/S2 names)

No imagery archive was fetched. Disk was checked before/after download and around the
write phase (`io.check_disk()`, ≥5 TB required; ~33 TB free throughout).

## Georeferencing (verified)

Reference maps are real GeoTIFFs: 120×120, uint16, **local UTM** CRS (e.g. EPSG:32633,
32635, 32629), **10 m/pixel**, with a genuine geotransform. A round-trip check confirmed
a written tile's pixel values, CRS, and geographic bounds match the source raster crop
exactly. No coordinate-free arrays — the dataset is accepted.

## Class mapping (CORINE CLC Level-3 → BigEarthNet-19)

Source pixels are CLC Level-3 codes. We remap to the official 19-class BigEarthNet
nomenclature (ids 0–18; matches the `labels` field in metadata.parquet and the manifest
"19/43 CORINE"):

| id | class | CLC codes |
|----|-------|-----------|
| 0 | Urban fabric | 111, 112 |
| 1 | Industrial or commercial units | 121 |
| 2 | Arable land | 211, 212, 213 |
| 3 | Permanent crops | 221, 222, 223, 241 |
| 4 | Pastures | 231 |
| 5 | Complex cultivation patterns | 242 |
| 6 | Land principally occupied by agriculture, w/ natural vegetation | 243 |
| 7 | Agro-forestry areas | 244 |
| 8 | Broad-leaved forest | 311 |
| 9 | Coniferous forest | 312 |
| 10 | Mixed forest | 313 |
| 11 | Natural grassland and sparsely vegetated areas | 321, 333 |
| 12 | Moors, heathland and sclerophyllous vegetation | 322, 323 |
| 13 | Transitional woodland, shrub | 324 |
| 14 | Beaches, dunes, sands | 331 |
| 15 | Inland wetlands | 411, 412 |
| 16 | Coastal wetlands | 421, 422, 423 |
| 17 | Inland waters | 511, 512 |
| 18 | Marine waters | 521, 522, 523 |

CLC codes outside the 19-class scheme — roads/rail (122), airports (124), mineral
extraction (131), dumps (132), construction (133), green urban (141), sport/leisure
(142), bare rocks (332), burnt areas (334), glaciers (335), and nodata (999) — are set to
**255 (nodata/ignore)**, exactly as the original BigEarthNet dropped those classes. These
account for ~0.2% of pixels in a sampled scan.

## Sampling (tiles-per-class balanced)

Dense multi-class rasters use tiles-per-class balanced sampling. Patches are indexed by
`metadata.parquet` (per-patch 19-class multi-labels), so balancing needs no full raster
scan. Selection is **greedy rarest-class-first**: for each class (ascending availability)
we add unselected patches containing it until it reaches `per_class`, incrementing every
class present in the patch. `per_class = min(1000, 25000//19) = 1000`; 25k total cap.

Because patches are multi-label and classes co-occur, all 19 classes reach ≥1000 label
coverage from just **6795 unique patches** (well under the 25k cap). For each selected
patch, the reference map is remapped to BEN-19 ids and the **64×64 window** (chosen from a
3×3 offset grid over the 120×120 patch) that best covers the patch's target class is
written — so each tile actually contains the class it was selected for. No reprojection is
needed (source is already local UTM at 10 m). `classes_present` is recorded from the
written crop.

**Per-class tile counts (tiles actually containing each class in its 64×64 crop):**

| class | tiles | class | tiles |
|-------|------:|-------|------:|
| Urban fabric | 766 | Mixed forest | 1082 |
| Industrial or commercial units | 971 | Natural grassland & sparsely veg. | 1004 |
| Arable land | 1725 | Moors, heathland & sclerophyllous | 972 |
| Permanent crops | 916 | Transitional woodland, shrub | 1378 |
| Pastures | 830 | Beaches, dunes, sands | 1012 |
| Complex cultivation patterns | 805 | Inland wetlands | 960 |
| Land princ. agriculture w/ nat. veg. | 844 | Coastal wetlands | 1016 |
| Agro-forestry areas | 902 | Inland waters | 824 |
| Broad-leaved forest | 1313 | Marine waters | 1186 |
| Coniferous forest | 1228 | | |

## Time range

Land cover is annual; `time_range` is the 1-year calendar window of the S2 acquisition
year (2017 or 2018) parsed from the patch id — matching the paired imagery pretraining
will co-locate. `change_time` is null. All source splits (train/val/test) are used as
pretraining labels (no split filtering).

## Caveats

- Labels are a **derived product** (CORINE 2018), not in-situ reference. CORINE's minimum
  mapping unit is 25 ha and it is a 2018 vintage, so fine features and 2017/2019 changes
  can be mislabeled at 10 m.
- The 64×64 crop covers ~28% of a 120×120 patch's area; the best-window heuristic favors
  the target class, so some non-target classes present in the full patch are under-counted
  relative to the metadata multi-labels (expected; actual per-crop counts reported above).
- Urban fabric is often small/fragmented within patches, so its tile count (766) is the
  lowest despite ample availability.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.bigearthnet_v2_reben
```

Idempotent: existing `locations/{id}.tif` are skipped. Requires
`raw/bigearthnet_v2_reben/{Reference_Maps/, metadata.parquet}` on weka (re-download via
the two Zenodo files above if absent).
