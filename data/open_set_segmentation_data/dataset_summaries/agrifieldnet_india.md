# AgriFieldNet India

- **slug**: `agrifieldnet_india`
- **status**: completed — **classification** (per-pixel crop type)
- **num_samples**: 3924 label patches (13 classes)
- **source**: Source Cooperative `radiantearth/agrifieldnet-competition` (STAC id
  `ref_agrifieldnet_competition_v1`), the AgriFieldNet India Competition training data
  (Radiant Earth Foundation & IDinsight, 2022; DOI 10.34911/rdnt.wu92p1). License
  **CC-BY-4.0**.
- **region / time**: Northern India (Uttar Pradesh, Rajasthan, Odisha, Bihar), 2021-22
  rabi (winter) crop season; anchored on a 1-year 2022 window.

## Source & access
Public, no credentials. AgriFieldNet was originally distributed via Radiant MLHub (now
retired); an **open, fully-georeferenced mirror** lives on Source Cooperative and was read
via the unsigned S3 proxy `https://data.source.coop` (bucket `radiantearth`, prefix
`agrifieldnet-competition/`). We pulled only the **train label rasters** — for each of the
1165 train chips: `..._labels_train_{chip}.tif` (crop-code raster) and
`..._labels_train_{chip}_field_ids.tif` (field-id raster). The Sentinel-2 imagery
(`source/`) was **not** downloaded — pretraining supplies its own imagery. Raw label tifs
land in `raw/agrifieldnet_india/train_labels/`.

Unlike the CV4A mirror, **this mirror is properly georeferenced**: each 256×256 chip is a
native 10 m UTM COG with an embedded CRS and transform, so no georeferencing recovery was
needed. Chips span UTM zones 43N/44N/45N (EPSG:32643/44/45); each output tile uses its
chip's native CRS. Sample centroids were confirmed to fall in northern India
(lon ≈ 76–82°E, lat ≈ 24–28°N).

## Classes
From **Documentation.pdf p.2** (authoritative legend). Crop codes are **non-contiguous**
(1,2,3,4,5,6,8,9,13,14,15,16,36); mapped to class ids 0–12 by ascending code. Label pixel
value 0 = unlabeled (no surveyed field) → nodata/ignore (255). Note "No crop/Fallow"
(code 4) is a **real labeled class**, not background. (The manifest's `classes` list has
the right 13 crop names but no code mapping; the codebook here is authoritative.)

| id | code | name | labeled fields | samples written |
|----|------|------|----------------|-----------------|
| 0 | 1 | Wheat | 2148 | 1000 (capped) |
| 1 | 2 | Mustard | 1041 | 1000 (capped) |
| 2 | 3 | Lentil | 105 | 105 |
| 3 | 4 | No crop/Fallow | 1707 | 1000 (capped) |
| 4 | 5 | Green pea | 25 | 25 |
| 5 | 6 | Sugarcane | 173 | 173 |
| 6 | 8 | Garlic | 49 | 49 |
| 7 | 9 | Maize | 304 | 304 |
| 8 | 13 | Gram | 64 | 64 |
| 9 | 14 | Coriander | 14 | 14 |
| 10 | 15 | Potato | 43 | 43 |
| 11 | 16 | Bersem (berseem) | 16 | 16 |
| 12 | 36 | Rice | 131 | 131 |

All 13 classes kept (well under the 254 uint8 cap). 5820 labeled fields total; Wheat,
Mustard, and No crop/Fallow are truncated to 1000 by balancing. Several classes are sparse
(Coriander 14, Bersem 16, Green pea 25) — kept per spec §5 (downstream assembly handles
rare-class filtering).

## Label encoding
Per-field label patches (EuroCrops/CV4A-style), one per **labeled train field**. Each
field's pixel bounding box within its chip defines a `≤64×64` UTM 10 m window centered on
the field; the crop class id is burned at **every labeled pixel** in that window
(neighboring labeled fields included), and **255 (nodata/ignore)** fills every pixel where
the field-id raster is 0 (unsurveyed land) — ground truth exists only inside surveyed
fields, so unlabeled land is ignore, not a background class (no synthetic negatives; spec
§5 positive-only handling). Fields are small smallholder plots, so output tiles are small
(typically ~7–15 px on a side; hard-capped 64). A field's crop id for balancing is the
majority crop code over its labeled pixels.

- **Test chips** carry only `field_ids` (crop labels withheld in the competition release)
  and are **excluded** — only the 1165 train chips have crop codes.
- **Time range**: 1-year window on 2022 (`[2022-01-01, 2023-01-01)`), the labeled rabi
  season.
- **Sampling**: tiles-per-class balanced, up to 1000 fields/class, 25k cap
  (`balance_by_class`). Per-class limit stays 1000 (13 classes → 25000//13 = 1923 > 1000).

## Judgment calls / caveats
- Manifest `label_type` is `polygons`, but the mirror ships per-pixel crop-code **rasters**
  aligned to the S2 grid; processed as dense per-field label rasters (the per-field
  approach is equivalent and yields cleanly-centered field tiles).
- A few fields span multiple chips; such a field yields one partial tile per chip it
  touches (footprints are small, so overlap/duplication is negligible).
- Used the Documentation.pdf codebook (non-contiguous crop codes) rather than assuming
  contiguous ids.
- Georeferencing is native/authoritative (embedded COG CRS), validated by centroid
  location in northern India; no reconstruction needed.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.agrifieldnet_india
```
Idempotent (skips already-written `locations/{id}.tif`). Public unsigned S3 read from
`https://data.source.coop/radiantearth/agrifieldnet-competition/` (no credentials).
