# Ethiopian Crop Type 2020 (EthCT2020)

- **Slug:** `ethiopian_crop_type_2020_ethct2020`
- **Status:** completed — classification (points) — 1,716 samples
- **Source:** CIMMYT / Data in Brief (Kerner et al. 2024, *Data in Brief* 54:110427).
  Data on Mendeley Data record `mfpvmk8cnm` v1, https://data.mendeley.com/datasets/mfpvmk8cnm/1
- **License:** CC-BY-4.0
- **Region / time:** Ethiopia (nationwide highlands), Meher (main) season 2020/21.

## Source data
Publicly downloadable ESRI shapefile (`EthCT2020.{shp,shx,dbf,prj}`) pulled unauthenticated
via the Mendeley public-files API to
`raw/ethiopian_crop_type_2020_ethct2020/`. 2,793 quality-controlled, georeferenced in-situ
crop-type samples at smallholder field level, all `annual cropland`. CRS EPSG:32637
(UTM 37N). Geometries are small (~20 m) circular field plots buffered around field
centroids. Sources: GDCC ground campaign (1,263), FHSD farm-household survey (796), WRTB
Wheat Rust Toolbox (734). Hierarchical taxonomy: 7 crop groups, 22 crop classes.

**Coordinate gotcha:** the shapefile's `lat`/`long` attribute columns actually hold UTM
northing/easting, not WGS84 degrees. We ignore them and reproject each geometry centroid
UTM37N -> WGS84 to obtain lon/lat.

## Processing
`label_type = points` -> sparse point classification -> one dataset-wide `points.json`
(spec §2a), not per-point GeoTIFFs. Each field plot becomes one point at its centroid
lon/lat with the crop-class id. Crop type is a seasonal/annual label -> 1-year time range
`[2020-01-01, 2021-01-01)` anchored on the Meher 2020/21 growing season, on every point.

Class ids assigned 0–21 by descending frequency (all 22 classes kept; well under the
254-class uint8 cap). `metadata.json` `description` records the source crop group per class.
Balanced with `balance_by_class(per_class=1000, total_cap=25000)`: only **wheat** is capped
(2,077 raw -> 1,000); all other classes are kept in full. Selected total 1,716.

## Class counts (selected)
wheat 1000, teff 255, barley 102, maize 96, broad/faba beans 50, triticale 46, other
oilseed crops 35, sugar cane 24, millets 21, potatoes 18, spice crops 16, other leguminous
crops 14, root/bulb/tuberous vegetables 9, sweet potatoes 7, peas 6, sorghum 6, chick peas
4, oats 3, fruit-bearing vegetables 1, groundnuts 1, lentils 1, leafy/stem vegetables 1.

(Several tail classes have only 1–6 samples — inherent to the source distribution, a
wheat-focused survey.)

## Verification
- `points.json`: 1,716 points, task_type classification, labels 0–21 (all 22 present),
  every time range ≤ 1 year, all lon/lat inside the Ethiopia bbox (36.2–40.7°E, 5.9–12.0°N).
- `metadata.json`: 22 classes, nodata 255; class ids cover all point labels.
- Point-only dataset, so no per-sample GeoTIFFs (spec §2/§2a). Coordinate range confirms
  points fall on Ethiopian highland cropland; no per-point S2 overlay done (1×1 point labels).

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ethiopian_crop_type_2020_ethct2020
```
Idempotent: re-running re-reads the raw shapefile and overwrites `points.json`/`metadata.json`.
