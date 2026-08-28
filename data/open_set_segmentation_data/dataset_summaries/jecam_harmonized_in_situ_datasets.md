# JECAM Harmonized In-Situ Datasets

- **Slug**: `jecam_harmonized_in_situ_datasets`
- **Status**: completed
- **Task type**: classification (per-pixel crop type + land cover)
- **Family / label_type**: crop_type / polygons
- **License**: CC-BY-4.0
- **Samples written**: 11,233 label patches across 86 classes

## Source

"Harmonized in situ JECAM datasets for agricultural land use mapping and monitoring in
tropical countries" (Jolivot et al. 2021, *Earth Syst. Sci. Data* 13, 5951–5967,
https://doi.org/10.5194/essd-13-5951-2021). Quality-controlled, field-scale land-use /
land-cover **polygons** collected by local experts under the GEOGLAM/JECAM initiative in
seven tropical/subtropical countries (Burkina Faso – Koumbia; Madagascar – Antsirabe;
Brazil – São Paulo & Tocantins; Senegal – several sites; Kenya – Muranga; Cambodia –
Kandal; South Africa – Mpumalanga), field-surveyed yearly/seasonally 2013–2022. 31,879
records total (24,287 cropland + 7,592 non-crop). Each record carries a precise field
polygon (WGS84), an acquisition date (`AcquiDate`), a broad `LandCover` class and, for
cropland, up to three `CropType` attributes plus season (`SOS`/`EOS`), irrigation,
intercrop and area attributes.

## Access method

The original CIRAD Dataverse DOI (`doi:10.18167/DVN1/P7OLAP`) is **DEACCESSIONED**
("dataset transferred to another repository"), so its file API returns no versions. The
current authoritative copy is on the CIRAD GeoNetwork/GeoServer at `geode.cirad.fr`
(GeoNetwork record `6855571d-677a-4852-afa8-7d7084ed2de8`), published as WFS layer
`TETIS:BD_JECAM_CIRAD_2023`. Downloaded label-only (no imagery) via one WFS GetFeature
call to a single GeoJSON:

```
https://geode.cirad.fr/geoserver/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=TETIS:BD_JECAM_CIRAD_2023&outputFormat=application/json&srsName=EPSG:4326
```

Saved to `raw/jecam_harmonized_in_situ_datasets/BD_JECAM_CIRAD_2023.geojson` (~27 MB,
31,879 MultiPolygon features). No credentials required (open CC-BY). Georeferencing
confirmed: features are precise WGS84 MultiPolygons; tile centers reproduce source
centroids to <15 m and land in the correct source countries (sanity-checked).

## Class mapping

Unified single class scheme (no separate targets):
- `LandCover == "Cropland"` → the field's **`CropType1`** value (the crop), e.g. Maize,
  Rice, Groundnut, Millet, Soybean, Cotton, Sugarcane, Sorghum, Cassava, Cowpea, Fallow,
  Eucalyptus/Pine (forest plantations), market-garden vegetables, orchard/tree crops, …
- non-cropland → the **`LandCover`** value: Built-up surface, Pasture, Bare soil,
  Herbaceous savannah, Forest, Water body, Savannah with shrubs/trees, Shrub land, Natural
  vegetation, Wetland, Mineral soil, Grassland.

86 distinct classes appear in the post-2016 subset (73 crop types + 13 non-crop
land-cover). This is comfortably under the 254 uint8 cap, so **no classes were dropped**.
Class ids are assigned 0..85 in **descending frequency** (id 0 = Maize, 1 = Rice,
2 = Groundnut, …). `metadata.json` carries the full `id→name` map and per-class counts.
Only labeled fields have ground truth, so there is **no background class**: pixels outside
the field polygon are nodata/ignore (255). Rare classes (e.g. Beet, Zucchini, Vineyard,
Coffee, single-sample crops) are kept — downstream assembly filters too-small classes.

## GeoTIFF spec

Single-band **uint8**, local UTM, 10 m/pixel, north-up. Each field polygon is rasterized
(`all_touched=True`) into a ≤64×64 tile sized to its footprint and centered on its
centroid: the class id is burned inside the polygon, 255 (nodata/ignore) elsewhere. Median
field ≈6 px on a side; ~4.4% of fields exceed 64 px and are cropped to the central 64×64
window. `nodata_value = 255`.

## Time range

1-year window `[Jan 1 year, Jan 1 year+1)` anchored on each record's **acquisition year**
(labeled growing season). `change_time = null` (not a change dataset).

## Post-2016 filtering

Records span 2013–2022. Only acquisition year **≥ 2016** are kept (Sentinel era). The
pre-2016 subset (2013: 748, 2014: 1,763, 2015: 5,515 → 8,025 records, plus 1 null
geometry) was filtered out, leaving 23,853 post-2016 labeled records as candidates.

## Sampling

Tiles-per-class balanced with the 25k per-dataset cap. With 86 classes the effective
per-class limit is `min(1000, 25000 // 86) = 290`. 20 classes hit the 290 cap; all other
classes contribute their full post-2016 count. Total selected = 11,240 parcels; 11,233
tiles written (7 tiny polygons rasterized empty and were skipped).

Per-class counts (top): Maize/Groundnut/Millet/Soybean/Pasture/Cotton/Sugarcane/Sorghum/
Eucalyptus/Fallow/Eggplant/Herbaceous savannah/Cowpea/Pine/Forest/Savannah with shrubs/
Potato/Young fallow/Weakly vegetated agricultural/Fruit crop = 290 each; Rice/Built-up
surface/Bare soil/Carrot/Water body = 289; tapering to single-sample classes (Beet,
Zucchini, Root/tuber crop …). Full counts in `metadata.json:class_counts`.

## Caveats

- Fields are small (surveyed for homogeneous ≥20×20 m entities); most tiles are only a few
  pixels of labeled crop surrounded by ignore.
- `CropType1` is a mix of granularity (specific crops like Maize/Rice vs generic "Annual
  crop", "Cereals", "Vegetables", "Market gardening"); kept as-is to preserve the source
  taxonomy. Fallow variants (Fallow / Young fallow / Mid fallow / Old fallow) kept
  separate.
- 4.4% of fields larger than 640 m are center-cropped to 64×64 (sub-window sample).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jecam_harmonized_in_situ_datasets
```

Idempotent: skips any already-written `locations/{id}.tif`. Outputs under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/jecam_harmonized_in_situ_datasets/`
(`metadata.json`, `locations/{id}.tif` + `.json`, `registry_entry.json`).
