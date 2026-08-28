# Sargassum Detection & Fractional Cover ML Dataset

- **Slug:** `sargassum_detection_fractional_cover_ml_dataset`
- **Status:** REJECTED (no recoverable geocoordinates for the label data; the only
  georeferenced content is demonstration model-output rasters, not a reference dataset)
- **Source:** Zenodo record 17246345 — "Sargassum Fractional Cover Estimation Models"
  (v1.1), Echevarría-Rubio, Martínez-Flores & Morales-Pérez (2025). DOI
  https://doi.org/10.5281/zenodo.17246345
- **License:** MIT (code/models)
- **Family:** kelp / floating macroalgae. **Region:** Tropical/Central W. Atlantic, Caribbean.

## What the release actually contains

The Zenodo release is a **model suite** (single 1.19 GB zip
`sargassum_detection_models.zip`), inspected cheaply via HTTP range requests
(`remotezip`) without downloading the full archive per SOP §8.2. Contents:

1. **`sargassum_data.csv`** — the "labeled spectra" (the actual training labels).
   196,037 rows, columns = `Blue, Green, Red, NIR, SWIR1, class`
   (11,597 `sargassum`, 184,440 `no_sargassum`). **No lon/lat, no pixel/tile index, no
   scene id — coordinate-free surface-reflectance spectra.** These cannot be placed on
   the Sentinel-2 grid, so they are unusable as open-set-segmentation labels (§8.2 fast
   reject for coordinate-free spectra).
2. **Trained models** (`output/models_classification/*.joblib|.keras`), model cards,
   training/classification scripts and notebooks. Not label data.
3. **`output/fractional_cover_maps/*.tif`** — **2 example rasters** (the only
   georeferenced content):
   - `..._Sentinel2_S2B_MSIL2A_20180827T160059_..._T16QGH_..._10m_xgboost_classifier.tif`
     — EPSG:32616, 10 m, 10980×10980, float32, nodata −9999, values 0–1.
   - `..._Landsat8_LC08_L1GT_016046_20150723_..._30m_xgboost_classifier.tif`
     — EPSG:32617, 30 m, 7681×7841, float32, nodata −9999, values 0–1.
4. **`satellite_data/`** — the 2 raw input scenes (1 Sentinel-2 SAFE, 1 Landsat-8) that
   the example rasters were produced from.

## Why rejected

- The **primary label product is coordinate-free spectra** (item 1). No geocoordinates
  are recoverable, so the bulk of the dataset cannot be georeferenced onto the S2 grid.
- The **only georeferenced label content is 2 example fractional-cover rasters** (item 3),
  and these are:
  - **The repo's own model predictions** (xgboost classifier probability output run over
    the 2 bundled example scenes for the tutorial), i.e. self-generated pseudo-labels — the
    weakest form of derived-product map, with no in-situ/reference basis. The design brief
    prefers manual/in-situ reference data and uses derived maps only as a fallback.
  - **Only 2 satellite acquisitions** = essentially 2 locations, with no spatial diversity.
    One of the two scenes is **Landsat-8 2015-07-23 (pre-2016)**, which the Sentinel-era
    rule would filter out anyway, leaving effectively a **single** Sentinel-2 scene
    (2018-08-27).
  - **Almost no positive signal**: pixels with fractional cover > 0.1 are ~5.7e-5 of the
    Sentinel-2 raster and ~2.8e-4 of the Landsat raster; the maps are overwhelmingly
    near-zero open-ocean/land probability.
- **A preferred georeferenced reference alternative is already in the manifest**: **MARIDA
  (Marine Debris Archive)** provides manually photo-interpreted, georeferenced Sentinel-2
  annotations that include dense/sparse **sargassum** classes. Per §8.2, defer to the
  reference product rather than ingest this derived/demonstration map suite.

Taken together this is not a usable georeferenced label dataset: the labeled data is
coordinate-free and the georeferenced data is 2 demonstration model-output rasters (one
pre-2016), not reference labels.

## How the assessment was reproduced

```python
from remotezip import RemoteZip
url = "https://zenodo.org/api/records/17246345/files/sargassum_detection_models.zip/content"
with RemoteZip(url) as z:
    print(z.namelist())                      # file listing (no full download)
    open("sargassum_data.csv","wb").write(z.read(
        "sargassum_detection_models/sargassum_data.csv"))
    # the two output/fractional_cover_maps/*.tif similarly
```
CSV header confirms `Blue,Green,Red,NIR,SWIR1,class` with no coordinate columns; the two
example .tif were opened with rasterio (CRS/res/value stats above).

## If revisited

Only worth reconsidering if a version of this dataset is published that ships the labeled
spectra **with per-pixel lon/lat** (so points could be encoded as sparse classification
points, sargassum vs no_sargassum), or a genuinely multi-scene georeferenced
fractional-cover reference product. For sargassum coverage in the meantime, use MARIDA.
