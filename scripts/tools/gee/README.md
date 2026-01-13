# Google Earth Engine Scripts

These scripts are meant to be run in Google Earth Engine after embeddings are generated using either `scripts/tools/generate_embeddings.ipynb` or `scripts/tools/cloud_run` Docker container.
The embeddings are expected to be stored in a Cloud Storage bucket.


**`gee_createOlmoEarth_cropland.js`**

Generates Togo cropland map from embeddings stored in a Google Cloud storage bucket.

**`gee_OlmoEarth_cropland_eval.js`**

Evaluates generated Togo cropland map against existing Togo cropland maps using data points from CropHarvest dataset.
