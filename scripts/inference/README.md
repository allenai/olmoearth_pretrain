# OlmoEarth Embeddings using data from Google Earth Engine (GEE)

**Author**: Ivan Zvonkov (ivan.zvonkov@gmail.com)

**Last modified**: Oct 27, 2025

## Contents

**[generate_embeddings.ipynb](generate_embeddings.ipynb)** 
One-stop shop Colab notebook for generating embeddings with OlmoEarth and Google Earth Engine. Inference can be done directly in the notebook or by deploying a Docker container.

**[gee_create_OlmoEarth_cropland.js](gee_create_OlmoEarth_cropland.js)**
Google Earth Engine script for creating a Togo cropland using OlmoEarth embeddings in Cloud Storage.

**[gee_OlmoEarth_cropland_eval.js](gee_OlmoEarth_cropland_eval.js)**
Google Earth Engine script for evaluating the OlmoEarth cropland map and comparing with other cropland maps in Togo.

**[deploy.sh](deploy.sh)**
Script to build and deploy OlmoEarth embedding generation docker container to Google Cloud.

**[Dockerfile](Dockerfile)**
Builds docker container for running job.py

**[job.py](job.py)**
Python script for running inference and generating embeddings using OlmoEarth and two Google Cloud Storage buckets. Expects Google Earth Engine data to already be exported using Colab notebook.
