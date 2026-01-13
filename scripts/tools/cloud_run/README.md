
# OlmoEarth Embeddings using Google Cloud Run

Deploying OlmoEarth to Google Cloud Run allows generating embeddings using Google's GPUs on data exported from Google Earth Engine.
This is an alternative to generating embeddings using the `generate_embeddings.ipynb` notebook.

### Folder Contents
- `job.py` - Python script for running inference using pretrained model and data from Google Earth Engine (exported to Google Cloud)
- `Dockerfile` - builds Docker container that can be deployed in Google Cloud Run
- `deploy.sh` - script for building docker container and deploying it to Google Cloud.

### Setup

**1. Push docker container to cloud**

Update the env variables in `deploy.sh`
```bash
GCLOUD_PROJECT="your Google Cloud project name"
REGION="region for Cloud Run deployment (us-central1 recommended)"
IN_BUCKET="bucket that contains data from Google Earth Engine"
OUT_BUCKET="bucket that will store embeddings"
```

Run the script:
```bash
sh scripts/tools/cloud_run/deploy.sh
```

**2. Create Cloud Run Job**

Navigate to https://console.cloud.google.com/run/jobs and select "Deploy container".

Specify parameters:
- `Container Image URL` select the OlmoEarth container most recently deployed.
- `Job Name` specify whatever you'd like.
- `Region` select the region of the buckets for optimal performance.

Select the dropdown `Containers, Volumes, Connections, Security`: and specify the container arguments following the below example:
```
--run Togo_v2_nano_2019-03-01_2020-03-01 --tiles 10.tif
```

Check the GPU checkbox to enable inference on an L4 GPU. Finally, click the button at the bottom to create the job.

## Inference
The Cloud Run job container arguments can be updated to run additional jobs with remaining tif files.
