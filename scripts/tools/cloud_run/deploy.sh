# Script to deploy OlmoEarth docker container
# Has to be run from olmoearth_pretrain root directory: sh scripts/tools/cloud_run/deploy.sh
set -e
GCLOUD_PROJECT="ai2-ivan"
REGION="us-central1"
IN_BUCKET="ai2-ivan-helios-input-data"
OUT_BUCKET="ai2-ivan-helios-output-data"
TAG="$REGION-docker.pkg.dev/$GCLOUD_PROJECT/olmoearth/olmoearth"
STEPS="5"

if [[ "$1" == "--skip-auth" || "$1" == "-s" ]]; then
        echo "1/$STEPS: Skipping Google Cloud authentication."
else
        echo "1/$STEPS: Authenticating Google Cloud."
        gcloud auth login
        gcloud auth application-default login
        gcloud auth application-default set-quota-project "$GCLOUD_PROJECT"
        gcloud config set project "$GCLOUD_PROJECT"
        gcloud auth configure-docker $REGION-docker.pkg.dev
fi

echo "2/$STEPS: Creating artifact registry project if it doesn't exist."
if [ -z "$(gcloud artifacts repositories list --format='get(name)' --filter "olmoearth")" ]; then
        gcloud artifacts repositories create "olmoearth" \
        --location "$REGION" \
        --repository-format docker
fi

echo "3/$STEPS: Building docker container."
docker build -t "$TAG" \
        -f scripts/tools/cloud_run/Dockerfile \
        --build-arg GCLOUD_PROJECT=$GCLOUD_PROJECT \
        --build-arg IN_BUCKET=$IN_BUCKET \
        --build-arg OUT_BUCKET=$OUT_BUCKET .

echo "4/$STEPS: Pushing docker container to cloud."
docker push "$TAG"

echo "You can now create a Google Cloud Run Job using this container."
