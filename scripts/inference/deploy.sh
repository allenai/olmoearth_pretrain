# Script to deploy Helios server to Vertex AI
# Has to be run from helios root directory: sh scripts/inference/deploy.sh
set -e
PROJECT="ai2-ivan"
REGION="us-central1"
TAG="$REGION-docker.pkg.dev/$PROJECT/helios/helios"
BUCKET="gee-input-data"
STEPS="6"


echo "1/$STEPS: Authenticating Google Cloud."
gcloud config set project "$PROJECT"
gcloud auth application-default set-quota-project "$PROJECT"
gcloud auth application-default login
gcloud auth configure-docker $REGION-docker.pkg.dev

echo "2/$STEPS: Creating artifact registry project if it doesn't exist."
if [ -z "$(gcloud artifacts repositories list --format='get(name)' --filter "helios")" ]; then
        gcloud artifacts repositories create "helios" \
        --location "$REGION" \
        --repository-format docker
fi

echo "3/$STEPS: Building docker container."
docker build -t "$TAG" -f scripts/litserve/Dockerfile .

echo "4/$STEPS: Pushing docker container to cloud."
docker push "$TAG"

echo "5/$STEPS Deploying inference docker image to Google Cloud Run"
gcloud run deploy "helios" --image "$TAG":latest \
        --cpu=4 \
        --memory=8Gi \
        --platform=managed \
        --region="$REGION" \
        --allow-unauthenticated \
        --concurrency 10 \
        --max-instances 50 \
        --port 8080

echo "6/$STEPS Deploying trigger function."
export URL=$(gcloud run services list --platform managed --filter helios --limit 1 --format='get(URL)')
gcloud functions deploy trigger-helios \
    --source="scripts/inference/trigger_function" \
    --trigger-bucket="$BUCKET" \
    --allow-unauthenticated \
    --entry-point=trigger \
    --runtime=python312 \
    --set-env-vars INFERENCE_HOST="$URL" 
