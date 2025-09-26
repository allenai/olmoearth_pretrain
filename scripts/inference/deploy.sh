# Script to deploy Helios server to Vertex AI
# Has to be run from helios root directory: sh scripts/inference/deploy.sh
set -e
PROJECT="ai2-ivan"
REGION="us-central1"
TAG="$REGION-docker.pkg.dev/$PROJECT/helios/helios"
IN_BUCKET="ai2-ivan-helios-input-data"
STEPS="5"

if [[ "$1" == "--skip-auth" || "$1" == "-s" ]]; then
        echo "1/$STEPS: Skipping Google Cloud authentication."
else
        echo "1/$STEPS: Authenticating Google Cloud."
        gcloud auth login
        gcloud auth application-default login
        gcloud auth application-default set-quota-project "$PROJECT"
        gcloud config set project "$PROJECT"
        gcloud auth configure-docker $REGION-docker.pkg.dev
fi

echo "2/$STEPS: Creating artifact registry project if it doesn't exist."
if [ -z "$(gcloud artifacts repositories list --format='get(name)' --filter "helios")" ]; then
        gcloud artifacts repositories create "helios" \
        --location "$REGION" \
        --repository-format docker
fi

echo "3/$STEPS: Building docker container."
docker build -t "$TAG" -f scripts/inference/Dockerfile .

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
