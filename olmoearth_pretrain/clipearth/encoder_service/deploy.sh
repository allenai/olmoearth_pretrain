#!/bin/bash
# Build the ClipEarth text encoder from this directory and deploy it to Cloud Run.
#   ./deploy.sh
# Sizing (override with env): 4 vCPU / 4 GiB, one instance kept warm (a cold start loads the
# model, a few seconds), up to 3.
set -euo pipefail
PROJECT=${PROJECT:-earthsystem-dev-c3po}
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-clipearth-encoder}
REPO=${AR_REPO:-gea}  # the Artifact Registry repo the tile server uses
CPU=${CPU:-4}
MEMORY=${MEMORY:-4Gi}
MIN_INSTANCES=${MIN_INSTANCES:-1}
MAX_INSTANCES=${MAX_INSTANCES:-3}
CONCURRENCY=${CONCURRENCY:-16}
ORT_THREADS=${ORT_THREADS:-4}
ORIGINS=${ALLOWED_ORIGINS:-https://olmoearth.allenai.org,https://staging.olmoearth.allenai.org}
HERE=$(cd "$(dirname "$0")" && pwd)

SHA=$(git -C "$HERE" rev-parse --short=8 HEAD)
git -C "$HERE" diff --quiet HEAD -- . || SHA="$SHA-dirty-$(date -u +%H%M%S)"
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:$SHA

echo "Building $IMAGE"
gcloud builds submit "$HERE" --tag "$IMAGE" --project "$PROJECT" --quiet
gcloud run deploy "$SERVICE" --image "$IMAGE" --project "$PROJECT" --region "$REGION" \
  --allow-unauthenticated --port 8080 --execution-environment gen2 \
  --cpu "$CPU" --memory "$MEMORY" --cpu-boost \
  --min-instances "$MIN_INSTANCES" --max-instances "$MAX_INSTANCES" --concurrency "$CONCURRENCY" \
  --timeout 60 \
  --update-labels "git-sha=${SHA%%-*}" \
  --set-env-vars "^@^ALLOWED_ORIGINS=$ORIGINS@ORT_THREADS=$ORT_THREADS" \
  --quiet
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)'
