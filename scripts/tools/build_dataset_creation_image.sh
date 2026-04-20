#!/usr/bin/env bash
# Build and register the dataset-creation Docker image with Beaker.
#
# Tags the image by the current git sha so every build is traceable.
# The resulting Beaker image is what every orchestrator-launched task runs.
#
# Usage:
#   scripts/tools/build_dataset_creation_image.sh            # uses git sha
#   scripts/tools/build_dataset_creation_image.sh mytag      # override tag

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TAG="${1:-$(git rev-parse --short HEAD)}"
LOCAL_NAME="oep-dataset-creation:${TAG}"
BEAKER_NAME="oep-dataset-creation-${TAG}"

echo "Building ${LOCAL_NAME}"
docker build \
    -f olmoearth_pretrain/dataset_creation/Dockerfile \
    -t "${LOCAL_NAME}" \
    .

echo "Registering as Beaker image: ${BEAKER_NAME}"
beaker image create --name "${BEAKER_NAME}" "${LOCAL_NAME}"

echo
echo "Done. Reference this image as:"
echo "  ${BEAKER_NAME}"
echo "or <your-beaker-user>/${BEAKER_NAME}"
