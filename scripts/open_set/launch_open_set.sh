#!/bin/bash
# Launch the open-set text-conditioned segmentation experiment on Beaker.
#
# Usage:
#   ./scripts/open_set/launch_open_set.sh CHECKPOINT_PATH [RUN_NAME] [CLUSTER]
#
# CHECKPOINT_PATH (required): path to an OlmoEarth distributed checkpoint
#   directory, e.g. /weka/dfive-default/helios/checkpoints/USER/RUN/step370000
# RUN_NAME (default: "open_set"): the Beaker / wandb run name.
# CLUSTER (default: "ai2/ceres-cirrascale"): the Beaker cluster.
#
# All trailing arguments after the third are forwarded as overrides to the
# python script (e.g. --train_module.optim_config.lr=1e-4).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CHECKPOINT_PATH [RUN_NAME] [CLUSTER] [-- OVERRIDES...]"
    echo ""
    echo "  CHECKPOINT_PATH  required. OlmoEarth distributed checkpoint dir."
    echo "  RUN_NAME         default: open_set"
    echo "  CLUSTER          default: ai2/ceres-cirrascale"
    echo ""
    echo "Example:"
    echo "  $0 /weka/dfive-default/helios/checkpoints/USER/RUN/step370000 my_run ai2/ceres-cirrascale"
    exit 1
fi

CHECKPOINT_PATH="$1"
RUN_NAME="${2:-open_set}"
CLUSTER="${3:-ai2/ceres-cirrascale}"

# Drop the first three positional args; everything else passes through as
# olmo-core dotlist overrides.
shift $(( $# < 3 ? $# : 3 ))

CLUSTERS="[ai2/jupiter-cirrascale-2,ai2/ceres-cirrascale]"
PRIORITY="high"
NUM_GPUS=8

python scripts/open_set/script.py launch "$RUN_NAME" "$CLUSTER" \
    --model.checkpoint_path="$CHECKPOINT_PATH" \
    --launch.clusters="$CLUSTERS" \
    --launch.priority="$PRIORITY" \
    --launch.num_gpus="$NUM_GPUS" \
    "$@"
