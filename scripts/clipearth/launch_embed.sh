#!/bin/bash
# Embed the open-set segmentation windows (Molmo's open_set_seg train + val, ~51.6k
# windows; BANK=1 for all ~1.43M windows of the bank, encoding only labeled tiles)
# with frozen OlmoEarth v1.3, for CLIP training in olmoearth_pretrain/clipearth.
#
# Each replica takes every REPLICAS-th window and writes r<rank>_p<part>.npz files to
# OUT_DIR; finished parts are skipped. An OUT_DIR that already has parts is refused
# unless RESUME=1.
#
# Gantry runs the committed code at the current git ref, so commit and push
# olmoearth_pretrain/clipearth first.
#
# Usage:
#   bash scripts/clipearth/launch_embed.sh             # launch
#   DRY_RUN=1 bash scripts/clipearth/launch_embed.sh   # print the experiment, no submit
set -euo pipefail
cd "$(dirname "$0")/../.."

CHECKPOINT="${CHECKPOINT:-/weka/dfive-default/agent/piperbots/checkpoints/v1_3_rc_ld1_pixtgt_pix512_step667200}"
PATCH_SIZE="${PATCH_SIZE:-4}"
LOCATION_QUERY_PATCH_SIZE="${LOCATION_QUERY_PATCH_SIZE:-2}"
SET="open_set${BANK:+_bank}"
OUT_DIR="${OUT_DIR:-/weka/dfive-default/agent/piperbots/clipearth/embeddings/${SET}_v1_3_ps${PATCH_SIZE}_lq${LOCATION_QUERY_PATCH_SIZE}}"
REPLICAS="${REPLICAS:-8}"
WINDOWS_PER_BATCH="${WINDOWS_PER_BATCH:-8}"
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
PRIORITY="${PRIORITY:-urgent}"
MIN_RUNTIME="${MIN_RUNTIME:-8h}"

if [[ -n "$(git status --porcelain -- olmoearth_pretrain/clipearth)" ]]; then
    echo "olmoearth_pretrain/clipearth has uncommitted changes; commit and push first." >&2
    exit 1
fi
if ! git branch -r --contains HEAD | grep -q .; then
    echo "HEAD ($(git rev-parse --short HEAD)) is not on any remote branch; push first." >&2
    exit 1
fi

if compgen -G "$OUT_DIR/*.npz" > /dev/null && [[ -z "${RESUME:-}${FILL:-}" ]]; then
    echo "$OUT_DIR already has parts; set RESUME=1 to continue it, FILL=1 to embed only the" \
        "windows missing from it, or pick another OUT_DIR." >&2
    exit 1
fi

EXTRA=()
if [[ -n "${BANK:-}" ]]; then
    EXTRA+=(--bank --labeled_tiles_only --regions_only)
fi
if [[ -n "${FILL:-}" ]]; then
    EXTRA+=(--fill)
fi

DRY=()
if [[ -n "${DRY_RUN:-}" ]]; then
    DRY=(--dry-run)
fi

gantry run "${DRY[@]}" \
    --name "clipearth-embed-${SET//_/-}-ps${PATCH_SIZE}-lq${LOCATION_QUERY_PATCH_SIZE}${FILL:+-fill}" \
    --description "Embed open-set windows with OlmoEarth v1.3 for CLIP (clipearth)" \
    --workspace ai2/earth-systems \
    --budget ai2/atec-olmoearth \
    --cluster "$CLUSTER" \
    --priority "$PRIORITY" \
    --min-runtime "$MIN_RUNTIME" \
    --gpus 1 \
    --replicas "$REPLICAS" \
    --shared-memory 64GiB \
    --weka dfive-default:/weka/dfive-default \
    --weka oe-training-default:/weka/oe-training-default \
    --python-manager uv \
    --uv-extra training \
    --default-python-version 3.12 \
    --yes \
    -- python -m olmoearth_pretrain.clipearth.embed \
        --checkpoint "$CHECKPOINT" \
        --out_dir "$OUT_DIR" \
        --patch_size "$PATCH_SIZE" \
        --location_query_patch_size "$LOCATION_QUERY_PATCH_SIZE" \
        --windows_per_batch "$WINDOWS_PER_BATCH" \
        --num_workers 8 \
        "${EXTRA[@]}"
