#!/usr/bin/env bash
# Launch Beaker jobs to run `rslearn dataset prepare` on the *_year_aligned
# staging datasets, via the rslearn_projects launcher.
#
# The launcher does NOT shard work: it starts N identical jobs and relies on
# rslearn skipping already-done windows (prepare_dataset_windows checks
# `layer_name in layer_datas`, manage.py:138). So more jobs = more duplicated
# scanning, and the useful parallelism comes from running different DATASETS
# concurrently rather than piling jobs onto one.
#
# RATE LIMIT, read this first: prepare is bound by the Planetary Computer STAC
# API, not CPU. Measured ~2 windows/s * 24 layers ~= 48 queries/s from a single
# box, already drawing occasional 403s. That limit is global to your account,
# so N hosts divide it rather than multiply it. launch_job() also cannot pass
# PC_SDK_SUBSCRIPTION_KEY (it forwards only NASA_EARTHDATA_* and *_PROXY), so
# these jobs query anonymously at the lowest tier. Keep JOBS_PER_DATASET at 1
# unless you have raised the ceiling -- see the note at the bottom.
#
# Prints the commands and exits unless LAUNCH=1. Beaker jobs cost real compute,
# so a dry run is the default.
#
# Usage:
#   scripts/tools/launch_year_aligned_prepare.sh                       # dry run
#   LAUNCH=1 HOSTS=jupiter-cs-aus-134.reviz.ai2.in scripts/tools/launch_year_aligned_prepare.sh
#   LAUNCH=1 CLUSTERS=ai2/jupiter-cirrascale-2 NUM_JOBS=2 ONLY=us_trees ...
#
# Env:
#   IMAGE               Beaker image (default favyen/rslpomp20260727a)
#   HOSTS               comma/space separated Beaker hosts (one job per host)
#   CLUSTERS            comma/space separated clusters; requires NUM_JOBS
#   NUM_JOBS            jobs per cluster target
#   JOBS_PER_DATASET    repeat each dataset's launch this many times (default 1)
#   ONLY                restrict to one dataset (staging name or base name)
#   WORKERS             rslearn --workers (default 16; 64 drew 403 storms)
#   PRIORITY            Beaker priority: low|normal|high|immediate|urgent (default high)
#   LAUNCH=1            actually launch instead of printing

set -uo pipefail

STAGE_ROOT="${STAGE_ROOT:-/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals}"
IMAGE="${IMAGE:-favyen/rslpomp20260727a}"
WORKERS="${WORKERS:-16}"
JOBS_PER_DATASET="${JOBS_PER_DATASET:-1}"
# BeakerJobPriority: low | normal | high | immediate | urgent. launch_jobs
# defaults to high; prepare is not latency-critical, so raise it only to jump a
# busy queue.
PRIORITY="${PRIORITY:-high}"
ONLY="${ONLY:-}"
LAUNCH="${LAUNCH:-}"

DATASETS=(
    africa_crop_mask_year_aligned
    canada_crops_coarse_year_aligned
    canada_crops_fine_year_aligned
    descals_year_aligned
    ethiopia_crops_year_aligned
    glance_year_aligned
    lcmap_lu_year_aligned
    us_trees_year_aligned
    pastis_year_aligned
)

# Restricting to the monthly layers is MANDATORY for pastis_year_aligned, whose
# config still carries the sentinel2_l2a_all / sentinel1_*_all fetch layers
# (max_matches 150/100, kept so tessera_v2 inference stays possible). A bare
# prepare there would try to fetch a year of scenes per window. It is a no-op
# for the other eight, whose remaining layers have no data_source, so it is
# applied uniformly rather than special-cased.
ENABLED_LAYERS=$(
    python3 - <<'EOF'
print(",".join([f"sentinel2_l2a_mo{i:02d}" for i in range(1, 13)]
             + [f"sentinel1_mo{i:02d}" for i in range(1, 13)]))
EOF
)

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

if [[ -z "$LAUNCH" ]]; then
    log "DRY RUN -- set LAUNCH=1 to actually submit. Nothing will be launched."
fi

# The launcher requires exactly one of hosts or clusters+num_jobs.
target_args=()
if [[ -n "${HOSTS:-}" ]]; then
    for host in ${HOSTS//,/ }; do
        target_args+=("--hosts+=$host")
    done
elif [[ -n "${CLUSTERS:-}" ]]; then
    if [[ -z "${NUM_JOBS:-}" ]]; then
        log "ERROR: CLUSTERS requires NUM_JOBS"
        exit 1
    fi
    for cluster in ${CLUSTERS//,/ }; do
        target_args+=("--clusters+=$cluster")
    done
    target_args+=("--num_jobs=$NUM_JOBS")
else
    log "ERROR: set HOSTS or CLUSTERS+NUM_JOBS"
    exit 1
fi

launched=0
skipped=0
for name in "${DATASETS[@]}"; do
    if [[ -n "$ONLY" && "$name" != "$ONLY" && "$name" != "${ONLY}_year_aligned" ]]; then
        continue
    fi
    ds_path="$STAGE_ROOT/$name"

    if [[ ! -d "$ds_path" ]]; then
        log "SKIP  $name -- not staged yet at $ds_path"
        skipped=$((skipped + 1))
        continue
    fi
    # Refuse to prepare a dataset that has not been re-anchored: without apply,
    # config.json still describes the old single sentinel2 layer and the run
    # would silently do nothing.
    if ! python3 -c "
import json, sys
layers = json.load(open('$ds_path/config.json'))['layers']
sys.exit(0 if 'sentinel2_l2a_mo12' in layers and 'sentinel1_mo12' in layers else 1)
" 2>/dev/null; then
        log "SKIP  $name -- monthly layers absent from config.json; run reanchor apply first"
        skipped=$((skipped + 1))
        continue
    fi

    command_json=$(
        ENABLED_LAYERS="$ENABLED_LAYERS" WORKERS="$WORKERS" python3 - <<'EOF'
import json, os
print(json.dumps([
    "rslearn", "dataset", "prepare",
    "--root", "{ds_path}",
    "--workers", os.environ["WORKERS"],
    "--no-use-initial-job",
    "--retry-max-attempts", "8",
    "--retry-backoff-seconds", "60",
    "--enabled-layers", os.environ["ENABLED_LAYERS"],
    "--ignore-errors",
]))
EOF
    )

    for ((i = 0; i < JOBS_PER_DATASET; i++)); do
        if [[ -z "$LAUNCH" ]]; then
            echo
            echo "python -m rslp.main common launch_data_materialization_jobs \\"
            echo "    --image $IMAGE \\"
            echo "    --ds_path $ds_path \\"
            echo "    --priority=$PRIORITY \\"
            for arg in "${target_args[@]}"; do echo "    $arg \\"; done
            echo "    --command '$command_json'"
        else
            log "LAUNCH $name (job $((i + 1))/$JOBS_PER_DATASET)"
            python -m rslp.main common launch_data_materialization_jobs \
                --image "$IMAGE" \
                --ds_path "$ds_path" \
                --priority="$PRIORITY" \
                "${target_args[@]}" \
                --command "$command_json" || log "FAILED to launch $name"
        fi
        launched=$((launched + 1))
    done
done

echo
log "$launched job(s) $([[ -z "$LAUNCH" ]] && echo 'would be launched' || echo launched), $skipped dataset(s) skipped"
log "Completion check per dataset: re-run prepare and confirm it logs"
log "  'Preparing 0 windows for layer ...' for all 24 layers. --ignore-errors"
log "  means a nonzero exit is not the signal to trust."
if [[ -z "$LAUNCH" ]]; then
    echo
    log "To raise the PC rate ceiling, patch rslp/common/beaker_data_materialization.py"
    log "to forward PC_SDK_SUBSCRIPTION_KEY the way it already forwards"
    log "NASA_EARTHDATA_USERNAME (~line 66). Without that these jobs are anonymous."
fi
