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
# MATERIALIZE is the opposite regime: worker-bound, not rate-bound, so it wants
# more workers and a long backoff. WORKERS and RETRY_BACKOFF default per
# COMMAND accordingly; JOBS_PER_DATASET stays at 1 either way, since the
# launcher starts N IDENTICAL jobs (it does not shard) and the extra ones spend
# their time re-scanning windows the others already finished.
#
# Prints the commands and exits unless LAUNCH=1. Beaker jobs cost real compute,
# so a dry run is the default.
#
# Usage:
#   scripts/tools/launch_year_aligned_prepare.sh                       # dry run
#   LAUNCH=1 HOSTS=jupiter-cs-aus-134.reviz.ai2.in scripts/tools/launch_year_aligned_prepare.sh
#   LAUNCH=1 CLUSTERS=ai2/jupiter-cirrascale-2 NUM_JOBS=2 ONLY=us_trees ...
#   LAUNCH=1 COMMAND=materialize HOSTS=... scripts/tools/launch_year_aligned_prepare.sh
#
# Env:
#   IMAGE               Beaker image (default favyen/rslpomp20260727a)
#   HOSTS               comma/space separated Beaker hosts (one job per host)
#   CLUSTERS            comma/space separated clusters; requires NUM_JOBS
#                       NB: prefer HOSTS. rslp's launch_job() sets
#                       resources={"gpuCount": 1} whenever hostname is None, so
#                       a CLUSTERS launch reserves a GPU per job -- and neither
#                       prepare nor materialize touches one.
#   NUM_JOBS            jobs per cluster target
#   JOBS_PER_DATASET    repeat each dataset's launch this many times (default 1)
#   ONLY                restrict to one dataset (staging name or base name)
#   COMMAND             prepare (default) or materialize
#   LAYER_SET           monthly (default) or tessera_v2_fetch -- see below
#   GROUP               restrict to one window group (default: all)
#   WORKERS             rslearn --workers (default 16 prepare / 64 materialize)
#   PRIORITY            Beaker priority: low|normal|high|immediate|urgent (default high)
#   RETRY_BACKOFF       --retry-backoff-seconds (default 2 prepare / 60 materialize)
#   RETRY_ATTEMPTS      --retry-max-attempts (default 12)
#   LAUNCH=1            actually launch instead of printing
#
# LAYER_SET=tessera_v2_fetch switches the job to the three all-scenes layers
# used for our own TESSERA v2 inference (docs/TesseraV2Inference.md). Pair it
# with GROUP=<dataset>_tessera_v2, the fetch group created by
# olmoearth_pretrain/evals/datasets/tessera_v2_export.py -- without GROUP the
# job would fetch a full year of scenes for the eval windows too. Example:
#
#   LAUNCH=1 LAYER_SET=tessera_v2_fetch GROUP=ethiopia_crops_tessera_v2 \
#       ONLY=ethiopia_crops HOSTS=... scripts/tools/launch_year_aligned_prepare.sh

set -uo pipefail

STAGE_ROOT="${STAGE_ROOT:-/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals}"
IMAGE="${IMAGE:-favyen/rslpomp20260727a}"
JOBS_PER_DATASET="${JOBS_PER_DATASET:-1}"
# BeakerJobPriority: low | normal | high | immediate | urgent. launch_jobs
# defaults to high; prepare is not latency-critical, so raise it only to jump a
# busy queue.
PRIORITY="${PRIORITY:-high}"
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-12}"

# prepare writes items.json (STAC queries); materialize downloads the pixels.
# Same flags either way -- both handlers take --workers/--retry-*/--enabled-layers/
# --ignore-errors -- but materialize is far heavier, and running it before prepare
# has finished just does partial work, so it gets a readiness pre-flight below.
COMMAND="${COMMAND:-prepare}"
if [[ "$COMMAND" != "prepare" && "$COMMAND" != "materialize" ]]; then
    echo "ERROR: COMMAND must be 'prepare' or 'materialize', got '$COMMAND'" >&2
    exit 1
fi

# The two commands are bound by DIFFERENT resources, so they want opposite
# settings and the defaults follow COMMAND rather than being one compromise.
#
# prepare is bound by the Planetary Computer STAC API, which is a per-ACCOUNT
# limit -- more workers past the ceiling only converts throughput into 403s.
# rslearn's retry() sleeps retry_backoff * (attempt + 1) * random(1..2), so the
# 60 that internal_docs.md gives parks a worker for 60-120s on its FIRST 403;
# with ~100ms queries and routine 403s that collapses effective parallelism
# toward one worker (measured: us_trees went 0.25 -> ~2 windows/s on dropping
# it to 2).
#
# materialize is bound by the workers themselves -- it downloads and writes
# pixels, so throughput scales with concurrency until the box's network or
# Weka saturates. Here a 60s backoff is right: a retry covers a real failure
# mid-download and a minute is negligible against the transfer.
if [[ "$COMMAND" == "prepare" ]]; then
    WORKERS="${WORKERS:-16}"
    RETRY_BACKOFF="${RETRY_BACKOFF:-2}"
else
    WORKERS="${WORKERS:-64}"
    RETRY_BACKOFF="${RETRY_BACKOFF:-60}"
fi
LAYER_SET="${LAYER_SET:-monthly}"
if [[ "$LAYER_SET" != "monthly" && "$LAYER_SET" != "tessera_v2_fetch" ]]; then
    echo "ERROR: LAYER_SET must be 'monthly' or 'tessera_v2_fetch', got '$LAYER_SET'" >&2
    exit 1
fi
GROUP="${GROUP:-}"
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

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# Restricting to the monthly layers is MANDATORY for pastis_year_aligned, whose
# config still carries the sentinel2_l2a_all / sentinel1_*_all fetch layers
# (max_matches 150/100, kept so tessera_v2 inference stays possible). A bare
# prepare there would try to fetch a year of scenes per window. It is a no-op
# for the other eight, whose remaining layers have no data_source, so it is
# applied uniformly rather than special-cased.
if [[ "$LAYER_SET" == "monthly" ]]; then
    ENABLED_LAYERS=$(
        python3 - <<'EOF'
print(",".join([f"sentinel2_l2a_mo{i:02d}" for i in range(1, 13)]
             + [f"sentinel1_mo{i:02d}" for i in range(1, 13)]))
EOF
    )
    REQUIRED_LAYER=sentinel2_l2a_mo12
else
    # The fetch layers live in their own config (written by
    # tessera_v2_export.py write_fetch_config), not in the dataset's
    # config.json, so the job is scoped by --config rather than
    # --enabled-layers. GROUP is still mandatory: the fetch config would
    # otherwise pull a year of scenes for every eval window too.
    ENABLED_LAYERS=""
    FETCH_CONFIG_NAME=config_tessera_v2_fetch.json
    if [[ -z "$GROUP" ]]; then
        log "ERROR: LAYER_SET=tessera_v2_fetch needs GROUP=<dataset>_tessera_v2;"
        log "  without it these layers fetch a year of scenes for the eval windows."
        exit 1
    fi
fi

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
    if [[ "$LAYER_SET" == "monthly" ]]; then
        # Refuse to prepare a dataset that has not been re-anchored: without
        # apply, config.json still describes the old single sentinel2 layer and
        # the run would silently do nothing.
        if ! python3 -c "
import json, sys
layers = json.load(open('$ds_path/config.json'))['layers']
sys.exit(0 if '$REQUIRED_LAYER' in layers else 1)
" 2>/dev/null; then
            log "SKIP  $name -- monthly layers absent from config.json; run reanchor apply first"
            skipped=$((skipped + 1))
            continue
        fi
    else
        fetch_config="$ds_path/$FETCH_CONFIG_NAME"
        if [[ ! -f "$fetch_config" ]]; then
            log "SKIP  $name -- no $FETCH_CONFIG_NAME; run"
            log "      'python -m olmoearth_pretrain.evals.datasets.tessera_v2_export write_fetch_config --ds_path $ds_path' first"
            skipped=$((skipped + 1))
            continue
        fi
    fi

    # Materializing before prepare has finished silently produces a partial
    # dataset, so sample a few windows and check they carry all 24 monthly
    # entries in items.json. Only meaningful for the monthly layer set: the
    # fetch layers have no fixed item count per window (it is however many
    # scenes intersected that year), so there is nothing to compare against.
    if [[ "$COMMAND" == "materialize" && "$LAYER_SET" == "monthly" ]]; then
        # items.json is a LIST of serialized WindowLayerData dicts, each with a
        # "layer_name" key -- not a mapping. Any exception prints, so the check
        # fails closed (non-empty output => skip) rather than passing silently.
        readiness=$(DS_PATH="$ds_path" python3 - <<'EOF'
import json, os, pathlib, random, traceback
try:
    root = pathlib.Path(os.environ["DS_PATH"]) / "windows"
    windows = [p for p in root.glob("*/*") if p.is_dir()]
    if not windows:
        print("no windows found")
        raise SystemExit(0)
    random.seed(0)
    sample = random.sample(windows, min(25, len(windows)))
    short = 0
    for w in sample:
        items = w / "items.json"
        if not items.exists():
            short += 1
            continue
        entries = json.loads(items.read_text())
        n = sum(
            1
            for e in entries
            if e["layer_name"].startswith(("sentinel2_l2a_mo", "sentinel1_mo"))
        )
        if n < 24:
            short += 1
    if short:
        print(f"{short}/{len(sample)} sampled windows lack all 24 monthly item entries")
except Exception:
    print("readiness check errored: " + traceback.format_exc(limit=1).replace("\n", " "))
EOF
)
        if [[ -n "$readiness" ]]; then
            log "SKIP  $name -- prepare looks incomplete: $readiness"
            skipped=$((skipped + 1))
            continue
        fi
    fi

    command_json=$(
        ENABLED_LAYERS="$ENABLED_LAYERS" WORKERS="$WORKERS" COMMAND="$COMMAND" \
            RETRY_BACKOFF="$RETRY_BACKOFF" RETRY_ATTEMPTS="$RETRY_ATTEMPTS" \
            GROUP="$GROUP" FETCH_CONFIG="${FETCH_CONFIG_NAME:-}" python3 - <<'EOF'
import json, os
command = [
    "rslearn", "dataset", os.environ["COMMAND"],
    "--root", "{ds_path}",
    "--workers", os.environ["WORKERS"],
    "--no-use-initial-job",
    "--retry-max-attempts", os.environ["RETRY_ATTEMPTS"],
    "--retry-backoff-seconds", os.environ["RETRY_BACKOFF"],
    "--ignore-errors",
]
if os.environ["ENABLED_LAYERS"]:
    command += ["--enabled-layers", os.environ["ENABLED_LAYERS"]]
if os.environ["FETCH_CONFIG"]:
    # {ds_path} is substituted by the launcher inside the job.
    command += ["--config", "{ds_path}/" + os.environ["FETCH_CONFIG"]]
if os.environ["GROUP"]:
    command += ["--group", os.environ["GROUP"]]
print(json.dumps(command))
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
            log "LAUNCH $COMMAND $name (job $((i + 1))/$JOBS_PER_DATASET)"
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
log "Completion check per dataset: re-run $COMMAND and confirm it reports nothing"
log "  left to do for any of: ${ENABLED_LAYERS:-the layers in $FETCH_CONFIG_NAME}"
log "  ('Preparing 0 windows for layer ...')."
log "  --ignore-errors means a nonzero exit is not the signal to trust."
if [[ -z "$LAUNCH" ]]; then
    echo
    log "To raise the PC rate ceiling, patch rslp/common/beaker_data_materialization.py"
    log "to forward PC_SDK_SUBSCRIPTION_KEY the way it already forwards"
    log "NASA_EARTHDATA_USERNAME (~line 66). Without that these jobs are anonymous."
fi
