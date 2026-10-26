#!/usr/bin/env bash
# Seed the *_year_aligned staging datasets from the ingested eval copies.
#
# Copies every eval dataset from /weka/.../olmoearth/eval_datasets (the only
# copy carrying the materialized gse/tessera layers) into a staging tree,
# leaving out the imagery that reanchor_year_aligned_dataset.py replaces.
# rslearn windows are tens of thousands of tiny files, so this streams through
# tar rather than rsync -- the same reason studio_ingest's _tar_copy_cmd does.
#
# Usage:
#   nohup scripts/tools/copy_year_aligned_seeds.sh > copy_seeds.log 2>&1 &
#   tail -f copy_seeds.log
#
#   JOBS=4 scripts/tools/copy_year_aligned_seeds.sh        # parallelism (default 3)
#   ONLY=us_trees scripts/tools/copy_year_aligned_seeds.sh # single dataset
#   FORCE=1 scripts/tools/copy_year_aligned_seeds.sh       # redo completed ones
#
# Existing non-empty destinations are skipped, so a re-run resumes rather than
# restarting. Each copy is verified for gse and for the absence of the dropped
# imagery; failures are reported in the summary and do not stop the run.

set -uo pipefail

EVAL_ROOT="${EVAL_ROOT:-/weka/dfive-default/olmoearth/eval_datasets}"
STAGE_ROOT="${STAGE_ROOT:-/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals}"
JOBS="${JOBS:-3}"
FORCE="${FORCE:-}"
ONLY="${ONLY:-}"

# seed dataset name : staging dataset name
DATASETS=(
    "africa_crop_mask:africa_crop_mask_year_aligned"
    "canada_crops_coarse:canada_crops_coarse_year_aligned"
    "canada_crops_fine:canada_crops_fine_year_aligned"
    "descals:descals_year_aligned"
    "ethiopia_crops:ethiopia_crops_year_aligned"
    "glance:glance_year_aligned"
    "lcmap_lu:lcmap_lu_year_aligned"
    "us_trees:us_trees_year_aligned"
    "pastis_rslearn:pastis_year_aligned"
)

# Item group N of layer L is stored as the SIBLING directory "L.N", so the
# groups need their own pattern. tar matches the whole member name, hence the
# */layers/ prefix. The pastis "*_all" layers are deliberately NOT excluded --
# tessera_v2 inference needs them.
EXCLUDES=(
    # rslearn caches its eval-time window list here and does NOT invalidate on
    # windows being added or removed (train/dataset_index.py) -- only on a
    # config.json hash change or version bump. A copied index describes the
    # SOURCE dataset, and a stale one silently makes the eval run on a fraction
    # of the windows. Never copy it; it rebuilds on demand.
    --exclude='.rslearn_dataset_index'
    --exclude='*/layers/sentinel2'
    --exclude='*/layers/sentinel2.*'
    --exclude='*/layers/sentinel1'
    --exclude='*/layers/sentinel1.*'
    --exclude='*/layers/sentinel2_l2a_mo*'
    --exclude='*/layers/sentinel1_mo*'
)

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

count_windows() { find "$1/windows" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l | tr -d ' '; }

# Check a staging copy is complete and holds the layers we care about. Applied
# to skipped destinations too, so an interrupted copy (e.g. a killed rsync) is
# reported rather than silently accepted as done.
verify_stage() {
    local dst="$1" stage="$2" n_src="$3"
    local window layers n_dst
    n_dst=$(count_windows "$stage")
    window=$(find "$stage/windows" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | head -1)
    layers=$(ls "$window/layers" 2>/dev/null | tr '\n' ' ')

    if [[ "$n_dst" -ne "$n_src" ]]; then
        log "FAILED  $dst -- has $n_dst windows, seed has $n_src (incomplete; FORCE=1 to redo)"
        return 1
    fi
    if [[ "$layers" != *gse* ]]; then
        log "FAILED  $dst -- no gse layer in $window (baselines would need re-fetching)"
        return 1
    fi
    if [[ "$layers" == *sentinel2\ * || "$layers" == *sentinel2.* ]]; then
        log "FAILED  $dst -- old sentinel2 layers survived the excludes: $layers"
        return 1
    fi
    log "        $dst -- $n_dst windows, layers: $layers"
    return 0
}

copy_one() {
    local src="$1" dst="$2"
    local seed="$EVAL_ROOT/$src" stage="$STAGE_ROOT/$dst"

    if [[ ! -d "$seed" ]]; then
        log "MISSING $src -- no seed at $seed"
        return 1
    fi

    local n_src
    n_src=$(count_windows "$seed")

    if [[ -d "$stage" && -n "$(ls -A "$stage" 2>/dev/null)" && -z "$FORCE" ]]; then
        log "SKIP    $dst -- destination exists, verifying"
        verify_stage "$dst" "$stage" "$n_src"
        return $?
    fi

    log "START   $dst -- $n_src windows from $src"
    mkdir -p "$stage"
    local t0=$SECONDS
    if ! tar cf - -C "$seed" "${EXCLUDES[@]}" . | tar xf - -C "$stage"; then
        log "FAILED  $dst -- tar exited non-zero after $((SECONDS - t0))s"
        return 1
    fi
    log "DONE    $dst -- $((SECONDS - t0))s"
    verify_stage "$dst" "$stage" "$n_src"
    return $?
}

# Worker mode: one dataset, invoked by the xargs fan-out below.
if [[ "${1:-}" == "--one" ]]; then
    copy_one "${2%%:*}" "${2##*:}"
    exit $?
fi

log "seeding from $EVAL_ROOT -> $STAGE_ROOT (JOBS=$JOBS)"

selected=()
for entry in "${DATASETS[@]}"; do
    if [[ -z "$ONLY" || "${entry%%:*}" == "$ONLY" || "${entry##*:}" == "$ONLY" ]]; then
        selected+=("$entry")
    fi
done
if [[ ${#selected[@]} -eq 0 ]]; then
    log "no datasets matched ONLY=$ONLY"
    exit 1
fi

printf '%s\n' "${selected[@]}" \
    | xargs -P "$JOBS" -I{} "$0" --one {}
status=$?

log "----- summary -----"
for entry in "${selected[@]}"; do
    src="${entry%%:*}"
    dst="${entry##*:}"
    stage="$STAGE_ROOT/$dst"
    if [[ ! -d "$stage" || -z "$(ls -A "$stage" 2>/dev/null)" ]]; then
        log "  MISSING $dst"
        status=1
    elif verify_stage "$dst" "$stage" "$(count_windows "$EVAL_ROOT/$src")" >/dev/null 2>&1; then
        log "  ok      $dst  ($(du -sh "$stage" 2>/dev/null | cut -f1))"
    else
        log "  BAD     $dst  -- failed verification, re-run with FORCE=1 ONLY=$dst"
        status=1
    fi
done

if [[ $status -ne 0 ]]; then
    log "one or more copies failed -- re-run with ONLY=<name> after fixing"
fi
exit $status
