#!/usr/bin/env bash
# Per-dataset landsat_qa coverage on the REGISTERED eval tree.
#
# Why: `l8_pixel_cloud_mask` leaves Landsat UNMASKED for any window missing its
# `landsat_qa` layers, warning only once per dataset instance (per dataloader
# worker), so a partial shortfall silently dilutes every _l8pixmask /
# _l8pixstrict score toward its unmasked sibling rather than failing. The
# warning count cannot measure it; this counts windows.
#
# The number that matters is with_qa / with_landsat: a window carrying no
# Landsat needs no QA. Under 100% means those windows ran unmasked.
#
#   bash scripts/tools/check_landsat_qa_coverage.sh            # sampled, seconds
#   bash scripts/tools/check_landsat_qa_coverage.sh --full     # exact, slow
#
# Sampled mode stats ~300 random windows per dataset instead of walking ~7M
# directory entries. It cannot give an exact percentage, but it answers the
# question actually being asked -- is coverage complete, or is there a real
# shortfall -- since any gap above ~1% is near-certain to show up in 300 draws.
set -uo pipefail

FULL=0
[ "${1:-}" = "--full" ] && { FULL=1; shift; }
ROOT=${1:-/weka/dfive-default/olmoearth/eval_datasets}
DATASETS="africa_crop_mask canada_crops_coarse canada_crops_fine descals
          ethiopia_crops glance lcmap_lu us_trees pastis"
SAMPLE=300

printf '%-24s %12s %10s %8s  %s\n' dataset with_landsat with_qa pct note

for ds in $DATASETS; do
  d="$ROOT/${ds}_year_aligned/windows"
  [ -d "$d" ] || { printf '%-24s %12s\n' "$ds" "NO TREE"; continue; }

  if [ "$FULL" = 1 ]; then
    # ONE traversal; awk tallies both layer families. windows/<group>/<window>/
    # layers/<layer>, so the window is $(NF-2). africa/ethiopia carry
    # *_tessera_v2 fetch groups at 2x the count -- evals pin the base group.
    read -r l8 qa part < <(
      find "$d" -mindepth 4 -maxdepth 4 -type d \
           \( -name 'landsat_mo*' -o -name 'landsat_qa_mo*' \) ! -path '*tessera_v2*' \
      | awk -F/ '
          { w = $(NF-2) }
          $NF ~ /^landsat_qa_mo/ { qa[w]++; next }
          $NF ~ /^landsat_mo/    { l8[w]++ }
          END {
            for (w in l8) n++
            for (w in qa) { m++; if (qa[w] < 12) p++ }
            print n+0, m+0, p+0
          }')
  else
    # Sample windows and stat only their layers dir.
    read -r l8 qa part < <(
      find "$d" -mindepth 2 -maxdepth 2 -type d ! -path '*tessera_v2*' \
      | shuf -n "$SAMPLE" \
      | while read -r w; do
          nl=$(ls -1 "$w/layers" 2>/dev/null | grep -c '^landsat_mo')
          nq=$(ls -1 "$w/layers" 2>/dev/null | grep -c '^landsat_qa_mo')
          echo "$nl $nq"
        done \
      | awk '{ if ($1>0) n++; if ($2>0) { m++; if ($2<12) p++ } } END { print n+0, m+0, p+0 }')
  fi

  if [ "${l8:-0}" -gt 0 ]; then pct=$(( 100 * qa / l8 )); else pct=0; fi
  note=""
  [ "$pct" -lt 100 ] && note="DILUTED: $(( l8 - qa )) landsat windows UNMASKED"
  if [ "${part:-0}" -gt 0 ]; then
    [ -n "$note" ] && note="$note; "
    # Expected rather than alarming: Landsat is ragged, so a window can hold
    # fewer than 12 QA months legitimately. Only a large tail here would dilute.
    note="${note}${part} window(s) with <12 QA months"
  fi
  printf '%-24s %12d %10d %7d%%  %s\n' "$ds" "$l8" "$qa" "$pct" "$note"
done

if [ "$FULL" != 1 ]; then
  printf '\nSampled %s windows/dataset. 100%% here means any shortfall is under ~1%%;\n' "$SAMPLE"
  printf 're-run with --full for exact counts.\n'
fi
