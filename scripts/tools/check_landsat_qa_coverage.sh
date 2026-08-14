#!/usr/bin/env bash
# Per-dataset landsat_qa coverage on the REGISTERED eval tree.
#
# Why: `l8_pixel_cloud_mask` falls back to LEAVING LANDSAT UNMASKED for any
# window whose `landsat_qa` layers are absent, warning only once per dataset
# instance (i.e. per dataloader worker), so a partial shortfall silently
# dilutes every _l8pixmask / _l8pixstrict score toward its unmasked sibling
# rather than failing. This counts the shortfall directly.
#
# The number that matters is `with_landsat` vs `with_qa`: a window carrying no
# Landsat imagery needs no QA. Anything below 100% there means the masked
# variants are a blend of masked and unmasked samples.
#
# Run on a weka host:  bash scripts/tools/check_landsat_qa_coverage.sh
set -uo pipefail

ROOT=${1:-/weka/dfive-default/olmoearth/eval_datasets}
DATASETS="africa_crop_mask canada_crops_coarse canada_crops_fine descals
          ethiopia_crops glance lcmap_lu us_trees pastis"

printf '%-24s %10s %14s %10s %9s  %s\n' dataset windows with_landsat with_qa pct note

for ds in $DATASETS; do
  d="$ROOT/${ds}_year_aligned/windows"
  [ -d "$d" ] || { printf '%-24s %10s\n' "$ds" "NO TREE"; continue; }

  # windows/<group>/<window>/layers/<layer>, so a window is depth 2 and a layer
  # dir depth 4. africa/ethiopia carry *_tessera_v2 fetch groups at 2x the
  # window count -- exclude them, evals pin the base group.
  tot=$(find "$d" -mindepth 2 -maxdepth 2 -type d ! -path '*tessera_v2*' | wc -l)
  l8=$(find "$d" -mindepth 4 -maxdepth 4 -type d -name 'landsat_mo*' ! -path '*tessera_v2*' \
       | awk -F/ '{print $(NF-2)}' | sort -u | wc -l)
  qa=$(find "$d" -mindepth 4 -maxdepth 4 -type d -name 'landsat_qa_mo*' ! -path '*tessera_v2*' \
       | awk -F/ '{print $(NF-2)}' | sort -u | wc -l)

  if [ "$l8" -gt 0 ]; then pct=$(( 100 * qa / l8 )); else pct=0; fi
  note=""
  [ "$pct" -lt 100 ] && note="DILUTED: $(( l8 - qa )) landsat windows run UNMASKED"
  printf '%-24s %10d %14d %10d %8d%%  %s\n' "$ds" "$tot" "$l8" "$qa" "$pct" "$note"
done

echo
echo "Per-window QA month histogram (12 = complete); anything under 12 is a"
echo "partially-masked window, which dilutes the same way:"
for ds in $DATASETS; do
  d="$ROOT/${ds}_year_aligned/windows"
  [ -d "$d" ] || continue
  h=$(find "$d" -mindepth 4 -maxdepth 4 -type d -name 'landsat_qa_mo*' ! -path '*tessera_v2*' \
      | awk -F/ '{c[$(NF-2)]++} END {for (w in c) print c[w]}' | sort -n | uniq -c \
      | awk '{printf "%s windows x %s months; ", $1, $2}')
  printf '  %-24s %s\n' "$ds" "${h:-none}"
done
