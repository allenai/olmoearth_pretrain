#!/usr/bin/env bash
# Repair the tessera_v2 fetch-group straggler windows on the *_year_aligned
# datasets: windows whose sentinel2_l2a_all layer never materialized because
# one scene's asset is missing from MPC blob storage (404 -- cataloged in
# STAC, gone from blob; re-preparing cannot fix it) and rslearn aborts the
# whole window-layer on any scene error. Measured on canada_crops_coarse
# 2026-08-14: every straggler traced to ONE dead granule
# (T11ULR 2021-07-06, B03_10m.tif missing).
#
# Per dataset, this walks:
#   1. scan the fetch group for windows with incomplete S2 (prepared item
#      groups without completed markers); S1 gaps are not scanned -- every
#      scan so far found S1 complete;
#   2. probe those windows' S2 asset hrefs (planetary_computer.sign first;
#      unsigned MPC URLs 404 unconditionally) and drop item groups with dead
#      assets from items.json. Guards: never touch a window that already has
#      completed S2 markers (dropping would shift group indices under them),
#      never drop >20% of a window's groups (that is not "a few bad scenes");
#      guarded windows are excluded and flagged for manual handling;
#   3. re-materialize just those windows, S2 only, low retries (leftover
#      404s would be deterministic; long backoffs only add hours).
#
# Dropping an unfetchable scene matches Tessera's own downloader behavior,
# so the baseline sees the same inputs their pipeline would produce.
#
# us_trees is excluded while its bulk materialize runs -- add it to DATASETS
# when done. Run in tmux: the glance scan alone is ~1-2h of metadata stats.
# Success = each dataset's Materialize Summary ends failed=0.
set -uo pipefail

ROOT_BASE=/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals
# Override the dataset list per run: REPAIR_DATASETS="us_trees" bash <this script>
# (space-separated base names). The default walks everything.
if [ -n "${REPAIR_DATASETS:-}" ]; then
  read -r -a DATASETS <<< "$REPAIR_DATASETS"
else
  DATASETS=(canada_crops_coarse canada_crops_fine descals lcmap_lu glance us_trees)
fi

for BASE in "${DATASETS[@]}"; do
  DS=${BASE}_year_aligned
  ROOT=$ROOT_BASE/$DS
  GROUP=${BASE}_tessera_v2

  echo "=== $DS: scan + dead-scene surgery ==="
  WINDOWS=$(DS_ROOT="$ROOT" FETCH_GROUP="$GROUP" python - <<'PYEOF'
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import planetary_computer
import requests
from rslearn.dataset import Dataset
from rslearn.dataset.window import WindowLayerData
from upath import UPath

LAYER = "sentinel2_l2a_all"
BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "SCL")
root = UPath(os.environ["DS_ROOT"])
group = os.environ["FETCH_GROUP"]


def log(*args):
    print(*args, file=sys.stderr, flush=True)


windows = Dataset(root).storage.get_windows(groups=[group], workers=16)
log(f"scanning {len(windows)} windows for incomplete {LAYER}...")


def incomplete(w):
    lds = w.load_layer_datas()
    if LAYER not in lds:
        return None
    want = len(lds[LAYER].serialized_item_groups)
    have = sum(1 for layer, _ in w.list_completed_layers() if layer == LAYER)
    return (w, lds, have) if have < want else None


targets = []
with ThreadPoolExecutor(max_workers=64) as pool:
    for result in pool.map(incomplete, windows):
        if result:
            targets.append(result)
log(f"{len(targets)} incomplete windows")


def urls_in(obj):
    if isinstance(obj, str):
        if obj.startswith("http") and any(b in obj for b in BANDS):
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from urls_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from urls_in(v)


def dead_urls(item_group):
    bad = []
    for url in urls_in(item_group):
        try:
            code = requests.head(planetary_computer.sign(url), timeout=30).status_code
        except Exception:
            code = -1
        if code != 200:
            bad.append((code, url))
    return bad


names = []
for w, lds, completed in targets:
    groups = lds[LAYER].serialized_item_groups
    with ThreadPoolExecutor(max_workers=32) as pool:
        dead = [gi for gi, bad in enumerate(pool.map(dead_urls, groups)) if bad]
    if dead:
        if completed:
            log(f"  {w.name}: {len(dead)} dead groups but {completed} completed "
                "markers -- NOT dropping (index shift); handle manually")
            continue
        if len(dead) > 0.2 * len(groups):
            log(f"  {w.name}: {len(dead)}/{len(groups)} groups dead (>20%) -- "
                "NOT dropping; investigate")
            continue
        keep = [gi for gi in range(len(groups)) if gi not in set(dead)]
        # group_time_ranges is PARALLEL to serialized_item_groups; their
        # length match is validated on every load, so both must be trimmed
        # with the same indices. Build the trimmed entry through the
        # CONSTRUCTOR, which enforces that invariant before anything is
        # written -- attribute mutation is what corrupted 35 windows on
        # 2026-08-14.
        gtr = lds[LAYER].group_time_ranges
        lds[LAYER] = WindowLayerData(
            layer_name=LAYER,
            serialized_item_groups=[groups[gi] for gi in keep],
            group_time_ranges=(
                [gtr[gi] for gi in keep] if gtr is not None else None
            ),
            materialized=lds[LAYER].materialized,
        )
        w.save_layer_datas(lds)
        w.load_layer_datas()  # round-trip check: raises if the write is bad
        log(f"  {w.name}: dropped {len(dead)}/{len(groups)} dead groups")
    else:
        log(f"  {w.name}: no dead assets (transient failure?) -- re-materializing")
    names.append(w.name)
print(" ".join(names))
PYEOF
  )
  if [ -z "$WINDOWS" ]; then
    echo "=== $DS: nothing to repair ==="
    continue
  fi
  N=$(wc -w <<< "$WINDOWS")
  echo "=== $DS: re-materializing $N windows ==="
  rslearn dataset materialize --root "$ROOT" \
    --config "$ROOT/config_tessera_v2_fetch.json" \
    --group "$GROUP" --enabled-layers sentinel2_l2a_all \
    --window $WINDOWS \
    --workers $(( N < 16 ? N : 16 )) --no-use-initial-job \
    --retry-max-attempts 2 --retry-backoff-seconds 5 --ignore-errors
done
echo "=== all datasets processed; check each Materialize Summary for failed=0 ==="
