"""Drop known-bad scenes from prepared (not yet materialized) allcap layers.

Some source scenes are corrupt at the provider (e.g. truncated ~14KB GeoTIFF stubs
in the Planetary Computer landsat-c2-l2 mirror). rslearn materializes a layer
all-or-nothing per window, so one bad scene loses every capture of that layer for
the window. This script removes the offending item groups from items.json so a
re-run of materialize succeeds with the remaining captures.

Scene ids can be given directly (--scene-ids) or extracted from a materialize log
(--from-log): any URL on an Error line yields the scene/product id path component.

Usage:
    python -m olmoearth_pretrain.dataset_creation.prune_bad_scenes \
        --ds_path /weka/.../osm_allcaptures_pilot1k --group res_10 \
        --layers landsat --from-log materialize.log
"""

from __future__ import annotations

import argparse
import re

from rslearn.data_sources import Item
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from upath import UPath

# Product-id-shaped path components in error URLs, e.g.
# LC08_L2SP_195021_20221105_20221115_02_T1 or S2A_MSIL2A_20220710T165911_...
SCENE_ID_PATTERN = re.compile(r"/((?:LC|LE|LT|S1|S2)[A-Z0-9]{0,2}_[A-Za-z0-9_]+)/")


def scene_ids_from_log(log_path: str) -> set[str]:
    """Extract scene/product ids from Error lines of a materialize log."""
    scene_ids = set()
    with open(log_path) as f:
        for line in f:
            if "Error" not in line:
                continue
            for match in SCENE_ID_PATTERN.finditer(line):
                scene_ids.add(match.group(1))
    return scene_ids


def matches_scene(item_name: str, scene_id: str) -> bool:
    """Check whether an item name refers to the given scene/product id.

    Item names may omit components of the on-disk product id (e.g. Landsat STAC ids
    drop the processing date: LC08_L2SP_026046_20221105_02_T1 vs the URL's
    LC08_L2SP_026046_20221105_20221115_02_T1), so we accept the item name's
    underscore tokens appearing as an ordered subsequence of the scene id's tokens.
    """
    if scene_id in item_name or item_name in scene_id:
        return True
    item_tokens = item_name.split("_")
    scene_tokens = scene_id.split("_")
    if len(item_tokens) < 4:
        return False
    it = iter(scene_tokens)
    return all(token in it for token in item_tokens)


def prune_window(
    window: Window, layers: list[str], scene_ids: set[str]
) -> dict[str, int]:
    """Remove item groups whose item name matches a bad scene id; returns drops."""
    layer_datas = window.load_layer_datas()
    dropped: dict[str, int] = {}
    changed = False
    for layer_name in layers:
        if layer_name not in layer_datas:
            continue
        layer_data = layer_datas[layer_name]
        if layer_data.materialized:
            continue
        kept_groups = []
        kept_time_ranges = []
        num_dropped = 0
        for group_idx, group in enumerate(layer_data.serialized_item_groups):
            item = Item.deserialize(group[0])
            if any(matches_scene(item.name, scene_id) for scene_id in scene_ids):
                num_dropped += 1
                continue
            kept_groups.append(group)
            if layer_data.group_time_ranges is not None:
                kept_time_ranges.append(layer_data.group_time_ranges[group_idx])
        if num_dropped:
            layer_datas[layer_name] = WindowLayerData(
                layer_name=layer_name,
                serialized_item_groups=kept_groups,
                group_time_ranges=(
                    kept_time_ranges
                    if layer_data.group_time_ranges is not None
                    else None
                ),
                materialized=layer_data.materialized,
            )
            dropped[layer_name] = num_dropped
            changed = True
    if changed:
        window.save_layer_datas(layer_datas)
    return dropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drop known-bad scenes from prepared allcap layers"
    )
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--group", type=str, default="res_10")
    parser.add_argument(
        "--layers", type=str, default="landsat,sentinel1,sentinel2_l2a"
    )
    parser.add_argument("--scene-ids", nargs="*", default=[])
    parser.add_argument("--from-log", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    scene_ids = set(args.scene_ids)
    if args.from_log:
        scene_ids |= scene_ids_from_log(args.from_log)
    if not scene_ids:
        parser.error("no scene ids given (use --scene-ids and/or --from-log)")
    print(f"pruning {len(scene_ids)} bad scenes: {sorted(scene_ids)}")

    layers = [layer.strip() for layer in args.layers.split(",")]
    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(groups=[args.group], workers=args.workers)

    total = 0
    for window in windows:
        dropped = prune_window(window, layers, scene_ids)
        for layer_name, count in dropped.items():
            print(f"{window.name} {layer_name}: dropped {count}")
            total += count
    print(f"dropped {total} item groups total")
