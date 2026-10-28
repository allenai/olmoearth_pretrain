"""Write the snow task configs and their ``direct_registry.json`` entries.

One data-only YAML per target property (``configs/snow_<target>_era5d.yaml``)
and registry entries ``snow_<target>_<family>_eval`` (the family's
``eval_<family>`` window group, for the in-loop linear probes) plus
``snow_<target>_<family>`` for the supervised targets (``all_<family>``).
Regression target statistics come from the built dataset's
``meta/label_stats.json`` (train split of the entry's group); probe LRs from
``PROBE_LRS`` (the step-50k LR sweep), else ``DEFAULT_PROBE_LR``.

Usage (repo root)::

    python scripts/era5_supervised/v0/snow/make_configs.py --label-stats <root>/meta/label_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

V0_DIR = Path(__file__).resolve().parents[1]
WEKA_PATH = "/weka/dfive-default/rslearn-eai/datasets/snow/rslearn_dataset"

FAMILIES = {
    "l0": "window ending on the cell's climatological melt-out date",
    "l30": "window ending 30 days before the cell's climatological melt-out date",
    "l60": "window ending 60 days before the cell's climatological melt-out date",
    "l90": "window ending 90 days before the cell's climatological melt-out date",
    "p30": "window ending 30 days after the cell's climatological melt-out date",
    "p60": "window ending 60 days after the cell's climatological melt-out date",
    "post": "window ending on the last day of the water year (whole season seen)",
}
# The lead / lag families register only the anomaly: their window ends a fixed
# offset from the cell's climatological melt-out, so the last tokens' day of year
# encodes that climatology and an absolute date target (``sdd_dowy``) reads it
# off the timestamps (random-init R2 0.86-0.89); ``days_to_melt`` is the anomaly
# plus the lead. Both stay as window properties for diagnostics.
TARGETS = {
    "forecast": {
        "sdd_anom": "snow disappearance date anomaly (days; SDD minus the cell's leave-one-year-out mean)",
    },
    "post": {
        "sdd_anom": "snow disappearance date anomaly (days; SDD minus the cell's leave-one-year-out mean)",
        "scd_anom": "snow-covered days anomaly (cell-mean days with snow minus the leave-one-year-out mean)",
        "sad_anom": "snow appearance date anomaly (days; windows whose season starts inside the water year)",
        "season_anom": "snow season length anomaly (days; windows whose season starts inside the water year)",
        "sdd_dowy": "snow disappearance date (day of water year; absolute, climate-dominated)",
        "scd": "snow-covered days (absolute, climate-dominated)",
    },
}
OPTIONAL = ("sad_anom", "season_anom")
# Supervised-training flavours on the full 'all_<family>' groups.
FULL: tuple[tuple[str, str], ...] = (("sdd_anom", "l0"), ("sdd_anom", "p30"))
# Probe inputs (registry ``probe_pooling``): the lead / lag anomalies are read
# from the window's end (target lab on the 20 v1.4-v1.6 arms x 3 seeds,
# 2026-10-28, l0: ridge val R2 floor 0.13 -> 1601 0.15 and F 1.0 on the pooled
# vector, 0.25 -> 0.29 and F 4.0 on the last 4 tokens; p30 alike).
PROBE_POOLINGS: dict[str, str] = {
    f"snow_sdd_anom_{family}_eval": "last4"
    for family in ("l0", "l30", "l60", "l90", "p30", "p60")
}
# Plateau of the step-50k probe-LR sweep (median val over the 20 arms x 3 seeds,
# emb_probe on dumped embeddings, 2026-10-28), per task with its pooling.
PROBE_LRS: dict[str, float] = {
    "snow_sdd_anom_l0_eval": 3e-3, "snow_sdd_anom_l30_eval": 1e-3,
    "snow_sdd_anom_p30_eval": 3e-3, "snow_sdd_anom_p60_eval": 3e-3,
    "snow_sdd_anom_post_eval": 3e-3, "snow_scd_anom_post_eval": 3e-3,
    "snow_scd_post_eval": 0.1, "snow_sdd_dowy_post_eval": 0.1,
}  # fmt: skip
DEFAULT_PROBE_LR = 0.01
# Eval window group per task when not the family's ``eval_<family>``.
EVAL_GROUPS: dict[str, str] = {}

YAML = """\
# Data-only config for the snow melt-timing eval: {what}
# from a 448-day ERA5-Land daily history ending at the window family's anchor
# (registry entry). Only the `data:` section is consumed (rslearn_builder).
#
# Dataset: built by olmoearth_pretrain/dataset_creation/snow (see its README).
# One window per (cell, water year, family): the cell's ERA5-Land series, 1x1 in a
# WGS84 0.1 deg/px window; label = a property of the vector `label` point.
# Splits cross 3 deg blocks with water years (tags written at creation).
# Registry entries pick the group (`eval_<family>` subset or `all_<family>`) and tags.
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: {weka_path}
    inputs:
      era5_daily:
        data_type: raster
        layers:
        - era5_daily
        use_all_bands_in_order_of_band_set_idx: 0
        passthrough: true
        dtype: FLOAT32
      targets:
        data_type: vector
        layers:
        - label
        is_target: true
    task:
      class_path: rslearn.train.tasks.regression.RegressionTask
      init_args:
        property_name: "{prop}"
        allow_invalid: false
        metrics:
        - rmse
    batch_size: 64
    num_workers: 16
    # ERA5 normalization is handled by the OlmoEarth Normalizer
    # (Strategy.ZSCORE, computed.json); no rslearn Normalize transform here.
    train_config:
      groups:
      - eval_l30
      tags:
        split: train
    val_config:
      groups:
      - eval_l30
      tags:
        split: val
    test_config:
      groups:
      - eval_l30
      tags:
        split: test
    predict_config:
      skip_targets: true
"""


def entry(target: str, family: str, group: str, stats: dict, notes: str) -> dict:
    """Registry entry of one target on one window group.

    Args:
        target: Property name.
        family: Window family.
        group: ``eval_<family>`` or ``all_<family>``.
        stats: ``label_stats.json`` content (may be empty).
        notes: Entry notes.

    Returns:
        The entry dict.
    """
    tags = {"has_sad": ""} if target in OPTIONAL else {}
    out: dict = {
        "groups": [group], "val_groups": [group], "test_groups": [group],
        "tags": {**tags, "split": "train"}, "val_tags": {**tags, "split": "val"},
        "test_tags": {**tags, "split": "test"},
        "height_width": 1, "modality_layer_name": "era5_daily",
        "model_yaml_path": f"configs/snow_{target}_era5d.yaml",
        "weka_path": WEKA_PATH, "notes": notes,
        "task_type": "regression", "num_classes": 1, "label_extractor_name": "default_regression",
    }  # fmt: skip
    name = (
        f"snow_{target}_{family}_eval"
        if group.startswith("eval")
        else f"snow_{target}_{family}"
    )
    out["probe_lr"] = PROBE_LRS.get(name, DEFAULT_PROBE_LR)
    if name in PROBE_POOLINGS:
        out["probe_pooling"] = PROBE_POOLINGS[name]
    train = stats.get(group, {}).get("train", {}).get(target)
    if train:
        out.update(
            target_mean=round(train["mean"], 6), target_std=round(train["std"], 6)
        )
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label-stats", default=None)
    args = ap.parse_args()
    stats = json.loads(Path(args.label_stats).read_text()) if args.label_stats else {}
    descriptions = {**TARGETS["forecast"], **TARGETS["post"]}
    for target, what in descriptions.items():
        (V0_DIR / "configs" / f"snow_{target}_era5d.yaml").write_text(
            YAML.format(what=what, weka_path=WEKA_PATH, prop=target)
        )
    registry_path = V0_DIR / "direct_registry.json"
    registry = json.loads(registry_path.read_text())
    tasks = {k: v for k, v in registry["tasks"].items() if not k.startswith("snow_")}
    base = (
        "Snow melt-timing eval (MODIS/Terra MOD10A2 8-day snow cover 2000-2026 on seasonal-snow "
        "ERA5-Land cells, per water year): {what}, from the 448-day ERA5-Land history of the {fam}. "
    )
    split_note = (
        "Splits cross 3 deg blocks (70/15/15 % of cells within 5 regions) with water years "
        "(WY % 5: 2 test, 4 val), so no test window shares a cell or a season with training. "
        "Built by olmoearth_pretrain/dataset_creation/snow."
    )
    n = 0
    for family, fam in FAMILIES.items():
        kind = "post" if family == "post" else "forecast"
        for target, what in TARGETS[kind].items():
            name = f"snow_{target}_{family}_eval"
            tasks[name] = entry(
                target, family, EVAL_GROUPS.get(name, f"eval_{family}"), stats, base.format(what=what, fam=fam) + (
                    f"'eval_{family}' group: 4000 / 1500 / 1500 (cell, water year) keys shared by "
                    "every family, spread over cells. " + split_note))  # fmt: skip
            n += 1
    for target, family in FULL:
        kind = "post" if family == "post" else "forecast"
        tasks[f"snow_{target}_{family}"] = entry(
            target, family, f"all_{family}", stats,
            base.format(what=TARGETS[kind][target], fam=FAMILIES[family]) + (
                f"Full 'all_{family}' group for supervised training. " + split_note))  # fmt: skip
        n += 1
    registry["tasks"] = dict(sorted(tasks.items()))
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(descriptions)} configs and {n} registry entries")


if __name__ == "__main__":
    main()
