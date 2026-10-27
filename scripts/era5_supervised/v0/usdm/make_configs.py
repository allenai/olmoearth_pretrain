"""Write the USDM task configs and their ``direct_registry.json`` entries.

One data-only YAML per target property (``configs/usdm_<target>_era5d.yaml``)
and registry entries ``usdm_<target>_eval`` (the ``eval`` group, for the
in-loop linear probes) plus ``usdm_<target>`` for the supervised targets (the
``all`` group). Regression target statistics come from the built dataset's
``meta/label_stats.json`` (train split of the entry's group); probe LRs from
``PROBE_LRS`` (the step-50k LR sweep), else ``DEFAULT_PROBE_LR``.

Usage (repo root)::

    python scripts/era5_supervised/v0/usdm/make_configs.py --label-stats <root>/meta/label_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

V0_DIR = Path(__file__).resolve().parents[1]
WEKA_PATH = "/weka/dfive-default/rslearn-eai/datasets/usdm/rslearn_dataset"

REGRESSION = {
    "sev": "current USDM severity (0 none, 1 D0 .. 5 D4; cell area average = cell DSCI / 100)",
    "sev_p8": "USDM severity 8 weeks after the anchor (forecast of the state)",
    "dpast4": "severity change over the 4 weeks before the anchor (recent evolution, inside the input window)",
    "dpast8": "severity change over the 8 weeks before the anchor (recent evolution, inside the input window)",
    "dpast13": "severity change over the 13 weeks before the anchor (recent evolution, inside the input window)",
    "dpast26": "severity change over the 26 weeks before the anchor (recent evolution, inside the input window)",
    "dfut4": "severity change over the 4 weeks after the anchor (forecast of the evolution)",
    "dfut8": "severity change over the 8 weeks after the anchor (forecast of the evolution)",
    "dfut13": "severity change over the 13 weeks after the anchor (forecast of the evolution)",
}
CLASSIFICATION = {
    "tpast8": (
        ["improve", "stable", "worsen"],
        "8-week past severity change as improve / stable / worsen (|change| >= 0.5)",
    ),
    "tfut8": (
        ["improve", "stable", "worsen"],
        "8-week coming severity change as improve / stable / worsen (|change| >= 0.5)",
    ),
    "drought": (["no", "yes"], "in drought: severity >= 2 (D1 or worse)"),
}
# Supervised-training flavours on the full 'all' group.
FULL = ("sev", "dpast8", "dpast26")
# Probe inputs (registry ``probe_pooling``): the recent-change targets read the
# mean of the last 4 tokens, which kept 1.3-2x the gain over the random-init
# floor and a 2-9x larger between-arm F than the pooled vector (dumped
# embeddings of the 20 v1.4-v1.6 arms x 3 seeds, 2026-10-27).
PROBE_POOLINGS = {
    "usdm_dpast8_eval": "last4",
    "usdm_dpast13_eval": "last4",
    "usdm_dpast26_eval": "last4",
}
# Plateau of the step-50k probe-LR sweep over the same arms (median val), for
# each task's own pooling.
PROBE_LRS = {
    "usdm_sev_eval": 0.3, "usdm_sev_p8_eval": 0.3, "usdm_drought_eval": 0.01,
    "usdm_dpast4_eval": 0.003, "usdm_dpast8_eval": 0.003, "usdm_dpast13_eval": 0.003,
    "usdm_dpast26_eval": 0.003, "usdm_dfut4_eval": 0.001, "usdm_dfut8_eval": 0.003,
    "usdm_dfut13_eval": 0.003, "usdm_tpast8_eval": 0.01, "usdm_tfut8_eval": 0.0001,
}  # fmt: skip
DEFAULT_PROBE_LR = 0.01

YAML = """\
# Data-only config for the USDM drought eval: {what}
# from a 448-day ERA5-Land daily history ending the day before the weekly map's
# valid date (the anchor). Only the `data:` section is consumed (rslearn_builder).
#
# Dataset: built by olmoearth_pretrain/dataset_creation/usdm (see its README).
# One window per (CONUS cell, map date): the cell's ERA5-Land series, 1x1 in a
# WGS84 0.1 deg/px window; label = a property of the vector `label` point.
# Splits cross 3 deg blocks with water years (tags written at creation).
# Registry entries pick the group (`eval` subset or `all`) and the split tags.
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
{task}
    batch_size: 64
    num_workers: 16
    # ERA5 normalization is handled by the OlmoEarth Normalizer
    # (Strategy.ZSCORE, computed.json); no rslearn Normalize transform here.
    train_config:
      groups:
      - eval
      tags:
        split: train
    val_config:
      groups:
      - eval
      tags:
        split: val
    test_config:
      groups:
      - eval
      tags:
        split: test
    predict_config:
      skip_targets: true
"""

REGRESSION_TASK = """\
      class_path: rslearn.train.tasks.regression.RegressionTask
      init_args:
        property_name: "{prop}"
        allow_invalid: false
        metrics:
        - rmse"""

CLASSIFICATION_TASK = """\
      class_path: rslearn.train.tasks.classification.ClassificationTask
      init_args:
        property_name: {prop}
        classes:
{classes}
        enable_f1_metric: true
        metric_kwargs:
          average: macro"""


def entry(target: str, group: str, stats: dict, notes: str) -> dict:
    """Registry entry of one target on one window group.

    Args:
        target: Property name.
        group: ``eval`` or ``all``.
        stats: ``label_stats.json`` content (may be empty).
        notes: Entry notes.

    Returns:
        The entry dict.
    """
    out: dict = {
        "groups": [group], "val_groups": [group], "test_groups": [group],
        "tags": {"split": "train"}, "val_tags": {"split": "val"}, "test_tags": {"split": "test"},
        "height_width": 1, "modality_layer_name": "era5_daily",
        "model_yaml_path": f"configs/usdm_{target}_era5d.yaml",
        "weka_path": WEKA_PATH, "notes": notes,
    }  # fmt: skip
    name = f"usdm_{target}_eval" if group == "eval" else f"usdm_{target}"
    out["probe_lr"] = PROBE_LRS.get(name, DEFAULT_PROBE_LR)
    if name in PROBE_POOLINGS:
        out["probe_pooling"] = PROBE_POOLINGS[name]
    if target in CLASSIFICATION:
        out.update(
            task_type="classification", num_classes=len(CLASSIFICATION[target][0])
        )
    else:
        out.update(
            task_type="regression",
            num_classes=1,
            label_extractor_name="default_regression",
        )
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
    for target, what in REGRESSION.items():
        task = REGRESSION_TASK.format(prop=target)
        (V0_DIR / "configs" / f"usdm_{target}_era5d.yaml").write_text(
            YAML.format(what=what, weka_path=WEKA_PATH, task=task)
        )
    for target, (classes, what) in CLASSIFICATION.items():
        # Quoted: YAML 1.1 reads bare yes / no as booleans.
        listed = "\n".join(f'        - "{c}"' for c in classes)
        task = CLASSIFICATION_TASK.format(prop=target, classes=listed)
        (V0_DIR / "configs" / f"usdm_{target}_era5d.yaml").write_text(
            YAML.format(what=what, weka_path=WEKA_PATH, task=task)
        )
    registry_path = V0_DIR / "direct_registry.json"
    registry = json.loads(registry_path.read_text())
    tasks = {k: v for k, v in registry["tasks"].items() if not k.startswith("usdm_")}
    base = (
        "USDM drought eval (weekly US Drought Monitor maps 2000-2026 on CONUS ERA5-Land cells): {what}, "
        "from the 448-day ERA5-Land history ending the day before the map date. "
    )
    split_note = (
        "Splits cross 3 deg blocks (70/15/15 % of cells within west/central/east) with water years "
        "(WY % 5: 2 test, 4 val), so no test window shares a cell or a season with training. "
        "Built by olmoearth_pretrain/dataset_creation/usdm."
    )
    descriptions = {**REGRESSION, **{k: v[1] for k, v in CLASSIFICATION.items()}}
    for target, what in descriptions.items():
        tasks[f"usdm_{target}_eval"] = entry(target, "eval", stats, base.format(what=what) + (
            "'eval' group: 4000 / 1500 / 1500 windows stratified by drought state x 8-week dynamics "
            "and spread over cells and water years. " + split_note))  # fmt: skip
    for target in FULL:
        tasks[f"usdm_{target}"] = entry(target, "all", stats, base.format(what=descriptions[target]) + (
            "Full 'all' group (every cell on a 12-week anchor grid) for supervised training. " + split_note))  # fmt: skip
    registry["tasks"] = dict(sorted(tasks.items()))
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {len(descriptions)} configs and {len(descriptions) + len(FULL)} registry entries"
    )


if __name__ == "__main__":
    main()
