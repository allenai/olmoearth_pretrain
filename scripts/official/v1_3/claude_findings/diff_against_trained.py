"""Diff a ported control recipe's config against the trained run's config.json.

Usage (from the repo root, with the repo venv)::

    python scripts/official/v1_3/claude_findings/diff_against_trained.py \
        scripts/official/v1_3/claude_findings/control_pixtgt.py \
        /weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt/v1_3_rc_ld1_pixtgt_pix512/step667200/config.json

The trained config's model section is first passed through
``convert_legacy_checkpoint.convert_model_config`` (the same renames the released
checkpoints get), and the remaining known API renames are applied to the other
sections (see ``KNOWN_RENAMES``). Every remaining difference is printed as
``path: trained -> ported``; the caller judges whether each is expected.
"""

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "official" / "v1_3"))

from convert_legacy_checkpoint import convert_model_config  # noqa: E402

from olmoearth_pretrain.internal.experiment import SubCmd, build_config  # noqa: E402

# Paths that identify a run or a machine rather than what is trained.
IGNORED_PREFIXES = (
    "run_name",
    "launch",
    "trainer.save_folder",
    "trainer.callbacks.wandb.name",
    "trainer.callbacks.wandb.group",
    "trainer.callbacks.wandb.tags",
    "data_loader.work_dir",
    "init_seed",
)

# Eval-task field renames (pre-rename name -> main name).
EVAL_TASK_RENAMES = {
    "eval_on_student_registers": "eval_on_student_location_tokens",
}


def _rename_eval_tasks(trained: dict) -> None:
    tasks = (
        trained.get("trainer", {})
        .get("callbacks", {})
        .get("downstream_evaluator", {})
        .get("tasks", {})
    )
    for task in tasks.values():
        for old, new in EVAL_TASK_RENAMES.items():
            if old in task:
                task[new] = task.pop(old)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        for i, v in enumerate(obj):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ported_recipe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_ported_config(script: Path, run_name: str) -> dict:
    """Build the recipe's full experiment config as a plain dict."""
    module = _load_module(script)
    common = module.build_common_components(
        str(script), SubCmd.dry_run, run_name, "ai2/jupiter", []
    )
    config = build_config(
        common=common,
        model_config_builder=module.build_model_config,
        dataset_config_builder=module.build_dataset_config,
        dataloader_config_builder=module.build_dataloader_config,
        trainer_config_builder=module.build_trainer_config,
        train_module_config_builder=module.build_train_module_config,
        overrides=[],
        visualize_config_builder=module.build_visualize_config,
    )
    return json.loads(json.dumps(config.as_config_dict(), default=str))


def normalize_trained(trained: dict) -> dict:
    """Apply the released-API renames to a trained run's config.json."""
    trained = copy.deepcopy(trained)
    trained["model"] = convert_model_config(trained["model"])
    _rename_eval_tasks(trained)
    return trained


def diff(trained: dict, ported: dict) -> list[str]:
    """``path: trained -> ported`` for every differing leaf."""
    a, b = _flatten(trained), _flatten(ported)
    lines = []
    for key in sorted(set(a) | set(b)):
        if key.startswith(IGNORED_PREFIXES):
            continue
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if va != vb:
            lines.append(f"{key}: {va!r} -> {vb!r}")
    return lines


def main() -> int:
    """Print the remaining differences; exit 1 if there are any."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path)
    parser.add_argument("trained_config", type=Path)
    args = parser.parse_args()
    trained = normalize_trained(json.loads(args.trained_config.read_text()))
    ported = build_ported_config(args.script, trained.get("run_name", "diff"))
    lines = diff(trained, ported)
    for line in lines:
        print(line)
    print(f"{len(lines)} differences")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main())
