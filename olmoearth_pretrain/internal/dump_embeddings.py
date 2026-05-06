r"""Launch embedding-dump runs that re-extract paper Table 2 embeddings.

For each (foundation model, downstream task) pair we re-create the val-best
hyperparameters from the paper sweep (loaded from a JSON written by
``scripts/tools/get_max_eval_metrics_from_wandb.py``), run the existing eval
pipeline with ``eval_mode=embedding_dump`` per task, and save embeddings +
labels for the train / valid / test splits.

Per-task knobs that are read from the JSON: ``pooling_type``,
``norm_stats_from_pretrained``. ``probe_lr`` is ignored (no probe runs in
EMBEDDING_DUMP mode). Per-model knobs (``norm_method``,
``--model.use_pretrained_normalizer``) come from the same model-specific
helpers used by ``full_eval_sweep.py``.

Example (external FM)::

    python3 olmoearth_pretrain/internal/dump_embeddings.py \\
        --model=galileo --size=base \\
        --settings_json=data/max_eval_settings/max_eval_settings_per_group_merged.json \\
        --settings_group=galileo_base \\
        --save_embeddings_dir=/weka/dfive-default/olmoearth_pretrain/paper_embeddings \\
        --cluster=ai2/saturn-cirrascale

Example (OlmoEarth)::

    python3 olmoearth_pretrain/internal/dump_embeddings.py \\
        --module_path=scripts/official/base.py \\
        --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \\
        --settings_json=data/max_eval_settings/base_settings.json \\
        --settings_group=phase2.0_base_lr0.0001_wd0.02 \\
        --save_embeddings_dir=/weka/dfive-default/olmoearth_pretrain/paper_embeddings \\
        --cluster=ai2/saturn-cirrascale
"""

import argparse
import json
import subprocess  # nosec
from logging import getLogger

from olmoearth_pretrain.evals.models import (
    get_launch_script_path,
)
from olmoearth_pretrain.internal.constants import EVAL_LAUNCH_PATH, EVAL_WANDB_PROJECT
from olmoearth_pretrain.internal.experiment import SubCmd
from olmoearth_pretrain.internal.full_eval_sweep import (
    _get_load_checkpoints_args,
    _get_model_size_args,
    _get_model_specific_args,
    _get_normalization_args,
    _parse_model_arg,
)

logger = getLogger(__name__)


def _per_task_overrides(
    task_name: str,
    settings: dict,
    save_embeddings_dir: str,
    embedding_dump_dtype: str,
) -> list[str]:
    """Build the dotlist overrides for a single task in EMBEDDING_DUMP mode."""
    pooling_type = settings.get("pooling_type", "mean")
    norm_from_pretrained = settings.get("norm_stats_from_pretrained", False)
    base = f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}"
    # olmo-core's dotlist parser validates against the enum *name*
    # (EMBEDDING_DUMP), not the value ("embedding_dump").
    return [
        f"{base}.eval_mode=EMBEDDING_DUMP",
        f"{base}.save_embeddings_dir={save_embeddings_dir}",
        f"{base}.embedding_dump_dtype={embedding_dump_dtype}",
        f"{base}.pooling_type={pooling_type}",
        f"{base}.norm_stats_from_pretrained={norm_from_pretrained}",
    ]


def _task_norm_mode(entry: dict) -> bool:
    """Return True if the val-best run was the ``pre_trained`` sweep arm.

    Prefer the ``norm_mode`` field (added by ``enrich_eval_settings_json.py``)
    because for a few models (Galileo, Tessera, ...) the per-task helper
    hardcodes ``norm_stats_from_pretrained=False`` regardless of which sweep
    arm produced the best result. ``norm_stats_from_pretrained`` is a correct
    proxy for OlmoEarth (whose ``get_olmoearth_args`` directly maps the mode
    to that flag).
    """
    settings = entry["settings"]
    if "norm_mode" in settings:
        return settings["norm_mode"] == "pre_trained"
    return bool(settings.get("norm_stats_from_pretrained", False))


def _partition_by_norm_mode(
    group_settings: dict[str, dict],
) -> dict[bool, list[str]]:
    """Group task names by their best norm mode (pretrained vs dataset).

    Why split the launch: ``--model.use_pretrained_normalizer`` and the matching
    ``norm_method`` are coupled to that flag, and the model arg is global, not
    per-task. To reproduce the paper's per-task best, we launch once per mode
    with the right tasks active.
    """
    by_mode: dict[bool, list[str]] = {True: [], False: []}
    for task, entry in group_settings.items():
        by_mode[_task_norm_mode(entry)].append(task)
    return {k: sorted(v) for k, v in by_mode.items() if v}


def _load_settings_group(settings_json: str, settings_group: str) -> dict[str, dict]:
    """Load settings for a single model-group from the merged or per-size JSON."""
    with open(settings_json) as f:
        data = json.load(f)
    if settings_group not in data:
        raise KeyError(
            f"Group '{settings_group}' not in {settings_json}. "
            f"Available groups: {sorted(data.keys())}"
        )
    return data[settings_group]


def _build_run_name(args: argparse.Namespace) -> str:
    """Build a run name for this dump."""
    if args.run_name:
        return args.run_name
    parts = ["dump", args.settings_group]
    if args.size:
        parts.append(args.size)
    return "_".join(parts).replace("/", "_")


def _get_module_path(args: argparse.Namespace) -> str:
    """Resolve the launch-script path for the model."""
    if args.module_path is not None:
        return args.module_path
    if args.model is None:
        raise ValueError("Either --module_path or --model must be provided")
    return get_launch_script_path(args.model)


def _build_one_command(
    args: argparse.Namespace,
    group_settings: dict[str, dict],
    tasks: list[str],
    norm_mode_pretrained: bool,
    run_name_suffix: str,
) -> str:
    """Build a single launch command for one (group, norm-mode) partition."""
    cmd_args: list[str] = []
    for task_name in tasks:
        cmd_args.extend(
            _per_task_overrides(
                task_name=task_name,
                settings=group_settings[task_name]["settings"],
                save_embeddings_dir=args.save_embeddings_dir,
                embedding_dump_dtype=args.embedding_dump_dtype,
            )
        )
    # Single-quote the JSON so the shell doesn't expand the brackets.
    cmd_args.append(
        f"--trainer.callbacks.downstream_evaluator.tasks_to_run='{json.dumps(tasks)}'"
    )
    cmd_args.append("--trainer.callbacks.downstream_evaluator.run_on_test=True")

    model_specific = _get_model_specific_args(args.model)
    # The norm_method / use_pretrained_normalizer pair is set globally by
    # ``_get_normalization_args``; we pick the mode that matches this partition
    # so it's coherent with the per-task ``norm_stats_from_pretrained`` flags.
    norm_args = _get_normalization_args(
        args.model,
        "pre_trained" if norm_mode_pretrained else "dataset",
    )
    size_args = _get_model_size_args(args.model, args.size)
    no_ckpt_args = _get_load_checkpoints_args(args.model)

    sub_command = (
        SubCmd.dry_run
        if args.dry_run
        else (SubCmd.train if args.cluster == "local" else SubCmd.launch)
    )
    launch_command = "torchrun" if sub_command == SubCmd.train else "python3"
    module_path = _get_module_path(args)
    run_name = _build_run_name(args) + run_name_suffix

    checkpoint_args = (
        f"--trainer.load_path={args.checkpoint_path}"
        if args.checkpoint_path is not None
        else ""
    )
    launch_overrides_parts: list[str] = []
    if sub_command == SubCmd.launch:
        launch_overrides_parts.extend(
            [
                f"--launch.priority={args.priority}",
                f"--launch.num_gpus={args.num_gpus}",
                "--launch.task_name=embed_dump",
            ]
        )
        if args.launch_clusters:
            cluster_list = [c.strip() for c in args.launch_clusters.split(",")]
            launch_overrides_parts.append(
                f"--launch.clusters=[{','.join(cluster_list)}]"
            )
    launch_overrides = " ".join(launch_overrides_parts)

    project = args.project_name or EVAL_WANDB_PROJECT
    return " ".join(
        [
            f"TRAIN_SCRIPT_PATH={module_path}",
            launch_command,
            EVAL_LAUNCH_PATH,
            sub_command,
            run_name,
            args.cluster,
            launch_overrides,
            checkpoint_args,
            f"--trainer.callbacks.wandb.project={project}",
            model_specific,
            norm_args,
            size_args,
            no_ckpt_args,
            *cmd_args,
        ]
    )


def _build_commands(args: argparse.Namespace) -> list[str]:
    """Build one command per (group, norm_stats_from_pretrained) partition."""
    group_settings = _load_settings_group(args.settings_json, args.settings_group)
    if args.exclude_tasks:
        excluded = set(t.strip() for t in args.exclude_tasks.split(",") if t.strip())
        keep = {t: v for t, v in group_settings.items() if t not in excluded}
        dropped = sorted(set(group_settings) & excluded)
        if dropped:
            logger.info(f"Dropping {len(dropped)} excluded tasks: {dropped}")
        group_settings = keep
    if not group_settings:
        logger.warning(
            f"No tasks left for group '{args.settings_group}' after applying excludes."
        )
        return []
    by_mode = _partition_by_norm_mode(group_settings)
    logger.info(
        f"Group '{args.settings_group}' partitions: "
        + ", ".join(f"pretrained={m}: {len(t)}" for m, t in by_mode.items())
    )

    cmds: list[str] = []
    # If only one partition exists, use no suffix to keep the run name clean.
    use_suffix = len(by_mode) > 1
    for mode, tasks in by_mode.items():
        suffix = ("_pretrained" if mode else "_dataset") if use_suffix else ""
        cmds.append(
            _build_one_command(
                args=args,
                group_settings=group_settings,
                tasks=tasks,
                norm_mode_pretrained=mode,
                run_name_suffix=suffix,
            )
        )
    return cmds


def main() -> None:
    """Build and (optionally) run the embedding-dump command."""
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True, help="Beaker cluster or 'local'")
    p.add_argument(
        "--settings_json",
        required=True,
        help="Path to per-group eval-settings JSON (e.g. "
        "data/max_eval_settings/max_eval_settings_per_group_merged.json or "
        "data/max_eval_settings/base_settings.json).",
    )
    p.add_argument(
        "--settings_group",
        required=True,
        help="Top-level key in the JSON (e.g. 'galileo_base', "
        "'phase2.0_base_lr0.0001_wd0.02').",
    )
    p.add_argument(
        "--save_embeddings_dir",
        required=True,
        help="Directory where embeddings will be written. Each task lands in "
        "{save_embeddings_dir}/{task_name}/{train,valid,test}.pt",
    )
    p.add_argument(
        "--embedding_dump_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Dtype for saved embedding tensors. Default matches the autocast "
        "precision used at extraction time.",
    )
    p.add_argument(
        "--model",
        type=_parse_model_arg,
        default=None,
        help="Baseline FM name (e.g. galileo, croma). Mutually exclusive with "
        "--module_path.",
    )
    p.add_argument(
        "--size",
        default=None,
        help="Model size for FMs that have multiple sizes (e.g. base, large).",
    )
    p.add_argument(
        "--module_path",
        default=None,
        help="Path to a script.py (e.g. scripts/official/base.py) for OlmoEarth "
        "checkpoints. Mutually exclusive with --model.",
    )
    p.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint dir to load (only used for OlmoEarth runs).",
    )
    p.add_argument(
        "--run_name",
        default=None,
        help="Override the auto-generated run name.",
    )
    p.add_argument(
        "--project_name",
        default=None,
        help="Wandb project (defaults to the constant in internal/constants.py).",
    )
    p.add_argument(
        "--exclude_tasks",
        default=None,
        help="Comma-separated list of task names to drop from the JSON group "
        "before launching (e.g. pastis128_sentinel1,pastis128_sentinel2).",
    )
    p.add_argument(
        "--priority",
        default="normal",
        choices=["low", "normal", "high", "urgent"],
        help="Beaker job priority. Ignored when --cluster=local.",
    )
    p.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="--launch.num_gpus passed to Beaker.",
    )
    p.add_argument(
        "--launch_clusters",
        default=None,
        help="Comma-separated Beaker clusters (e.g. ai2/jupiter,ai2/saturn). "
        "If set, becomes --launch.clusters=[...]; the positional --cluster is "
        "still required (Beaker uses the override to pick).",
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument(
        "--print_only",
        action="store_true",
        help="Print the commands and exit without running them.",
    )

    args = p.parse_args()
    if (args.model is None) == (args.module_path is None):
        raise ValueError("Exactly one of --model or --module_path must be set.")

    cmds = _build_commands(args)
    for cmd in cmds:
        print(cmd)
    if args.print_only:
        return
    failures = 0
    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, check=False)  # nosec
        if result.returncode != 0:
            print(f"  -> partition failed (rc={result.returncode}); continuing")
            failures += 1
    if failures:
        # Surface non-zero so the outer fan-out also records this group as failed.
        raise SystemExit(f"{failures} of {len(cmds)} partitions failed")


if __name__ == "__main__":
    main()
