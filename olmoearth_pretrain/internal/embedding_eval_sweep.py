"""Sweep the embedding-product evals (EMBEDDING_EVAL_TASKS).

The third sweep type next to the KNN/LP sweep (full_eval_sweep.py) and the
finetuning sweep (full_eval_sweep_finetune.py). It runs only the per-pixel
ws16/ps1 embedding-convention tasks (EMBEDDING_EVALS=1 selects them in
all_evals.py) and differs from the full sweep in that:

- normalization is held fixed: pretraining stats for OlmoEarth, and the
  precomputed embedding products are consumed exactly as stored (NO_NORM);
- only the probe LR is swept, and only for the linear-probe tasks — the KNN
  twins have no hyperparameters and run once in their own job;
- only OlmoEarth checkpoints and the precomputed embedding products (aef,
  tessera_precomputed) are supported.

e.g.
  # OlmoEarth checkpoint
  python -m olmoearth_pretrain.internal.embedding_eval_sweep \
      --cluster=ai2/saturn-cirrascale \
      --checkpoint_path=/weka/.../step370000 \
      --module_path=scripts/.../nano.py

  # Precomputed baselines
  python -m olmoearth_pretrain.internal.embedding_eval_sweep \
      --cluster=ai2/saturn-cirrascale --model=aef

Pass --task_names=<name>[,<name>...] to run a subset of EMBEDDING_EVAL_TASKS.
"""

import argparse
import json
import os
import subprocess  # nosec
import uuid
from logging import getLogger

from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.embedding_transforms import EmbeddingNormalization
from olmoearth_pretrain.evals.linear_probe import ProbeInputNorm
from olmoearth_pretrain.evals.models import BaselineModelName, get_launch_script_path
from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS
from olmoearth_pretrain.internal.constants import EVAL_LAUNCH_PATH, EVAL_WANDB_PROJECT
from olmoearth_pretrain.internal.experiment import SubCmd
from olmoearth_pretrain.internal.full_eval_sweep import (
    LAUNCH_OVERRIDES,
    MAX_DURATION_OVERRIDE,
    PRECOMPUTED_MODEL_TO_MODALITY,
    QUANTIZE_AT_EVAL_MODALITIES,
    QUANTIZE_SCHEME_BY_MODALITY,
    LP_LRs,
    _get_checkpoint_args,
    _get_sub_command,
    parse_task_names,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import EvalMode

logger = getLogger(__name__)

SUPPORTED_BASELINES = tuple(PRECOMPUTED_MODEL_TO_MODALITY)

LP_TASK_NAMES = [
    name
    for name, task in EMBEDDING_EVAL_TASKS.items()
    if task.eval_mode == EvalMode.LINEAR_PROBE
]
KNN_TASK_NAMES = [
    name
    for name, task in EMBEDDING_EVAL_TASKS.items()
    if task.eval_mode == EvalMode.KNN
]


def _task_arg(task_name: str, field_name: str, value: object) -> str:
    """Build one per-task downstream-evaluator override."""
    return (
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}"
        f".{field_name}={value}"
    )


def _capable_tasks(task_names: list[str], modality: str) -> list[str]:
    """Task names whose dataset carries the given precomputed modality.

    A precomputed baseline reads only the embedding modality, so tasks that
    differ solely in imagery input_modalities (e.g. the S2-only and S1+S2
    PASTIS variants) collapse to the same run — keep one task per dataset.
    """
    capable = []
    seen_datasets = set()
    for name in task_names:
        dataset = EMBEDDING_EVAL_TASKS[name].dataset
        if dataset in seen_datasets:
            continue
        if modality in dataset_to_config(dataset).supported_modalities:
            capable.append(name)
            seen_datasets.add(dataset)
    return capable


def _model_task_names(
    model: BaselineModelName | None,
) -> tuple[list[str], list[str]]:
    """(LP, KNN) task names the given model can run; fail fast on zero tasks."""
    if model is None:
        return LP_TASK_NAMES, KNN_TASK_NAMES
    modality, product = PRECOMPUTED_MODEL_TO_MODALITY[model]
    lp = _capable_tasks(LP_TASK_NAMES, modality)
    knn = _capable_tasks(KNN_TASK_NAMES, modality)
    skipped = sorted(set(LP_TASK_NAMES + KNN_TASK_NAMES) - set(lp + knn))
    if skipped:
        logger.warning(
            f"--model={model}: skipping tasks whose dataset does not carry the "
            f"'{modality}' modality: {', '.join(skipped)}"
        )
    if not lp and not knn:
        raise SystemExit(
            f"No embedding eval task's dataset supports the precomputed "
            f"'{modality}' modality, so --model={model} would run zero tasks. "
            f"Bake the embeddings into the eval datasets first (embedding "
            f"materializer / pastis_processor --embedding_products {product}) "
            f"and list '{modality}' in the dataset's supported_modalities."
        )
    return lp, knn


def _filter_selected_tasks(
    task_names_arg: str | None, lp_tasks: list[str], knn_tasks: list[str]
) -> tuple[list[str], list[str]]:
    """Restrict (LP, KNN) task names to a --task_names selection, if given."""
    selected = parse_task_names(task_names_arg)
    if not selected:
        return lp_tasks, knn_tasks
    unknown = sorted(set(selected) - set(EMBEDDING_EVAL_TASKS))
    if unknown:
        raise SystemExit(
            f"Unknown embedding eval task(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(EMBEDDING_EVAL_TASKS)}."
        )
    lp = [name for name in lp_tasks if name in selected]
    knn = [name for name in knn_tasks if name in selected]
    if not lp and not knn:
        raise SystemExit(
            "--task_names selected zero runnable tasks (for precomputed "
            "baselines, tasks whose dataset does not carry the embedding "
            "modality are skipped)."
        )
    return lp, knn


def _model_args(
    model: BaselineModelName | None,
    task_names: list[str],
    quantization: str | None = None,
) -> str:
    """Per-task and trainer args pinning each model's normalization/quantization.

    The task configs already carry these values, but the sweep pins them
    explicitly so the convention cannot drift: OlmoEarth's forward-pass
    embeddings are always int8 round-tripped (scored as an embedding product,
    pretraining-stats normalization), while the precomputed products are
    consumed exactly as stored — no re-normalization, reading the embedding
    modality instead of imagery.

    Quantization is per-product rather than blanket-off, because "already int8
    at source" is true of the downloaded products but NOT of tessera_v2, which
    we bake ourselves in float32. See ``QUANTIZE_AT_EVAL_MODALITIES`` in
    full_eval_sweep for the rule and its caveat.

    ``quantization`` overrides the OlmoEarth side of that convention, so the
    round trip itself can be measured rather than assumed: "none" scores the
    float embeddings (the ceiling -- every arm to date has round-tripped, so the
    cost of quantizing has never actually been observed), and "tessera" swaps
    AEF's power scheme for Tessera's linear per-vector one, which is clip-free
    by construction and isolates the companding curve from the value range.
    Baselines are unaffected: they ship at a fixed precision and are scored at
    it. None keeps AEF's power scheme, which is what every existing number used.
    """
    if model is None:
        quantize = (quantization or "aef_power") != "none"
        args = [" --trainer.no_checkpoints=False"]
        for task_name in task_names:
            args.append(_task_arg(task_name, "norm_stats_from_pretrained", "True"))
            args.append(_task_arg(task_name, "quantize_embeddings", str(quantize)))
            if quantize and quantization == "tessera":
                args.append(
                    _task_arg(
                        task_name,
                        "quantization_scheme",
                        "QuantizationScheme.TESSERA_PER_VECTOR",
                    )
                )
        return " ".join(args)
    modality, _ = PRECOMPUTED_MODEL_TO_MODALITY[model]
    quantize = modality in QUANTIZE_AT_EVAL_MODALITIES
    args = [" --trainer.no_checkpoints=True"]
    for task_name in task_names:
        args.append(_task_arg(task_name, "norm_stats_from_pretrained", "False"))
        args.append(_task_arg(task_name, "norm_method", "NormMethod.NO_NORM"))
        args.append(_task_arg(task_name, "input_modalities", f"[{modality}]"))
        args.append(_task_arg(task_name, "quantize_embeddings", str(quantize)))
        if quantize:
            args.append(
                _task_arg(
                    task_name,
                    "quantization_scheme",
                    # StrEnum interpolates to its VALUE ("tessera_per_vector"),
                    # which OmegaConf rejects -- it parses enums by member NAME.
                    f"QuantizationScheme.{QUANTIZE_SCHEME_BY_MODALITY[modality].name}",
                )
            )
    return " ".join(args)


def _normalization_args(args: argparse.Namespace, task_names: list[str]) -> str:
    """Per-task embedding-normalization overrides.

    Applies to OlmoEarth checkpoints and the precomputed baselines alike: the
    question "does this embedding space need normalizing before a probe reads
    it" is as fair to ask of AEF/Tessera as of us, and holding the arm's
    transform identical across models keeps the comparison honest.

    Read through ``getattr`` like ``priority``/``window_size``: the cluster-side
    submitters build this namespace by hand.
    """
    normalization = getattr(args, "embedding_normalization", None)
    if not normalization or normalization == "none":
        return ""
    stats_path = getattr(args, "embedding_norm_stats_path", None)
    overrides = []
    for name in task_names:
        overrides.append(
            _task_arg(
                name,
                "embedding_normalization",
                f"EmbeddingNormalization.{normalization.upper()}",
            )
        )
        if stats_path is not None:
            overrides.append(_task_arg(name, "embedding_norm_stats_path", stats_path))
    return " " + " ".join(overrides)


def _probe_input_norm_args(args: argparse.Namespace, task_names: list[str]) -> str:
    """Per-LP-task overrides for the probe's input norm.

    LP only: KNN has no probe, and it L2-normalizes internally regardless, so a
    KNN job would be an exact duplicate of the default arm under any setting
    here.
    """
    probe_input_norm = getattr(args, "probe_input_norm", None)
    if not probe_input_norm or probe_input_norm == "batchnorm":
        return ""
    return " " + " ".join(
        _task_arg(
            name, "probe_input_norm", f"ProbeInputNorm.{probe_input_norm.upper()}"
        )
        for name in task_names
    )


def _tasks_to_run_arg(task_names: list[str]) -> str:
    """Restrict the evaluator to the given tasks (compact JSON; see full sweep)."""
    return (
        " --trainer.callbacks.downstream_evaluator.tasks_to_run="
        f"'{json.dumps(task_names, separators=(',', ':'))}'"
    )


def _window_size_args(window_size: int | None, task_names: list[str]) -> str:
    """Per-task window_size overrides for windowed-sampling tasks.

    Only tasks whose config already sets window_size are overridden. Tiled
    (tile_samples) datasets require window_size to divide the stored sample
    size (128 for pastis_rslearn).
    """
    if window_size is None:
        return ""
    overrides = [
        _task_arg(name, "window_size", window_size)
        for name in task_names
        if EMBEDDING_EVAL_TASKS[name].window_size is not None
    ]
    if not overrides:
        return ""
    return " " + " ".join(overrides)


def _balanced_trial_args(args: argparse.Namespace, task_names: list[str]) -> str:
    """Per-KNN-task overrides for the AEF balanced trials.

    Only the KNN tasks carry a ``balanced_trial`` config (see
    ``_aef_ps1_task``), so these must not be applied to the LP tasks -- there is
    no nested config there to override.

    Read through ``getattr`` like ``priority``/``window_size`` above: the
    cluster-side submitters in olmoearth_plus_cropharvest build this namespace
    by hand, so a new field must not become a required attribute.
    """
    overrides = []
    max_folds = getattr(args, "balanced_trial_max_folds", None)
    draw_pool = getattr(args, "balanced_trial_draw_pool", None)
    eval_split = getattr(args, "balanced_trial_eval_split", None)
    disabled = getattr(args, "no_balanced_trials", False)
    for name in task_names:
        if EMBEDDING_EVAL_TASKS[name].balanced_trial is None:
            continue
        if disabled:
            overrides.append(_task_arg(name, "balanced_trial.enabled", "False"))
        if max_folds is not None:
            overrides.append(_task_arg(name, "balanced_trial.max_folds", max_folds))
        if draw_pool is not None:
            pool = "[" + ",".join(draw_pool.split(",")) + "]"
            overrides.append(_task_arg(name, "balanced_trial.draw_pool", pool))
        if eval_split is not None:
            overrides.append(_task_arg(name, "balanced_trial.eval_split", eval_split))
    if not overrides:
        return ""
    return " " + " ".join(overrides)


def _select_best_val_args(task_names: list[str]) -> str:
    """Per-LP-task early-stopping args (best epoch by primary val metric)."""
    return " " + " ".join(
        f"{_task_arg(name, 'select_best_by_primary_metric', 'True')} "
        f"{_task_arg(name, 'linear_probe_eval_interval', '5')}"
        for name in task_names
    )


def _base_run_name(args: argparse.Namespace) -> str:
    """Base run name from --model_name, checkpoint path, or model."""
    if args.model_name is not None:
        return args.model_name
    if args.checkpoint_path is not None:
        parent_dir = os.path.basename(os.path.dirname(args.checkpoint_path))[:100]
        step_num = os.path.basename(args.checkpoint_path)
        return f"{parent_dir}_{step_num}"
    return f"{args.model}_{str(uuid.uuid4())[:4]}"


def build_commands(args: argparse.Namespace, extra_cli: list[str]) -> list[str]:
    """Build one command per LP learning rate plus one KNN command."""
    model: BaselineModelName | None = args.model
    if model is None and args.module_path is None:
        raise ValueError("Provide --module_path (and --checkpoint_path) or --model")

    lp_tasks, knn_tasks = _model_task_names(model)
    lp_tasks, knn_tasks = _filter_selected_tasks(args.task_names, lp_tasks, knn_tasks)

    module_path = args.module_path if model is None else get_launch_script_path(model)
    sub_command = _get_sub_command(args)
    launch_command = "torchrun" if sub_command == SubCmd.evaluate else "python3"
    launch_overrides = LAUNCH_OVERRIDES if sub_command == SubCmd.launch_evaluate else ""
    priority = getattr(args, "priority", None)
    if priority:
        launch_overrides = launch_overrides.replace(
            "--launch.priority=high", f"--launch.priority={priority}"
        )
    checkpoint_args = _get_checkpoint_args(args.checkpoint_path)
    project_name = args.project_name or EVAL_WANDB_PROJECT
    extra = " " + " ".join(extra_cli) if extra_cli else ""
    base_run_name = _base_run_name(args) + "_emb"
    quantization = getattr(args, "quantization", None)
    if quantization and quantization != "aef_power":
        # The arm changes what the probe consumes, so it must be separable in
        # W&B from the runs that used AEF's scheme.
        base_run_name += f"_q{quantization}"
    window_size = getattr(args, "window_size", None)
    if window_size is not None:
        base_run_name += f"_ws{window_size}"
    normalization = getattr(args, "embedding_normalization", None)
    if normalization and normalization != "none":
        # The arm belongs in the run name: every normalization arm evaluates
        # the same checkpoint on the same tasks, so the W&B groups are only
        # separable by name.
        base_run_name += f"_norm{normalization.replace('_', '')}"

    env_prefix = f"TRAIN_SCRIPT_PATH={module_path} EMBEDDING_EVALS=1"
    if getattr(args, "landsat_reflectance", False):
        # The radiometry is a property of the arm, not of the checkpoint path,
        # so name it: running the DN checkpoint under this flag (deliberately
        # or by mistake) must not collide with its own baseline run. Kept to
        # two letters because these run names are already near Beaker's limit.
        base_run_name += "_rf"
        env_prefix += " LANDSAT_REFLECTANCE=1"
    common = (
        f"{env_prefix} {launch_command} {EVAL_LAUNCH_PATH} "
        f"{sub_command} {{run_name}} {args.cluster} {launch_overrides} "
        f"{checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra}"
        f" {MAX_DURATION_OVERRIDE}"
    )

    probe_input_norm = getattr(args, "probe_input_norm", None)
    lp_run_name = base_run_name
    if probe_input_norm and probe_input_norm != "batchnorm":
        # LP-only suffix: the arm changes the probe, so it must not silently
        # rename an otherwise-identical KNN run.
        lp_run_name += f"_probe{probe_input_norm.replace('_', '')}"

    commands = []
    if lp_tasks:
        lp_model_args = _model_args(model, lp_tasks, quantization)
        for lr in LP_LRs:
            cmd = common.format(run_name=f"{lp_run_name}_lr{lr}")
            cmd += lp_model_args
            cmd += _window_size_args(window_size, lp_tasks)
            cmd += _normalization_args(args, lp_tasks)
            cmd += _probe_input_norm_args(args, lp_tasks)
            cmd += " " + " ".join(_task_arg(name, "probe_lr", lr) for name in lp_tasks)
            if args.select_best_val:
                cmd += _select_best_val_args(lp_tasks)
            cmd += _tasks_to_run_arg(lp_tasks)
            commands.append(cmd)
    if knn_tasks:
        cmd = common.format(run_name=f"{base_run_name}_knn")
        cmd += _model_args(model, knn_tasks, quantization)
        cmd += _window_size_args(window_size, knn_tasks)
        cmd += _normalization_args(args, knn_tasks)
        cmd += _balanced_trial_args(args, knn_tasks)
        cmd += _tasks_to_run_arg(knn_tasks)
        commands.append(cmd)
    return commands


def _parse_model_arg(value: str) -> BaselineModelName:
    """Parse --model, restricted to the precomputed embedding products."""
    try:
        model = BaselineModelName(value)
    except ValueError:
        model = None
    if model not in SUPPORTED_BASELINES:
        raise argparse.ArgumentTypeError(
            f"Invalid model: {value}. The embedding sweep supports "
            f"{[m.value for m in SUPPORTED_BASELINES]} (or omit --model and pass "
            "--checkpoint_path/--module_path for an OlmoEarth checkpoint)."
        )
    return model


# Beaker rejects experiment names over 128 characters, and it appends
# "-<task_name>-<8 hex>" to the run name we pass. These run names are built by
# concatenating the checkpoint dir, the step, and one suffix per arm, so they
# sit close to the limit; catching it here turns a slow 400 from the Beaker API
# (one per command, after the config has already been resolved) into an
# immediate local error naming the offending run.
BEAKER_MAX_NAME_LEN = 128
_BEAKER_NAME_OVERHEAD = len("-evaluate-") + 8


def _check_run_names_fit_beaker(commands: list[str]) -> None:
    """Fail before launch if any run name would exceed Beaker's name limit."""
    too_long = []
    for cmd in commands:
        # "<subcmd> <run_name> <cluster>" -- the run name follows the subcommand.
        # Split on the launch path as it actually appears in the command
        # (".../internal/all_evals.py"); matching a bare " all_evals.py " never
        # fires, which silently turns this whole check into a no-op.
        _, sep, tail = cmd.partition(f"{EVAL_LAUNCH_PATH} ")
        if not sep:
            continue
        parts = tail.split()
        if len(parts) < 2:
            continue
        run_name = parts[1]
        total = len(run_name) + _BEAKER_NAME_OVERHEAD
        if total > BEAKER_MAX_NAME_LEN:
            too_long.append((run_name, total))
    if too_long:
        detail = "\n".join(
            f"  {name} -> {total} chars (over by {total - BEAKER_MAX_NAME_LEN})"
            for name, total in too_long
        )
        raise ValueError(
            f"{len(too_long)} run name(s) exceed Beaker's {BEAKER_MAX_NAME_LEN}-char "
            f"limit once it appends its task/id suffix:\n{detail}\n"
            "Shorten the checkpoint directory name or the arm suffixes."
        )


def main() -> None:
    """Run the embedding-product eval sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="OlmoEarth checkpoint path (omit for precomputed baselines)",
    )
    parser.add_argument(
        "--module_path",
        type=str,
        default=None,
        help="Path to the OlmoEarth model-config module .py",
    )
    parser.add_argument(
        "--model",
        type=_parse_model_arg,
        default=None,
        help=f"Precomputed baseline: {[m.value for m in SUPPORTED_BASELINES]}. "
        "Omit to evaluate an OlmoEarth checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="If set, use this as the base run name",
    )
    parser.add_argument(
        "--project_name", type=str, required=False, help="Wandb project name"
    )
    parser.add_argument(
        "--task_names",
        type=str,
        default=None,
        help="Comma-separated subset of EMBEDDING_EVAL_TASKS to run (default: all)",
    )
    parser.add_argument(
        "--select_best_val",
        action="store_true",
        help="Select the best test epoch by the primary validation metric",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default=None,
        help="Beaker job priority (default high), e.g. urgent",
    )
    parser.add_argument(
        "--landsat_reflectance",
        action="store_true",
        help=(
            "For checkpoints pretrained on the Landsat-reflectance h5: convert "
            "eval Landsat DN to TOA reflectance at load time and use the "
            "reflectance-scale norm stats. Applies to Landsat-bearing tasks "
            "only, leaving the S2/S1+S2 tasks as a shared control. Requires a "
            "landsat_calibration.json in each dataset root "
            "(build_landsat_calibration_sidecar.py); the loader refuses to run "
            "without it rather than feeding DN to reflectance-scale stats."
        ),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help=(
            "Override window_size on every selected task. Must divide the "
            "stored sample size for tiled datasets (128 for pastis_rslearn). "
            "NOTE: ws16/8/4/1 variants already run by default "
            "(EMBEDDING_EVAL_WINDOW_SIZES) — this override forces ALL "
            "selected tasks to one size, so combine it with --task_names to "
            "avoid running duplicates."
        ),
    )
    parser.add_argument(
        "--embedding_normalization",
        type=str,
        default=None,
        choices=[m.value for m in EmbeddingNormalization],
        help=(
            "Normalize embeddings before the int8 round-trip and the probe: "
            "none (default, as the model emits them), l2 (per-embedding, AEF's "
            "convention), center, center_l2, zscore. The fitted modes (center*, "
            "zscore) take their stats from each task's TRAIN split unless "
            "--embedding_norm_stats_path is given -- per-dataset stats measure "
            "whether geometry is the problem; fixed stats are what a global run "
            "could actually deploy. Tags the run name with the arm."
        ),
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["aef_power", "tessera", "none"],
        help=(
            "How OlmoEarth embeddings are quantized before the probe: aef_power "
            "(default, the scheme every existing number used), tessera (linear, "
            "per-vector scale, clip-free by construction -- isolates the "
            "companding curve), or none (float32; the ceiling, since every arm "
            "so far has round-tripped). Baselines are unaffected: they are "
            "scored at the precision they ship. Tags the run names."
        ),
    )
    parser.add_argument(
        "--probe_input_norm",
        type=str,
        default=None,
        choices=[m.value for m in ProbeInputNorm],
        help=(
            "What sits in front of the linear probe: batchnorm (default, "
            "BatchNorm1d on classification tasks and nothing on segmentation) "
            "or none (classification scored exactly like the dense probes, so "
            "embedding geometry reaches the weights and no batch statistics are "
            "involved). LP jobs only; tags the LP run names with the arm."
        ),
    )
    parser.add_argument(
        "--embedding_norm_stats_path",
        type=str,
        default=None,
        help=(
            "Fixed normalization constants for the fitted modes, written by "
            "scripts/tools/fit_embedding_norm_stats.py. Per-model, not "
            "per-dataset: the same file is used for every task."
        ),
    )
    parser.add_argument(
        "--no_balanced_trials",
        action="store_true",
        help=(
            "Skip the AEF balanced trials that the KNN job runs by default "
            "(class-balanced draw from the pooled splits, scored on the "
            "remainder, over AEF's k draws)"
        ),
    )
    parser.add_argument(
        "--balanced_trial_max_folds",
        type=int,
        default=None,
        help=(
            "Cap the balanced trials' fold count (default: AEF's "
            "k = 1000 / (2 * log10(least class)), which is 200-500)"
        ),
    )
    parser.add_argument(
        "--balanced_trial_draw_pool",
        type=str,
        default=None,
        help=(
            "Comma-separated splits the balanced draw is taken from "
            "(default train,val,test -- AEF pools everything). Use "
            "'train,val' with --balanced_trial_eval_split=test to keep the "
            "test split unspent."
        ),
    )
    parser.add_argument(
        "--balanced_trial_eval_split",
        type=str,
        default=None,
        help=(
            "Split the trials report on: 'remainder' (default, AEF's "
            "structure) or a split held out of --balanced_trial_draw_pool, "
            "which holds the eval rows fixed across folds"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print the configs that would be run",
    )
    args, extra_cli = parser.parse_known_args()

    commands_to_run = build_commands(args, extra_cli)
    _check_run_names_fit_beaker(commands_to_run)
    logger.info(f"Running {len(commands_to_run)} commands")
    for cmd in commands_to_run:
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)  # nosec
    logger.info(f"Finished running {len(commands_to_run)} commands")


if __name__ == "__main__":
    main()
