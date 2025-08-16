"""Run an evaluation sweep for an arbitrary helios checkpoint.

e.g. python3 scripts/run_all_evals/full_eval_sweep.py --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0078125/step450000  --module_path=scripts/2025_06_26_dataset_percentage_experiments/latent_mim_all_data.py (extra args here e.g --model.decoder_config.depth=1)
"""

import argparse
import os
import subprocess  # nosec
from logging import getLogger

from all_evals import EVAL_TASKS

from helios.evals.datasets.configs import dataset_to_config, get_eval_mode
from helios.internal.experiment import SubCmd
from helios.nn.flexihelios import PoolingType

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
Normalization_MODES = ["dataset", "helios"]
pooling_types = [PoolingType.MAX, PoolingType.MEAN]

logger = getLogger(__name__)


def create_linear_probe_arg(task_name: str, field_name: str) -> str:
    """Create a linear probe argument for a given task name."""
    initial_str = (
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.{field_name}="
    )
    return initial_str + "{arg}"


lr_args = " ".join(
    [
        create_linear_probe_arg(task_name, "probe_lr")
        for task_name, task in EVAL_TASKS.items()
        if get_eval_mode(dataset_to_config(task.dataset).task_type) == "linear_probe"
    ]
)

pooling_args = " ".join(
    [" "]
    + [
        create_linear_probe_arg(task_name, "pooling_type")
        for task_name, task in EVAL_TASKS.items()
    ]
)

dataset_args = " ".join(
    [" "]
    + [
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_stats_from_pretrained=False"
        for task_name in EVAL_TASKS.keys()
    ]
)

helios_args = " ".join(
    [""]
    + [
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_stats_from_pretrained=True"
        for task_name in EVAL_TASKS.keys()
    ]
)


def loop_through_params():
    """Yield a dict of the hps we are sweeping over."""
    for lr in LP_LRs:
        for norm_mode in Normalization_MODES:
            for pooling_type in pooling_types:
                yield {
                    "lr": lr,
                    "norm_mode": norm_mode,
                    "pooling_type": pooling_type,
                }


def main():
    """Run the full evaluation sweep or just the defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Checkpoint path"
    )
    parser.add_argument(
        "--module_path", type=str, required=True, help="Path to module .py"
    )
    parser.add_argument(
        "--project_name", type=str, required=False, help="Wandb project name"
    )
    parser.add_argument(
        "--defaults_only",
        action="store_true",
        help="If set, only run with default values (no sweep)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print the commands that would be run",
    )
    args, extra_cli = parser.parse_known_args()

    cluster = args.cluster
    checkpoint_path = args.checkpoint_path
    module_path = args.module_path
    project_name = args.project_name
    extra = " " + " ".join(extra_cli) if extra_cli else ""
    logger.info(
        f"Running with checkpoint path {checkpoint_path} and module path {module_path} on cluster {cluster}"
    )
    sub_command = SubCmd.launch if not args.dry_run else SubCmd.dry_run
    if project_name is None:
        project_name = "helios_in_loop_evals"

    parent_dir = os.path.basename(os.path.dirname(checkpoint_path))[:100]
    # the step number is the last part of the checkpoint path
    step_num = os.path.basename(checkpoint_path)
    if args.defaults_only:
        # Just run with the first/default values
        lr = LP_LRs[0]
        norm_mode = Normalization_MODES[0]
        pooling_type = pooling_types[0]
        logger.info(
            f"Running defaults: {norm_mode} normalization, lr={lr}, pooling={pooling_type}"
        )
        base_run_name = f"{parent_dir}_{step_num}_defaults"
        run_name = base_run_name

        cmd = (
            f"TRAIN_SCRIPT_PATH={module_path} python3 scripts/run_all_evals/all_evals.py "
            f"{sub_command} {run_name} {cluster} --launch.priority=high "
            f"--launch.task_name=eval --trainer.load_path={checkpoint_path} --trainer.callbacks.wandb.project={project_name}{extra}"
        )
        logger.info(cmd)
        subprocess.run(cmd, shell=True, check=True)  # nosec
    else:
        hp_params = loop_through_params()
        for params in hp_params:
            lr = params["lr"]
            norm_mode = params["norm_mode"]
            pooling_type = params["pooling_type"]
            logger.info(
                f"Running with {norm_mode} normalization and {lr} learning rate"
            )
            base_run_name = f"{parent_dir}_{step_num}_{norm_mode}_lr{lr}"
            run_name = base_run_name
            cmd_args = lr_args.format(arg=lr)
            cmd_args += pooling_args.format(arg=pooling_type)
            if norm_mode == "dataset":
                cmd_args += dataset_args
            elif norm_mode == "helios":
                cmd_args += helios_args

            cmd = (
                f"TRAIN_SCRIPT_PATH={module_path} python3 scripts/run_all_evals/all_evals.py "
                f"{sub_command} {run_name} {cluster} --launch.priority=high {cmd_args} "
                f"--launch.task_name=eval --trainer.load_path={checkpoint_path} --trainer.callbacks.wandb.project={project_name}{extra}"
            )
            logger.info(cmd)
            subprocess.run(cmd, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
