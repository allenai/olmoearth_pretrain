"""Run an evaluation sweep for an arbitrary helios checkpoint.

e.g. python3 scripts/run_all_evals/full_eval_sweep.py --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0078125/step450000  --module_path=scripts/2025_06_26_dataset_percentage_experiments/latent_mim_all_data.py (extra args here e.g --model.decoder_config.depth=1)
"""

import argparse
import os
import subprocess  # nosec
from logging import getLogger
import uuid
from all_evals import EVAL_TASKS

from helios.evals.datasets.configs import dataset_to_config, get_eval_mode
from helios.evals.datasets.normalize import NormMethod
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


# TODO: Need to add filtering sentinel 1 data

def get_dino_v3_args():
    """Get the dino v3 arguments."""
    # DATASET ARGS + NORM METHOD ARGS
    dino_v3_args = dataset_args
    dino_v3_args += " " + " ".join(
        [
            f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.norm_method={NormMethod.NORM_YES_CLIP_3_STD_INT}"
            for task_name in EVAL_TASKS.keys()
        ]
    )
    return dino_v3_args



def main():
    """Run the full evaluation sweep or just the defaults."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, required=False, help="Checkpoint path"
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
    parser.add_argument(
        "--dino_v3",
        action="store_true",
        help="If set, use the dino v3 normalization settings",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="If set, use this as the  base run name",
    )
    args, extra_cli = parser.parse_known_args()

    cluster = args.cluster
    checkpoint_path = args.checkpoint_path
    module_path = args.module_path
    project_name = args.project_name
    extra = " " + " ".join(extra_cli) if extra_cli else ""

    # If we are using a torch hub model we should not specify a checkpoint path
    if checkpoint_path is None:
        logger.info(
            f"Running with module path {module_path} on cluster {cluster}"
        )
    else:
        logger.info(
            f"Running with checkpoint path {checkpoint_path} and module path {module_path} on cluster {cluster}"
        )
    if args.dry_run:
        sub_command = SubCmd.dry_run
    elif cluster == "local":
        sub_command = SubCmd.train
    else:
        sub_command = SubCmd.launch

    if project_name is None:
        project_name = "helios_in_loop_evals"

    if checkpoint_path is not None:
        parent_dir = os.path.basename(os.path.dirname(checkpoint_path))[:100]
        # the step number is the last part of the checkpoint path
        step_num = os.path.basename(checkpoint_path)
        base_run_name = f"{parent_dir}_{step_num}"
    else:
        if args.model_name is not None:
            base_run_name = args.model_name
        else:
            logger.warning("No model name provided, using random run name")
            base_run_name = str(uuid.uuid4())[:8]


    launch_command = "python3" if not sub_command == SubCmd.train else "torchrun"
    if checkpoint_path is not None:
        checkpoint_args = f"--trainer.load_path={checkpoint_path}"
    else:
        checkpoint_args = ""
    if args.defaults_only:
        # Just run with the first/default values
        lr = LP_LRs[0]
        norm_mode = Normalization_MODES[0]
        pooling_type = pooling_types[0]
        logger.info(
            f"Running defaults: {norm_mode} normalization, lr={lr}, pooling={pooling_type}"
        )
        run_name = f"{base_run_name}_defaults"

        if args.dino_v3:
            cmd_args = get_dino_v3_args()
        else:
            cmd_args = ""
        cmd = (
            f"TRAIN_SCRIPT_PATH={module_path} {launch_command} scripts/run_all_evals/all_evals.py "
            f"{sub_command} {run_name} {cluster} --launch.priority=high "
            # TODO: Make a debugging mode
            f"--launch.task_name=eval {checkpoint_args} --trainer.callbacks.wandb.enabled=False --trainer.callbacks.wandb.project={project_name}{extra} {cmd_args}"
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
            run_name = f"{base_run_name}_{norm_mode}_lr{lr}"
            cmd_args = lr_args.format(arg=lr)
            cmd_args += pooling_args.format(arg=pooling_type)

            if args.dino_v3:
                cmd_args += get_dino_v3_args()
            elif norm_mode == "dataset":
                cmd_args += dataset_args
            elif norm_mode == "helios":
                cmd_args += helios_args

            cmd = (
                f"TRAIN_SCRIPT_PATH={module_path} {launch_command} scripts/run_all_evals/all_evals.py "
                f"{sub_command} {run_name} {cluster} --launch.priority=high {cmd_args} "
                f"--launch.task_name=eval {checkpoint_args} --trainer.callbacks.wandb.project={project_name}{extra}"
            )
            logger.info(cmd)
            subprocess.run(cmd, shell=True, check=True)  # nosec


if __name__ == "__main__":
    main()
