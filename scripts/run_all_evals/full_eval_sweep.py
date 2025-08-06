"""Run an evaluation sweep for DINOv2."""

import argparse
import os
import subprocess  # nosec

from all_evals import EVAL_TASKS

from helios.evals.datasets.configs import dataset_to_config, get_eval_mode

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
Normalization_MODES = ["dataset", "helios"]


def create_linear_probe_arg(task_name: str) -> str:
    """Create a linear probe argument for a given task name."""
    initial_str = (
        f"--trainer.callbacks.downstream_evaluator.tasks.{task_name}.probe_lr="
    )
    return initial_str + "{lr}"


lr_args = " ".join(
    [
        create_linear_probe_arg(task_name)
        for task_name, task in EVAL_TASKS.items()
        if get_eval_mode(dataset_to_config(task.dataset).task_type) == "linear_probe"
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


def main():
    """Run the full evaluation sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Checkpoint path"
    )
    parser.add_argument(
        "--module_path", type=str, required=True, help="Path to module .py"
    )
    args = parser.parse_args()

    cluster = args.cluster
    checkpoint_path = args.checkpoint_path
    module_path = args.module_path
    print(
        f"Running with checkpoint path {checkpoint_path} and module path {module_path} on cluster {cluster}"
    )
    for lr in LP_LRs:
        for norm_mode in Normalization_MODES:
            print(f"Running with {norm_mode} normalization and {lr} learning rate")

            parent_dir = os.path.basename(os.path.dirname(checkpoint_path))
            base_run_name = f"test_{parent_dir}_{norm_mode}_lr{lr}"
            run_name = base_run_name[:100]
            cmd_args = lr_args.format(lr=lr)
            if norm_mode == "dataset":
                cmd_args += dataset_args
            elif norm_mode == "helios":
                cmd_args += helios_args
            cmd = (
                f"TRAIN_SCRIPT_PATH={module_path} python3 scripts/run_all_evals/all_evals.py "
                f"launch {run_name} {cluster} --launch.priority=high {cmd_args} "
                f"--launch.task_name=eval --trainer.load_path={checkpoint_path}"
            )
            print(cmd)
            subprocess.run(cmd, shell=True)  # nosec
            break
        break


if __name__ == "__main__":
    main()
