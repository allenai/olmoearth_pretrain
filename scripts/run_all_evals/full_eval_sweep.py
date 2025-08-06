"""Run an evaluation sweep for an arbitrary helios checkpoint.

e.g. python3 scripts/run_all_evals/full_eval_sweep.py --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0078125/step450000  --module_path=scripts/2025_06_26_dataset_percentage_experiments/latent_mim_all_data.py
"""

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
    parser.add_argument(
        "--project_name", type=str, required=False, help="Wandb project name"
    )
    args = parser.parse_args()

    cluster = args.cluster
    checkpoint_path = args.checkpoint_path
    module_path = args.module_path
    project_name = args.project_name
    print(
        f"Running with checkpoint path {checkpoint_path} and module path {module_path} on cluster {cluster}"
    )
    for lr in LP_LRs:
        for norm_mode in Normalization_MODES:
            print(f"Running with {norm_mode} normalization and {lr} learning rate")
            # crop the parent dirname if needed
            parent_dir = os.path.basename(os.path.dirname(checkpoint_path))[:100]
            base_run_name = f"{parent_dir}_{norm_mode}_lr{lr}"
            run_name = base_run_name
            cmd_args = lr_args.format(lr=lr)
            if norm_mode == "dataset":
                cmd_args += dataset_args
            elif norm_mode == "helios":
                cmd_args += helios_args

            if project_name is None:
                project_name = "helios_in_loop_evals"
            cmd = (
                f"TRAIN_SCRIPT_PATH={module_path} python3 scripts/run_all_evals/all_evals.py "
                f"launch {run_name} {cluster} --launch.priority=high {cmd_args} "
                f"--launch.task_name=eval --trainer.load_path={checkpoint_path} --trainer.callbacks.wandb.project={project_name}"
            )
            print(cmd)
            subprocess.run(cmd, shell=True)  # nosec


if __name__ == "__main__":
    main()
