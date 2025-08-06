"""Run an evaluation sweep for DINOv2."""

import argparse
import subprocess  # nosec

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
Normalization_MODES = ["imagenet", "dataset", "helios"]

lr_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.breizhcrops.probe_lr={lr}",
    ]
)

dataset_args = " ".join(
    [
        " ",
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_forestnet.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=False",
    ]
)

helios_args = " ".join(
    [
        " ",
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_forestnet.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=True",
    ]
)


def main():
    """Run the full evaluation sweep."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, required=True, help="Cluster name")
    parser.add_argument(
        "--checkpoint-path", type=str, required=True, help="Checkpoint path"
    )
    parser.add_argument(
        "--module-path", type=str, required=True, help="Path to module .py"
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
            run_name = f"eval_{norm_mode}_lr{lr}"
            cmd_args = lr_args.format(lr=lr)
            if norm_mode == "dataset":
                cmd_args += dataset_args
            elif norm_mode == "helios":
                cmd_args += helios_args
            cmd = (
                f"python3 scripts/run_all_evals/all_evals.py -m {module_path} "
                f"launch {run_name} {cluster} --launch.priority=high {cmd_args} "
                f"--launch.task_name=eval --trainer.load_path={checkpoint_path}"
            )
            print(cmd)
            subprocess.run(cmd, shell=True)  # nosec


if __name__ == "__main__":
    main()
