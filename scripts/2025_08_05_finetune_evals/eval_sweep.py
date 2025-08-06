"""Eval sweep."""

import subprocess
import argparse

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
lr_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.breizhcrops.probe_lr={lr}",
    ]
)
checkpoints = {
    "base": "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000"
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cluster", type=str, default="ai2/titan-cirrascale", help="Cluster to run on"
)
args = parser.parse_args()

run_cmd = "launch"
for ckpt_name, checkpoint in checkpoints.items():
    for probe_lr in LP_LRs:
        run_name = f"{ckpt_name}__lr{probe_lr}"
        start_command = "python3" if run_cmd == "launch" else "torchrun"
        formatted_lr_args = lr_args.format(lr=probe_lr)
        cmd = f"{start_command} scripts/2025_08_05_finetune_evals/eval.py {run_cmd} {run_name} {args.cluster} --launch.priority=high {formatted_lr_args} --trainer.load_path={checkpoint} --launch.task_name=eval"
        print(cmd)
        subprocess.run(cmd, shell=True)  # nosec
        print()
