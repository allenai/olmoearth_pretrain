"""Run sweep over learning rates."""

import argparse
import subprocess  # nosec

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cluster", type=str, default="ai2/titan-cirrascale", help="Cluster to use"
)
parser.add_argument("--priority", type=str, help="Priority for the launch")
args = parser.parse_args()

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr}",
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr}",
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr}",
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr}",
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr}",
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr}",
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr}",
]

for probe_lr in LP_LRs:
    subprocess.call(
        [
            "python",
            "scripts/2025_07_21_gse/eval.py",
            "launch",
            f"favyen_decode_gse_worldcover_osm_srtm_titan_eval_{probe_lr}",
            args.cluster,
            "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/favyen_decode_gse_worldcover_osm_srtm_titan/step370000",
        ]
        + [arg.format(lr=probe_lr) for arg in lr_args]
    )  # nosec
