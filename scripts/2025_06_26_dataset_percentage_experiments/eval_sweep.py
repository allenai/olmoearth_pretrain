"""Eval sweep."""

import argparse

# LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
LP_LRs = [0.1]

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
checkpoints = [
    "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_0.0004/step375000",
    "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0001/step195000",
    "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0004/step420000",
    "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0078125/step450000",
    "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.015625/step450000",
    "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0625/step450000",
    "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.125/step450000",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cluster", type=str, default="ai2/titan-cirrascale", help="Cluster to run on"
)
args = parser.parse_args()

run_cmd = "launch"
for checkpoint in checkpoints:
    for probe_lr in LP_LRs:
        run_name = (
            checkpoint.split("/")[-2][20:]
            + checkpoint.split("/")[-1]
            + f"_eval_{probe_lr}"
        )
        if "0.0004" in run_name:
            # the value should be 0.00004 for the ds percentage
            run_name = run_name.replace("0.0004", "0.00004")
        start_command = "python3" if run_cmd == "launch" else "torchrun"
        formatted_lr_args = lr_args.format(lr=probe_lr)
        print(formatted_lr_args)
        cmd = f"{start_command} scripts/2025_06_26_dataset_percentage_experiments/eval.py {run_cmd} {run_name} {args.cluster} --launch.priority=high {formatted_lr_args} --trainer.load_path={checkpoint} --launch.task_name=eval"
        print(cmd)
        # subprocess.run(cmd, shell=True)  # nosec
