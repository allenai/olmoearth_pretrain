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
    # "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr}",
]

dataset_paritions = [
    "0.01x_train",
    "0.05x_train",
    "0.20x_train",
    "0.50x_train",
]
dataset_partition_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.m_sa_crop_type.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.m_cashew_plant.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.mados.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.partition={dataset_partition}",
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.partition={dataset_partition}",
    # "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.partition={dataset_partition}",
]
for probe_lr in LP_LRs:
    for dataset_partition in dataset_paritions:
        # Add priority argument if provided
        priority_args = []
        if args.priority:
            priority_args = [f"--launch.priority={args.priority}"]

        # subprocess.call(
        #     [
        #         "python",
        #         "scripts/2025_06_23_naip/eval.py",
        #         "launch",
        #         f"v0.2_base_latent_mim_128_moredata_random_fixed_modality_0.decodes1landsat_eval_{probe_lr}_dp_{dataset_percentage}",
        #         args.cluster,
        #         "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_moredata_random_fixed_modality_0.decodes1landsat/step340000",
        #         "--common.training_modalities=[sentinel2_l2a,sentinel1,worldcover,latlon,srtm,landsat,openstreetmap_raster]",
        #     ]
        #     + priority_args
        #     + [arg.format(lr=probe_lr) for arg in lr_args]
        #     + [
        #         arg.format(dataset_percentage=dataset_percentage)
        #         for arg in dataset_percentage_args
        #     ],
        # )  # nosec
        # subprocess.call(
        #     [
        #         "python",
        #         "scripts/2025_06_23_naip/eval.py",
        #         "launch",
        #         f"v0.2_small_latent_mim_128_naip_moredata_random_fixed_modality_0.5_nonaip_eval_{probe_lr}_dp_{dataset_percentage}",
        #         args.cluster,
        #         "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_small_latent_mim_128_naip_moredata_random_fixed_modality_0.5_nonaip/step310000",
        #         "--common.training_modalities=[sentinel2_l2a,sentinel1,worldcover,latlon,srtm,landsat,openstreetmap_raster]",
        #         "--model.encoder_config.embedding_size=384",
        #         "--model.decoder_config.encoder_embedding_size=384",
        #         "--model.decoder_config.decoder_embedding_size=384",
        #         "--model.encoder_config.num_heads=6",
        #         "--model.decoder_config.num_heads=6",
        #     ]
        #     + priority_args
        #     + [arg.format(lr=probe_lr) for arg in lr_args]
        #     + [
        #         arg.format(dataset_percentage=dataset_percentage)
        #         for arg in dataset_percentage_args
        #     ],
        # )  # nosec
        subprocess.call(
            [
                "python",
                "scripts/2025_06_23_naip/eval_alldata.py",
                "launch",
                f"1gal_partition_v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5_eval_{probe_lr}_dp_{dataset_partition}",
                args.cluster,
                "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
                "--common.training_modalities=[sentinel2_l2a,sentinel1,worldcover,latlon,srtm,landsat,openstreetmap_raster]",
            ]
            + priority_args
            + [arg.format(lr=probe_lr) for arg in lr_args]
            + [
                arg.format(dataset_partition=dataset_partition)
                for arg in dataset_partition_args
            ],
        )  # nosec
        # subprocess.call(
        #     [
        #         "python",
        #         "scripts/2025_06_23_naip/eval.py",
        #         "launch",
        #         f"v0.2_base_latent_mim_128_naip_moredata_random_fixed_modality_0.5_nonaip_eval_{probe_lr}_dp_{dataset_percentage}",
        #         args.cluster,
        #         "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_naip_moredata_random_fixed_modality_0.5_nonaip/step340000",
        #         "--common.training_modalities=[sentinel2_l2a,sentinel1,worldcover,latlon,srtm,landsat,openstreetmap_raster]",
        #     ]
        #     + priority_args
        #     + [arg.format(lr=probe_lr) for arg in lr_args]
        #     + [
        #         arg.format(dataset_percentage=dataset_percentage)
        #         for arg in dataset_percentage_args
        #     ],
        # )  # nosec
