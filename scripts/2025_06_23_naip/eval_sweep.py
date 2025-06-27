"""Run sweep over learning rates."""

import subprocess  # nosec

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr}",
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

for probe_lr in LP_LRs:
    subprocess.call(
        [
            "python",
            "scripts/2025_06_23_naip/eval.py",
            "launch",
            f"v0.2_base_latent_mim_128_naip_moredata_random_fixed_modality_0.5_eval_{probe_lr}",
            "ai2/titan-cirrascale",
            "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_naip_moredata_random_fixed_modality_0.5/step320000",
        ]
        + [arg.format(lr=probe_lr) for arg in lr_args],
    )  # nosec
    subprocess.call(
        [
            "python",
            "scripts/2025_06_23_naip/eval.py",
            "launch",
            f"v0.2_base_latent_mim_128_moredata_random_fixed_modality_0.decodes1landsat_eval_{probe_lr}",
            "ai2/titan-cirrascale",
            "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_moredata_random_fixed_modality_0.decodes1landsat/step340000",
            "--common.training_modalities=[sentinel2_l2a,sentinel1,worldcover,latlon,srtm,landsat,openstreetmap_raster]",
        ]
        + [arg.format(lr=probe_lr) for arg in lr_args],
    )  # nosec
    subprocess.call(
        [
            "python",
            "scripts/2025_06_23_naip/eval.py",
            "launch",
            f"v0.2_small_latent_mim_128_naip_moredata_random_fixed_modality_0.5_nonaip_eval_{probe_lr}",
            "ai2/titan-cirrascale",
            "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_small_latent_mim_128_naip_moredata_random_fixed_modality_0.5_nonaip/step310000",
            "--common.training_modalities=[sentinel2_l2a,sentinel1,worldcover,latlon,srtm,landsat,openstreetmap_raster]",
            "--model.encoder_config.embedding_size=384",
            "--model.decoder_config.encoder_embedding_size=384",
            "--model.decoder_config.decoder_embedding_size=384",
            "--model.encoder_config.num_heads=6",
            "--model.decoder_config.num_heads=6",
        ]
        + [arg.format(lr=probe_lr) for arg in lr_args],
    )  # nosec
    subprocess.call(
        [
            "python",
            "scripts/2025_06_23_naip/eval_alldata.py",
            "launch",
            f"v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5_eval_{probe_lr}",
            "ai2/titan-cirrascale",
            "--trainer.load_path=/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
            "--common.training_modalities=[sentinel2_l2a,sentinel1,worldcover,latlon,srtm,landsat,openstreetmap_raster]",
        ]
        + [arg.format(lr=probe_lr) for arg in lr_args],
    )  # nosec
