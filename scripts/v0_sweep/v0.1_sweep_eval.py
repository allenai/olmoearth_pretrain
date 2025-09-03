"""Run evaluation sweep for v0.1 models."""

import subprocess  # nosec

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

# v0.1_base_latent_mim_contrastive_random
base_cmd = (
    "python scripts/v0_sweep/contrastive_latent_mim.py launch v0.1_base_latent_mim_contrastive_random_use_new_evals_pr_{lr} ai2/titan-cirrascale "
    "--model.decoder_config.depth=4 --common.launch.num_gpus=1 "
    "--train_module.masking_config.strategy_config.type=random "
    "--model.reconstructor_config=null --train_module.mae_loss_config=null "
    "--train_module.ema_decay=[1,1] "
    "--trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_latent_mim_contrastive_random__/step157000 "
)

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.breizhcrops.probe_lr={lr}"
]

for lr in LP_LRs:
    full_cmd = base_cmd.format(lr=lr) + lr_args[0].format(lr=lr)
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)  # nosec

# # v0.1_base_latent_mim_space_time
# python scripts/v0_sweep/latent_mim.py launch v0.1_base_latent_mim_space_time ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.type=space_time --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.ema_decay=\[1,1\] --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_latent_mim_space_time/step165000

# v0.1_base_latent_mim_contrastive_random
base_cmd = (
    "python scripts/v0_sweep/latent_mim.py launch v0.1_base_latent_mim_space_time_all_evals_{lr} ai2/titan-cirrascale "
    "--model.decoder_config.depth=4 --common.launch.num_gpus=1 "
    "--train_module.masking_config.strategy_config.type=space_time "
    "--model.reconstructor_config=null --train_module.mae_loss_config=null "
    "--train_module.ema_decay=[1,1] "
    "--trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_latent_mim_space_time/step165000 "
)

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}"
]

for lr in LP_LRs:
    full_cmd = base_cmd.format(lr=lr) + lr_args[0].format(lr=lr)
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)  # nosec


# # v0.1_base_galileo_random_x_space_time
# python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_random_x_space_time ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=1 --train_module.masking_config_a.strategy_config.type=space_time --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.contrastive_config=null --train_module.ema_decay=\[1,1\] --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_galileo_random_x_space_time/step146750

base_cmd = (
    "python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_random_x_space_time_all_evals_{lr} ai2/titan-cirrascale "
    "--model.decoder_config.depth=4 --common.launch.num_gpus=1 "
    "--train_module.masking_config_a.strategy_config.type=space_time "
    "--model.reconstructor_config=null --train_module.mae_loss_config=null "
    "--train_module.contrastive_config=null --train_module.ema_decay=[1,1] "
    "--trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_galileo_random_x_space_time/step146750 "
)

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}"
]

for lr in LP_LRs:
    full_cmd = base_cmd.format(lr=lr) + lr_args[0].format(lr=lr)
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)  # nosec

# v0.1_base_spt_latentmim_contrastive_space_time
# python scripts/v0_sweep/contrastive_latent_mim_st.py launch v0.1_base_SpT_latentmim_contrastive_space_time ai2/jupiter-cirrascale-2 --train_module.masking_config.strategy_config.type=space_time --common.launch.num_gpus=8  --model.reconstructor_config=null --data_loader.token_budget=3000 --train_module.rank_microbatch_size=16 --train_module.mae_loss_config=null --train_module.ema_decay=\[1,1\]

base_cmd = (
    "python scripts/v0_sweep/contrastive_latent_mim_st.py launch v0.1_base_SpT_latentmim_contrastive_space_time_all_evals_0_{lr} ai2/titan-cirrascale "
    "--model.decoder_config.depth=4 --common.launch.num_gpus=1 "
    "--train_module.masking_config.strategy_config.type=space_time "
    "--model.reconstructor_config=null --train_module.mae_loss_config=null "
    "--data_loader.token_budget=3000 --train_module.rank_microbatch_size=16 --train_module.ema_decay=[1,1] "
    "--trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_SpT_latentmim_contrastive_space_time/step140000 "
)

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}"
]

for lr in LP_LRs:
    full_cmd = base_cmd.format(lr=lr) + lr_args[0].format(lr=lr)
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)  # nosec


# v0.1_base_galileo_random_x_cross_space_time
# python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_random_x_cross_space_time ai2/jupiter-cirrascale-2 --model.decoder_config.depth=4 --common.launch.num_gpus=8 --train_module.masking_config_a.strategy_config.type=modality_cross_space_time --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.contrastive_config=null --train_module.ema_decay=\[1,1\]

base_cmd = (
    "python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_random_x_cross_space_time_all_evals_0_{lr} ai2/titan-cirrascale "
    "--model.decoder_config.depth=4 --common.launch.num_gpus=1 "
    "--train_module.masking_config_a.strategy_config.type=modality_cross_space_time "
    "--model.reconstructor_config=null --train_module.mae_loss_config=null "
    "--train_module.contrastive_config=null --train_module.ema_decay=[1,1] "
    "--trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_galileo_random_x_cross_space_time/step190000 "
)

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}"
]

for lr in LP_LRs:
    full_cmd = base_cmd.format(lr=lr) + lr_args[0].format(lr=lr)
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)  # nosec

# v0.1_base_galileo_random_x_cross_space_time_ema
base_cmd = (
    "python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_random_x_cross_space_time_ema_all_evals_0_{lr} ai2/titan-cirrascale "
    "--model.decoder_config.depth=4 --common.launch.num_gpus=1 "
    "--train_module.masking_config_a.strategy_config.type=modality_cross_space_time "
    "--model.reconstructor_config=null --train_module.mae_loss_config=null "
    "--train_module.contrastive_config=null "
    "--trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_galileo_random_x_cross_space_time_ema/step210000 "
)

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}"
]

for lr in LP_LRs:
    full_cmd = base_cmd.format(lr=lr) + lr_args[0].format(lr=lr)
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)  # nosec


# v0.1_base_galileo_contrastive_random_x_space_time
# python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_contrastive_random_x_space_time ai2/jupiter-cirrascale-2 --model.decoder_config.depth=4 --common.launch.num_gpus=8 --train_module.masking_config_a.strategy_config.type=space_time --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.ema_decay=\[1,1\]

base_cmd = (
    "python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_contrastive_random_x_space_time_all_evals_0_{lr} ai2/titan-cirrascale "
    "--model.decoder_config.depth=4 --common.launch.num_gpus=1 "
    "--train_module.masking_config_a.strategy_config.type=space_time "
    "--model.reconstructor_config=null --train_module.mae_loss_config=null "
    "--train_module.ema_decay=[1,1] "
    "--trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_galileo_contrastive_random_x_space_time/step210000 "
)

lr_args = [
    "--trainer.callbacks.downstream_evaluator.tasks.m-sa-crop-type.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.m-cashew-plant.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1_sentinel2.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1_landsat.probe_lr={lr}"
]

for lr in LP_LRs:
    full_cmd = base_cmd.format(lr=lr) + lr_args[0].format(lr=lr)
    print(f"Running: {full_cmd}")
    subprocess.run(full_cmd, shell=True)  # nosec


# favyencheck_base_wattn_latentmim_contrastive_random_0
