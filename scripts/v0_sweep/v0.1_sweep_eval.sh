# v0.1_base_latent_mim_contrastive_random
python scripts/v0_sweep/contrastive_latent_mim.py launch v0.1_base_latent_mim_contrastive_random ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.type=random  --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.ema_decay=\[1,1\] --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_latent_mim_contrastive_random__/step157000


# v0.1_base_latent_mim_space_time
python scripts/v0_sweep/latent_mim.py launch v0.1_base_latent_mim_space_time ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.type=space_time --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.ema_decay=\[1,1\] --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_latent_mim_space_time/step165000


# v0.1_base_galileo_random_x_space_time
python scripts/v0_sweep/galileo.py launch v0.1_base_galileo_random_x_space_time ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=1 --train_module.masking_config_a.strategy_config.type=space_time --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.contrastive_config=null --train_module.ema_decay=\[1,1\] --trainer.load_path=/weka/dfive-default/helios/checkpoints/joer/v0.1_base_galileo_random_x_space_time/step146750
