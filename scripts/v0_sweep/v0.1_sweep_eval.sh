# v0.1_base_latent_mim_contrastive_random
python scripts/v0_sweep/contrastive_latent_mim.py launch v0.1_base_latent_mim_contrastive_random ai2/titan-cirrascale --model.decoder_config.depth=4 --common.launch.num_gpus=1 --train_module.masking_config.strategy_config.type=random  --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.ema_decay=\[1,1\]
