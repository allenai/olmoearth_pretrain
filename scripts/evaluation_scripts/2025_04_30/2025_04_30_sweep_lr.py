"""This script sweeps learning rates for the Galileo + Contrastive model ladder."""

import subprocess  # nosec

from helios.internal.utils import MODEL_SIZE_ARGS

# Model size configurations
MODEL_SIZES = {
    "base": MODEL_SIZE_ARGS["base_shallow_decoder"],
    "large": MODEL_SIZE_ARGS["large_super_shallow_decoder"],
    "giga": MODEL_SIZE_ARGS["giga_shallow_decoder"],
}

# Checkpoint paths
CHECKPOINT_PATHS = {
    "base": "/weka/dfive-default/helios/checkpoints/henryh/3_galileo_contrastive_base_decoder_4_lr_0.0001_weight_0.05/step312400",
    "large": "/weka/dfive-default/helios/checkpoints/henryh/1_galileo_contrastive_0.05_s2_s1_wc_large_dec2_lr0.0001_titan/step109250",
    "giga": "/weka/dfive-default/helios/checkpoints/henryh/1_galileo_contrastive_0.05_s2_s1_wc_giga_dec4_lr0.0001_jupiter/step140500",
}

# Base command template
BASE_COMMAND = (
    "python3 scripts/evaluation_scripts/galileo_eval.py launch {run_name} ai2/jupiter-cirrascale-2 "
    "--model.encoder_config.embedding_size={encoder_embedding_size} "
    "--model.encoder_config.depth={encoder_depth} "
    "--model.encoder_config.num_heads={encoder_num_heads} "
    "--model.encoder_config.mlp_ratio={mlp_ratio} "
    "--model.decoder_config.encoder_embedding_size={encoder_embedding_size} "
    "--model.decoder_config.decoder_embedding_size={decoder_embedding_size} "
    "--model.decoder_config.depth={decoder_depth} "
    "--model.decoder_config.num_heads={decoder_num_heads} "
    "--model.decoder_config.mlp_ratio={mlp_ratio} "
    "--train_module.token_exit_cfg_a.sentinel2_l2a={encoder_depth} "
    "--train_module.token_exit_cfg_a.latlon={encoder_depth} "
    "--train_module.token_exit_cfg_a.sentinel1={encoder_depth} "
    "--train_module.token_exit_cfg_a.srtm={encoder_depth} "
    "--train_module.token_exit_cfg_a.landsat={encoder_depth} "
    "--trainer.load_path={checkpoint_path} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.dataset=mados "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=False "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.dataset=sen1floods11 "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis.dataset=pastis "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis.batch_size=8 "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis.num_workers=2 "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis-r.dataset=pastis-r "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis-r.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis-r.batch_size=8 "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis-r.num_workers=2 "
    "--launch.priority=urgent "
)

# Learning rates to sweep for linear probe
LP_LRs = [5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

for lr in LP_LRs:
    for model_size in MODEL_SIZES:
        run_name = f"20250430_galileo_all_evals_{model_size}_linear_probe_lr_{lr}"
        checkpoint_path = CHECKPOINT_PATHS[model_size]
        command = BASE_COMMAND.format(
            run_name=run_name,
            encoder_depth=MODEL_SIZE_ARGS[model_size]["encoder_depth"],
            encoder_embedding_size=MODEL_SIZE_ARGS[model_size][
                "encoder_embedding_size"
            ],
            encoder_num_heads=MODEL_SIZE_ARGS[model_size]["encoder_num_heads"],
            mlp_ratio=MODEL_SIZE_ARGS[model_size]["mlp_ratio"],
            decoder_depth=MODEL_SIZE_ARGS[model_size]["decoder_depth"],
            decoder_embedding_size=MODEL_SIZE_ARGS[model_size][
                "decoder_embedding_size"
            ],
            decoder_num_heads=MODEL_SIZE_ARGS[model_size]["decoder_num_heads"],
            lr=lr,
            checkpoint_path=checkpoint_path,
        )
        print(f"Launching: {command}")
        # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec
