"""This script sweeps learning rates for the Galileo + Contrastive model ladder."""

import subprocess  # nosec

from helios.internal.utils import MODEL_SIZE_ARGS

# Model size configurations
MODEL_SIZES = {
    "base": MODEL_SIZE_ARGS["base_shallow_decoder"],
    "large": MODEL_SIZE_ARGS["large_super_shallow_decoder"],
    "giga": MODEL_SIZE_ARGS["giga_shallow_decoder"],
}

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
    "--trainer.load_path={load_path} "
    "--trainer.callback.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callback.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callback.downstream_evaluator.tasks.pastis.probe_lr={lr} "
    "--trainer.callback.downstream_evaluator.tasks.pastis-r.probe_lr={lr} "
    "--launch.priority=urgent "
)

# Learning rates to sweep
LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

for lr in LP_LRs:
    for size_name, size_config in MODEL_SIZES.items():
        for load_path in CHECKPOINT_PATHS.values():
            run_name = f"galileo_all_evals_{size_name}_linear_probe_lr_{lr}"
            command = BASE_COMMAND.format(
                run_name=run_name,
                load_path=load_path,
                encoder_embedding_size=size_config["encoder_embedding_size"],
                encoder_depth=size_config["encoder_depth"],
                encoder_num_heads=size_config["encoder_num_heads"],
                mlp_ratio=size_config["mlp_ratio"],
                decoder_embedding_size=size_config["decoder_embedding_size"],
                decoder_depth=size_config["decoder_depth"],
                decoder_num_heads=size_config["decoder_num_heads"],
            )

            print(f"Launching: {command}")

            # Execute the command
            subprocess.run(command, shell=True, check=True)  # nosec
