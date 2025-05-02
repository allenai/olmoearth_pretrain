"""This script sweeps learning rates for the Galileo + Contrastive model ladder."""

import subprocess  # nosec

# # Model size configurations
# MODEL_SIZES = {
#     "base": MODEL_SIZE_ARGS["base_shallow_decoder"],
#     "large": MODEL_SIZE_ARGS["large_super_shallow_decoder"],
#     "giga": MODEL_SIZE_ARGS["giga_shallow_decoder"],
# }

# CHECKPOINT_PATHS = {
#     "base": "/weka/dfive-default/helios/checkpoints/henryh/3_galileo_contrastive_base_decoder_4_lr_0.0001_weight_0.05/step312400",
#     "large": "/weka/dfive-default/helios/checkpoints/henryh/1_galileo_contrastive_0.05_s2_s1_wc_large_dec2_lr0.0001_titan/step109250",
#     "giga": "/weka/dfive-default/helios/checkpoints/henryh/1_galileo_contrastive_0.05_s2_s1_wc_giga_dec4_lr0.0001_jupiter/step140500",
# }

# /weka/dfive-default/helios/checkpoints/henryh/1_galileo_contrastive_0.05_s2_s1_wc_large_dec4_lr0.0001_titan/step179500
# /weka/dfive-default/helios/checkpoints/henryh/1_galileo_contrastive_0.05_s2_s1_wc_large_dec4_lr0.0001_titan/step179750

CHECKPOINT_PATHS = {
    "large": "/weka/dfive-default/helios/checkpoints/henryh/1_galileo_contrastive_0.05_s2_s1_wc_large_dec4_lr0.0001_titan/step179500",
}

scripts = [
    # "galileo_base_eval.py",
    "galileo_large_eval.py",
    # "galileo_giga_eval.py",
]

# Base command template
BASE_COMMAND = (
    "python3 scripts/evaluation_scripts/{script} launch {run_name} ai2/jupiter-cirrascale-2 "
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

# Learning rates to sweep
# LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
LP_LRs = [5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

for lr in LP_LRs:
    for script in scripts:
        run_name = f"20250502_galileo_all_evals_{script.split('_')[1]}_linear_probe_lr_{lr}_step179500"
        command = BASE_COMMAND.format(
            run_name=run_name,
            script=script,
            lr=lr,
        )

        print(f"Launching: {command}")

        # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec
