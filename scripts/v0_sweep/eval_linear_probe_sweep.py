"""This script sweeps learning rates for a given model from the v0 sweep for linear probing."""

import subprocess  # nosec

# Checkpoint paths args dict
CHECKPOINT_PATHS = {
    "/weka/dfive-default/helios/checkpoints/joer/v0.1_base_galileo_random_x_space_time/step130000": " --model.decoder_config.depth=4  --model.decoder_config.depth=4 --train_module.masking_config_a.strategy_config.type=space_time --model.reconstructor_config=null --train_module.mae_loss_config=null --train_module.contrastive_config=null"
}


# Base command template
BASE_COMMAND = (
    "python3 scripts/v0_sweep/galileo.py {cmd} {run_name} {cluster} "
    "--trainer.load_path={checkpoint_path} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_batch_size={batch_size} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sen1floods11.probe_batch_size={batch_size} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis.probe_batch_size={batch_size} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel1.probe_batch_size={batch_size} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_r.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.pastis_r.probe_batch_size={batch_size} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_batch_size={batch_size} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_lr={lr} "
    "--trainer.callbacks.downstream_evaluator.tasks.sickle_sentinel1.probe_batch_size={batch_size} "
    "--trainer.callbacks.wandb.project=v0_sweep_eval_debug "
    "--launch.priority=low "
    "--launch.task_name=eval "
)

# Learning rates to sweep for linear probe
PROBE_BATCH_SIZES = [4, 8, 16, 32, 64, 128]
LP_LRs = [
    1e-4,
    2e-4,
    5e-4,
    7e-4,
    1e-3,
    2e-3,
    5e-3,
    7e-3,
    1e-2,
    2e-2,
    5e-2,
    7e-2,
    1e-1,
    2e-1,
    5e-1,
    7e-1,
]
print(f"Sweeping {len(LP_LRs) * len(PROBE_BATCH_SIZES)} linear probe configurations")
for lr in LP_LRs:
    for probe_batch_size in PROBE_BATCH_SIZES:
        for checkpoint_path, model_specific_args in CHECKPOINT_PATHS.items():
            # get the second to last directory name
            print(checkpoint_path.split("/"))
            training_run_name = checkpoint_path.split("/")[-2]
            run_name = f"eval_{training_run_name}_lr_{lr}_bs_{probe_batch_size}"
        command = (
            BASE_COMMAND.format(
                run_name=run_name,
                lr=lr,
                batch_size=probe_batch_size,
                checkpoint_path=checkpoint_path,
                cmd="launch",
                cluster="ai2/titan-cirrascale",
            )
            + model_specific_args
        )
        print(f"Launching: {command}")
        # Execute the command
        subprocess.run(command, shell=True, check=True)  # nosec
