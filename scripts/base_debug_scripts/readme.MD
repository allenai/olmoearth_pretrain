# How to Use these scripts

These scripts contain a reccomended set up to get relatively fast start up and training.
These settings should do about 20 minute per epoch on 1 GPU with batch size 128 (Eval not timed)

For faster Training use more gpus bigger batch size potentially more workers until you are saturated

IF You want to share the runs with others please set the wandb project use override `--trainer.wandb.project=helios-train`

If debugging different architectures or datasets, please remove checkpoints in `./local_output/checkpoints/<user>/test_run` (for sessions) or `/weka/dfive-default/helios/checkpoints/<user>/test_run` (for batch jobs).
