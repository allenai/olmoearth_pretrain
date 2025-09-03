"""Run an evaluation sweep for DINOv2."""

import subprocess  # nosec

# Evaluation That sweeps over the following:
# Learning Rate
# Normalization
# helios pretrained, dataset norms

LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
PROBE_BATCH_SIZE = [128, 64, 32, 16, 8]

Normalization_MODES = ["dataset"]

lr_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr}",
    ]
)

probe_batch_size_args = " ".join(
    [
        " ",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_batch_size={probe_batch_size}",
    ]
)

dataset_args = " ".join(
    [
        " ",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=False",
    ]
)


for lr in LP_LRs:
    for norm_mode in Normalization_MODES:
        for probe_batch_size in PROBE_BATCH_SIZE:
            print(
                f"Running with {norm_mode} normalization and {lr} learning rate and {probe_batch_size} probe batch size"
            )
            run_name = (
                f"1_panopticon_1v2_eval_bs_{probe_batch_size}_norm{norm_mode}_{lr}"
            )
            args = lr_args.format(lr=lr)
            if norm_mode == "dataset":
                args += dataset_args
            args += probe_batch_size_args.format(probe_batch_size=probe_batch_size)
            # change to launch and saturn when we are ready to launch
            cmd = f"python3 scripts/dino_v2_evals/panopticon_eval.py launch {run_name}  ai2/titan-cirrascale  --launch.priority=high {args} --launch.task_name=eval"
            print(cmd)
            subprocess.run(cmd, shell=True)  # nosec
