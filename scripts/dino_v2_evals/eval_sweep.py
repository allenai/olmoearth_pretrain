"""Run an evaluation sweep for DINOv2."""

import subprocess  # nosec

# Evaluation That sweeps over the following:
# Learning Rate
# Normalization
# helios pretrained, dataset norms and imagenet normalization

dino_v2_torchub_ids = [
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dinov2_vitb14_reg",
]
LP_LRs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

Normalization_MODES = ["imagenet", "dataset", "helios"]

lr_args = " ".join(
    [
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.sickle_landsat.probe_lr={lr}",
        "--trainer.callbacks.downstream_evaluator.tasks.breizhcrops.probe_lr={lr}",
    ]
)

dataset_args = " ".join(
    [
        " ",
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_forestnet.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=False",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=False",
    ]
)

# This should be set in the model config
imagenet_args = (
    " ".join(
        [
            " ",
            "--trainer.callbacks.downstream_evaluator.tasks.m_forestnet.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_method=no_norm",
            "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=False",
            "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=False",
            "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=False",
            "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=False",
            "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=False",
            "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=False",
            "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=False",
            "--model.apply_imagenet_normalization=True",
        ]
    )
    + dataset_args
)


helios_args = " ".join(
    [
        " ",
        "--trainer.callbacks.downstream_evaluator.tasks.m_eurosat.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_forestnet.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_bigearthnet.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_so2sat.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_brick_kiln.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.m_cashew-plant.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.mados.norm_stats_from_pretrained=True",
        "--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.norm_stats_from_pretrained=True",
    ]
)


for dino_v2_torchub_id in dino_v2_torchub_ids:
    for lr in LP_LRs:
        for norm_mode in Normalization_MODES:
            print(f"Running with {norm_mode} normalization and {lr} learning rate")
            run_name = f"1_dino_v2_eval_model_{dino_v2_torchub_id}norm{norm_mode}_{lr}"
            args = lr_args.format(lr=lr)
            if norm_mode == "imagenet":
                args += imagenet_args
            elif norm_mode == "dataset":
                args += dataset_args
            elif norm_mode == "helios":
                args += helios_args
            args += f" --model.torchhub_id={dino_v2_torchub_id}"
            # change to launch and saturn when we are ready to launch
            cmd = f"python3 scripts/dino_v2_evals/dino_v2_eval.py launch {run_name}  ai2/saturn-cirrascale  --launch.priority=high {args} --launch.task_name=eval"
            print(cmd)
            subprocess.run(cmd, shell=True)  # nosec
