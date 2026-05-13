# Run ablations

```shell
python3 scripts/candidate_ablations/run_candidate_ablation.py launch candidate_combined jupiter \
    --candidate_columns in_top_combined \
    --candidate_parquet /weka/.../selection_top250000.parquet \
    --trainer.load_path=/weka/.../checkpoint/step_XXXXX \
    --train_module.optim_config.lr=0.0008 \
    --train_module.scheduler.warmup_steps=0 \
    --train_module.scheduler.alpha_f=0.0125 \
    --train_module.scheduler.t_max=200000 \
    --trainer.max_duration.value=200000 \
    --trainer.max_duration.unit=steps \
    --launch.num_gpus=8 \
    --launch.num_nodes=1 \
    --trainer.callbacks.wandb.project=candidate_ablations \
    --trainer.callbacks.wandb.name=candidate_combined
```


    --train_module.contrastive_config=null \
    --train_module.sigreg_config.loss_config.type=SIGReg \
    --train_module.sigreg_config.loss_config.weight=0.05 \
    --train_module.sigreg_config.loss_config.num_slices=256
