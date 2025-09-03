# Overide Commands and formulas

```
python3 scripts/latent_mim.py dry_run --run_name=tester_fun \
python3 scripts/latent_mim.py dry_run train_module.optim.lr=0.01 \
python3 scripts/latent_mim.py dry_run train_module.optim.weight_decay=0.00221 \
python3 scripts/latent_mim.py dry_run train_module.masking_config.strategy_config.encode_ratio=0.5 \
python3 scripts/latent_mim.py dry_run --common.save_folder="./test_data" \
python3 scripts/latent_mim.py dry_run --common.supported_modality_names=\[sentinel2\] \
python3 scripts/latent_mim.py dry_run --trainer.max_duration.value=10 --trainer.max_duration.unit=steps \
```

Make sure to replace all variables for evaluator task:

```
python3 scripts/latent_mim.py  --trainer.callbacks.downstream_evaluator.tasks="[{name: my_task, num_workers: 2, ...}]" \
```

Targeting specific Beaker host example:

```
python scripts/X.py launch [name] ai2/titan-cirrascale --launch.hostnames=["titan-cs-aus-465.reviz.ai2.in","titan-cs-aus-466.reviz.ai2.in"]
```

To add mixup, add the following CLI arguments:
```
--train_module.transform_config.transform_type=mixup --train_module.transform_config.transform_kwargs={"alpha": 1.3}
```
