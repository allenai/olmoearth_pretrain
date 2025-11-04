#!/bin/bash

# Random seed values for reproducibility
SEEDS=(1337 4242 7890 2468 5831 9124 3706 6549 8013 1729)

# Launch ALL jobs in parallel (all models, all seeds)
for seed in "${SEEDS[@]}"; do
    python3 olmoearth_pretrain/internal/full_eval_sweep.py --model=all --select_best_val --trainer.callbacks.downstream_evaluator.run_on_test=True --cluster=ai2/saturn --launch.priority=urgent --all_sizes --trainer.callbacks.wandb.project=linear_probe_random_seeds --load_eval_settings_from_json=max_settings_per_group_merged.json --trainer.callbacks.downstream_evaluator.filter_for_eval_mode=LINEAR_PROBE --init_seed=$seed
done

for seed in "${SEEDS[@]}"; do
    python -m olmoearth_pretrain.internal.full_eval_sweep --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000 --module_path=scripts/official/nano.py --select_best_val ---trainer.callbacks.downstream_evaluator.run_on_test=True --launch.priority=urgent --trainer.callbacks.wandb.project=linear_probe_random_seeds --load_eval_settings_from_json=nano_settings.json --trainer.callbacks.downstream_evaluator.filter_for_eval_mode=LINEAR_PROBE --init_seed=$seed
done

for seed in "${SEEDS[@]}"; do
    python -m olmoearth_pretrain.internal.full_eval_sweep --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000 --module_path=scripts/official/tiny.py --select_best_val ---trainer.callbacks.downstream_evaluator.run_on_test=True --launch.priority=urgent --trainer.callbacks.wandb.project=linear_probe_random_seeds --load_eval_settings_from_json=tiny_settings.json --trainer.callbacks.downstream_evaluator.filter_for_eval_mode=LINEAR_PROBE --init_seed=$seed
done

for seed in "${SEEDS[@]}"; do
    python -m olmoearth_pretrain.internal.full_eval_sweep --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 --module_path=scripts/official/base.py --select_best_val ---trainer.callbacks.downstream_evaluator.run_on_test=True --launch.priority=urgent --trainer.max_duration.value=700000 --trainer.max_duration.unit=steps --trainer.callbacks.wandb.project=linear_probe_random_seeds --load_eval_settings_from_json=base_settings.json --trainer.callbacks.downstream_evaluator.filter_for_eval_mode=LINEAR_PROBE --init_seed=$seed
done

for seed in "${SEEDS[@]}"; do
    python -m olmoearth_pretrain.internal.full_eval_sweep --cluster=ai2/saturn-cirrascale --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/step560000 --module_path=scripts/official/large.py --select_best_val ---trainer.callbacks.downstream_evaluator.run_on_test=True --launch.priority=urgent --trainer.max_duration.value=700000 --trainer.max_duration.unit=steps --trainer.callbacks.wandb.project=linear_probe_random_seeds --load_eval_settings_from_json=large_settings.json --trainer.callbacks.downstream_evaluator.filter_for_eval_mode=LINEAR_PROBE --init_seed=$seed
done

# Wait for all submissions to complete
wait

echo "All jobs submitted!"
