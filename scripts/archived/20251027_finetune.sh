#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="2025_10_26_olmoearth_finetune"
CLUSTER="ai2/titan"
SCRIPT="python olmoearth_pretrain/internal/full_eval_sweep_finetune.py"

# Dinov3
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py --cluster $CLUSTER --model dino_v3 --finetune_seed 1234
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py --cluster $CLUSTER --model dino_v3 --finetune_seed 42
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py --cluster $CLUSTER --model dino_v3 --finetune_seed 0

# Anysat
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat --finetune_seed 0
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat --finetune_seed 1234
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat --finetune_seed 42

$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat_ps16 --finetune_seed 0 --launch.preemptible=False
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat_ps16 --finetune_seed 1234 --launch.preemptible=False
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat_ps16 --finetune_seed 42 --launch.preemptible=False

$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat_ps8 --finetune_seed 0
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat_ps8 --finetune_seed 1234
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/anysat/anysat_launch.py --cluster $CLUSTER --model anysat_ps8 --finetune_seed 42

# Panopticon
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/panopticon/panopticon_launch.py --cluster $CLUSTER --model panopticon --finetune_seed 0
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/panopticon/panopticon_launch.py --cluster $CLUSTER --model panopticon --finetune_seed 42
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/panopticon/panopticon_launch.py --cluster $CLUSTER --model panopticon --finetune_seed 1234

# Croma
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/croma/croma_launch.py --cluster $CLUSTER --model croma --finetune_seed 0 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/croma/croma_launch.py --cluster $CLUSTER --model croma --finetune_seed 42 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/croma/croma_launch.py --cluster $CLUSTER --model croma --finetune_seed 1234 --launch.priority=high

# Croma Large
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/croma/croma_launch.py --cluster $CLUSTER --model croma_large --finetune_seed 0 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/croma/croma_launch.py --cluster $CLUSTER --model croma_large --finetune_seed 42 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/croma/croma_launch.py --cluster $CLUSTER --model croma_large --finetune_seed 1234 --launch.priority=high

# CopernicusFM
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/copernicusfm/copernicusfm_launch.py --cluster $CLUSTER --model copernicusfm --finetune_seed 0 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/copernicusfm/copernicusfm_launch.py --cluster $CLUSTER --model copernicusfm --finetune_seed 42 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/copernicusfm/copernicusfm_launch.py --cluster $CLUSTER --model copernicusfm --finetune_seed 1234 --launch.priority=high

# OlmoEarth Large / Base / Tiny / Nano from checkpoints (final piece to RUN, on jupiter!!)
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/step560000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/large.py --cluster $CLUSTER --finetune_seed 0 --launch.priority=high
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/step560000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/large.py --cluster $CLUSTER --finetune_seed 42 --launch.priority=high
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/step560000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/large.py --cluster $CLUSTER --finetune_seed 1234 --launch.priority=high

$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/base.py --cluster $CLUSTER --finetune_seed 0
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/base.py --cluster $CLUSTER --finetune_seed 42
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/base.py --cluster $CLUSTER --finetune_seed 1234

$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/tiny.py --cluster $CLUSTER --finetune_seed 0
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/tiny.py --cluster $CLUSTER --finetune_seed 42
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/tiny.py --cluster $CLUSTER --finetune_seed 1234

$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/nano.py --cluster $CLUSTER --finetune_seed 0
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/nano.py --cluster $CLUSTER --finetune_seed 42
$SCRIPT --checkpoint_path /weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000 --project_name $PROJECT_NAME --module_path scripts/2025_10_02_phase2/nano.py --cluster $CLUSTER --finetune_seed 1234

# Satlas
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/satlas/satlas_launch.py --cluster $CLUSTER --model satlas --finetune_seed 1234
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/satlas/satlas_launch.py --cluster $CLUSTER --model satlas --finetune_seed 42
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/satlas/satlas_launch.py --cluster $CLUSTER --model satlas --finetune_seed 0

# Satlas, dataset norm
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/satlas/satlas_launch.py --cluster $CLUSTER --model satlas --finetune_seed 1234 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/satlas/satlas_launch.py --cluster $CLUSTER --model satlas --finetune_seed 42 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/satlas/satlas_launch.py --cluster $CLUSTER --model satlas --finetune_seed 0 --use_dataset_normalizer

# Galileo
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo --finetune_seed 1234
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo --finetune_seed 42
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo --finetune_seed 0

# Galileo, dataset norm
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo_ps16 --finetune_seed 1234 --use_dataset_normalizer --launch.preemptible=False
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo_ps16 --finetune_seed 42 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo_ps16 --finetune_seed 0 --use_dataset_normalizer

$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo_ps8 --finetune_seed 1234 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo_ps8 --finetune_seed 42 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/galileo/galileo_launch.py --cluster $CLUSTER --model galileo_ps8 --finetune_seed 0 --use_dataset_normalizer

# Clay
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/clay/clay_launch.py --cluster $CLUSTER --model clay --finetune_seed 1234 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/clay/clay_launch.py --cluster $CLUSTER --model clay --finetune_seed 42 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/clay/clay_launch.py --cluster $CLUSTER --model clay --finetune_seed 0 --launch.priority=high

# Clay, dataset norm
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/clay/clay_launch.py --cluster $CLUSTER --model clay --finetune_seed 1234 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/clay/clay_launch.py --cluster $CLUSTER --model clay --finetune_seed 42 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/clay/clay_launch.py --cluster $CLUSTER --model clay --finetune_seed 0 --use_dataset_normalizer

# Prithvi V2
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py --cluster $CLUSTER --model prithvi_v2 --finetune_seed 1234 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py --cluster $CLUSTER --model prithvi_v2 --finetune_seed 42 --launch.priority=high
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py --cluster $CLUSTER --model prithvi_v2 --finetune_seed 0 --launch.priority=high

# Prithvi V2, dataset norm
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py --cluster $CLUSTER --model prithvi_v2 --finetune_seed 1234 --use_dataset_normalizer  --launch.preemptible=False
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py --cluster $CLUSTER --model prithvi_v2 --finetune_seed 42 --use_dataset_normalizer --launch.preemptible=False
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py --cluster $CLUSTER --model prithvi_v2 --finetune_seed 0 --use_dataset_normalizer --launch.preemptible=False

# Terramind
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind --finetune_seed 1234
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind --finetune_seed 42
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind --finetune_seed 0

# Terramind, dataset norm
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind --finetune_seed 1234 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind --finetune_seed 42 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind --finetune_seed 0 --use_dataset_normalizer

# Terramind_large
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind_large --finetune_seed 1234
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind_large --finetune_seed 42
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind_large --finetune_seed 0

# Terramind_large, dataset norm
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind_large --finetune_seed 1234 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind_large --finetune_seed 42 --use_dataset_normalizer
$SCRIPT --project_name $PROJECT_NAME --module_path olmoearth_pretrain/evals/models/terramind/terramind_launch.py --cluster $CLUSTER --model terramind_large --finetune_seed 0 --use_dataset_normalizer

# Get metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_25_phase2_finetune --run_prefix nano_lr0.001_wd0.002_step370000_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_25_phase2_finetune --run_prefix tiny_lr0.0002_wd0.02_step360000_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_25_phase2_finetune --run_prefix phase2.0_base_lr0.0001_wd0.02_step667200_seed --finetune --get_test_metrics

python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix dino_v3_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix anysat_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix panopticon_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix croma_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix croma_large_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix copernicusfm_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix prithvi_v2_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_phase2_finetune --run_prefix galileo_seed --finetune --get_test_metrics

python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix nano_lr0.001_wd0.002_step370000_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix tiny_lr0.0002_wd0.02_step360000_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix phase2.0_base_lr0.0001_wd0.02_step667200_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix phase2.0_large_lr0.0001_wd0.002_step560000_seed --finetune --get_test_metrics

python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix anysat_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix clay_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix copernicusfm_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix croma_seed --finetune --get_test_metrics

python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix croma_large_seed --finetune --get_test_metrics

python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix dino_v3_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix panopticon_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix satlas_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix galileo_seed --finetune --get_test_metrics

python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix prithvi_v2_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix terramind_seed --finetune --get_test_metrics
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix terramind_large_seed --finetune --get_test_metrics

# Galileo ps16
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix galileo_ps16_seed --finetune --get_test_metrics
# AnySat ps16
python scripts/get_max_eval_metrics_from_wandb.py --project_name 2025_10_26_olmoearth_finetune --run_prefix anysat_ps16_seed --finetune --get_test_metrics
