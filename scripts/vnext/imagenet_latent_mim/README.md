# ImageNet Latent MIM

This experiment trains the existing contrastive latent MIM pipeline on a single
natural-image modality, `imagenet`, with the target encoder EMA fixed at `1.0`
for the whole run.

## What This Runs

- Training script: `scripts/vnext/imagenet_latent_mim/base.py`
- Tiny smoke script: `scripts/vnext/imagenet_latent_mim/tiny.py`
- Modality: `imagenet`, RGB only, non-temporal
- Dataset: raw ImageFolder-style ImageNet train images
- Normalization: standard ImageNet RGB mean/std
- Train module: `ContrastiveLatentMIMTrainModuleConfig`
- EMA: `ema_decay=(1.0, 1.0)`
- Masking: random encode/decode masking

The fixed EMA setting means the target encoder is the initialization-time copy of
the online encoder and does not move during training.

## Data Layout

The pretraining dataset expects ImageFolder-style data:

```text
/weka/dfive-default/olmoearth/dataset/imagenet/train/
  n01440764/
    n01440764_18.JPEG
    ...
  n01443537/
    n01443537_2.JPEG
    ...
```

Labels and class names are ignored for pretraining; the class folders only give
us a standard ImageFolder layout.

## Download ImageNet

ImageNet-1k is access-controlled. Download the ILSVRC2012 archives from the
official ImageNet site after accepting the terms, or copy them from an internal
AI2-approved mirror if one is available.

Expected official archive names:

```text
ILSVRC2012_img_train.tar
ILSVRC2012_img_val.tar
ILSVRC2012_devkit_t12.tar.gz
```

Place the train archive somewhere on Weka, then expand it into class folders:

```bash
export IMAGENET_ROOT=/weka/dfive-default/olmoearth/dataset/imagenet
mkdir -p "$IMAGENET_ROOT/train"

tar -xf /path/to/ILSVRC2012_img_train.tar -C "$IMAGENET_ROOT/train"

for class_tar in "$IMAGENET_ROOT"/train/*.tar; do
  class_name="$(basename "$class_tar" .tar)"
  mkdir -p "$IMAGENET_ROOT/train/$class_name"
  tar -xf "$class_tar" -C "$IMAGENET_ROOT/train/$class_name"
  rm "$class_tar"
done
```

Do not commit or symlink the data into the repo.

## Image Size

Native ImageNet JPEGs have variable dimensions. The dataset loader converts each
image to RGB and uses `ImageOps.fit(..., (256, 256), BICUBIC)`, so the model sees
square `256 x 256` RGB crops.

ImageNet-1k train is about 1.28M images. The train archive is roughly 138 GB
compressed; validation is roughly 6 GB compressed. Leave extra room for the
extracted class-folder tree.

## Patch And Crop Sizes

The model still uses `MAX_PATCH_SIZE = 8`, matching the repo's existing FlexiViT
defaults. For ImageNet, the dataloader samples larger crop grids than the geo
experiments:

```python
MIN_PATCH_SIZE = 4
MAX_PATCH_SIZE = 8
sampled_hw_p_list = list(range(8, 33))
```

That gives crop sizes from `4 * 8 = 32` pixels up to `8 * 32 = 256` pixels. In
other words, the run can see full 256x256 ImageNet crops without increasing the
model's maximum patch size.

I would start here before trying patch size 16. A 16-pixel patch is more
ImageNet-conventional and cheaper (`16 x 16 = 256` tokens for a 256 crop), but it
changes the model patch embedding range and should be a separate ablation:

```bash
--model.encoder_config.max_patch_size=16 \
--data_loader.max_patch_size=16 \
--data_loader.min_patch_size=8 \
--data_loader.sampled_hw_p_list='[8,9,10,11,12,13,14,15,16]'
```

## Dry Run

Always dry-run with the real data path first:

```bash
uv run python scripts/vnext/imagenet_latent_mim/base.py dry_run imagenet_lmim_ema1 local \
  --dataset.root_dir=/weka/dfive-default/olmoearth/dataset/imagenet/train \
  --trainer.callbacks.wandb.enabled=false
```

Check the printed config for:

- `training_modalities=['imagenet']`
- `dataset=ImageNetDatasetConfig(...)`
- `ema_decay=[1.0, 1.0]`
- `num_masked_views=2`
- `sampled_hw_p_list=[8, ..., 32]`

## Tiny Smoke Launch

For the first Beaker run, disable downstream eval until an ImageNet eval dataset
has been registered:

```bash
uv run python scripts/vnext/imagenet_latent_mim/tiny.py launch imagenet_lmim_ema1_tiny ai2/jupiter \
  --dataset.root_dir=/weka/dfive-default/olmoearth/dataset/imagenet/train \
  --dataset.dataset_percentage=0.01 \
  --trainer.max_duration.steps=500 \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --launch.num_gpus=4
```

Watch for dataloader errors, GPU memory, and whether the masked patch
discrimination / InfoNCE losses move.

## Full Launch

Commit and push before launching; Beaker pulls from git.

```bash
uv run python scripts/vnext/imagenet_latent_mim/base.py launch imagenet_lmim_ema1_base ai2/jupiter \
  --dataset.root_dir=/weka/dfive-default/olmoearth/dataset/imagenet/train \
  --launch.num_gpus=8 \
  --launch.clusters='[ai2/jupiter,ai2/ceres]' \
  --trainer.callbacks.downstream_evaluator.enabled=false
```

Enable the downstream evaluator once an eval dataset named `imagenet` is
registered in the eval dataset registry.

## ImageNet Eval

`olmoearth_pretrain/evals/datasets/configs.py` has an `imagenet` config entry
for a 1000-class classification task over the `imagenet` modality. That is only
metadata. The actual loop eval still needs a registered eval dataset named
`imagenet`.

Until that registry entry exists, keep this override on training launches:

```bash
--trainer.callbacks.downstream_evaluator.enabled=false
```
