# Experiment Development Guide

This guide documents best practices for developing and launching new pretraining experiments in OlmoEarth, based on production workflows.

---

## Table of Contents

1. [Planning Your Experiment](#planning-your-experiment)
2. [Architecture Changes](#architecture-changes)
3. [Creating Experiment Scripts](#creating-experiment-scripts)
4. [Testing](#testing)
5. [Launch Configuration](#launch-configuration)
6. [Git Workflow](#git-workflow)
7. [Experiment Organization](#experiment-organization)
8. [Example: Per-Modality Projection Experiments](#example-per-modality-projection-experiments)

---

## Planning Your Experiment

### 1. Define Your Hypothesis

**Be specific about what you're testing:**
- ❌ "Try to improve the model"
- ✅ "Test whether per-modality output projections improve representations by allowing modality-specific feature transformations"

### 2. Identify the Minimal Change

**Isolate the variable you're testing:**
- Change only what's necessary to test your hypothesis
- Keep all other hyperparameters identical to baseline
- This enables clear attribution of performance differences

### 3. Plan Ablations

**Test variants to understand contributions:**
- If testing encoder AND decoder changes, create separate experiments:
  - Encoder-only variant
  - Decoder-only variant
  - Both combined
- This reveals whether benefits are additive and where they come from

### 4. Document the Plan

Create a plan document (e.g., `.plan.md`) that includes:
- Goal/hypothesis
- Architecture changes
- Files to create/modify
- Expected outcomes

---

## Architecture Changes

### Principle: Subclass, Don't Modify

**Always subclass existing classes rather than modifying them:**

```python
# ✅ GOOD: Subclass existing encoder
class EncoderWithPerModalityProjection(Encoder):
    """Encoder with per-modality transformations after attention."""

    def __init__(self, ...):
        super().__init__(...)
        # Add your new functionality
        self.per_modality_transforms = nn.ModuleDict()
        for modality_name in self.supported_modality_names:
            self.per_modality_transforms[modality_name] = nn.Linear(...)

# ✅ GOOD: Subclass config too
@dataclass
class EncoderWithPerModalityProjectionConfig(EncoderConfig):
    """Configuration for EncoderWithPerModalityProjection."""

    def build(self) -> "EncoderWithPerModalityProjection":
        # Custom build logic
        return EncoderWithPerModalityProjection(**kwargs)
```

**Why subclassing?**
- ✅ Zero risk of breaking existing experiments
- ✅ No new required parameters in old configs
- ✅ Easy to compare: old vs new is explicit
- ✅ Can reuse all parent functionality

### Override Only What Changes

**Minimize the surface area of your changes:**

```python
class EncoderWithPerModalityProjection(Encoder):
    def __init__(self, ...):
        super().__init__(...)
        # Only add what's new

    def apply_attn(self, ...):
        # Only override methods that need to change
        # Reuse parent logic wherever possible
```

### Maintain Code Correctness

**Get the order of operations right:**
- Draw out the data flow
- Verify tensor shapes at each step
- Test with simplified inputs before full integration

Example from per-modality projection:
```python
# ✅ CORRECT ORDER:
# 1. Add removed tokens back (restore full sequence)
# 2. Split to per-modality
# 3. Apply per-modality transformations
# 4. Recombine
# 5. Apply shared normalization
# 6. Split back to per-modality

# ❌ WRONG: Applying transforms before adding tokens back
# This would apply transforms to incomplete sequences
```

---

## Creating Experiment Scripts

### Structure of an Experiment Script

Each experiment script should:

```python
"""Brief description of what this experiment tests."""

import logging

from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    YourNewEncoderConfig,
    YourNewDecoderConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# Copy constants from base.py
MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with your changes."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    # Use your new encoder config
    encoder_config = YourNewEncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        # ... all the same params as base.py
    )

    # Keep decoder standard (or use your new decoder config)
    decoder_config = PredictorConfig(
        # ... all the same params as base.py
    )

    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,  # Your custom builder
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
```

### Naming Convention

**Use descriptive names that indicate what's being tested:**

```
base_<component>_<modification>.py

Examples:
- base_encoder_per_mod_proj.py
- base_decoder_per_mod_proj.py
- base_both_per_mod_proj.py
- base_no_contrastive.py
- base_s2_only.py
```

### Keep Hyperparameters Identical

**Copy all hyperparameters from the baseline exactly:**
- Same learning rate
- Same batch size
- Same optimizer settings
- Same training duration
- Same random seeds (if applicable)

This ensures any performance difference is due to your architectural change, not a hyperparameter difference.

---

## Testing

### Write Unit Tests First

**Test your new components in isolation before full training:**

```python
class TestYourNewClass:
    """Test your new encoder/decoder."""

    def test_initialization(self) -> None:
        """Test that your class initializes correctly."""
        model = YourNewClass(...)
        assert hasattr(model, "your_new_attribute")

    def test_config_builds_correctly(self) -> None:
        """Test that config builds the right class."""
        config = YourNewConfig(...)
        model = config.build()
        assert isinstance(model, YourNewClass)

    def test_forward_pass(self) -> None:
        """Test forward pass with mock data."""
        # Use simple tensor inputs, not full data pipeline
        input_tokens = torch.randn(B, H, W, T, C, D)
        output = model.some_method(input_tokens)
        assert output.shape == expected_shape

    def test_gradients_flow(self) -> None:
        """Test that gradients are computed correctly."""
        # Verify your parameters are learnable
        output = model(...)
        loss = output.sum()
        loss.backward()
        assert model.your_parameter.grad is not None
```

### Test Execution Order

Run the unit tests:
```bash
source .venv/bin/activate
pytest tests/unit/nn/test_your_new_class.py -vv
```

### Run Linting

Always run pre-commit before committing:
```bash
pre-commit run --all-files
```

---

## Launch Configuration

### Create a Launch Script

**Make it easy to launch all variants at once:**

```bash
#!/bin/bash
# Launch script for <experiment_name>

set -e  # Exit on error

CLUSTERS='[ai2/jupiter,ai2/ceres]'
NUM_GPUS=8
PRIORITY=normal
WANDB_PROJECT=YYYY_MM_DD_experiment_name

echo "=================================="
echo "Launching <Experiment Name>"
echo "Clusters: ${CLUSTERS}"
echo "GPUs: ${NUM_GPUS}"
echo "Priority: ${PRIORITY}"
echo "W&B Project: ${WANDB_PROJECT}"
echo "=================================="
echo ""

# Launch variant 1
echo "Launching variant 1/3: <description>..."
python3 scripts/official/your_script_v1.py launch run_name_v1 ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" \
  --launch.priority=${PRIORITY} \
  --trainer.callbacks.wandb.project=${WANDB_PROJECT}
echo "✓ Variant 1 launched"
echo ""

# Repeat for other variants...

echo "All experiments launched successfully!"
```

### Launch Script Best Practices

1. **Set variables at the top** - Makes it easy to modify
2. **Use descriptive run names** - Include variant info
3. **Group related experiments** - Same W&B project
4. **Set appropriate priority** - `normal` for experiments, `high` only when needed
5. **Make it executable** - `chmod +x script.sh`
6. **Include progress messages** - User knows what's happening

### Launch Flags to Always Include

```bash
--launch.num_gpus=8                                    # Match base model
--launch.clusters="[ai2/jupiter,ai2/ceres]"           # Multi-cluster for availability
--launch.priority=normal                               # Appropriate priority
--trainer.callbacks.wandb.project=YYYY_MM_DD_exp_name # Organize experiments
```

---

## Git Workflow

### Branch Naming

```
<username>/<descriptive-name>

Example: henryh/per-modality-output-projection
```

### Commit Strategy

1. **Create the branch first:**
   ```bash
   git checkout -b username/experiment-name
   ```

2. **Make incremental commits:**
   ```bash
   git add <files>
   git commit -m "Descriptive message"
   ```

3. **Good commit messages:**
   ```
   Add per-modality projection experiments

   - Add EncoderWithPerModalityProjection: applies per-modality linear transforms
   - Add PredictorWithPerModalityOutput: uses per-modality output heads
   - Create 3 experiment scripts: encoder-only, decoder-only, and both
   - Add unit tests validating per-modality transforms

   All experiments maintain identical hyperparameters to base.py for comparison.
   ```

### Push Before Launch

**⚠️ CRITICAL: Always commit and push before launching Beaker jobs**

```bash
git push origin username/experiment-name
```

Beaker will pull your code from the repository, so uncommitted changes won't be included!

---

## Experiment Organization

### W&B Project Naming

**Use date-prefixed project names:**
```
YYYY_MM_DD_experiment_description

Examples:
- 2025_11_21_per_modality_projection_experiments
- 2025_11_15_ablation_studies
- 2025_10_30_patch_size_experiments
```

**Benefits:**
- ✅ Chronologically sorted
- ✅ Easy to find related runs
- ✅ Clear experiment grouping

### Run Naming

**Include variant information in run names:**
```
<base_model>_<variant_description>

Examples:
- base_encoder_per_mod_proj
- base_decoder_per_mod_proj
- base_both_per_mod_proj
- base_no_contrastive
```

### Documentation

After launching, document:
1. **Hypothesis** - What you're testing
2. **Changes** - What's different from baseline
3. **Expected outcome** - What would validate your hypothesis
4. **Run names** - How to find the runs in W&B
5. **Comparison baseline** - What to compare against

---

## Example: Per-Modality Projection Experiments

This section walks through the complete workflow for the per-modality projection experiments.

### 1. Hypothesis

**Question:** Do modality-specific transformations improve representations by allowing the model to learn specialized feature refinements for each modality?

**Reasoning:** Different modalities (optical vs SAR vs elevation) have fundamentally different characteristics. Shared projections force the model to use the same transformation for all modalities, potentially limiting representation quality.

### 2. Architecture Design

**Encoder change:**
- Add `nn.Linear` transformation per modality after attention, before final norm
- Each modality gets its own learnable `embedding_size → embedding_size` transformation
- Shared normalization ensures stable training

**Decoder change:**
- Replace single `to_output_embed` with per-modality output heads
- Each modality gets its own `decoder_embedding_size → output_embedding_size` projection
- Allows modality-specific output space transformations

### 3. Implementation

```python
# Step 1: Create new encoder class (subclass existing)
class EncoderWithPerModalityProjection(Encoder):
    def __init__(self, ...):
        super().__init__(...)
        self.per_modality_transforms = nn.ModuleDict()
        for modality_name in self.supported_modality_names:
            self.per_modality_transforms[modality_name] = nn.Linear(
                embedding_size, embedding_size, bias=True
            )

    def apply_attn(self, ...):
        # Override to insert per-modality transforms
        # ... parent attention logic ...

        # Add removed tokens back (full sequence)
        tokens = self._maybe_add_removed_tokens(...)

        # Split to per-modality and apply transforms
        tokens_per_modality = self.split_and_expand_per_modality(...)
        for modality, modality_tokens in tokens_per_modality.items():
            if modality in self.per_modality_transforms:
                tokens_per_modality[modality] = self.per_modality_transforms[
                    modality
                ](modality_tokens)

        # Recombine and normalize
        tokens, _ = self.collapse_and_combine_hwtc(tokens_per_modality)
        tokens = self.norm(tokens)

        # Split back to per-modality format
        return self.split_and_expand_per_modality(...)

# Step 2: Create config class
@dataclass
class EncoderWithPerModalityProjectionConfig(EncoderConfig):
    def build(self) -> "EncoderWithPerModalityProjection":
        return EncoderWithPerModalityProjection(**kwargs)

# Step 3: Repeat for decoder
class PredictorWithPerModalityOutput(PredictorBase):
    # Similar pattern...
```

### 4. Create Experiment Scripts

Three scripts to isolate effects:
- `base_encoder_per_mod_proj.py` - Only encoder has per-modality projections
- `base_decoder_per_mod_proj.py` - Only decoder has per-modality outputs
- `base_both_per_mod_proj.py` - Both have per-modality projections

Each script:
- Imports from `script.py` (reuse common components)
- Only defines custom `build_model_config()`
- Keeps all hyperparameters identical to `base.py`

### 5. Write Tests

```python
# Test initialization
def test_encoder_initialization():
    encoder = EncoderWithPerModalityProjection(...)
    assert len(encoder.per_modality_transforms) == num_modalities
    assert "sentinel2_l2a" in encoder.per_modality_transforms

# Test gradients
def test_per_modality_transforms_are_learnable():
    encoder = EncoderWithPerModalityProjection(...)
    input_tokens = torch.randn(..., requires_grad=True)
    output = encoder.per_modality_transforms["sentinel2_l2a"](input_tokens)
    output.sum().backward()
    assert encoder.per_modality_transforms["sentinel2_l2a"].weight.grad is not None
```

### 6. Create Launch Script

```bash
#!/bin/bash
set -e

CLUSTERS='[ai2/jupiter,ai2/ceres]'
NUM_GPUS=8
PRIORITY=normal
WANDB_PROJECT=2025_11_21_per_modality_projection_experiments

python3 scripts/official/base_encoder_per_mod_proj.py launch base_encoder_per_mod_proj ai2/jupiter \
  --launch.num_gpus=${NUM_GPUS} \
  --launch.clusters="${CLUSTERS}" \
  --launch.priority=${PRIORITY} \
  --trainer.callbacks.wandb.project=${WANDB_PROJECT}

# Repeat for decoder-only and both variants...
```

### 7. Git Workflow

```bash
# Create branch
git checkout -b henryh/per-modality-output-projection

# Add files
git add olmoearth_pretrain/nn/flexi_vit.py
git add scripts/official/base_*_per_mod_proj.py
git add tests/unit/nn/test_encoder_per_modality_projection.py
git add scripts/official/launch_per_modality_experiments.sh

# Commit with descriptive message
git commit -m "Add per-modality projection experiments

- Add EncoderWithPerModalityProjection: per-modality transforms after attention
- Add PredictorWithPerModalityOutput: per-modality output heads
- Create 3 variants: encoder-only, decoder-only, both
- Add unit tests (4 tests, all passing)
- Add launch script for easy deployment

All experiments maintain identical hyperparameters to base.py."

# Push before launching
git push origin henryh/per-modality-output-projection
```

### 8. Launch Experiments

```bash
# Launch all three variants
bash scripts/official/launch_per_modality_experiments.sh
```

### 9. Monitor and Compare

**In W&B:**
- Project: `2025_11_21_per_modality_projection_experiments`
- Runs:
  - `base_encoder_per_mod_proj`
  - `base_decoder_per_mod_proj`
  - `base_both_per_mod_proj`
- Baseline: `base` (from main experiments)

**Compare metrics:**
- Pretraining loss curves
- Downstream task performance
- Convergence speed
- Parameter efficiency (added parameters vs performance gain)

---

## Checklist for New Experiments

Use this checklist for every new experiment:

### Planning Phase
- [ ] Hypothesis clearly stated
- [ ] Minimal change identified
- [ ] Ablation variants planned
- [ ] Plan documented

### Implementation Phase
- [ ] New classes subclass existing ones
- [ ] Configs subclass existing configs
- [ ] Only necessary methods overridden
- [ ] Tensor shapes verified at each step
- [ ] Order of operations correct

### Testing Phase
- [ ] Unit tests written and passing
- [ ] Linting passes (`pre-commit run --all-files`)
- [ ] Gradients verified to flow correctly

### Script Creation Phase
- [ ] Experiment scripts created
- [ ] All hyperparameters match baseline
- [ ] Only `build_model_config()` differs
- [ ] Scripts tested with `dry_run` subcommand

### Launch Phase
- [ ] Launch script created
- [ ] Appropriate priority set
- [ ] W&B project name includes date
- [ ] Run names are descriptive
- [ ] Code committed and pushed
- [ ] Launch script tested

### Documentation Phase
- [ ] Hypothesis documented
- [ ] Changes documented
- [ ] Run names recorded
- [ ] Baseline for comparison identified

---

## Common Pitfalls to Avoid

### ❌ Modifying Existing Classes Directly
**Problem:** Breaks existing experiments and configs
**Solution:** Always subclass

### ❌ Changing Multiple Things at Once
**Problem:** Can't attribute performance changes
**Solution:** Isolate variables with separate experiments

### ❌ Different Hyperparameters from Baseline
**Problem:** Unfair comparison
**Solution:** Copy all hyperparameters exactly

### ❌ Forgetting to Push Code Before Launch
**Problem:** Beaker runs old code
**Solution:** Always `git push` before launching

### ❌ Using Generic Run Names
**Problem:** Hard to find and compare runs
**Solution:** Use descriptive names with variant info

### ❌ Not Testing with Simple Inputs First
**Problem:** Hard to debug in full training
**Solution:** Write unit tests with mock data

### ❌ Incorrect Tensor Operation Order
**Problem:** Silent bugs, wrong results
**Solution:** Verify shapes and logic at each step

---

## Additional Resources

- [Pretraining Guide](Pretraining.md) - How to launch training jobs
- [Evaluation Guide](Evaluation.md) - How to evaluate models
- [Dataset Creation](Dataset-Creation.md) - How to create datasets
- [Setup Internal](Setup-Internal.md) - AI2 Beaker setup

---

**Remember:** Good experiments are:
1. **Minimal** - Change only what's necessary
2. **Comparable** - Keep everything else identical
3. **Testable** - Verify components work in isolation
4. **Organized** - Group related runs, use clear names
5. **Documented** - Record what you tested and why
