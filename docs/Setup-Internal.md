# Internal Setup Guide (AI2 Researchers)

> **This guide is for AI2 researchers with access to Beaker, Weka, and internal infrastructure.**
> External users should see [Pretraining.md](Pretraining.md) instead.

## Quick Start (10 Minutes)

If you're an AI2 researcher, follow these steps to get running quickly:

1. Configure Beaker (see [Beaker Setup](#beaker-setup))
2. Use existing datasets on Weka (see [Internal Datasets](#internal-datasets))
3. Launch via Beaker or Sessions (see [Launch Methods](#launch-methods))

---

## Beaker Setup

### 1. Create GitHub Token

Create a GitHub token that can clone this repo on Beaker. Generate a token [here](https://github.com/settings/tokens) with the following permissions:
- `repo`
- `read:packages`
- `read:org`
- `write:org`
- `read:project`

**Important:** Authorize this token for the allenai org by clicking on the "Configure SSO" dropdown [here](https://github.com/settings/tokens) for the token you created.

### 2. Configure Beaker Workspace and Budget

```bash
beaker config set default_workspace ai2/earth-systems
beaker workspace set-budget ai2/earth-systems ai2/es-platform
```

### 3. Set Beaker Secrets

```bash
ACCOUNT=$(beaker account whoami --format json | jq -r '.[0].name')
beaker secret write ${ACCOUNT}_WANDB_API_KEY <your_key>
beaker secret write ${ACCOUNT}_BEAKER_TOKEN <your_token>
beaker secret write ${ACCOUNT}_GITHUB_TOKEN <your_key>
```
/* Make sure you have `jq` installed: https://stedolan.github.io/jq/ */
```

### 4. Create Your Script


BELOW NEEDS UPDATING


Create a script based on `scripts/latent_mim.py` and configure your experiment (you can override specific changes as needed).

---

## Launch Methods

### Pre-emptible Jobs

To launch pre-emptible jobs, we use the main entrypoint in `olmoearth_pretrain/internal/experiment.py` and write python configuration files that use it like `scripts/latent_mim.py`.

**⚠️ Important:** Before launching your script, **MAKE SURE YOUR CODE IS COMMITTED AND PUSHED** as we clone the code on top of a docker image when we launch the job.

#### Launch Command

```bash
python3 scripts/base_debug_scripts/latent_mim.py launch test_run ai2/saturn-cirrascale
```

This will launch a Beaker job and stream the logs to your console until you cancel. Add additional overrides as needed.

---

### Beaker Sessions

For interactive development and debugging, you can use Beaker sessions.

#### Setup Workflow

See the [VSCode/Cursor workflow setup document](https://docs.google.com/document/d/1ydiCqIn45xlbrIcfPi8bILn_y00adTAHhIY1MPh9szE/edit?tab=t.0#heading=h.wua78h35aq1n) for detailed instructions.

#### Session Creation

When creating a session, include the following args:

```bash
--secret-env WANDB_API_KEY=<your_beaker_username>_WANDB_API_KEY \
--secret-env BEAKER_TOKEN=<your_beaker_username>_BEAKER_TOKEN
```

#### Flash Attention Setup

To use flash attention in a session:

1. Use `beaker://petew/olmo-core-tch270cu128` as your base Beaker image
2. Set up a conda environment:
   ```bash
   conda init
   exec bash
   conda shell.bash activate base
   pip install -e '.[all]'
   ```

#### Running in Sessions

For debugging in sessions, use:

```bash
torchrun scripts/base_debug_scripts/latent_mim.py train test_run local
```

Add additional overrides as needed.

---

## Internal Datasets

### Dataset Locations on Weka

Internal datasets are stored on Weka at:

```
/weka/dfive-default/helios/dataset/
```

You can reference these paths directly in your launch commands with `--dataset.h5py_dir`.

See the main [README.md](../README.md#olmoearth-pretrain-dataset) for specific dataset paths and details.

### Evaluation Datasets

Evaluation datasets have default paths configured in `olmoearth_pretrain/evals/datasets/paths.py` that point to internal AI2 infrastructure. You typically don't need to override these.

---

## Beaker Information

**Quick Reference:**
- **Budget:** `ai2/es-platform`
- **Workspace:** `ai2/earth-systems`
- **Weka:** `weka://dfive-default`

---

## Internal-Specific Gotchas

### 1. Code Must Be Committed

When launching Beaker jobs, the code is cloned fresh. Always commit and push your changes before launching.

### 2. Beaker Sessions vs Jobs

- **Jobs:** For long-running training, use pre-emptible jobs
- **Sessions:** For interactive debugging, use sessions with the torchrun command

### 3. Weka Performance

If you're experiencing slow data loading, consider using the 128x128 tile versions of datasets (4x more samples, better for GB/s bottlenecks on Weka).

---

## See Also

- [Pretraining.md](Pretraining.md) - Main training guide (launching, overrides, experiments)
- [Reference.md](Reference.md) - Configuration reference and troubleshooting
- [README.md](../README.md) - Project overview and dataset details
