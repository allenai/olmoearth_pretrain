"""Mode A loss noise diagnosis runner.

Loads a checkpoint, iterates the real `OlmoEarthDataLoader`, runs the same
forward pass as training (encoder + decoder + target encoder), and records
per-sample, per-modality loss statistics to parquet on weka.

Triggered via the standard experiment framework:

    # Dry-run (validate config):
    python scripts/official/base.py dry_run_loss_diagnose run_name local \
        --loss_diagnose.checkpoint_path=/weka/.../step500000 \
        --loss_diagnose.output_dir=/weka/.../loss_noise/run

    # Local (8x GPU via torchrun):
    torchrun --nproc_per_node=8 scripts/official/base.py loss_diagnose run_name local \
        --loss_diagnose.checkpoint_path=/weka/.../step500000 \
        --loss_diagnose.output_dir=/weka/.../loss_noise/run

    # Beaker launch:
    python scripts/official/base.py launch_loss_diagnose run_name ai2/jupiter \
        --launch.num_gpus=8 \
        --loss_diagnose.checkpoint_path=/weka/.../step500000 \
        --loss_diagnose.output_dir=/weka/.../loss_noise/run
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.utils import get_default_device, seed_all
from torch import Tensor

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.utils import unpack_encoder_output

if TYPE_CHECKING:
    from olmoearth_pretrain.internal.experiment import OlmoEarthExperimentConfig

logger = logging.getLogger(__name__)


# =============================================================================
# User-facing config (subcommand override target)
# =============================================================================


@dataclass
class LossNoiseDiagnoseConfig(Config):
    """Options for the `loss_diagnose` subcommand.

    Override on the CLI with `--loss_diagnose.<field>=...`.
    """

    checkpoint_path: str = ""
    output_dir: str = ""
    max_steps: int = 20000
    flush_every_steps: int = 100
    log_every: int = 50

    # Repeated-sample mode: evaluate specific samples many times with different
    # random masks/modality selections to measure per-sample loss distributions.
    # Set num_diagnosis_samples > 0 to randomly pick samples from the dataset,
    # or provide sample_indices_file for specific indices.
    num_diagnosis_samples: int = 0
    sample_indices_file: str = ""
    num_repeats_per_sample: int = 100

    def check_required(self) -> None:
        """Verify required fields are set. Called by the runner, not Config.apply()."""
        if not self.checkpoint_path:
            raise ValueError("loss_diagnose.checkpoint_path must be set")
        if not self.output_dir:
            raise ValueError("loss_diagnose.output_dir must be set")
        if self.max_steps <= 0:
            raise ValueError("loss_diagnose.max_steps must be > 0")
        if self.sample_indices_file:
            p = Path(self.sample_indices_file)
            if not p.exists():
                raise ValueError(f"sample_indices_file not found: {p}")
        if (
            self.num_diagnosis_samples > 0 or self.sample_indices_file
        ) and self.num_repeats_per_sample < 1:
            raise ValueError("num_repeats_per_sample must be >= 1")


# =============================================================================
# Instrumented loss: same math as ModalityPatchDiscriminationLoss but emits
# per-sample, per-modality records instead of discarding them.
# =============================================================================


@dataclass
class _PerModalityRecord:
    modality: str
    h5_index: int
    sample_idx: int
    total_tokens: int
    num_present_tokens: int
    num_missing_tokens: int
    decoder_token_count: int
    sample_loss: float


@dataclass
class _BatchRecord:
    total_loss: float
    contributing_modality_count: int
    modality_losses: dict[str, float]
    present_modalities: list[str]


class InstrumentedModalityPatchDiscLoss:
    """Mirrors `ModalityPatchDiscriminationLoss` but records per-sample stats."""

    def __init__(self, tau: float = 0.1, pred2unit: bool = False, weight: float = 1.0):
        self.tau = tau
        self.pred2unit = pred2unit
        self.weight = weight
        self.last_sample_records: list[_PerModalityRecord] = []
        self.last_batch_record: _BatchRecord | None = None

    def compute(
        self,
        predictions: TokensAndMasks,
        targets: TokensAndMasks,
        h5_indices: Tensor | None,
    ) -> Tensor:
        modality_preds, modality_masks = (
            predictions.flatten_tokens_and_masks_per_modality()
        )
        modality_targets = targets.flatten_tokens_and_masks_per_modality()[0]

        sample_records: list[_PerModalityRecord] = []
        modality_losses: dict[str, float] = {}
        contributing = 0
        total_loss = torch.tensor(0.0, device=predictions.device)

        for all_preds, all_masks, all_targets, modality in zip(
            modality_preds, modality_masks, modality_targets, targets.modalities
        ):
            decoder_mask = all_masks == MaskValue.DECODER.value
            pred = all_preds[decoder_mask].unsqueeze(dim=0)
            target = all_targets[decoder_mask].unsqueeze(dim=0)

            if self.pred2unit:
                pred_mu = pred.mean(1, keepdims=True)
                pred_std = pred.std(1, keepdims=True)
                pred = (pred - pred_mu) / (pred_std + 1e-4)

            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)

            decoder_counts = decoder_mask.sum(dim=-1)
            missing_counts = (all_masks == MaskValue.MISSING.value).sum(dim=-1)
            total_per_sample = int(all_masks.shape[-1])

            modality_sample_losses: list[Tensor] = []
            start = 0
            for sample_i, (c, n_missing) in enumerate(
                zip(decoder_counts, missing_counts)
            ):
                c_val = int(c.item())
                n_miss = int(n_missing.item())
                h5_idx = (
                    int(h5_indices[sample_i].item()) if h5_indices is not None else -1
                )

                if c_val == 0:
                    sample_records.append(
                        _PerModalityRecord(
                            modality=modality,
                            h5_index=h5_idx,
                            sample_idx=sample_i,
                            total_tokens=total_per_sample,
                            num_present_tokens=total_per_sample - n_miss,
                            num_missing_tokens=n_miss,
                            decoder_token_count=0,
                            sample_loss=float("nan"),
                        )
                    )
                    continue

                end = start + c_val
                pred_s = pred[:, start:end, :]
                target_s = target[:, start:end, :]
                score = torch.einsum("npd,nqd->npq", pred_s, target_s) / self.tau
                labels = torch.arange(c_val, dtype=torch.long, device=pred.device)[None]
                token_losses = F.cross_entropy(
                    score.flatten(0, 1),
                    labels.flatten(0, 1),
                    reduction="none",
                ) * (self.tau * 2)
                sample_loss = token_losses.mean()
                modality_sample_losses.append(sample_loss)

                sample_records.append(
                    _PerModalityRecord(
                        modality=modality,
                        h5_index=h5_idx,
                        sample_idx=sample_i,
                        total_tokens=total_per_sample,
                        num_present_tokens=total_per_sample - n_miss,
                        num_missing_tokens=n_miss,
                        decoder_token_count=c_val,
                        sample_loss=float(sample_loss.detach().item()),
                    )
                )
                start = end

            if not modality_sample_losses:
                continue

            mod_loss = torch.stack(modality_sample_losses).mean()
            modality_losses[modality] = float(mod_loss.detach().item())
            contributing += 1
            total_loss = total_loss + mod_loss

        self.last_sample_records = sample_records
        self.last_batch_record = _BatchRecord(
            total_loss=float((self.weight * total_loss).detach().item()),
            contributing_modality_count=contributing,
            modality_losses=modality_losses,
            present_modalities=list(modality_losses.keys()),
        )
        return self.weight * total_loss


# =============================================================================
# Buffered parquet writer (flush hook)
# =============================================================================


class BufferedParquetWriter:
    """Append per-step records; flush to a new parquet file every N steps.

    Each flush produces a separate file, so a crash at step 19,500 still leaves
    all previously-flushed chunks intact on weka.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike,
        prefix: str,
        rank: int,
        flush_every_steps: int,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.rank = rank
        self.flush_every_steps = flush_every_steps
        self.buffer: list[dict[str, Any]] = []
        self.steps_since_flush = 0
        self.flush_count = 0
        self.total_rows_written = 0

    def add(self, records: list[dict[str, Any]]) -> None:
        self.buffer.extend(records)
        self.steps_since_flush += 1
        if self.steps_since_flush >= self.flush_every_steps:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            self.steps_since_flush = 0
            return
        df = pd.DataFrame(self.buffer)
        path = (
            self.output_dir
            / f"{self.prefix}_rank{self.rank:02d}_chunk{self.flush_count:06d}.parquet"
        )
        df.to_parquet(path, index=False, compression="snappy")
        self.total_rows_written += len(df)
        logger.info(
            f"[rank {self.rank}] flush {self.flush_count}: "
            f"+{len(df)} rows -> {path.name} "
            f"(total: {self.total_rows_written})"
        )
        self.buffer.clear()
        self.flush_count += 1
        self.steps_since_flush = 0


# =============================================================================
# Helpers
# =============================================================================


def _extract_step(checkpoint_path: str) -> int:
    m = re.search(r"step(\d+)/?$", checkpoint_path.rstrip("/"))
    return int(m.group(1)) if m else -1


# =============================================================================
# Main entry point (called from experiment.py SubCmd dispatch)
# =============================================================================


def run_loss_diagnose(config: OlmoEarthExperimentConfig) -> None:
    """Run Mode A loss noise diagnosis using the train script's builders.

    Expects `config.loss_diagnose` to be set with at least `checkpoint_path` and
    `output_dir`. Reuses `config.model`, `config.dataset`, `config.data_loader`,
    and `config.train_module` exactly as training does, so the forward pass
    matches what the trainer would run.
    """
    if config.loss_diagnose is None:
        raise ValueError(
            "config.loss_diagnose is required; set --loss_diagnose.checkpoint_path "
            "and --loss_diagnose.output_dir on the CLI"
        )
    diag = config.loss_diagnose
    diag.check_required()

    seed_all(config.init_seed)
    device = get_default_device()
    rank = get_rank()
    world_size = get_world_size()
    is_rank0 = rank == 0
    logger.info(f"rank={rank}/{world_size} device={device}")

    # Build model
    model = config.model.build()
    model = model.to(device)

    # Load checkpoint
    ckpt_dir = os.path.join(diag.checkpoint_path, "model_and_optim")
    logger.info(f"Loading checkpoint from {ckpt_dir}")
    load_model_and_optim_state(ckpt_dir, model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Build dataset + dataloader using the train script's exact configs.
    # We force num_masked_views=1 since we only need a single view for diagnosis,
    # and override the dataloader work_dir to live under the diagnosis output dir.
    config.data_loader.num_masked_views = 1
    config.data_loader.work_dir = str(Path(diag.output_dir) / "_dl_workdir")

    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset)
    data_loader.reshuffle(epoch=1, in_memory=True)

    # Repeated-sample mode: override global indices with repeated specific samples
    repeated_mode = diag.num_diagnosis_samples > 0 or diag.sample_indices_file
    if repeated_mode:
        if diag.sample_indices_file:
            sample_indices_path = Path(diag.sample_indices_file)
            if sample_indices_path.suffix == ".npy":
                target_indices = np.load(sample_indices_path).astype(np.uint32).ravel()
            else:
                target_indices = np.loadtxt(
                    sample_indices_path, dtype=np.uint32
                ).ravel()
        else:
            rng = np.random.default_rng(config.init_seed)
            all_indices = np.arange(len(dataset), dtype=np.uint32)
            target_indices = rng.choice(
                all_indices, size=diag.num_diagnosis_samples, replace=False
            )
            # Save chosen indices for reproducibility
            idx_path = Path(diag.output_dir) / "sampled_indices.npy"
            idx_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(idx_path, target_indices)
            if is_rank0:
                logger.info(f"Saved sampled indices to {idx_path}")

        n_samples = len(target_indices)
        repeated = np.repeat(target_indices, diag.num_repeats_per_sample)
        total = (
            len(repeated) // data_loader.global_batch_size
        ) * data_loader.global_batch_size
        repeated = repeated[:total]
        data_loader._global_indices = repeated
        n_batches = total // data_loader.global_batch_size
        if is_rank0:
            logger.info(
                f"Repeated-sample mode: {n_samples} samples x "
                f"{diag.num_repeats_per_sample} repeats = {len(repeated)} instances "
                f"({n_batches} batches)"
            )
        diag.max_steps = max(diag.max_steps, n_batches)

    # Extract loss params from train_module config
    loss_cfg = config.train_module.loss_config.loss_config
    tau = float(loss_cfg.get("tau", 0.1))
    pred2unit = bool(loss_cfg.get("pred2unit", False))
    token_exit_cfg = getattr(config.train_module, "token_exit_cfg", {})
    logger.info(f"Loss config: tau={tau}, pred2unit={pred2unit}")

    instrumented = InstrumentedModalityPatchDiscLoss(tau=tau, pred2unit=pred2unit)

    # Load latlon distribution for geographic lookup
    h5py_dir = Path(getattr(config.dataset, "h5py_dir", ""))
    latlon_path = h5py_dir / "latlon_distribution.npy"
    latlon_dist = None
    if latlon_path.exists():
        latlon_dist = np.load(latlon_path)
        logger.info(f"Loaded latlon distribution: {latlon_dist.shape}")
    else:
        logger.warning(f"No latlon distribution at {latlon_path}")

    checkpoint_step = _extract_step(diag.checkpoint_path)

    sample_writer = BufferedParquetWriter(
        diag.output_dir,
        prefix=f"per_sample_step{checkpoint_step}",
        rank=rank,
        flush_every_steps=diag.flush_every_steps,
    )
    batch_writer = BufferedParquetWriter(
        diag.output_dir,
        prefix=f"per_batch_step{checkpoint_step}",
        rank=rank,
        flush_every_steps=diag.flush_every_steps,
    )

    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    t_start = time.monotonic()
    t_last_log = t_start
    step = 0

    for batch in data_loader:
        if step >= diag.max_steps:
            break

        # Single-view collator returns (patch_size, MaskedOlmoEarthSample)
        patch_size, masked_batch = batch
        masked_batch = masked_batch.to_device(device)
        h5_indices = masked_batch.h5_indices

        with torch.no_grad(), autocast_ctx:
            _, decoded, _, _, _ = model(masked_batch, patch_size)
            target_dict = model.target_encoder.forward(
                masked_batch.unmask(),
                patch_size=patch_size,
                token_exit_cfg=token_exit_cfg,
            )
            target_output, _, _ = unpack_encoder_output(target_dict)
            instrumented.compute(decoded, target_output, h5_indices=h5_indices)

        num_timestamps = (
            int(masked_batch.timestamps.shape[1])
            if masked_batch.timestamps is not None
            else -1
        )

        sample_rows: list[dict[str, Any]] = []
        for rec in instrumented.last_sample_records:
            row: dict[str, Any] = {
                "checkpoint_step": checkpoint_step,
                "step": step,
                "rank": rank,
                "h5_index": rec.h5_index,
                "sample_idx": rec.sample_idx,
                "modality": rec.modality,
                "total_tokens": rec.total_tokens,
                "num_present_tokens": rec.num_present_tokens,
                "num_missing_tokens": rec.num_missing_tokens,
                "decoder_token_count": rec.decoder_token_count,
                "sample_loss": rec.sample_loss,
                "patch_size": int(patch_size),
                "num_timestamps": num_timestamps,
            }
            if latlon_dist is not None and 0 <= rec.h5_index < len(latlon_dist):
                row["lat"] = float(latlon_dist[rec.h5_index, 0])
                row["lon"] = float(latlon_dist[rec.h5_index, 1])
            else:
                row["lat"] = float("nan")
                row["lon"] = float("nan")
            sample_rows.append(row)
        sample_writer.add(sample_rows)

        br = instrumented.last_batch_record
        if br is not None:
            batch_row = {
                "checkpoint_step": checkpoint_step,
                "step": step,
                "rank": rank,
                "patch_size": int(patch_size),
                "batch_size": int(masked_batch.batch_size),
                "num_timestamps": num_timestamps,
                "total_loss": br.total_loss,
                "contributing_modality_count": br.contributing_modality_count,
                "present_modalities": ",".join(br.present_modalities),
            }
            for mod, val in br.modality_losses.items():
                batch_row[f"loss_{mod}"] = val
            batch_writer.add([batch_row])

        step += 1

        if is_rank0 and (step % diag.log_every == 0):
            now = time.monotonic()
            interval = now - t_last_log
            total = now - t_start
            steps_per_s = diag.log_every / max(interval, 1e-6)
            remaining = (diag.max_steps - step) / max(steps_per_s, 1e-6)
            logger.info(
                f"step {step}/{diag.max_steps} "
                f"({steps_per_s:.2f} steps/s, elapsed {total / 60:.1f}m, "
                f"eta {remaining / 60:.1f}m) "
                f"loss={br.total_loss:.4f} contrib_mods={br.contributing_modality_count}"
            )
            t_last_log = now

    sample_writer.flush()
    batch_writer.flush()

    if is_rank0:
        total_time = time.monotonic() - t_start
        logger.info(
            f"Done. {step} steps in {total_time / 60:.1f} min "
            f"({step / max(total_time, 1e-6):.2f} steps/s)"
        )
