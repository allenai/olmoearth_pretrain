"""Multi-objective train module for the daily ERA5 encoder.

This is the central piece that wires the dedicated `Era5DailyEncoder`
(`nn/era5_encoder.py`) to per-objective heads and losses.  The module is
structured around an `_Objective` interface so that objectives can be
freely composed (A only, B only, A+B, and eventually C).

Objectives implemented:

* **A — Supervised** (`SupervisedObjective`): multi-task classification /
  regression over pooled encoder embeddings.
* **B — Reconstruction** (`ReconstructionObjective`): corrupt the raw ERA5
  input, encode, decode via time-query cross-attention, and supervise with
  Huber + band-normalized undecimated-SWT multiscale loss.

Each training step runs *all* objectives whose `applies_to(batch)` returns
True, accumulates their weighted losses, and does a single backward pass.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from torch import Tensor, nn
from torch.distributed import DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.multi_task_era5_dataset import Era5SupervisedBatch
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.era5_decoder import (
    Era5TimeQueryDecoder,
    Era5TimeQueryDecoderConfig,
)
from olmoearth_pretrain.nn.era5_encoder import (
    Era5DailyEncoder,
    Era5DailyEncoderConfig,
)
from olmoearth_pretrain.nn.era5_heads import SupervisedHeadRegistry, build_head
from olmoearth_pretrain.nn.transforms.era5_corruption import (
    CorruptionConfig,
    corrupt_era5,
)
from olmoearth_pretrain.nn.transforms.era5_swt import (
    StationaryWaveletTransform1d,
    multiscale_swt_loss,
)
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model wrapper (encoder + per-objective sub-modules)
# ---------------------------------------------------------------------------


class Era5MultiObjectiveModel(nn.Module):
    """Top-level model holding the encoder and per-objective sub-modules.

    The `OlmoEarthTrainModule` base class introspects `self.model` for things
    like DDP wrapping, parameter counts, and optionally checkpoint
    compatibility. By exposing `apply_ddp` / `apply_compile` here we plug
    straight into that machinery.

    Per-objective sub-modules (e.g. the supervised head registry) are
    registered as a `nn.ModuleDict` so the optimizer + DDP/compile
    machinery picks them up automatically. The list of `_Objective`
    instances is stashed on the model so the train module can pull the
    same instances back out after the model has been DDP-wrapped —
    avoiding a duplicate head registry.
    """

    def __init__(
        self,
        encoder: Era5DailyEncoder,
        objectives: list[_Objective] | None = None,
    ) -> None:
        """Initialize with an encoder and optional objective modules."""
        super().__init__()
        self.encoder = encoder
        objective_modules: dict[str, nn.Module] = {}
        objectives = objectives or []
        for objective in objectives:
            module = objective.get_module()
            if module is not None:
                if objective.name in objective_modules:
                    raise KeyError(f"Objective {objective.name!r} already registered")
                objective_modules[objective.name] = module
        self.objectives = nn.ModuleDict(objective_modules)
        # Plain python attribute (not a parameter container) so DDP doesn't
        # try to treat the `_Objective` instances as submodules.
        self._objective_list: list[_Objective] = list(objectives)

    @property
    def objective_list(self) -> list[_Objective]:
        """The objectives that participate in `train_batch`."""
        return self._objective_list

    def add_objective(self, objective: _Objective) -> None:
        """Register a new objective post-construction (currently unused)."""
        if any(obj.name == objective.name for obj in self._objective_list):
            raise KeyError(f"Objective {objective.name!r} already registered")
        module = objective.get_module()
        if module is not None:
            self.objectives[objective.name] = module
        self._objective_list.append(objective)

    # -- DDP / compile plumbing ------------------------------------------------

    def apply_ddp(
        self,
        dp_mesh: DeviceMesh | None = None,
        compile_enabled: bool = False,
        find_unused_parameters: bool = True,
    ) -> None:
        """Apply DDP in-place using olmo-core's composable replicate pattern.

        Matches the convention used by `DistributedMixins.apply_ddp` so that
        `OlmoEarthTrainModule` can keep using `self.model` after wrapping.
        """
        from torch.distributed._composable.replicate import replicate

        if compile_enabled:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"  # type: ignore[attr-defined]
        replicate(
            self,
            device_mesh=dp_mesh,
            bucket_cap_mb=100,
            find_unused_parameters=find_unused_parameters,
        )

    def apply_compile(self) -> None:
        """Apply torch.compile to the encoder + each objective module."""
        self.encoder.apply_compile()
        for module in self.objectives.values():
            if hasattr(module, "apply_compile"):
                module.apply_compile()
            else:
                module.compile(dynamic=True)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward delegates to the encoder by default.

        The train module never calls `self.model(...)` directly — it calls
        the encoder/objective explicitly inside `_Objective.compute_loss`.
        This method exists so that DDP can wrap the model without
        complaining about a missing forward.
        """
        return self.encoder(*args, **kwargs)


# ---------------------------------------------------------------------------
# Objective interface
# ---------------------------------------------------------------------------


class _Objective(ABC):
    """An objective owns its own loss head(s) and metric routing.

    Objectives are intentionally tiny: they pull what they need from the
    batch, run the encoder, compute their loss, and return both the loss
    tensor and a dict of metrics. The train module owns the optimization
    loop (microbatching, DDP context, backward, optimizer step).
    """

    name: str
    weight: float

    def applies_to(self, batch: Any) -> bool:
        """Return True if this objective should fire for *batch*.

        The default returns ``True`` unconditionally.  Subclasses override
        to restrict to specific batch types (e.g. supervised requires
        labels).
        """
        return True

    @abstractmethod
    def compute(
        self,
        encoder: Era5DailyEncoder,
        batch: Any,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the objective's loss and metrics for one microbatch."""

    def get_module(self) -> nn.Module | None:
        """Return any nn.Module owned by the objective (heads, decoders).

        The returned module is registered on the top-level model so its
        parameters are tracked by the optimizer + DDP/compile passes.
        """
        return None


# ---------------------------------------------------------------------------
# Objective A: multi-task supervised
# ---------------------------------------------------------------------------


@dataclass
class SupervisedTaskConfig(Config):
    """Per-task configuration consumed by `SupervisedObjective`."""

    name: str
    task_type: str  # TaskType value: classification | regression
    num_classes: int | None = None
    is_multilabel: bool = False
    regression_loss: str = "l2"
    weight: float = 1.0


@dataclass
class SupervisedObjectiveConfig(Config):
    """Config for the multi-task supervised objective (objective A)."""

    name: str = "supervised"
    weight: float = 1.0
    tasks: list[SupervisedTaskConfig] = field(default_factory=list)

    def build(self) -> SupervisedObjective:
        """Instantiate the objective and its head registry."""
        if not self.tasks:
            raise ValueError("SupervisedObjectiveConfig requires at least one task")
        registry = SupervisedHeadRegistry()
        task_weights: dict[str, float] = {}
        for task in self.tasks:
            head = build_head(
                task_type=task.task_type,
                num_classes=task.num_classes,
                is_multilabel=task.is_multilabel,
                regression_loss=task.regression_loss,
            )
            registry.register(task.name, head)
            task_weights[task.name] = task.weight
        return SupervisedObjective(
            name=self.name,
            weight=self.weight,
            registry=registry,
            task_weights=task_weights,
        )


class SupervisedObjective(_Objective):
    """Multi-task supervised objective: per-batch pick the head + compute loss."""

    def __init__(
        self,
        name: str,
        weight: float,
        registry: SupervisedHeadRegistry,
        task_weights: dict[str, float],
    ) -> None:
        """Initialize supervised objective with head registry and task weights."""
        self.name = name
        self.weight = weight
        self.registry = registry
        self.task_weights = task_weights

    def applies_to(self, batch: Any) -> bool:
        """Only fire when the batch carries supervised labels."""
        return isinstance(batch, Era5SupervisedBatch)

    def get_module(self) -> nn.Module | None:
        """The head registry holds all per-task parameters."""
        return self.registry

    def compute(
        self,
        encoder: Era5DailyEncoder,
        batch: Era5SupervisedBatch,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run encoder + per-task head + supervised loss for ``batch``."""
        if not isinstance(batch, Era5SupervisedBatch):
            raise TypeError(
                f"SupervisedObjective expects Era5SupervisedBatch, got "
                f"{type(batch).__name__}"
            )
        out = encoder(
            era5=batch.era5,
            timestamps=batch.timestamps,
            ignore_mask=batch.ignore_mask,
        )
        pooled = out["pooled"]
        loss, head_metrics = self.registry.compute_loss(
            task_name=batch.task_name,
            pooled=pooled,
            labels=batch.labels,
        )
        task_weight = self.task_weights.get(batch.task_name, 1.0)
        loss = loss * task_weight
        metrics: dict[str, Tensor] = {
            f"{self.name}/{batch.task_name}/loss": loss.detach(),
            f"{self.name}/{batch.task_name}/task_weight": torch.tensor(
                task_weight, device=loss.device
            ),
        }
        for key, value in head_metrics.items():
            metrics[f"{self.name}/{batch.task_name}/{key}"] = value
        return loss, metrics


# ---------------------------------------------------------------------------
# Objective B: ERA5 reconstruction
# ---------------------------------------------------------------------------


class _ReconstructionModule(nn.Module):
    """Container holding the decoder and SWT so both are device-tracked."""

    def __init__(
        self, decoder: Era5TimeQueryDecoder, swt: StationaryWaveletTransform1d
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.swt = swt

    def apply_compile(self) -> None:
        self.decoder.apply_compile()


@dataclass
class ReconstructionObjectiveConfig(Config):
    """Config for the ERA5 reconstruction objective (objective B).

    Args:
        name: Objective name used for metric keys.
        weight: Objective-level weight multiplied into the loss.
        decoder: Decoder config (cross-attention depth, heads, …).
        corruption: Input corruption config (time masks, var-group masks).
        huber_delta: Delta for the raw Huber loss.
        raw_loss_on_masked_only: If True, compute raw Huber only over
            positions that were corrupted; otherwise over the full sequence.
        swt_lambda: Weight of the wavelet multiscale loss term.
        swt_levels: Which SWT decomposition levels to use.
        swt_wavelet: Wavelet family for the SWT (``db2`` or ``haar``).
    """

    name: str = "reconstruction"
    weight: float = 1.0
    decoder: Era5TimeQueryDecoderConfig = field(
        default_factory=Era5TimeQueryDecoderConfig
    )
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    huber_delta: float = 1.0
    raw_loss_on_masked_only: bool = True
    swt_lambda: float = 0.1
    swt_levels: list[int] = field(default_factory=lambda: [0, 1, 2])
    swt_wavelet: str = "db2"

    def build(self) -> ReconstructionObjective:
        """Instantiate decoder, SWT, and the objective."""
        decoder = self.decoder.build()
        num_channels = self.decoder.num_output_channels
        max_level = max(self.swt_levels) + 1 if self.swt_levels else 1
        swt = StationaryWaveletTransform1d(
            num_channels=num_channels,
            max_levels=max_level,
            wavelet=self.swt_wavelet,
        )
        module = _ReconstructionModule(decoder=decoder, swt=swt)
        return ReconstructionObjective(
            name=self.name,
            weight=self.weight,
            module=module,
            corruption_config=self.corruption,
            huber_delta=self.huber_delta,
            raw_loss_on_masked_only=self.raw_loss_on_masked_only,
            swt_lambda=self.swt_lambda,
            swt_levels=self.swt_levels,
        )


class ReconstructionObjective(_Objective):
    """ERA5 reconstruction: corrupt → encode → decode → Huber + SWT loss."""

    def __init__(
        self,
        name: str,
        weight: float,
        module: _ReconstructionModule,
        corruption_config: CorruptionConfig,
        huber_delta: float = 1.0,
        raw_loss_on_masked_only: bool = True,
        swt_lambda: float = 0.1,
        swt_levels: list[int] | None = None,
    ) -> None:
        """Initialize the reconstruction objective."""
        self.name = name
        self.weight = weight
        self._module = module
        self.corruption_config = corruption_config
        self.huber_delta = huber_delta
        self.raw_loss_on_masked_only = raw_loss_on_masked_only
        self.swt_lambda = swt_lambda
        self.swt_levels = swt_levels or [0, 1, 2]

    def applies_to(self, batch: Any) -> bool:
        """Fire on any batch that carries ERA5 data."""
        return isinstance(batch, Era5SupervisedBatch)

    def get_module(self) -> nn.Module | None:
        """Return the decoder module owned by this objective."""
        return self._module

    def compute(
        self,
        encoder: Era5DailyEncoder,
        batch: Era5SupervisedBatch,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Corrupt → encode → decode → loss."""
        x = batch.era5  # [B, T, V]
        ignore_mask = batch.ignore_mask  # [B, T]

        # 1. Corrupt
        x_corrupted, mask = corrupt_era5(x, ignore_mask, self.corruption_config)

        # 2. Encode the corrupted input
        out = encoder(
            era5=x_corrupted,
            timestamps=batch.timestamps,
            ignore_mask=ignore_mask,
        )

        # 3. Decode
        decoder = self._module.decoder
        x_hat = decoder(
            tokens=out["tokens"],
            token_ignore_mask=out["ignore_mask"],
            timestamps=batch.timestamps,
        )

        # 4. Raw Huber loss
        if self.raw_loss_on_masked_only:
            flat_pred = x_hat[mask]
            flat_targ = x[mask]
            if flat_pred.numel() == 0:
                raw_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            else:
                raw_loss = F.huber_loss(
                    flat_pred, flat_targ, reduction="mean", delta=self.huber_delta
                )
        else:
            raw_loss = F.huber_loss(x_hat, x, reduction="mean", delta=self.huber_delta)

        # 5. Multiscale SWT loss
        swt_loss: Tensor
        swt_metrics: dict[str, Tensor]
        if self.swt_lambda > 0 and self.swt_levels:
            swt_loss, swt_metrics = multiscale_swt_loss(
                x_hat=x_hat,
                x=x,
                swt=self._module.swt,
                levels=self.swt_levels,
                huber_delta=self.huber_delta,
                mask=mask if self.raw_loss_on_masked_only else None,
            )
            swt_loss = self.swt_lambda * swt_loss
        else:
            swt_loss = torch.zeros((), device=x.device, dtype=x.dtype)
            swt_metrics = {}

        total_loss = raw_loss + swt_loss

        metrics: dict[str, Tensor] = {
            f"{self.name}/raw_loss": raw_loss.detach(),
            f"{self.name}/swt_loss": swt_loss.detach(),
            f"{self.name}/masked_fraction": mask.float().mean().detach(),
        }
        for k, v in swt_metrics.items():
            metrics[f"{self.name}/{k}"] = v

        return total_loss, metrics


# ---------------------------------------------------------------------------
# Model config (built by `experiment.train()` before the train module)
# ---------------------------------------------------------------------------


@dataclass
class Era5MultiObjectiveModelConfig(Config):
    """Config that builds `Era5MultiObjectiveModel`.

    Each objective sub-config contributes both an `_Objective` instance and
    its parameter-bearing `nn.Module`.  Any combination of A and B (and
    eventually C) is supported — at least one must be enabled.
    """

    encoder_config: Era5DailyEncoderConfig = field(
        default_factory=Era5DailyEncoderConfig
    )
    supervised_objective: SupervisedObjectiveConfig | None = None
    reconstruction_objective: ReconstructionObjectiveConfig | None = None

    def build(self) -> Era5MultiObjectiveModel:
        """Build the encoder, build each objective, return the wrapped model."""
        encoder = self.encoder_config.build()
        objectives: list[_Objective] = []
        if self.supervised_objective is not None:
            objectives.append(self.supervised_objective.build())
        if self.reconstruction_objective is not None:
            objectives.append(self.reconstruction_objective.build())
        if not objectives:
            raise ValueError(
                "Era5MultiObjectiveModelConfig requires at least one "
                "objective (supervised_objective and/or "
                "reconstruction_objective)"
            )
        return Era5MultiObjectiveModel(encoder=encoder, objectives=objectives)


# ---------------------------------------------------------------------------
# Train module config + class
# ---------------------------------------------------------------------------


@dataclass
class MultiObjectiveEra5TrainModuleConfig(OlmoEarthTrainModuleConfig):
    """Config for `MultiObjectiveEra5TrainModule`.

    The model is built separately (see `Era5MultiObjectiveModelConfig`) and
    passed in by `experiment.train()`. The train module pulls the
    `_Objective` instances back off the model so heads and objectives stay
    in lock-step.
    """

    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> MultiObjectiveEra5TrainModule:
        """Build the train module. ``model`` must be an `Era5MultiObjectiveModel`."""
        if not isinstance(model, Era5MultiObjectiveModel):
            raise TypeError(
                "MultiObjectiveEra5TrainModuleConfig.build expects an "
                "Era5MultiObjectiveModel, got "
                f"{type(model).__name__}"
            )
        kwargs = self.prepare_kwargs()
        return MultiObjectiveEra5TrainModule(
            model=model,
            device=device,
            objectives=model.objective_list,
            **kwargs,
        )


class MultiObjectiveEra5TrainModule(OlmoEarthTrainModule):
    """Train module that runs N objectives over the shared ERA5 encoder.

    Currently supports objective A (supervised) and objective B
    (reconstruction), individually or combined.  Each ``train_batch``
    call evaluates every objective whose ``applies_to(batch)`` returns
    True, sums their weighted losses, and performs a single backward.
    """

    def __init__(
        self,
        model: Era5MultiObjectiveModel,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        objectives: list[_Objective],
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        find_unused_parameters: bool = True,
    ) -> None:
        """Initialize the multi-objective training module."""
        super().__init__(
            model=model,
            optim_config=optim_config,
            transform_config=transform_config,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            dp_config=dp_config,
            compile_loss=compile_loss,
            autocast_precision=autocast_precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
            find_unused_parameters=find_unused_parameters,
        )
        if not objectives:
            raise ValueError(
                "MultiObjectiveEra5TrainModule requires at least one objective"
            )
        self.objectives = objectives
        names = [obj.name for obj in objectives]
        if len(names) != len(set(names)):
            raise ValueError(f"Objective names must be unique, got {names}")
        logger.info("MultiObjectiveEra5TrainModule built with objectives: %s", names)

    # ---- override `on_attach` so we don't require image-style patch sizes ----
    def on_attach(self) -> None:
        """Validate batch size, skip image-encoder-specific checks."""
        from olmo_core.distributed.utils import get_world_size

        if (
            self.trainer.global_batch_size
            % (
                self.rank_microbatch_size
                * (ws := get_world_size(self.trainer.dp_process_group))
            )
            != 0
        ):
            raise ValueError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must "
                f"be divisible by micro-batch size ({self.rank_microbatch_size:,d}) "
                f"x DP world size ({ws})"
            )

    @property
    def eval_batch_spec(self) -> Any:
        """Return the default eval batch spec (unused by objective A v0)."""
        from olmo_core.distributed.utils import get_world_size
        from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec

        rank_batch_size = self.trainer.global_batch_size // get_world_size(
            self.trainer.dp_process_group
        )
        return EvalBatchSpec(
            rank_batch_size=rank_batch_size,
            batch_size_unit=EvalBatchSizeUnit.instances,
        )

    def _split_batch(self, batch: Era5SupervisedBatch) -> list[Era5SupervisedBatch]:
        """Split a batch along dim 0 into microbatches of size ``rank_microbatch_size``."""
        bsz = batch.era5.shape[0]
        if bsz <= self.rank_microbatch_size:
            return [batch]
        mb = self.rank_microbatch_size
        microbatches: list[Era5SupervisedBatch] = []
        for start in range(0, bsz, mb):
            end = min(start + mb, bsz)
            microbatches.append(
                Era5SupervisedBatch(
                    era5=batch.era5[start:end],
                    timestamps=batch.timestamps[start:end],
                    ignore_mask=batch.ignore_mask[start:end],
                    labels=batch.labels[start:end],
                    task_name=batch.task_name,
                )
            )
        return microbatches

    def _to_device(self, batch: Era5SupervisedBatch) -> Era5SupervisedBatch:
        return Era5SupervisedBatch(
            era5=batch.era5.to(self.device, non_blocking=True),
            timestamps=batch.timestamps.to(self.device, non_blocking=True),
            ignore_mask=batch.ignore_mask.to(self.device, non_blocking=True),
            labels=batch.labels.to(self.device, non_blocking=True),
            task_name=batch.task_name,
        )

    # ---- main train_batch ----
    def train_batch(
        self,
        batch: Era5SupervisedBatch,
        dry_run: bool = False,
    ) -> None:
        """Run a single training step over *batch*.

        All objectives whose ``applies_to(batch)`` returns True are
        executed; their weighted losses are summed and a single backward
        pass is performed per microbatch.
        """
        # When DDP wraps via `replicate` (composable), `self.model` is still
        # the original `Era5MultiObjectiveModel`, so the encoder is accessible
        # directly. For classic nn.parallel.DDP, peel `.module` to reach it.
        if isinstance(self.model, DDP):
            encoder = self.model.module.encoder
        else:
            encoder = self.model.encoder

        applicable = [obj for obj in self.objectives if obj.applies_to(batch)]
        if not applicable:
            raise RuntimeError(
                f"No objective applies to batch type {type(batch).__name__}"
            )

        self.model.train()
        microbatches = self._split_batch(batch)
        num_micro = len(microbatches)

        per_obj_totals: dict[str, Tensor] = {
            obj.name: torch.zeros((), device=self.device) for obj in applicable
        }
        agg_metrics: dict[str, list[Tensor]] = {}

        for mb_idx, micro in enumerate(microbatches):
            micro = self._to_device(micro)
            with self._train_microbatch_context(mb_idx, num_micro):
                mb_obj_losses: list[Tensor] = []
                for objective in applicable:
                    with self._model_forward_context():
                        loss, metrics = objective.compute(encoder, micro)
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        logger.warning(
                            "Non-finite loss at microbatch %d for objective %s, "
                            "skipping this objective for this microbatch.",
                            mb_idx,
                            objective.name,
                        )
                        continue
                    scaled = (loss * objective.weight) / num_micro
                    per_obj_totals[objective.name] = (
                        per_obj_totals[objective.name] + scaled.detach()
                    )
                    mb_obj_losses.append(scaled)
                    for key, value in metrics.items():
                        agg_metrics.setdefault(key, []).append(value)
                if mb_obj_losses:
                    combined = torch.stack(mb_obj_losses).sum()
                    combined.backward()

        if dry_run:
            return

        for objective in applicable:
            self.trainer.record_metric(
                f"train/{objective.name}/loss",
                per_obj_totals[objective.name],
                ReduceType.mean,
            )
        for key, values in agg_metrics.items():
            stacked = torch.stack(values).float().mean()
            self.trainer.record_metric(f"train/{key}", stacked, ReduceType.mean)

    def eval_batch(
        self, batch: Any, labels: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor | None]:
        """Eval is wired via DownstreamEvaluatorCallback; not run here."""
        del batch, labels
        return None, None
