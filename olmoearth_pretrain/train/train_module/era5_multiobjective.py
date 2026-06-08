"""Multi-objective train module for the daily ERA5 encoder.

This is the central piece that wires the dedicated `Era5DailyEncoder`
(`nn/era5_encoder.py`) to per-objective heads and losses. The module is
structured around an `_Objective` interface so that objective A (supervised
multi-task) — the only objective implemented in this iteration — can later
share the loop with objective B (ERA5 reconstruction) and objective C
(latent S2 prior) without rewriting `train_batch`.

For objective A, each batch from `MultiTaskEra5DataLoader` is an
`Era5SupervisedBatch` tagged with a single task name. The supervised
objective routes the batch to the right head, computes the loss, and
emits per-task metrics through `Trainer.record_metric`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
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
from olmoearth_pretrain.nn.era5_encoder import (
    Era5DailyEncoder,
    Era5DailyEncoderConfig,
)
from olmoearth_pretrain.nn.era5_heads import SupervisedHeadRegistry, build_head
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
            padding_mask=batch.padding_mask,
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
# Model config (built by `experiment.train()` before the train module)
# ---------------------------------------------------------------------------


@dataclass
class Era5MultiObjectiveModelConfig(Config):
    """Config that builds `Era5MultiObjectiveModel`.

    Each objective sub-config (currently only `supervised_objective` for A)
    contributes both an `_Objective` instance and its parameter-bearing
    `nn.Module` (e.g. the supervised head registry). Adding B/C is just a
    matter of adding their objective configs here.
    """

    encoder_config: Era5DailyEncoderConfig = field(
        default_factory=Era5DailyEncoderConfig
    )
    supervised_objective: SupervisedObjectiveConfig | None = None

    def build(self) -> Era5MultiObjectiveModel:
        """Build the encoder, build each objective, return the wrapped model."""
        encoder = self.encoder_config.build()
        objectives: list[_Objective] = []
        if self.supervised_objective is not None:
            objectives.append(self.supervised_objective.build())
        if not objectives:
            raise ValueError(
                "Era5MultiObjectiveModelConfig requires at least one "
                "objective (supervised_objective for objective A)"
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

    Today only objective A (`SupervisedObjective`) is implemented. The
    `train_batch` loop is structured so adding objectives B / C is just a
    matter of:

    1. Implementing a new `_Objective` subclass.
    2. Adding it to the `objectives` list passed in at construction.
    3. Routing the right batch type to its `compute(...)`.
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
                    padding_mask=batch.padding_mask[start:end],
                    labels=batch.labels[start:end],
                    task_name=batch.task_name,
                )
            )
        return microbatches

    def _to_device(self, batch: Era5SupervisedBatch) -> Era5SupervisedBatch:
        return Era5SupervisedBatch(
            era5=batch.era5.to(self.device, non_blocking=True),
            timestamps=batch.timestamps.to(self.device, non_blocking=True),
            padding_mask=batch.padding_mask.to(self.device, non_blocking=True),
            labels=batch.labels.to(self.device, non_blocking=True),
            task_name=batch.task_name,
        )

    def _route_batch_to_objective(self, batch: Any) -> tuple[_Objective, Any]:
        """Route a batch to the objective that knows how to consume it.

        For objective A the batch is an `Era5SupervisedBatch`. As B / C land,
        this routing layer is where their batch types get matched to their
        respective objectives.
        """
        if isinstance(batch, Era5SupervisedBatch):
            for objective in self.objectives:
                if isinstance(objective, SupervisedObjective):
                    return objective, batch
            raise RuntimeError(
                "Received an Era5SupervisedBatch but no SupervisedObjective is registered"
            )
        raise TypeError(f"Unsupported batch type: {type(batch).__name__}")

    # ---- main train_batch ----
    def train_batch(
        self,
        batch: Era5SupervisedBatch,
        dry_run: bool = False,
    ) -> None:
        """Run a single training step over `batch`.

        Mirrors the microbatching pattern used by
        `ContrastiveLatentMIMTrainModule.train_batch`: split the batch into
        rank-microbatch-sized chunks, run encoder + per-task head + loss in
        each chunk, divide by `num_micro_batches`, backward, and accumulate
        the loss / metric tensors for the trainer's reduce.
        """
        objective, batch = self._route_batch_to_objective(batch)
        # When DDP wraps via `replicate` (composable), `self.model` is still
        # the original `Era5MultiObjectiveModel`, so the encoder is accessible
        # directly. For classic nn.parallel.DDP, peel `.module` to reach it.
        if isinstance(self.model, DDP):
            encoder = self.model.module.encoder
        else:
            encoder = self.model.encoder

        self.model.train()
        microbatches = self._split_batch(batch)
        num_micro = len(microbatches)

        total_loss = torch.zeros((), device=self.device)
        agg_metrics: dict[str, list[Tensor]] = {}

        for mb_idx, micro in enumerate(microbatches):
            micro = self._to_device(micro)
            with self._train_microbatch_context(mb_idx, num_micro):
                with self._model_forward_context():
                    loss, metrics = objective.compute(encoder, micro)
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        "Non-finite loss at microbatch %d for objective %s, "
                        "skipping backward for this microbatch.",
                        mb_idx,
                        objective.name,
                    )
                    continue
                scaled = (loss * objective.weight) / num_micro
                total_loss = total_loss + scaled.detach()
                scaled.backward()
            for key, value in metrics.items():
                agg_metrics.setdefault(key, []).append(value)

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{objective.name}/loss",
            total_loss,
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
