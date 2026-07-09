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
from olmoearth_pretrain.data.multi_task_era5_dataset import (
    Era5Batch,
    Era5SupervisedBatch,
)
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
    DEFAULT_VARIABLE_GROUPS,
    GROUP_RECON_MODE,
    RECON_MODE_SPEC,
    MaskPolicy,
    corrupt_era5,
)
from olmoearth_pretrain.nn.transforms.era5_swt import StationaryWaveletTransform1d
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
    target_mean: float | None = None
    target_std: float | None = None


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
                target_mean=task.target_mean,
                target_std=task.target_std,
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
        mask_policy: Two-stage masking policy (temporal interpolation +
            cross-variable reconstruction).
        variable_groups: Mapping from group name to list of band indices.
        huber_delta: Delta for the raw Huber loss.
        raw_loss_on_masked_only: If True, compute raw Huber only over
            positions that were corrupted; otherwise over the full sequence.
        raw_lambda: Weight of the raw (time-domain) reconstruction loss term.
            Set to 0.0 for wavelet-only reconstruction.
        swt_lambda: Weight of the wavelet multiscale loss term.
        swt_levels: Which SWT decomposition levels to use.
        swt_wavelet: Wavelet family for the SWT (``db2`` or ``haar``).
        group_recon_mode: Per-group reconstruction mode (see
            :data:`~olmoearth_pretrain.nn.transforms.era5_corruption.GROUP_RECON_MODE`).
            Controls how each variable group's reconstruction loss is
            weighted across raw vs. wavelet bands.
    """

    name: str = "reconstruction"
    weight: float = 1.0
    decoder: Era5TimeQueryDecoderConfig = field(
        default_factory=Era5TimeQueryDecoderConfig
    )
    mask_policy: MaskPolicy = field(default_factory=MaskPolicy)
    variable_groups: dict[str, list[int]] = field(
        default_factory=lambda: dict(DEFAULT_VARIABLE_GROUPS)
    )
    huber_delta: float = 1.0
    raw_loss_on_masked_only: bool = True
    raw_lambda: float = 1.0
    swt_lambda: float = 0.1
    swt_levels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    swt_wavelet: str = "haar"
    swt_buffer_days: int = 83
    group_recon_mode: dict[str, str] = field(
        default_factory=lambda: dict(GROUP_RECON_MODE)
    )

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
            mask_policy=self.mask_policy,
            variable_groups=dict(self.variable_groups),
            group_recon_mode=dict(self.group_recon_mode),
            huber_delta=self.huber_delta,
            raw_loss_on_masked_only=self.raw_loss_on_masked_only,
            raw_lambda=self.raw_lambda,
            swt_lambda=self.swt_lambda,
            swt_levels=self.swt_levels,
            swt_buffer_days=self.swt_buffer_days,
        )


def _parse_recon_mode(
    mode: str, all_swt_levels: list[int]
) -> tuple[bool, list[int], bool]:
    """Parse a ``group_recon_mode`` string into ``(include_raw, swt_detail_levels, include_lowpass)``.

    Looks up *mode* in :data:`RECON_MODE_SPEC` and intersects the spec's
    ``swt_detail_levels`` with *all_swt_levels* (the levels actually
    computed by the SWT module).
    """
    spec = RECON_MODE_SPEC.get(mode)
    if spec is None:
        logger.warning(
            "Unknown group_recon_mode %r, falling back to raw_plus_all_swt", mode
        )
        spec = RECON_MODE_SPEC["raw_plus_all_swt"]
    detail_levels = [lv for lv in spec["swt_detail_levels"] if lv in all_swt_levels]
    return spec["include_raw"], detail_levels, spec["include_lowpass"]


class ReconstructionObjective(_Objective):
    """ERA5 reconstruction: corrupt → encode → decode → Huber + SWT loss."""

    def __init__(
        self,
        name: str,
        weight: float,
        module: _ReconstructionModule,
        mask_policy: MaskPolicy,
        variable_groups: dict[str, list[int]],
        group_recon_mode: dict[str, str],
        huber_delta: float = 1.0,
        raw_loss_on_masked_only: bool = True,
        raw_lambda: float = 1.0,
        swt_lambda: float = 0.1,
        swt_levels: list[int] | None = None,
        swt_buffer_days: int = 83,
    ) -> None:
        """Initialize the reconstruction objective."""
        self.name = name
        self.weight = weight
        self._module = module
        self.mask_policy = mask_policy
        self.variable_groups = variable_groups
        self.group_recon_mode = group_recon_mode
        self.huber_delta = huber_delta
        self.raw_loss_on_masked_only = raw_loss_on_masked_only
        self.raw_lambda = raw_lambda
        self.swt_lambda = swt_lambda
        self.swt_levels = swt_levels or [0, 1, 2, 3, 4, 5]
        self.swt_buffer_days = swt_buffer_days
        # Static variable-count weights for loss averaging. Each (group, scale)
        # term is weighted by the number of variables in the group
        self._raw_weight, self._swt_weight = self._compute_loss_weights()

    def _compute_loss_weights(self) -> tuple[float, float]:
        """Precompute variable-count weights for raw and SWT loss averaging.

        Returns ``(raw_weight, swt_weight)`` where each is the total number of
        per-variable terms contributing to that objective: raw counts one term
        per variable in every ``include_raw`` group; SWT counts one term per
        variable per allowed detail level, plus the deepest-approximation term
        for ``include_lowpass`` groups.
        """
        raw_weight = 0
        swt_weight = 0
        for group_name, bi in self.variable_groups.items():
            n_vars = len(bi)
            mode = self.group_recon_mode.get(group_name, "raw_plus_all_swt")
            include_raw, allowed_levels, include_lowpass = _parse_recon_mode(
                mode, self.swt_levels
            )
            if include_raw:
                raw_weight += n_vars
            swt_weight += n_vars * len(allowed_levels)
            if include_lowpass and allowed_levels:
                swt_weight += n_vars
        return float(raw_weight), float(swt_weight)

    def applies_to(self, batch: Any) -> bool:
        """Fire on any batch that carries ERA5 data (supervised or SSL)."""
        return isinstance(batch, Era5Batch)

    def get_module(self) -> nn.Module | None:
        """Return the decoder module owned by this objective."""
        return self._module

    def _group_huber(
        self,
        pred: Tensor,
        targ: Tensor,
        group_mask: Tensor | None,
    ) -> Tensor:
        """Huber loss over a channel subset, optionally masked to corrupted positions."""
        if group_mask is not None:
            fp = pred[group_mask]
            ft = targ[group_mask]
        else:
            fp, ft = pred, targ
        if fp.numel() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return F.huber_loss(fp, ft, reduction="mean", delta=self.huber_delta)

    def compute(
        self,
        encoder: Era5DailyEncoder,
        batch: Era5Batch,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Corrupt → encode → decode → per-group loss.

        Loss terms are gated per variable group by ``group_recon_mode``.
        All losses are computed only on the target window (after
        ``swt_buffer_days``).
        """
        x = batch.era5  # [B, T, V]
        ts = self.swt_buffer_days  # target_start index

        # No-data validity ([B, T, V], True = valid). No-data cells were already
        # mean-imputed (0 post z-score) in the dataset; here we use the mask to
        # keep them out of the reconstruction loss.
        valid = getattr(batch, "valid_mask", None)
        if valid is None:
            valid = torch.ones_like(x, dtype=torch.bool)
        else:
            valid = valid.to(dtype=torch.bool)

        # 1. Generate corruption mask (only target window is masked)
        mask = corrupt_era5(x, self.mask_policy, self.variable_groups, ts)
        # Never spend the reconstruction budget on no-data cells.
        mask = mask & valid

        # 2. Encode with corruption mask
        out = encoder(
            era5=x,
            timestamps=batch.timestamps,
            corruption_mask=mask,
        )

        # 3. Decode
        decoder = self._module.decoder
        x_hat = decoder(
            tokens=out["tokens"],
            timestamps=batch.timestamps,
        )

        # 4. Compute SWT bands on full sequence, crop to target window
        pred_bands: list | None = None
        targ_bands: list | None = None
        if self.swt_lambda > 0 and self.swt_levels:
            swt_mod = self._module.swt
            pred_bands = swt_mod(
                x_hat.transpose(1, 2), levels=self.swt_levels, target_start=ts
            )
            targ_bands = swt_mod(
                x.transpose(1, 2), levels=self.swt_levels, target_start=ts
            )

        # Slice to target window for raw loss
        x_hat_tgt = x_hat[:, ts:, :]
        x_tgt = x[:, ts:, :]
        mask_tgt = mask[:, ts:, :]
        valid_tgt = valid[:, ts:, :]  # [B, T_win, V]
        # Conservative SWT validity: drop an entire
        # (sample, variable) from the SWT loss whenever that variable has ANY
        # no-data over the full sequence.
        var_valid = valid.all(dim=1)  # [B, V]

        variable_groups = self.variable_groups

        # 5. Per-group loss accumulation. Each (group, scale) Huber is weighted
        # by the group's variable count `n_vars`
        raw_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        swt_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        level_loss_sums: dict[int, Tensor] = {}
        level_loss_counts: dict[int, int] = {}

        for group_name, bi in variable_groups.items():
            n_vars = len(bi)
            mode = self.group_recon_mode.get(group_name, "raw_plus_all_swt")
            include_raw, allowed_levels, include_lowpass = _parse_recon_mode(
                mode, self.swt_levels
            )

            g_pred = x_hat_tgt[:, :, bi]  # [B, T_win, |bi|]
            g_targ = x_tgt[:, :, bi]
            # Always exclude no-data cells
            if self.raw_loss_on_masked_only:
                g_mask = mask_tgt[:, :, bi] & valid_tgt[:, :, bi]
            else:
                g_mask = valid_tgt[:, :, bi]

            # Raw Huber (band-normalized like the SWT path)
            if include_raw and self.raw_lambda > 0:
                with torch.no_grad():
                    raw_std = g_targ.std(dim=(0, 1)).clamp(min=1e-6)
                g_pred_n = g_pred / raw_std[None, None, :]
                g_targ_n = g_targ / raw_std[None, None, :]
                raw_loss = raw_loss + n_vars * self._group_huber(
                    g_pred_n, g_targ_n, g_mask
                )

            # SWT wavelet loss (bands already cropped to target window)
            if pred_bands is not None and targ_bands is not None and allowed_levels:
                # [B, |bi|, T_win] loss mask: drop invalid (sample, variable)
                # rows; require corruption too when scoring masked-only.
                gvar_valid_vt = var_valid[:, bi].unsqueeze(-1)  # [B, |bi|, 1]
                if self.raw_loss_on_masked_only:
                    g_mask_vt = mask_tgt[:, :, bi].transpose(1, 2) & gvar_valid_vt
                else:
                    g_mask_vt = gvar_valid_vt.expand(-1, -1, valid_tgt.shape[1])
                deepest_allowed = max(allowed_levels)
                for level_idx, level in enumerate(self.swt_levels):
                    if level not in allowed_levels:
                        continue
                    a_pred, d_pred = pred_bands[level_idx]  # [B, V, T_win]
                    a_targ, d_targ = targ_bands[level_idx]
                    d_pred_g = d_pred[:, bi, :]
                    d_targ_g = d_targ[:, bi, :]

                    with torch.no_grad():
                        std = d_targ_g.std(dim=(0, 2)).clamp(min=1e-6)
                    d_pred_n = d_pred_g / std[None, :, None]
                    d_targ_n = d_targ_g / std[None, :, None]

                    lvl = self._group_huber(d_pred_n, d_targ_n, g_mask_vt)
                    swt_loss = swt_loss + n_vars * lvl
                    level_loss_sums[level] = (
                        level_loss_sums.get(
                            level, torch.zeros((), device=x.device, dtype=x.dtype)
                        )
                        + lvl
                    )
                    level_loss_counts[level] = level_loss_counts.get(level, 0) + 1

                    # Deepest-level approximation (lowpass) loss
                    if include_lowpass and level == deepest_allowed:
                        a_pred_g = a_pred[:, bi, :]
                        a_targ_g = a_targ[:, bi, :]
                        with torch.no_grad():
                            a_std = a_targ_g.std(dim=(0, 2)).clamp(min=1e-6)
                        a_pred_n = a_pred_g / a_std[None, :, None]
                        a_targ_n = a_targ_g / a_std[None, :, None]

                        a_lvl = self._group_huber(a_pred_n, a_targ_n, g_mask_vt)
                        swt_loss = swt_loss + n_vars * a_lvl
                        approx_key = -1  # sentinel for deepest approx
                        level_loss_sums[approx_key] = (
                            level_loss_sums.get(
                                approx_key,
                                torch.zeros((), device=x.device, dtype=x.dtype),
                            )
                            + a_lvl
                        )
                        level_loss_counts[approx_key] = (
                            level_loss_counts.get(approx_key, 0) + 1
                        )

        # Variable-weighted mean over all (group, scale) terms, then apply the
        # per-objective weights (raw_lambda=0 => wavelet-only reconstruction).
        raw_loss = self.raw_lambda * (raw_loss / max(self._raw_weight, 1.0))
        swt_loss = self.swt_lambda * (swt_loss / max(self._swt_weight, 1.0))

        total_loss = raw_loss + swt_loss

        metrics: dict[str, Tensor] = {
            f"{self.name}/raw_loss": raw_loss.detach(),
            f"{self.name}/swt_loss": swt_loss.detach(),
            f"{self.name}/masked_fraction": mask_tgt.float().mean().detach(),
            f"{self.name}/nodata_fraction": (~valid).float().mean().detach(),
        }
        for level in self.swt_levels:
            cnt = level_loss_counts.get(level, 0)
            if cnt > 0:
                metrics[f"{self.name}/swt_level_{level}_loss"] = (
                    level_loss_sums[level].detach() / cnt
                )
            else:
                metrics[f"{self.name}/swt_level_{level}_loss"] = torch.zeros(
                    (), device=x.device, dtype=x.dtype
                )
        approx_cnt = level_loss_counts.get(-1, 0)
        if approx_cnt > 0:
            metrics[f"{self.name}/swt_deepest_approx_loss"] = (
                level_loss_sums[-1].detach() / approx_cnt
            )

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

    def _split_batch(self, batch: Era5Batch) -> list[Era5Batch]:
        """Split a batch along dim 0 into microbatches of size ``rank_microbatch_size``.

        Uses the batch's generic ``microbatch`` helper so both supervised and
        SSL batch types (and any future extra tensor fields) are sliced
        uniformly while preserving the concrete batch type.
        """
        bsz = batch.era5.shape[0]
        if bsz <= self.rank_microbatch_size:
            return [batch]
        mb = self.rank_microbatch_size
        microbatches: list[Era5Batch] = []
        for start in range(0, bsz, mb):
            end = min(start + mb, bsz)
            microbatches.append(batch.microbatch(start, end))
        return microbatches

    def _to_device(self, batch: Era5Batch) -> Era5Batch:
        return batch.to_device(self.device)

    # ---- main train_batch ----
    def train_batch(
        self,
        batch: Era5Batch,
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
