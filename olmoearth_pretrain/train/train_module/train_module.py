"""Training and optimizer abstraction for OlmoEarth Pretrain."""

import contextlib
import dataclasses
import itertools
import json
import os
from collections.abc import Generator
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.config import DType
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_world_mesh,
    get_dp_mesh,
    get_dp_process_group,
)
from olmo_core.distributed.utils import get_local_tensor, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec, TrainModule
from olmo_core.utils import gc_cuda, get_default_device
from torch import nn
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias
from olmoearth_pretrain.config import Config
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.model_loader import patch_legacy_encoder_config
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import FrozenTargetProjection
from olmoearth_pretrain.train.loss import LossConfig

logger = getLogger(__name__)


def _strip_unknown_fields(d: Any) -> Any:
    """Recursively strip fields from config dicts that don't exist on the target class.

    Old checkpoints may contain fields that have since been removed (e.g.
    ``merge_bandsets``). OmegaConf's strict merge rejects unknown keys, so we
    strip them before calling ``Config.from_dict``.
    """
    if not isinstance(d, dict):
        return d

    from dataclasses import fields as dc_fields
    from importlib import import_module

    # Recurse first
    cleaned = {k: _strip_unknown_fields(v) for k, v in d.items()}

    cls_name = cleaned.get("_CLASS_")
    if cls_name is None or "." not in cls_name:
        return cleaned

    try:
        *modules, name = cls_name.split(".")
        cls = getattr(import_module(".".join(modules)), name)
        valid_keys = {f.name for f in dc_fields(cls)} | {"_CLASS_"}
        removed = set(cleaned) - valid_keys
        if removed:
            logger.info("Stripping unknown fields %s from %s", removed, cls_name)
            cleaned = {k: v for k, v in cleaned.items() if k in valid_keys}
    except Exception:
        logger.debug("Could not resolve class %s for field stripping", cls_name)

    return cleaned


@dataclass
class OlmoEarthTrainModuleConfig(Config):
    """A configuration class for building :class:`OlmoEarthTrainModule` instances.

    Args:
        rank_microbatch_size: The micro batch size per rank in instances.
        optim: The optimizer configuration.
        transform_config: The transform configuration for the model.
        compile_model: Whether to compile the model using torch.compile.
        dp_config: Data parallel configuration for distributed training.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
        find_unused_parameters: Whether to find unused parameters for DDP.
    """

    # Training settings

    optim_config: OptimConfig
    rank_microbatch_size: int

    transform_config: TransformConfig = field(
        default_factory=lambda: TransformConfig(transform_type="flip_and_rotate")
    )
    # Model settings
    compile_model: bool = False
    dp_config: DataParallelConfig | None = None

    # Loss function settings
    compile_loss: bool = False

    # Training settings
    autocast_precision: DType | None = None
    max_grad_norm: float | None = None
    scheduler: Scheduler | None = None
    find_unused_parameters: bool = True

    # Checkpoint settings
    state_dict_save_opts: dict[str, Any] | None = None
    state_dict_load_opts: dict[str, Any] | None = None
    regularizer_config: LossConfig | None = None

    def prepare_kwargs(self) -> dict[str, Any]:
        """Prepare the kwargs for the train module."""
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (
            state_dict_save_opts := kwargs.pop("state_dict_save_opts", None)
        ) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(
                **state_dict_save_opts
            )
        if (
            state_dict_load_opts := kwargs.pop("state_dict_load_opts", None)
        ) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(
                **state_dict_load_opts
            )
        return kwargs

    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> "OlmoEarthTrainModule":
        """Build the corresponding :class:`OlmoEarthTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return OlmoEarthTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class OlmoEarthTrainModule(TrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim_config: The corresponding optimizer config.
        transform_config: The transform configuration for the model.
        rank_microbatch_size: The rank micro batch size in instances.
        compile_model: Whether to compile to the model.
        dp_config: Data parallel configuration for the model.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        device: The device to train on.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
    """

    def __init__(
        self,
        model: Any,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        find_unused_parameters: bool = True,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
            rank_microbatch_size: The rank batch size in instances.
            compile_model: Whether to compile to the model.
            dp_config: Data parallel configuration for the model.
            compile_loss: Whether to compile the loss function.
            find_unused_parameters: Whether to find unused parameters for DDP.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
        """
        super().__init__()

        self.model = model

        # Band dropout is disabled by default in the encoder so it never activates
        # during fine-tuning. We enable it here since the train module is only
        # instantiated during pre-training; note that, for LatentMIM/Galileo, this
        # happens after the target encoder was deepcopied from the online encoder,
        # so it stays off for the target encoder which is correct.
        if hasattr(self.model, "encoder") and hasattr(
            self.model.encoder, "enable_band_dropout"
        ):
            self.model.encoder.enable_band_dropout()

        self.transform = transform_config.build()
        if hasattr(self.model, "encoder"):
            logger.info(
                "Number of encoder parameters: %d",
                sum(p.numel() for p in self.model.encoder.parameters()),
            )
        if hasattr(self.model, "decoder") and self.model.decoder is not None:
            logger.info(
                "Number of decoder parameters: %d",
                sum(p.numel() for p in self.model.decoder.parameters()),
            )

        self.device = device or get_default_device()

        if dp_config is not None:
            self.world_mesh = build_world_mesh(
                dp=dp_config, device_type=self.device.type
            )
            logger.info(
                f"Data parallel world size = {get_world_size(self.dp_process_group):,d}"
            )
        else:
            self.world_mesh = None

        # Maybe compile.
        if compile_model:
            if torch.cuda.is_available():
                self.model.apply_compile()
                logger.info("Applied torch.compile() to the model")
            else:
                logger.warning(
                    "torch.compile() not applied because CUDA is not available"
                )

        # Maybe shard/replicate according to data parallel config.
        self._dp_config = dp_config
        self._replicate_grad_sync = False
        if dp_config is not None:
            dp_mesh = get_dp_mesh(self.world_mesh)
            if dp_config.name in (DataParallelType.fsdp):
                param_dtype = (
                    dp_config.param_dtype.as_pt()
                    if dp_config.param_dtype is not None
                    else None
                )
                # TODO: MIXED PRecision is not yet supported
                self.model.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                )
                logger.info("Applied FSDP to the model")
            elif dp_config.name == DataParallelType.ddp:
                # NOTE: We intentionally do NOT wrap with torch DDP / composable
                # replicate(). Our train modules can run multiple forwards per
                # backward (e.g. the two contrastive views in
                # ContrastiveLatentMIMTrainModule), which the DDP reducer does not
                # support (params get marked ready twice). Instead the model stays
                # a plain replicated module (already materialized on-device) and we
                # all-reduce gradients once per step in optim_step(). This matches
                # DDP numerics (mean-reduced gradients, like FSDP) with a single
                # coalesced collective per step.
                self._replicate_grad_sync = True
                self._broadcast_params()
                logger.info(
                    "Applied replicated data parallelism (manual gradient "
                    "all-reduce) to the model"
                )
            else:
                raise NotImplementedError(dp_config.name)

        # Materialize and init parameters.
        logger.info("Initializing model weights...")
        # model.init_weights(max_seq_len=max_sequence_length, device=self.device)

        # Build optimizer(s).
        logger.info("Building optimizer(s)...")
        self.optimizer: Optimizer = optim_config.build(
            self.model,
        )
        self.rank_microbatch_size = rank_microbatch_size
        self.autocast_precision = autocast_precision
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, strict=True
        )

    @property
    def dp_process_group(self) -> dist.ProcessGroup | None:
        """Get the data parallel process group."""
        if self.world_mesh is None:
            return None
        return get_dp_process_group(self.world_mesh)

    @property
    def is_fsdp(self) -> bool:
        """Check if the model is FSDP."""
        return self._dp_config is not None and self._dp_config.name in (
            DataParallelType.fsdp
        )

    @property
    def eval_batch_spec(self) -> EvalBatchSpec:
        """Get the evaluation batch spec."""
        # Determine the number of micro-batches.
        rank_batch_size = self.trainer.global_batch_size // get_world_size(
            self.trainer.dp_process_group
        )
        rank_batch_size_instances = rank_batch_size
        return EvalBatchSpec(
            rank_batch_size=rank_batch_size_instances,
            batch_size_unit=EvalBatchSizeUnit.instances,
        )

    @property
    def local_rank(self) -> int:
        """Get the local rank."""
        return self.trainer.data_loader.dp_rank

    @property
    def logits_dtype(self) -> torch.dtype:
        """Get the logits dtype."""
        if self.autocast_precision is not None:
            return self.autocast_precision
        elif self._dp_config is not None and self._dp_config.param_dtype is not None:
            return self._dp_config.param_dtype.as_pt()
        else:
            for param in self.model.parameters():
                return param.dtype
        raise RuntimeError("Should not get here")

    def on_attach(self) -> None:
        """Called when the train module is attached to the trainer."""
        # Validate batch size.
        if (
            self.trainer.global_batch_size
            % (
                self.rank_microbatch_size
                * (ws := get_world_size(self.trainer.dp_process_group))
            )
            != 0
        ):
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size:,d}) x DP world size ({ws})"
            )
        if not hasattr(self.model, "encoder"):
            # hack to allow other models for EVAL
            return
        if self.trainer.data_loader.min_patch_size != self.model.encoder.min_patch_size:
            raise ValueError(
                f"min_patch_size of dataloader ({self.trainer.data_loader.min_patch_size}) must match min_patch_size of model ({self.model.encoder.min_patch_size})"
            )
        if self.trainer.data_loader.max_patch_size != self.model.encoder.max_patch_size:
            raise ValueError(
                f"max_patch_size of dataloader ({self.trainer.data_loader.max_patch_size}) must match max_patch_size of model ({self.model.encoder.max_patch_size})"
            )

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict."""
        return self._get_state_dict(self.state_dict_save_opts)

    def state_dict_to_load(
        self, metadata: Metadata, optim: bool | None = None
    ) -> dict[str, Any]:
        """Get the state dict to load."""
        load_opts = self.state_dict_load_opts
        state_dict = self._get_state_dict(load_opts)
        if os.environ.get("OE_LOAD_SKIP_MISMATCHED_KEYS"):
            self._drop_mismatched_keys(state_dict, metadata)
        return state_dict

    def _drop_mismatched_keys(
        self, state_dict: dict[str, Any], metadata: Metadata
    ) -> None:
        """Drop keys whose checkpoint shape mismatches the current model.

        Opt-in via ``OE_LOAD_SKIP_MISMATCHED_KEYS=1``, for loading a checkpoint
        whose architecture has drifted in ways that do not matter for the run —
        e.g. evaluating S2-only tasks with a checkpoint whose srtm
        patch-embedding was trained at a different band count than the current
        model builds (say [64, 4] saved vs [64, 1] now). Dropped parameters keep
        their fresh initialization; optimizer state is skipped entirely (the
        flattened optimizer format requires an entry per current param, so a
        partial optimizer load is not possible), and missing-key strictness is
        relaxed so the partial load succeeds. Never set this when resuming
        training: a silently unloaded weight — and a fresh optimizer — is a
        bug there, not a convenience.
        """
        dropped: list[str] = []
        for key, value in list(state_dict["model"].items()):
            shape = getattr(value, "shape", None)
            if shape is None:
                continue
            meta = metadata.state_dict_metadata.get(f"model.{key}")
            saved_size = getattr(meta, "size", None)
            if meta is not None and (
                saved_size is None or tuple(saved_size) == tuple(shape)
            ):
                continue
            del state_dict["model"][key]
            dropped.append(key)
            logger.warning(
                "OE_LOAD_SKIP_MISMATCHED_KEYS: not loading '%s' "
                "(checkpoint: %s, model: %s); keeping fresh initialization.",
                key,
                tuple(saved_size) if saved_size is not None else "<absent>",
                tuple(shape),
            )
        if not dropped:
            return
        if state_dict.pop("optim", None) is not None:
            logger.warning(
                "OE_LOAD_SKIP_MISMATCHED_KEYS: skipping optimizer state entirely "
                "(%d model key(s) were dropped, and the flattened optimizer "
                "format cannot be partially loaded).",
                len(dropped),
            )
        self.state_dict_load_opts = dataclasses.replace(
            self.state_dict_load_opts, strict=False
        )

    def state_dict_to_save(self) -> dict[str, Any]:
        """Get the state dict to save."""
        return self._get_state_dict(self.state_dict_save_opts)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict."""
        dist_cp_sd.set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if "optim" not in state_dict:
            # _drop_mismatched_keys removed it (OE_LOAD_SKIP_MISMATCHED_KEYS);
            # the optimizer keeps its fresh state.
            logger.warning("No optimizer state in the load plan; not loading any.")
            return
        dist_cp_sd.set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state_dict["optim"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()

    def zero_grads(self) -> None:
        """Zero the gradients."""
        self.optimizer.zero_grad(set_to_none=True)

    def optim_step(self) -> None:
        """Optimize the model."""
        # For replicated data parallelism, average gradients across ranks first
        # so clipping and the optimizer step see the same gradients DDP/FSDP would.
        if self._replicate_grad_sync:
            self._all_reduce_grads()

        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            if isinstance(self.optimizer, SkipStepOptimizer):
                self.optimizer.latest_grad_norm = grad_norm

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optimizer.param_groups):
                new_lr = self.scheduler.set_lr(group, self.trainer)
                self.trainer.record_metric(
                    f"LR (group {group_idx})", new_lr, namespace="optim"
                )

        # Step optimizer.
        self.optimizer.step()
        if isinstance(self.optimizer, SkipStepOptimizer):
            self.trainer.record_metric(
                "step skipped", self.optimizer.step_skipped, namespace="optim"
            )

    def _broadcast_params(self) -> None:
        """Broadcast params and buffers from rank 0 so replicas start identical."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        group = self.dp_process_group
        for t in itertools.chain(self.model.parameters(), self.model.buffers()):
            dist.broadcast(t.detach(), src=0, group=group)

    def _all_reduce_grads(self) -> None:
        """Mean-reduce gradients across the data parallel group.

        Gradients are flattened into one buffer per dtype so the sync is a
        couple of large collectives instead of one per parameter.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        group = self.dp_process_group
        world_size = get_world_size(group)
        if world_size == 1:
            return
        grads_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for p in self.model.parameters():
            if p.grad is not None:
                grads_by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
        for grads in grads_by_dtype.values():
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat, group=group)
            flat.div_(world_size)
            offset = 0
            for g in grads:
                g.copy_(flat[offset : offset + g.numel()].view_as(g))
                offset += g.numel()

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            is_last_mb = micro_batch_idx == num_micro_batches - 1
            if isinstance(self.model, FSDPModule):
                assert self._dp_config is not None
                # On the last backward FSDP waits on pending gradient reduction and clears internal data
                # data structures for backward prefetching.
                self.model.set_is_last_backward(is_last_mb)
            if isinstance(self.model, DDP) and micro_batch_idx != num_micro_batches - 1:
                # For DDP, only sync gradients on the final micro batch.
                stack.enter_context(self.model.no_sync())
            yield

    @contextlib.contextmanager
    def _model_forward_context(
        self, no_sync: bool = False
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(
                    torch.autocast(self.device.type, dtype=self.autocast_precision)
                )
            if isinstance(self.model, DDP) and no_sync:
                # If we do multiple forwards through the  encoder we only want to sunc on the last one
                stack.enter_context(self.model.no_sync())
            yield

    def _clear_loss_buffers(self) -> None:
        """Clear the loss buffers."""
        logger.warning("clear loss buffers not implemented")
        pass

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions
    ) -> dict[str, Any]:
        """Get the state dict."""
        # This is a sanity check to make sure the checkpoint is compatible
        # with the current model architecture, mainly useful when running evaluation beaker jobs
        if self.trainer.load_path is not None:
            with open(f"{self.trainer.load_path}/config.json") as f:
                config_dict = json.load(f)
            config_dict = patch_legacy_encoder_config(config_dict)
            model_config = Config.from_dict(_strip_unknown_fields(config_dict["model"]))
            model = model_config.build()
            # Check if any keys are missing
            for key in self.model.state_dict().keys():
                if key not in model.state_dict():
                    logger.info("Key %s not in checkpoint", key)
                    raise RuntimeError("Model and checkpoint are not compatible")
            logger.info("Model and checkpoint are compatible")

        model_state_dict = dist_cp_sd.get_model_state_dict(
            self.model, options=sd_options
        )
        optim_state_dict = dist_cp_sd.get_optimizer_state_dict(
            self.model, self.optimizer, options=sd_options
        )
        return {
            "model": model_state_dict,
            "optim": optim_state_dict,
        }

    def _clip_grad_norm(
        self,
        max_grad_norm: float,
        norm_type: float = 2.0,
        foreach: bool | None = None,
    ) -> torch.Tensor:
        """Clip the gradients."""
        # Pipeline parallel grad clipping required nightly torch
        # return torch.nn.utils.clip_grad_norm_(
        #     self.model.parameters(), max_grad_norm, norm_type=norm_type, foreach=foreach
        # )
        parameters = [p for p in self.model.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = nn.utils.get_total_norm(
            grads, norm_type=norm_type, error_if_nonfinite=False, foreach=foreach
        )

        # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
        # We can simply reduce the DTensor to get the total norm in this tensor's process group
        # and then convert it to a local tensor.
        # NOTE: It has two purposes:
        #       1. to make sure the total norm is computed correctly when PP is used (see below)
        #       2. to return a reduced total_norm tensor whose .item() would return the correct value
        if isinstance(total_norm, DTensor):
            # Will reach here if any non-PP parallelism is used.
            # If only using PP, total_norm will be a local tensor.
            total_norm = total_norm.full_tensor()

        torch.nn.utils.clip_grads_with_norm_(
            parameters, max_grad_norm, total_norm, foreach=foreach
        )
        return total_norm

    def update_target_encoder(self) -> None:
        """Update the target encoder."""
        # Update target encoder with EMA this should be a callback
        if self.start_ema == 1.0 and self.end_ema == 1.0:
            return

        if isinstance(
            getattr(self.model, "target_encoder", None), FrozenTargetProjection
        ):
            raise OLMoConfigurationError(
                "EMA updates require a full target-encoder copy: "
                "projection_only_target=True only supports ema_decay=(1.0, 1.0)."
            )

        cur_ema_value = (
            self.start_ema
            + self.trainer.global_step
            * (self.end_ema - self.start_ema)
            / self.trainer.max_steps
        )
        with torch.no_grad():
            self.trainer.record_metric(
                "train/ema_decay",
                cur_ema_value,
                ReduceType.mean,
            )
            for p, tp in zip(
                self.model.encoder.parameters(), self.model.target_encoder.parameters()
            ):
                if isinstance(p.data, DTensor):
                    # get the local shard, update it in place
                    p_local = p.data.to_local()
                    tp_local = tp.data.to_local()
                    tp_local.mul_(cur_ema_value).add_(
                        p_local, alpha=(1 - cur_ema_value)
                    )
                else:
                    # fallback for any plain Tensor
                    tp.data.mul_(cur_ema_value).add_(p.data, alpha=(1 - cur_ema_value))

    def eval_batch(
        self, batch: dict[str, Any], labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch not implemented")

    def compute_regularization(self, latent: TokensAndMasks) -> torch.Tensor | None:
        """If a regularizer is present, compute it."""
        regularizer = getattr(self, "regularizer", None)
        if regularizer is None:
            return None
        return regularizer.compute(latent, None)

    def log_regularization(self, total_batch_reg: torch.Tensor) -> None:
        """If a regularizer is present, log its values."""
        regularizer = getattr(self, "regularizer", None)
        if regularizer is None:
            return None
        self.trainer.record_metric(
            f"train/{regularizer.name}",
            total_batch_reg,
            ReduceType.mean,
        )

    def accumulate_extra_metrics(
        self,
        accumulator: dict[str, Any],
        counts: dict[str, int],
        extra_metrics: dict[str, Any],
    ) -> None:
        """Accumulate extra metrics across microbatches.

        Extra metrics used to be logged once per microbatch, which trips
        olmo-core's duplicate-metric warning (it keeps the first value for a
        given step/name and warns on the rest). We instead accumulate here and
        log once per step, like the train loss.

        Flattens one level of nesting to match ``log_extra_metrics`` key naming,
        and tracks a per-key count so metrics present in only some microbatches
        (e.g. a supervision modality absent from a microbatch) are averaged over
        the microbatches that actually contributed them.
        """

        def _add(name: str, value: Any) -> None:
            if isinstance(value, torch.Tensor):
                value = get_local_tensor(value.detach())
            accumulator[name] = accumulator.get(name, 0) + value
            counts[name] = counts.get(name, 0) + 1

        for key, value in extra_metrics.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _add(f"{key}/{sub_key}", sub_value)
            else:
                _add(key, value)

    def log_accumulated_extra_metrics(
        self,
        accumulator: dict[str, Any],
        counts: dict[str, int],
        reduce_type: ReduceType | None = None,
    ) -> None:
        """Log metrics accumulated across microbatches, averaged per key."""
        if not counts:
            return
        averaged = {name: total / counts[name] for name, total in accumulator.items()}
        self.log_extra_metrics(averaged, reduce_type=reduce_type)

    def log_extra_metrics(
        self, extra_metrics: dict[str, Any], reduce_type: ReduceType | None = None
    ) -> None:
        """Log the extra metrics."""
        for key, value in extra_metrics.items():
            if isinstance(value, dict):
                name_space = key
                for sub_key, sub_value in value.items():
                    self.trainer.record_metric(
                        f"{name_space}/{sub_key}", sub_value, reduce_type=reduce_type
                    )
            else:
                self.trainer.record_metric(f"{key}", value, reduce_type=reduce_type)


HeliosTrainModuleConfig = _deprecated_class_alias(
    OlmoEarthTrainModuleConfig,
    "helios.train.train_module.train_module.HeliosTrainModuleConfig",
)
HeliosTrainModule = _deprecated_class_alias(
    OlmoEarthTrainModule,
    "helios.train.train_module.train_module.HeliosTrainModule",
)
