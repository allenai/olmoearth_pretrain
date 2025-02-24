"""Training and optimizer abstraction for Helios."""

import contextlib
from collections.abc import Generator
from dataclasses import dataclass
from logging import getLogger
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.config import Config, DType
from olmo_core.distributed.parallel import (
    DataParallelConfig,
    DataParallelType,
    build_device_mesh,
    get_dp_mesh,
    get_dp_process_group,
)
from olmo_core.distributed.utils import get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config, Float8Handler
from olmo_core.optim import OptimConfig, SkipStepOptimizer
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec, TrainModule
from olmo_core.train.train_module.transformer import (
    TransformerActivationCheckpointingConfig,
)
from olmo_core.utils import gc_cuda, get_default_device
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

logger = getLogger(__name__)


@dataclass
class HeliosTrainModuleConfig(Config):
    """A configuration class for building :class:`HeliosTrainModule` instances.

    Args:
        rank_microbatch_size: The micro batch size per rank in instances.
        optim: The optimizer configuration.
        compile_model: Whether to compile the model using torch.compile.
        float8_config: Configuration for Float8 training if enabled.
        dp_config: Data parallel configuration for distributed training.
        ac_config: Activation checkpointing configuration.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
    """

    # Training settings
    rank_microbatch_size: int
    optim: OptimConfig

    # Model settings
    compile_model: bool = False
    float8_config: Float8Config | None = None  # UNTESTED for helios
    dp_config: DataParallelConfig | None = None
    ac_config: TransformerActivationCheckpointingConfig | None = (
        None  # UNTESTED for helios
    )

    # Loss function settings
    compile_loss: bool = False

    # Training settings
    autocast_precision: DType | None = None  # UNTESTED for helios
    max_grad_norm: float | None = None
    scheduler: Scheduler | None = None

    # Checkpoint settings
    state_dict_save_opts: dict[str, Any] | None = None
    state_dict_load_opts: dict[str, Any] | None = None

    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> "HeliosTrainModule":
        """Build the corresponding :class:`HeliosTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
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
        return HeliosTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class HeliosTrainModule(TrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        rank_microbatch_size: The rank micro batch size in instances.
        compile_model: Whether to compile to the model.
        float8_config: Float8 configuration for the model.
        dp_config: Data parallel configuration for the model.
        ac_config: Activation checkpointing configuration for the model.
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
        optim: OptimConfig,
        rank_microbatch_size: int,
        compile_model: bool = False,
        float8_config: Float8Config | None = None,
        dp_config: DataParallelConfig | None = None,
        ac_config: TransformerActivationCheckpointingConfig | None = None,
        compile_loss: bool = False,
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
            optim: The corresponding optimizer config.
            rank_microbatch_size: The rank batch size in instances.
            compile_model: Whether to compile to the model.
            float8_config: Float8 configuration for the model.
            dp_config: Data parallel configuration for the model.
            ac_config: Activation checkpointing configuration for the model.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
        """
        super().__init__()

        self.model = model
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"number of parameters: {num_params:,d}")
        self.device = device or get_default_device()
        self.world_mesh = build_device_mesh(dp=dp_config, device_type=self.device.type)
        logger.info(
            f"Data parallel world size = {get_world_size(self.dp_process_group):,d}"
        )

        self.float8_handler: Float8Handler | None = None
        # float8_enabled = False
        if float8_config is not None:
            # float8_enabled = float8_config.enabled
            float8_config.compile = compile_model
            self.float8_handler = float8_config.build()

        # Maybe convert linear layers to FP8 linear.
        if self.float8_handler is not None and self.float8_handler.enabled:
            self.float8_handler.convert_to_float8_training(
                self.model, modules_to_ignore={"lm_head.w_out"}
            )
            logger.info("Swapped linear layers to Float8 linear layers")

        # Maybe apply activation checkpointing.
        if ac_config is not None:
            self.model.apply_activation_checkpointing(
                ac_config.mode,
                block_interval=ac_config.block_interval,
                modules=ac_config.modules,
            )
            logger.info(
                f"Applied '{ac_config.mode}' activation checkpointing to the model"
            )

        # Maybe compile.
        if compile_model:
            self.model.apply_compile()
            logger.info("Applied torch.compile() to the model")

        # Maybe shard/replicate according to data parallel config.
        self._dp_config = dp_config
        if dp_config is not None:
            dp_mesh = get_dp_mesh(self.world_mesh)
            if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
                self.model.apply_fsdp(
                    dp_mesh=dp_mesh,
                    param_dtype=(
                        dp_config.param_dtype.as_pt()
                        if dp_config.param_dtype is not None
                        else None
                    ),
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                    wrapping_strategy=dp_config.wrapping_strategy,
                    pp_enabled=False,
                )
                logger.info("Applied FSDP to the model")
            elif dp_config.name == DataParallelType.ddp:
                self.model.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model)
                logger.info("Applied DDP to the model")
            else:
                raise NotImplementedError(dp_config.name)

        # Materialize and init parameters.
        logger.info("Initializing model weights...")
        # model.init_weights(max_seq_len=max_sequence_length, device=self.device)

        # Build optimizer(s).
        logger.info("Building optimizer(s)...")
        self.optimizer: Optimizer = optim.build(
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
        return get_dp_process_group(self.world_mesh)

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

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict."""
        return self._get_state_dict(self.state_dict_save_opts)

    def state_dict_to_load(self, metadata: Metadata) -> dict[str, Any]:
        """Get the state dict to load."""
        load_opts = self.state_dict_load_opts
        return self._get_state_dict(load_opts)

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
        # Maybe clip gradients.
        if self.max_grad_norm is not None:
            grad_norm = self._clip_grad_norm(self.max_grad_norm)
            # NOTE: grad norm is already reduced over ranks, so we set `reduce_type` to `None`.
            self.trainer.record_metric(
                "total grad norm", grad_norm, reduce_type=None, namespace="optim"
            )
            if isinstance(self.optimizer, SkipStepOptimizer):
                self.optimizer.latest_grad_norm = grad_norm

        # Sync Float8 AMAXs (argmax of abs(max)) and scales.
        if self.float8_handler is not None:
            self.float8_handler.sync_float8_amax_and_scale_history(self.model)

        # Maybe adjust learning rate.
        if self.scheduler is not None:
            for group_idx, group in enumerate(self.optimizer.param_groups):
                if (lr_field := self.scheduler.lr_field) not in group and (
                    initial_lr_field := self.scheduler.initial_lr_field
                ) not in group:
                    group_fields_list = "\n - ".join(
                        [f"{k}: {v}" for k, v in group.items() if k != "params"]
                    )
                    raise RuntimeError(
                        f"learning rate field '{lr_field}' and initial learning rate field "
                        f"'{initial_lr_field}' not found in optimizer param group {group_idx} "
                        f"with {len(group['params'])} parameter(s):\n"
                        f" - {group_fields_list}"
                    )

                # Ensure 'initial_lr' is set.
                if group.get(self.scheduler.initial_lr_field) is None:
                    group[self.scheduler.initial_lr_field] = group["lr"]

                # Set new LR.
                new_lr = self.scheduler.get_lr(
                    group[self.scheduler.initial_lr_field],
                    self.trainer.global_step,
                    self.trainer.max_steps,
                )

                if isinstance(
                    current_lr := group.get(self.scheduler.lr_field), torch.Tensor
                ):
                    current_lr.fill_(new_lr)
                else:
                    group[self.scheduler.lr_field] = new_lr

                self.trainer.record_metric(
                    f"LR (group {group_idx})",
                    group[self.scheduler.lr_field],
                    namespace="optim",
                )

        # Step optimizer.
        self.optimizer.step()
        if isinstance(self.optimizer, SkipStepOptimizer):
            self.trainer.record_metric(
                "step skipped", self.optimizer.step_skipped, namespace="optim"
            )

        # Calculate Float8 dynamic AMAX/scale for all parameters.
        # For FSDP2 this issues a single all-reduce for all parameters at once.
        if self.float8_handler is not None:
            self.float8_handler.precompute_float8_dynamic_scale_for_fsdp(self.model)

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, DDP) and micro_batch_idx != num_micro_batches - 1:
                # For DDP, only sync gradients on the final micro batch.
                stack.enter_context(self.model.no_sync())
            yield

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(
                    torch.autocast(self.device.type, dtype=self.autocast_precision)
                )
            yield

    def _clear_loss_buffers(self) -> None:
        """Clear the loss buffers."""
        logger.warning("clear loss buffers not implemented")
        pass

    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions
    ) -> dict[str, Any]:
        return {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=sd_options),
            "optim": dist_cp_sd.get_optimizer_state_dict(
                self.model, self.optimizer, options=sd_options
            ),
        }

    def _clip_grad_norm(
        self,
        max_grad_norm: float,
        norm_type: float = 2.0,
        foreach: bool | None = None,
    ) -> torch.Tensor:
        """Clip the gradients."""
        if isinstance(self.model, FSDP):
            return self.model.clip_grad_norm_(max_grad_norm)
        # Pipeline parallel grad clipping required nightly torch
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm, norm_type=norm_type, foreach=foreach
        )
