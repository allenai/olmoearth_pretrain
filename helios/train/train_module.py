"""Training and optimizer abstraction for Helios."""

import contextlib
import math
from collections.abc import Callable, Generator
from logging import getLogger
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from einops import rearrange
from olmo_core.distributed.parallel import (
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
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec, TrainModule
from olmo_core.train.train_module.transformer import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)
from olmo_core.utils import gc_cuda, get_default_device, move_to_device
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

logger = getLogger(__name__)

TRAIN_PATCH_DISC_LOSS_METRIC = "train/patch_disc_loss"


# I can also build a helios config that follows omo core config however we want
class HeliosTrainModule(TrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        rank_batch_size: The rank batch size in instances.
        compile_model: Whether to compile to the model.
        float8_config: Float8 configuration for the model.
        dp_config: Data parallel configuration for the model.
        ac_config: Activation checkpointing configuration for the model.
        loss_fn: Loss function to use.
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
        rank_batch_size: int,
        compile_model: bool = False,
        float8_config: Float8Config | None = None,
        dp_config: TransformerDataParallelConfig | None = None,
        ac_config: TransformerActivationCheckpointingConfig | None = None,
        loss_fn: Callable | None = None,
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
            rank_batch_size: The rank batch size in instances.
            compile_model: Whether to compile to the model.
            float8_config: Float8 configuration for the model.
            dp_config: Data parallel configuration for the model.
            ac_config: Activation checkpointing configuration for the model.
            loss_fn: Loss function to use.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
        """
        super().__init__()
        self.moe_handler = None
        self.ema_decay = 0.99
        self.model = model
        self.device = device or get_default_device()
        self.world_mesh = build_device_mesh(dp=dp_config, device_type=self.device.type)
        logger.info(
            f"Data parallel world size = {get_world_size(self.dp_process_group):,d}"
        )

        self.base_loss_fn = loss_fn
        if compile_loss:
            self.base_loss_fn = torch.compile(self.base_loss_fn)

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

        self.rank_batch_size = rank_batch_size
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

    def loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        # TODO: add more generic and configurable loss function support
        if self.base_loss_fn is None:
            raise ValueError("base loss fn is not set")
        return self.base_loss_fn(pred, target)

    def eval_loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        raise NotImplementedError("eval loss fn not implemented")

    def on_attach(self) -> None:
        """Called when the train module is attached to the trainer."""
        # Validate batch size.
        dp_ws = get_world_size(self.trainer.dp_process_group)
        if self.trainer.global_batch_size % (self.rank_batch_size * dp_ws) != 0:
            raise OLMoConfigurationError(
                f"global batch size ({self.trainer.global_batch_size:,d}) must be divisible by "
                f"micro-batch size ({self.rank_batch_size:,d}) x DP world size ({dp_ws})"
            )

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict."""
        return self._get_state_dict(self.state_dict_save_opts)

    def state_dict_to_load(self, metadata: Metadata) -> dict[str, Any]:
        """Get the state dict to load."""
        # TODO: Unclear how we want to use this metadata
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

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False) -> None:
        """Train a batch."""
        # Record how many instances are going to be skipped (masked out).
        # if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
        #     self.record_metric("train/masked instances", (~instance_mask).sum(), ReduceType.sum)

        # Move tensors to the right device.
        # we may want to modify this
        batch = move_to_device(batch, self.device)

        # Run Encoder and decoder on the augmented input
        decoded, loss = self.model_forward(batch)

        self.trainer.record_metric(
            TRAIN_PATCH_DISC_LOSS_METRIC,
            loss / get_world_size(self.dp_process_group),
            ReduceType.sum,
        )

        # Backpropagate and optimize
        if loss is not None:
            loss.backward()
        # Update target encoder with EMA this should be a callback
        with torch.no_grad():
            for param, target_param in zip(
                self.model.encoder.parameters(), self.model.target_encoder.parameters()
            ):
                target_param.data = (
                    self.ema_decay * target_param.data
                    + (1 - self.ema_decay) * param.data
                )

        del batch  # In case this helps with memory utilization.

        if dry_run:
            self._clear_loss_buffers()
            return

        # TODO: Record loss metrics.
        # NOTE: losses could be none for pipeline parallelism if rank doesn't have the final stage.

        # Lastly, clear internal loss buffers.
        self._clear_loss_buffers()

    def eval_batch(
        self, batch: dict[str, Any], labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch not implemented")

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

    def model_forward(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run a forward pass."""
        with self._model_forward_context():
            with torch.no_grad():
                s2_data = batch["sentinel2"]
                input = rearrange(s2_data, "b h w t c -> b c t h w")
                target_output = self.model.target_encoder.forward(input)

            # Run Encoder and decoder on the augmented input
            decoded = self.model.forward(input)
            loss = self.loss_fn(decoded, target_output)
            return decoded, loss

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        raise NotImplementedError("train microbatch context not implemented")

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(
                    torch.autocast(self.device.type, dtype=self.autocast_precision)
                )
            yield

    def _clear_loss_buffers(self) -> None:
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
        logger.info("clip grad norm has not been adapted for helios")
        if isinstance(self.model, FSDP):
            return self.model.clip_grad_norm_(max_grad_norm)

        # Adapted from https://github.com/pytorch/torchtitan/blob/2a4437014e66bcf88a3f0419b816266e6326d539/torchtitan/utils.py#L348

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

        if self.train_pp_schedule is not None:
            pp_mesh = self.train_pp_schedule.pp_mesh
            if math.isinf(norm_type):
                dist.all_reduce(
                    total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group()
                )
            else:
                total_norm **= norm_type
                dist.all_reduce(
                    total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group()
                )
                total_norm **= 1.0 / norm_type

        torch.nn.utils.clip_grads_with_norm_(
            parameters, max_grad_norm, total_norm, foreach=foreach
        )
        return total_norm
