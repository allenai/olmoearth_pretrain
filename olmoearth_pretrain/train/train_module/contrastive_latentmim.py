"""Training and optimizer abstraction for OlmoEarth Pretrain."""

import math
import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType
from torch.distributed.tensor import DTensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskingConfig,
    propagate_tokenization_config,
)
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

logger = getLogger(__name__)


@dataclass
class ContrastiveLatentMIMTrainModuleConfig(OlmoEarthTrainModuleConfig):
    """A configuration class for building :class:`LatentMIMTrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
        ema_decay: EMA decay rate for target encoder (default: 0.99).
    """

    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "patch_discrimination"})
    )
    mae_loss_config: LossConfig | None = None
    masking_config: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    token_exit_cfg: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    ema_decay: tuple[float, float] = (0.996, 1.0)
    max_grad_norm: float = 1.0
    contrastive_config: LossConfig | None = None
    reinit_targets: bool = False
    # Clamp for the model's learned log logit_scale (CLIP temperature):
    # exp(4.6) ~ 99.5, matching CLIP's max scale of 100.
    max_logit_scale: float = 4.6
    # Separate (lower) clamp for the instance loss temperature: its 64-way
    # cross-sample softmax saturates far earlier than the token loss's, and
    # the effective class-token gradient scales with weight x temperature —
    # exp(3.4) ~ 30 bounds that drift.
    max_instance_logit_scale: float = 3.4

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "ContrastiveLatentMIMTrainModuleConfig":
        """Build the corresponding :class:`ContrastiveLatentMIMTrainModuleConfig`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return ContrastiveLatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class ContrastiveLatentMIMTrainModule(OlmoEarthTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.
    """

    def __init__(
        self,
        model: LatentMIM,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg: dict[str, int],
        mae_loss_config: LossConfig | None = None,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        ema_decay: tuple[float, float] = (0.996, 1.0),
        regularizer_config: LossConfig | None = None,
        contrastive_config: LossConfig | None = None,
        find_unused_parameters: bool = True,
        reinit_targets: bool = False,
        max_logit_scale: float = 4.6,
        max_instance_logit_scale: float = 3.4,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
            masking_config: The masking configuration for the model.
            loss_config: The loss configuration for the model.
            mae_loss_config: Optional loss config for masked auto-encoding.
            rank_microbatch_size: The rank microbatch size in instances.
            compile_model: Whether to compile to the model.
            dp_config: Data parallel configuration for the model.
            loss_fn: Loss function to use.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
            ema_decay: EMA decay rate for target encoder, as a tuple of (start_ema_decay, end_ema_decay)
            token_exit_cfg: The token exit configuration for the model.
            regularizer_config: An optional regularizer configuration for the model.
            contrastive_config: An optional contrastive configration for the model.
            find_unused_parameters: Whether to find unused parameters in the model, only used for DDP.
            reinit_targets: Whether or not to reinitialize the target encoder.
            max_logit_scale: Clamp (in log space) for the model's learned CLIP
                temperature; the exp'd scale is passed to the base loss.
            max_instance_logit_scale: Clamp (in log space) for the instance
                loss temperature (lower than the token loss's).
        """
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
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg = token_exit_cfg
        self.max_logit_scale = max_logit_scale
        self.max_instance_logit_scale = max_instance_logit_scale
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()
        tokenization_config = getattr(self.model.encoder, "tokenization_config", None)
        if tokenization_config is not None:
            propagate_tokenization_config(self.masking_strategy, tokenization_config)
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )
        self.contrastive_loss = (
            contrastive_config.build() if contrastive_config is not None else None
        )
        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
        if self.mae_loss is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.mae_loss.name}"
        if reinit_targets:
            if ema_decay != (0.0, 0.0):
                logger.warning(
                    "Applying EMA updates to a randomly initialized target encoder."
                )
            self.model.target_encoder.apply(self.model.target_encoder._init_weights)

    def loss_fn(self, pred: Any, targets: Any, **kwargs: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets, **kwargs)

    def _maybe_capture_exploding_batch(
        self,
        capture_dir: str,
        microbatch_idx: int,
        microbatch_a: MaskedOlmoEarthSample,
        microbatch_b: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> None:
        """Debug hook: dump the microbatch when decoder-head grads explode.

        Enabled via the OE_DEBUG_CAPTURE_DIR env var; checks the parameters
        the production explosions concentrate in.
        """
        from olmo_core.distributed.utils import get_local_tensor as _glt

        head_norm = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if (
                "to_output_embed" in name
                or name.endswith("decoder.norm.bias")
                or name.endswith("mask_token")
            ):
                grad = _glt(p.grad)
                if grad.numel():
                    val = grad.float().norm().item()
                    head_norm = max(head_norm, val)
        # Gradients are FSDP-reduced across ranks by the time we check, so a
        # single guilty batch poisons every rank's shards — and the guilty
        # rank's own shards may look fine. All-reduce the flag so every
        # rank's batch is captured whenever any rank trips.
        import torch.distributed as dist

        if not (head_norm > 1e4 or not math.isfinite(head_norm)):
            # also reset the stash on clean microbatches
            pass
        bad = head_norm > 1e4 or not math.isfinite(head_norm)
        if dist.is_initialized():
            flag = torch.tensor(
                [1.0 if bad else 0.0], device=next(self.model.parameters()).device
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            bad = flag.item() > 0
        if not bad:
            self._debug_decoded: list[TokensAndMasks] = []
        if bad:
            from olmo_core.distributed.utils import get_rank

            step = getattr(getattr(self, "trainer", None), "global_step", -1)
            rank = get_rank()
            if len(os.listdir(capture_dir)) > 40:
                self._debug_decoded = []
                return
            decoded_grads = []
            for view_idx, decoded in enumerate(getattr(self, "_debug_decoded", [])):
                dec_dict = decoded.as_dict()
                for m in decoded.modalities:
                    t = dec_dict[m]
                    if t.grad is None:
                        continue
                    mask = dec_dict[decoded.get_masked_modality_name(m)]
                    for mask_value in (0, 1, 2, 3):
                        sel = mask == mask_value
                        if not sel.any():
                            continue
                        g = t.grad[sel].float()
                        vals = t.detach()[sel].float().norm(dim=-1)
                        decoded_grads.append(
                            dict(
                                view=view_idx,
                                modality=m,
                                mask_value=mask_value,
                                count=int(sel.sum()),
                                grad_norm=g.norm().item(),
                                grad_absmax=g.abs().max().item(),
                                value_norm_min=vals.min().item(),
                                value_norm_max=vals.max().item(),
                            )
                        )
            self._debug_decoded = []
            path = os.path.join(
                capture_dir,
                f"exploding_batch_step{step}_rank{rank}_mb{microbatch_idx}.pt",
            )
            if not os.path.exists(path):
                # Write atomically: ranks share the filesystem and a partial
                # file from a raced/killed writer is useless.
                tmp_path = f"{path}.rank{rank}.tmp"
                torch.save(
                    {
                        "microbatch_a": microbatch_a.as_dict(),
                        "microbatch_b": microbatch_b.as_dict(),
                        "head_norm": head_norm,
                        "step": step,
                        "patch_size": patch_size,
                        "rank": rank,
                        "decoded_grads": decoded_grads,
                    },
                    tmp_path,
                )
                os.replace(tmp_path, path)
                logger.warning(
                    "Captured exploding batch (head grad norm %.3e) to %s",
                    head_norm,
                    path,
                )

    def _effective_logit_scale(self, param_name: str = "logit_scale") -> torch.Tensor:
        """CLIP temperature as the final multiplicative scale (clamp + exp).

        The parameter may be an FSDP DTensor whose local shard is empty on
        some ranks, so it is gathered to a full tensor (differentiable) first.

        Args:
            param_name: Which temperature parameter on the model to read —
                "logit_scale" (token loss) or "instance_logit_scale".
        """
        if os.environ.get("OE_DEBUG_CONST_SCALE"):
            # Bisection probe: fixed scale, no parameter read, no DTensor
            # collective in the loss graph.
            return torch.tensor(1.0 / 0.07, device=self.device)
        max_scale = (
            self.max_instance_logit_scale
            if param_name == "instance_logit_scale"
            else self.max_logit_scale
        )
        logit_scale = getattr(self.model, param_name)
        if isinstance(logit_scale, DTensor):
            logit_scale = logit_scale.full_tensor()
        return logit_scale.clamp(max=max_scale).exp()

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For non contrastive losses, the loss is invariant to the global batch size across GPUS as well

        Args:
            batch: A (patch_size, MaskedOlmoEarthSample_a, MaskedOlmoEarthSample_b) tuple from the dataloader.
            dry_run: If True, skip metric recording and just run forward/backward.
        """
        if not dry_run:
            self.update_target_encoder()
        # The loss-side clamp is forward-only; re-clamp the parameters
        # themselves so the optimizer cannot walk the log-scales arbitrarily
        # far past the cap (CLIP clamps param data after each optimizer step).
        with torch.no_grad():
            get_local_tensor(self.model.logit_scale).clamp_(max=self.max_logit_scale)
            get_local_tensor(self.model.instance_logit_scale).clamp_(
                max=self.max_instance_logit_scale
            )
        # Set the model to train mode
        self.model.train()
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        total_batch_con = torch.zeros([], device=self.device)
        total_batch_con_raw = torch.zeros([], device=self.device)
        total_batch_acc = torch.zeros([], device=self.device)

        # Unpack batch
        patch_size = batch[0]
        batch_data_a = batch[1]
        batch_data_b = batch[2]
        microbatches_a = split_masked_batch(batch_data_a, self.rank_microbatch_size)
        microbatches_b = split_masked_batch(batch_data_b, self.rank_microbatch_size)
        num_microbatches = len(microbatches_a)

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                microbatch_a = microbatches_a[microbatch_idx]
                microbatch_b = microbatches_b[microbatch_idx]
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} "
                    f"with batch size {microbatch_a.batch_size}"
                )
                masked_batch_a = microbatch_a.to_device(self.device)
                masked_batch_b = microbatch_b.to_device(self.device)

                # Run Encoder and decoder on the augmented input
                loss_a, latent_a, decoded_a, target_output_a, pooled_a = (
                    self.model_forward(masked_batch_a, patch_size, self.token_exit_cfg)
                )
                loss_b, latent_b, decoded_b, target_output_b, pooled_b = (
                    self.model_forward(masked_batch_b, patch_size, self.token_exit_cfg)
                )
                loss = (loss_a + loss_b) / 2

                # Scale loss by number of microbatches
                reg_term_a = self.compute_regularization(pooled_a)
                reg_term_b = self.compute_regularization(pooled_b)
                if reg_term_a is not None:
                    assert reg_term_b is not None
                    loss = loss + (reg_term_a + reg_term_b) / 2
                    total_batch_reg += (
                        get_local_tensor(
                            (reg_term_a.detach() + reg_term_b.detach()) / 2
                        )
                        / num_microbatches
                    )

                if self.contrastive_loss is not None:
                    contrastive_loss = self.contrastive_loss.compute(
                        pooled_a,
                        pooled_b,
                        logit_scale=self._effective_logit_scale("instance_logit_scale"),
                    )
                    loss += contrastive_loss
                    total_batch_con += (
                        get_local_tensor(contrastive_loss.detach()) / num_microbatches
                    )
                    with torch.no_grad():
                        # Diagnostics for the instance objective: 64-way
                        # retrieval accuracy of the cross-view softmax and the
                        # raw (unweighted) CE, which the 0.05 weight obscures.
                        a_n = F.normalize(pooled_a.detach().float(), dim=-1)
                        b_n = F.normalize(pooled_b.detach().float(), dim=-1)
                        match = (a_n @ b_n.T).argmax(dim=1) == torch.arange(
                            a_n.shape[0], device=a_n.device
                        )
                        total_batch_acc += match.float().mean() / num_microbatches
                        weight = getattr(self.contrastive_loss, "weight", 1.0)
                        total_batch_con_raw += (
                            get_local_tensor(contrastive_loss.detach())
                            / max(weight, 1e-8)
                            / num_microbatches
                        )

                loss = loss / num_microbatches

                # Skip bad batches — synchronized across ranks: a rank-local
                # break desyncs FSDP collectives (the broken rank issues fewer
                # reduce-scatters and never finalizes grad reduction) and the
                # subsequent optim step would consume partial gradients.
                bad = (torch.isnan(loss) | torch.isinf(loss)).any()
                if dist.is_initialized():
                    bad_flag = bad.detach().float()
                    dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
                    bad = bad_flag > 0
                if bad:
                    logger.warning(
                        "Non-finite loss at microbatch %s on some rank; zeroing "
                        "gradients and skipping the rest of this batch on all ranks.",
                        microbatch_idx,
                    )
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.detach().zero_()
                    del latent_a, latent_b
                    break

                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                del latent_a, latent_b
                loss.backward()

                capture_dir = os.environ.get("OE_DEBUG_CAPTURE_DIR")
                if capture_dir:
                    self._maybe_capture_exploding_batch(
                        capture_dir,
                        microbatch_idx,
                        microbatch_a,
                        microbatch_b,
                        patch_size,
                    )

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        if self.contrastive_loss is not None:
            self.trainer.record_metric(
                f"train/{self.contrastive_loss.name}",
                total_batch_con,
                ReduceType.mean,
            )
            self.trainer.record_metric(
                f"train/{self.contrastive_loss.name}_raw_CE",
                total_batch_con_raw,
                ReduceType.mean,
            )
            self.trainer.record_metric(
                "train/instance_top1_acc",
                total_batch_acc,
                ReduceType.mean,
            )
        # The effective CLIP temperatures are the best early-warning signal
        # for shortcut/false-negative problems in the contrastive losses.
        with torch.no_grad():
            self.trainer.record_metric(
                "train/logit_scale",
                self._effective_logit_scale().mean(),
                ReduceType.mean,
            )
            self.trainer.record_metric(
                "train/instance_logit_scale",
                self._effective_logit_scale("instance_logit_scale").mean(),
                ReduceType.mean,
            )
        self.log_regularization(total_batch_reg)

        del batch  # In case this helps with memory utilization.
        del masked_batch_a, masked_batch_b

    def model_forward(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> tuple[
        torch.Tensor, TokensAndMasks, TokensAndMasks, TokensAndMasks, torch.Tensor
    ]:
        """Run a forward pass."""
        with self._model_forward_context():
            (
                latent,
                decoded,
                latent_projected_and_pooled,
                reconstructed,
                extra_metrics,
            ) = self.model(batch, patch_size)
            if os.environ.get("OE_DEBUG_CAPTURE_DIR"):
                # Keep d(loss)/d(decoded) so the capture hook can attribute
                # gradient explosions to positions/mask values.
                for _m in decoded.modalities:
                    getattr(decoded, _m).retain_grad()
                if not hasattr(self, "_debug_decoded"):
                    self._debug_decoded = []
                self._debug_decoded.append(decoded)
            if extra_metrics is not None:
                self.log_extra_metrics(extra_metrics)
            with torch.no_grad():
                logger.debug("Target Encoder forward pass...")
                output_dict = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
                target_output, _, _ = unpack_encoder_output(output_dict)
            loss = self.loss_fn(
                decoded, target_output, logit_scale=self._effective_logit_scale()
            )
            if self.mae_loss is not None and reconstructed is not None:
                loss += self.mae_loss.compute(reconstructed, batch)
            return loss, latent, decoded, target_output, latent_projected_and_pooled
