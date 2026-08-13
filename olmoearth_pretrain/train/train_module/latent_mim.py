"""Training and optimizer abstraction for OlmoEarth Pretrain."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_rank, get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.supervision_head import compute_supervision_loss
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)
from olmoearth_pretrain.train.utils import split_masked_batch

logger = getLogger(__name__)


def compute_register_uniformity_loss(
    registers: torch.Tensor,
    weight: float,
    num_rotations: int = 2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Spread the register grid over the sphere (AlphaEarth's "batch uniformity").

    AEF compare each embedding against batch-rotated copies by dot product and
    minimize the absolute value, a necessary condition for uniformity in
    ``S^{D-1}`` (their S2.2.4). Unit norm alone says only that embeddings live on
    the sphere; this is what makes them cover it, which is the same thing as
    information being spread across dimensions rather than concentrated on a few.

    Pairs are strictly CROSS-SCENE. Rolling the batch axis pairs cell ``(b, n)``
    with ``(b + r, n)`` -- the same grid position in a different sample -- so no
    pair ever relates two cells of one scene. That asymmetry is deliberate and is
    the opposite of the Gram terms above, which want both scales: cells within a
    scene genuinely ARE similar (a homogeneous field is homogeneous), the dense
    probes read exactly that spatial smoothness, and penalizing it would fight the
    data. Holding the cell index fixed also controls for grid position, so the term
    measures scene-to-scene spread rather than position effects.

    Unlike the distillation terms, this is NOT detached: it shapes the encoder.

    Args:
        registers: ``[B, N, D]`` register grid (or student output) to spread. The
            grid axis is kept, so cross-scene pairing stays well defined.
        weight: Multiplier on the term. 0 disables it (checked by the caller).
        num_rotations: How many distinct batch offsets to average over. Each costs
            one ``O(B * N * D)`` pass; more offsets give a lower-variance estimate
            of the same population quantity.

    Returns:
        total: The weighted term.
        metrics: Detached value for logging.
    """
    if registers.shape[0] < 2:
        # A single scene has no cross-scene pair; skip rather than silently
        # falling back to within-scene pairs.
        zero = torch.zeros([], device=registers.device, dtype=torch.float32)
        return zero, {}
    z = F.normalize(registers.float(), dim=-1)
    batch_size = z.shape[0]
    # Distinct, non-zero offsets only: shift 0 would pair every cell with itself
    # (dot product 1) and the term would be minimized by nothing.
    offsets = [r for r in range(1, num_rotations + 1) if r < batch_size]
    if not offsets:
        offsets = [1]
    terms = [
        (z * torch.roll(z, shifts=offset, dims=0)).sum(dim=-1).abs().mean()
        for offset in offsets
    ]
    uniformity = torch.stack(terms).mean()
    return weight * uniformity, {"register/uniformity": uniformity.detach()}


def compute_projection_distill_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    back_projections: dict[str, torch.nn.Module],
    cosine_weight: float,
    gram_weight: float,
    gram_max_tokens: int,
    gram_within_weight: float = 0.0,
    gram_within_max_cells: int = 256,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Distill the (detached) teacher register grid into the low-dim student.

    Each entry of ``back_projections`` is a Matryoshka prefix width ``d`` (as a
    string key): the first ``d`` dims of the student are distilled onto the full
    teacher through their own back-projection (cosine) and their own relational
    Gram term, so every listed prefix is trained to be self-sufficient
    (Tessera-v2 per-prefix heads). Terms are summed unweighted across prefixes.

    Args:
        teacher: Register grid ``[B, N, D]``; detached here, so this loss never
            reaches the encoder.
        student: Projected register grid ``[B, N, max_d]`` (the detached-input
            student; gradients flow into the projection + back-projections only).
        back_projections: Per-prefix learned ``d -> D`` maps, keyed by ``str(d)``.
        cosine_weight: Weight of ``1 - cos(back_projection_d(student[..., :d]),
            teacher)`` (each prefix).
        gram_weight: Weight of the MSE between each student prefix's and the
            teacher's token-token cosine-similarity matrices (relational/RKD term).
        gram_max_tokens: Max register cells entering the Gram terms (one random
            subsample shared across prefixes; bounds the O(n^2) matrices).
        gram_within_weight: Weight of the WITHIN-SCENE (block-diagonal) Gram term:
            the same relational MSE, but computed per scene so every pair relates
            two cells of the SAME sample. The flat term above is built over the
            flattened ``[B * N]`` grid, so only ~1/B of its pairs are within-scene;
            the block-diagonal form is all of them, and cheaper per pair by a factor
            of B (m blocks of k cells cost O(m * k^2) against O((m * k)^2)). Dense
            probes discriminate WITHIN a scene, so this asks whether the pairs that
            metric depends on were simply too rare to matter. 0.0 (default) leaves
            the loss exactly as the flat-only runs saw it.
        gram_within_max_cells: Register cells sampled PER SCENE for the within-scene
            term (one subsample of cell positions, shared across scenes and prefixes
            so the blocks stay spatially aligned). Scenes with fewer cells use all of
            them; a single-cell grid skips the term (no pairs exist).

    Returns:
        total: The weighted sum of the enabled terms across prefixes.
        metrics: Detached per-term, per-prefix values for logging.
    """
    teacher = teacher.detach().float()
    student = student.float()
    metrics: dict[str, torch.Tensor] = {}
    total = torch.zeros([], device=student.device, dtype=student.dtype)
    flat_teacher: torch.Tensor | None = None
    teacher_gram: torch.Tensor | None = None
    idx: torch.Tensor | None = None
    if gram_weight > 0:
        flat_teacher = F.normalize(teacher.reshape(-1, teacher.shape[-1]), dim=-1)
        num_tokens = flat_teacher.shape[0]
        if num_tokens > gram_max_tokens:
            idx = torch.randperm(num_tokens, device=flat_teacher.device)[
                :gram_max_tokens
            ]
            flat_teacher = flat_teacher[idx]
        teacher_gram = flat_teacher @ flat_teacher.T
    # Within-scene (block-diagonal) Gram: keep the [B, N, D] layout so each matrix
    # only ever relates two cells of the same sample. The register grid is shared
    # across the microbatch (the dynamic bottleneck sizes one (h, w) per batch), so
    # a single cell subsample applies to every scene and this stays a plain bmm.
    within_idx: torch.Tensor | None = None
    teacher_within_gram: torch.Tensor | None = None
    if gram_within_weight > 0 and teacher.shape[1] >= 2:
        num_cells = teacher.shape[1]
        if num_cells > gram_within_max_cells:
            within_idx = torch.randperm(num_cells, device=teacher.device)[
                :gram_within_max_cells
            ]
        within_teacher = teacher if within_idx is None else teacher[:, within_idx]
        within_teacher = F.normalize(within_teacher, dim=-1)
        teacher_within_gram = within_teacher @ within_teacher.transpose(1, 2)
    for dim_str, back_projection in back_projections.items():
        prefix = student[..., : int(dim_str)]
        if cosine_weight > 0:
            back = back_projection(prefix)
            cosine = (1.0 - F.cosine_similarity(back, teacher, dim=-1)).mean()
            total = total + cosine_weight * cosine
            metrics[f"projection/distill_cosine_d{dim_str}"] = cosine.detach()
        if gram_weight > 0:
            assert teacher_gram is not None
            flat_prefix = F.normalize(prefix.reshape(-1, prefix.shape[-1]), dim=-1)
            if idx is not None:
                flat_prefix = flat_prefix[idx]
            gram = F.mse_loss(flat_prefix @ flat_prefix.T, teacher_gram)
            total = total + gram_weight * gram
            metrics[f"projection/distill_gram_d{dim_str}"] = gram.detach()
        if teacher_within_gram is not None:
            within_prefix = prefix if within_idx is None else prefix[:, within_idx]
            within_prefix = F.normalize(within_prefix, dim=-1)
            gram_within = F.mse_loss(
                within_prefix @ within_prefix.transpose(1, 2), teacher_within_gram
            )
            total = total + gram_within_weight * gram_within
            metrics[f"projection/distill_gram_within_d{dim_str}"] = gram_within.detach()
    return total, metrics


@dataclass
class LatentMIMTrainModuleConfig(OlmoEarthTrainModuleConfig):
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
    # Distillation losses for the encoder's detached register projection (the low-dim
    # "student"; see EncoderConfig.register_projection_dim). Only used when the model
    # produces projection outputs. Cosine: 1 - cos(back_projection(student), teacher)
    # per register cell (Tessera-v2 style, via the encoder's learned back-projection).
    # Gram: MSE between the student's and teacher's token-token cosine-similarity
    # matrices over a random subsample of register cells (relational/RKD term --
    # preserves the teacher's geometry, the property dense probes use).
    projection_distill_cosine_weight: float = 1.0
    projection_distill_gram_weight: float = 1.0
    # Max register cells (across the microbatch) entering the Gram term; bounds the
    # O(n^2) similarity matrices.
    projection_distill_gram_max_tokens: int = 2048
    # Within-scene (block-diagonal) Gram: the same relational term computed per
    # sample, so 100% of its pairs are between cells of one scene rather than the
    # ~1/B the flat term gives. 0.0 keeps the flat-only behaviour of earlier runs.
    projection_distill_gram_within_weight: float = 0.0
    projection_distill_gram_within_max_cells: int = 256
    # Uniformity (AlphaEarth's "batch uniformity", S2.2.4): push the register grid
    # to cover the sphere instead of crowding into a cap of it, using strictly
    # cross-scene pairs. Only meaningful alongside EncoderConfig.register_unit_norm
    # (on an unnormalized grid the term can be minimized by shrinking magnitudes
    # rather than by spreading directions). 0.0 (default) leaves the loss exactly as
    # every run so far has seen it. The second weight applies the same term to the
    # distillation student, which is the served embedding when one is deployed.
    register_uniformity_weight: float = 0.0
    projection_uniformity_weight: float = 0.0
    # Distinct batch offsets averaged over per step; each is one O(B*N*D) pass.
    register_uniformity_rotations: int = 2

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "LatentMIMTrainModule":
        """Build the corresponding :class:`LatentMIMTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return LatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class LatentMIMTrainModule(OlmoEarthTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
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
        token_exit_cfg: The token exit configuration for the model.
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
        scheduler_overrides: dict[str, Scheduler] | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        ema_decay: tuple[float, float] = (0.996, 1.0),
        regularizer_config: LossConfig | None = None,
        find_unused_parameters: bool = True,
        projection_distill_cosine_weight: float = 1.0,
        projection_distill_gram_weight: float = 1.0,
        projection_distill_gram_max_tokens: int = 2048,
        projection_distill_gram_within_weight: float = 0.0,
        projection_distill_gram_within_max_cells: int = 256,
        register_uniformity_weight: float = 0.0,
        projection_uniformity_weight: float = 0.0,
        register_uniformity_rotations: int = 2,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
            masking_config: The masking configuration for the model.
            loss_config: The loss configuration for the model.
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
            mae_loss_config: Optional loss config for masked auto-encoding.
            regularizer_config: An optional regularizer configuration for the model.
            find_unused_parameters: Whether to find unused parameters in the model, only used for DDP.
            projection_distill_cosine_weight: Weight of the cosine distillation term
                for the detached register projection (see the config docstring).
            projection_distill_gram_weight: Weight of the Gram (relational)
                distillation term for the detached register projection.
            projection_distill_gram_max_tokens: Max register cells entering the Gram
                term per microbatch (bounds the O(n^2) similarity matrices).
            projection_distill_gram_within_weight: Weight of the within-scene
                (block-diagonal) Gram term. 0.0 disables it.
            projection_distill_gram_within_max_cells: Register cells sampled per
                scene for the within-scene Gram term.
            register_uniformity_weight: Weight of the cross-scene uniformity term on
                the register grid (AlphaEarth's batch uniformity). Shapes the
                encoder. Expects EncoderConfig.register_unit_norm.
            projection_uniformity_weight: The same term applied to the distillation
                student's own output.
            register_uniformity_rotations: Batch offsets averaged per step.
            scheduler_overrides: Optional per-param-group schedulers, keyed by the
                group's "group_name" tag; groups without a match use `scheduler`.
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
            scheduler_overrides=scheduler_overrides,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
            find_unused_parameters=find_unused_parameters,
        )
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg = token_exit_cfg
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )

        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
        if self.mae_loss is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.mae_loss.name}"

        self._supervised_modality_names: list[str] = []
        if self.model.supervision_head is not None:
            self._supervised_modality_names = list(
                self.model.supervision_head.modality_configs.keys()
            )
            self.total_loss_name = f"{self.total_loss_name}+supervision"

        self.projection_distill_cosine_weight = projection_distill_cosine_weight
        self.projection_distill_gram_weight = projection_distill_gram_weight
        self.projection_distill_gram_max_tokens = projection_distill_gram_max_tokens
        self.projection_distill_gram_within_weight = (
            projection_distill_gram_within_weight
        )
        self.projection_distill_gram_within_max_cells = (
            projection_distill_gram_within_max_cells
        )
        self.register_uniformity_weight = register_uniformity_weight
        self.projection_uniformity_weight = projection_uniformity_weight
        self.register_uniformity_rotations = register_uniformity_rotations
        if register_uniformity_weight > 0 or projection_uniformity_weight > 0:
            if not getattr(self.model.encoder, "use_register_bottleneck", False):
                raise ValueError(
                    "uniformity weights require a register bottleneck (there is no "
                    "register grid to spread otherwise)"
                )
            if not getattr(self.model.encoder, "register_unit_norm", False):
                # Without the sphere the term has a degenerate solution: shrink the
                # magnitudes and every dot product goes to zero without any
                # directions moving.
                logger.warning(
                    "register uniformity is enabled without "
                    "EncoderConfig.register_unit_norm; the term can be minimized by "
                    "shrinking register magnitudes rather than spreading directions"
                )
            self.total_loss_name = f"{self.total_loss_name}+uniformity"
        if getattr(self.model.encoder, "register_projection_dims", None) is not None:
            self.total_loss_name = f"{self.total_loss_name}+projection"

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample],
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
            batch: A (patch_size, MaskedOlmoEarthSample) tuple from the dataloader.
            dry_run: If True, skip metric recording and just run forward/backward.
        """
        if not dry_run:
            self.update_target_encoder()
        # Set the model to train mode
        self.model.train()
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        accumulated_extra_metrics: dict[str, Any] = {}
        extra_metric_counts: dict[str, int] = {}
        patch_size = batch[0]
        batch_data = batch[1]

        # Split batch into microbatches
        masked_microbatches = split_masked_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(masked_microbatches)

        for microbatch_idx in range(num_microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                microbatch_masked = masked_microbatches[microbatch_idx]
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} "
                    f"with batch size {microbatch_masked.batch_size} on rank {get_local_rank()}"
                )
                masked_batch = microbatch_masked.to_device(self.device)

                # Run Encoder and decoder on the augmented input
                loss, latent, decoded, target_output, extra_metrics = (
                    self.model_forward(masked_batch, patch_size, self.token_exit_cfg)
                )
                if extra_metrics is not None:
                    self.accumulate_extra_metrics(
                        accumulated_extra_metrics, extra_metric_counts, extra_metrics
                    )
                reg_term = self.compute_regularization(latent)
                if reg_term is not None:
                    loss = loss + reg_term
                    total_batch_reg += (
                        get_local_tensor(reg_term.detach()) / num_microbatches
                    )
                # Scale loss by number of microbatches
                loss = loss / num_microbatches

                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                # Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    print(f"rank {get_local_rank()} has nan or inf")

                loss.backward()

        if dry_run:
            return

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.log_accumulated_extra_metrics(
            accumulated_extra_metrics, extra_metric_counts
        )
        self.log_regularization(total_batch_reg)

        del batch, batch_data  # In case this helps with memory utilization.
        del masked_batch
        del latent, decoded, target_output

    def model_forward(
        self,
        batch: MaskedOlmoEarthSample,
        patch_size: int,
        token_exit_cfg: dict[str, int],
    ) -> tuple[
        torch.Tensor,
        TokensAndMasks,
        TokensAndMasks,
        TokensAndMasks,
        dict[str, Any] | None,
    ]:
        """Run a forward pass."""
        with self._model_forward_context():
            (
                latent,
                decoded,
                _,
                reconstructed,
                extra_metrics,
                supervision_preds,
                projection_outputs,
            ) = self.model(batch, patch_size)

            with torch.no_grad():
                logger.info("Target Encoder forward pass...")
                output_dict = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
                target_output, _, _ = unpack_encoder_output(output_dict)
            # Compute losses outside autocast: the loss internals cast to fp32
            # explicitly, but under autocast ops like bmm get re-cast to bf16,
            # silently changing the loss values (e.g. the same_target_threshold
            # similarity masking). This keeps the loss identical across
            # FSDP-param-cast and autocast (ddp) precision modes.
            with torch.autocast(torch.device(self.device).type, enabled=False):
                loss = self.loss_fn(decoded, target_output)
                if self.mae_loss is not None:
                    loss += self.mae_loss.compute(reconstructed, batch)

                if (
                    supervision_preds is not None
                    and self.model.supervision_head is not None
                ):
                    sup_loss, per_modality_losses = compute_supervision_loss(
                        supervision_preds,
                        batch,
                        self.model.supervision_head,
                    )
                    loss = loss + sup_loss
                    for mod_name, mod_loss in per_modality_losses.items():
                        if extra_metrics is None:
                            extra_metrics = {}
                        extra_metrics[f"supervision/{mod_name}"] = mod_loss

                if (
                    projection_outputs is not None
                    and self.register_uniformity_weight > 0
                ):
                    # Shapes the encoder (not detached, unlike everything below):
                    # the registers here are the served embedding.
                    uniformity_loss, uniformity_metrics = (
                        compute_register_uniformity_loss(
                            registers=projection_outputs["registers"],
                            weight=self.register_uniformity_weight,
                            num_rotations=self.register_uniformity_rotations,
                        )
                    )
                    loss = loss + uniformity_loss
                    if extra_metrics is None:
                        extra_metrics = {}
                    extra_metrics.update(uniformity_metrics)

                if (
                    projection_outputs is not None
                    and projection_outputs["projected_registers"] is not None
                ):
                    if self.projection_uniformity_weight > 0:
                        # The student is what gets served when it is deployed, so it
                        # needs the spread in its own space, not just the teacher's.
                        student_uniformity, student_metrics = (
                            compute_register_uniformity_loss(
                                registers=projection_outputs["projected_registers"],
                                weight=self.projection_uniformity_weight,
                                num_rotations=self.register_uniformity_rotations,
                            )
                        )
                        loss = loss + student_uniformity
                        if extra_metrics is None:
                            extra_metrics = {}
                        extra_metrics.update(
                            {
                                f"projection_{k.split('/')[-1]}": v
                                for k, v in student_metrics.items()
                            }
                        )
                    # Detached-student losses: the teacher registers are detached and
                    # the student's inputs were detached inside the encoder, so none
                    # of this reaches the encoder or the primary bottleneck.
                    distill_loss, distill_metrics = compute_projection_distill_loss(
                        teacher=projection_outputs["registers"],
                        student=projection_outputs["projected_registers"],
                        back_projections=dict(
                            self.model.encoder.register_back_projections
                        ),
                        cosine_weight=self.projection_distill_cosine_weight,
                        gram_weight=self.projection_distill_gram_weight,
                        gram_max_tokens=self.projection_distill_gram_max_tokens,
                        gram_within_weight=self.projection_distill_gram_within_weight,
                        gram_within_max_cells=(
                            self.projection_distill_gram_within_max_cells
                        ),
                    )
                    loss = loss + distill_loss
                    if extra_metrics is None:
                        extra_metrics = {}
                    extra_metrics.update(distill_metrics)
                    projection_supervision_preds = projection_outputs[
                        "supervision_preds"
                    ]
                    if (
                        projection_supervision_preds is not None
                        and self.model.projection_supervision_heads is not None
                    ):
                        for dim_str, dim_preds in projection_supervision_preds.items():
                            proj_sup_loss, proj_per_modality = compute_supervision_loss(
                                dim_preds,
                                batch,
                                self.model.projection_supervision_heads[dim_str],
                            )
                            loss = loss + proj_sup_loss
                            for mod_name, mod_loss in proj_per_modality.items():
                                extra_metrics[
                                    f"supervision_projection_d{dim_str}/{mod_name}"
                                ] = mod_loss

            return loss, latent, decoded, target_output, extra_metrics
