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
from olmoearth_pretrain.nn.flexi_vit import (
    DEFAULT_REGISTER_STUDENT_NAME,
    TokensAndMasks,
)
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


@dataclass
class TeacherGramState:
    """Teacher-side Gram matrices and the cell subsamples they were built from.

    Built ONCE per microbatch and shared by every student, for two reasons. The
    teacher is identical across students, so recomputing its O(n^2) matrices per
    student is pure waste. More importantly the subsample is random: if each student
    drew its own, two arms of a multi-student run would differ by their Gram
    sampling noise as well as by the thing under test, which is exactly the
    cross-arm variance a shared-teacher run exists to remove.

    The permutations are drawn at the LARGEST size any student asked for, and a
    student wanting ``k`` takes the first ``k`` entries -- so students sharing a size
    see identical pairs, and students with different sizes see NESTED samples
    (the smaller is a subset of the larger). Since ``gram = X @ X.T``, restricting
    to the first ``k`` rows of ``X`` is the leading ``k x k`` block of ``gram``.
    """

    idx: torch.Tensor | None = None
    gram: torch.Tensor | None = None
    within_idx: torch.Tensor | None = None
    within_gram: torch.Tensor | None = None

    def flat_sample(
        self, max_tokens: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The flat subsample and teacher Gram, narrowed to ``max_tokens``."""
        if self.gram is None:
            return None, None
        k = min(max_tokens, self.gram.shape[0])
        idx = None if self.idx is None else self.idx[:k]
        return idx, self.gram[:k, :k]

    def within_sample(
        self, max_cells: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The per-scene cell subsample and teacher Gram, narrowed to ``max_cells``."""
        if self.within_gram is None:
            return None, None
        k = min(max_cells, self.within_gram.shape[-1])
        within_idx = None if self.within_idx is None else self.within_idx[:k]
        return within_idx, self.within_gram[:, :k, :k]


def build_teacher_gram_state(
    teacher: torch.Tensor,
    *,
    gram_max_tokens: int,
    gram_within_max_cells: int,
    build_flat: bool,
    build_within: bool,
) -> TeacherGramState:
    """Draw this microbatch's Gram subsamples and build the teacher's matrices.

    Args:
        teacher: Register grid ``[B, N, D]``, already detached and float.
        gram_max_tokens: Largest flat sample any student will take.
        gram_within_max_cells: Largest per-scene sample any student will take.
        build_flat: Whether any student uses the flat (cross-scene) Gram term.
        build_within: Whether any student uses the within-scene Gram term.

    Returns:
        The shared state; unused halves are left as None.
    """
    state = TeacherGramState()
    if build_flat:
        flat_teacher = F.normalize(teacher.reshape(-1, teacher.shape[-1]), dim=-1)
        num_tokens = flat_teacher.shape[0]
        if num_tokens > gram_max_tokens:
            state.idx = torch.randperm(num_tokens, device=flat_teacher.device)[
                :gram_max_tokens
            ]
            flat_teacher = flat_teacher[state.idx]
        state.gram = flat_teacher @ flat_teacher.T
    # Within-scene (block-diagonal) Gram: keep the [B, N, D] layout so each matrix
    # only ever relates two cells of the same sample. The register grid is shared
    # across the microbatch (the dynamic bottleneck sizes one (h, w) per batch), so
    # a single cell subsample applies to every scene and this stays a plain bmm.
    if build_within and teacher.shape[1] >= 2:
        num_cells = teacher.shape[1]
        if num_cells > gram_within_max_cells:
            state.within_idx = torch.randperm(num_cells, device=teacher.device)[
                :gram_within_max_cells
            ]
        within_teacher = (
            teacher if state.within_idx is None else teacher[:, state.within_idx]
        )
        within_teacher = F.normalize(within_teacher, dim=-1)
        state.within_gram = within_teacher @ within_teacher.transpose(1, 2)
    return state


def compute_projection_distill_loss(
    teacher: torch.Tensor,
    student: torch.Tensor,
    back_projections: dict[str, torch.nn.Module],
    cosine_weight: float,
    gram_weight: float,
    gram_max_tokens: int,
    gram_within_weight: float = 0.0,
    gram_within_max_cells: int = 256,
    teacher_gram_state: TeacherGramState | None = None,
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
            This matrix is built over the FLATTENED ``[B * N]`` grid, so with a
            microbatch of B scenes only ~1/B of its pairs are within-scene.
        gram_max_tokens: Max register cells entering the Gram terms (one random
            subsample shared across prefixes; bounds the O(n^2) matrices).
        gram_within_weight: Weight of the WITHIN-SCENE (block-diagonal) Gram term:
            the same relational MSE, but computed per scene so every pair is
            between two cells of the SAME sample. Cheaper per pair than the flat
            term by a factor of B -- m blocks of k cells cost O(m * k^2) against
            O((m * k)^2) for one flat matrix over the same cells -- so at equal
            FLOPs it buys ~B times as many within-scene pairs. Set > 0 to enable;
            the default 0.0 leaves the loss exactly as the flat-only runs saw it.
        gram_within_max_cells: Register cells sampled PER SCENE for the
            within-scene term (one random subsample of cell positions, shared
            across scenes and prefixes so the blocks stay spatially aligned).
        teacher_gram_state: Pre-built teacher matrices and subsamples. Pass the SAME
            state to every student of a multi-student run so the arms are compared
            on identical pairs; None builds a fresh (independently sampled) one,
            which is correct only for a single student.

    Returns:
        total: The weighted sum of the enabled terms across prefixes.
        metrics: Detached per-term, per-prefix values for logging.
    """
    teacher = teacher.detach().float()
    student = student.float()
    metrics: dict[str, torch.Tensor] = {}
    total = torch.zeros([], device=student.device, dtype=student.dtype)
    if teacher_gram_state is None:
        teacher_gram_state = build_teacher_gram_state(
            teacher,
            gram_max_tokens=gram_max_tokens,
            gram_within_max_cells=gram_within_max_cells,
            build_flat=gram_weight > 0,
            build_within=gram_within_weight > 0,
        )
    idx, teacher_gram = teacher_gram_state.flat_sample(gram_max_tokens)
    within_idx, teacher_within_gram = teacher_gram_state.within_sample(
        gram_within_max_cells
    )
    if gram_weight <= 0:
        teacher_gram = None
    if gram_within_weight <= 0:
        teacher_within_gram = None
    for dim_str, back_projection in back_projections.items():
        prefix = student[..., : int(dim_str)]
        if cosine_weight > 0:
            back = back_projection(prefix)
            cosine = (1.0 - F.cosine_similarity(back, teacher, dim=-1)).mean()
            total = total + cosine_weight * cosine
            metrics[f"projection/distill_cosine_d{dim_str}"] = cosine.detach()
        if teacher_gram is not None:
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


#: The distillation weights a per-student override may set, mapped to the train
#: module attribute each one falls back to.
_STUDENT_DISTILL_WEIGHT_DEFAULTS = {
    "cosine_weight": "projection_distill_cosine_weight",
    "gram_weight": "projection_distill_gram_weight",
    "gram_max_tokens": "projection_distill_gram_max_tokens",
    "gram_within_weight": "projection_distill_gram_within_weight",
    "gram_within_max_cells": "projection_distill_gram_within_max_cells",
}


def validate_distill_overrides(
    overrides: dict[str, dict[str, Any]], student_names: set[str]
) -> None:
    """Reject per-student distillation overrides that would silently do nothing.

    A misspelt student name or weight key would otherwise leave the arm running on
    the global defaults, and the run would read as "this configuration made no
    difference" rather than "this configuration was never applied".
    """
    unknown = set(overrides) - student_names
    if unknown:
        raise ValueError(
            f"projection_distill_overrides names unknown students {sorted(unknown)}; "
            f"the encoder has {sorted(student_names)}"
        )
    allowed = set(_STUDENT_DISTILL_WEIGHT_DEFAULTS)
    for name, override in overrides.items():
        unknown_keys = set(override) - allowed
        if unknown_keys:
            raise ValueError(
                f"projection_distill_overrides[{name!r}] has unknown keys "
                f"{sorted(unknown_keys)}; valid keys are {sorted(allowed)}"
            )


def _namespace_student_metrics(
    metrics: dict[str, torch.Tensor], student_name: str
) -> dict[str, torch.Tensor]:
    """Scope a student's distillation metrics under its name.

    ``projection/distill_cosine_d128`` becomes
    ``projection/<student>/distill_cosine_d128``, so several students logging the
    same terms never collide. The lone student of a single-student run keeps the
    unscoped names, so its curves stay comparable with every run recorded before
    students became a list.
    """
    if student_name == DEFAULT_REGISTER_STUDENT_NAME:
        return metrics
    return {
        key.replace("projection/", f"projection/{student_name}/", 1): value
        for key, value in metrics.items()
    }


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
    # ~1/B the flat term gives. Costs a factor B fewer FLOPs per pair. 0.0 keeps
    # the flat-only behaviour the in-flight proj128 runs were trained with.
    projection_distill_gram_within_weight: float = 0.0
    projection_distill_gram_within_max_cells: int = 256
    # Per-student overrides of the four weights above, keyed by student name (see
    # EncoderConfig.register_students). Any key omitted for a student falls back to
    # the global value, so {"gram_heavy": {"gram_within_weight": 1.0}} varies exactly
    # one knob. Unknown student names are rejected at build time rather than silently
    # ignored -- a typo here would otherwise read as "this arm did nothing".
    projection_distill_overrides: dict[str, dict[str, Any]] | None = None

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
        projection_distill_overrides: dict[str, dict[str, Any]] | None = None,
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
            scheduler_overrides: Optional per-param-group schedulers, keyed
                by the group's "group_name" tag.
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
                (block-diagonal) Gram term, whose pairs are all drawn from a single
                sample rather than the ~1/B the flat term gives. 0.0 disables it.
            projection_distill_gram_within_max_cells: Register cells sampled per
                scene for the within-scene Gram term.
            projection_distill_overrides: Per-student overrides of the four
                distillation weights, keyed by student name; omitted keys fall back
                to the global values. Unknown student names raise.
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
        self.projection_distill_overrides: dict[str, dict[str, Any]] = (
            projection_distill_overrides or {}
        )
        validate_distill_overrides(
            self.projection_distill_overrides,
            set(getattr(self.model.encoder, "register_students", None) or {}),
        )
        if getattr(self.model.encoder, "register_projection_dims", None) is not None:
            self.total_loss_name = f"{self.total_loss_name}+projection"

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def _student_distill_weights(self, student_name: str) -> dict[str, Any]:
        """Distillation weights for one student: the globals, then its overrides."""
        weights = {
            key: getattr(self, attr)
            for key, attr in _STUDENT_DISTILL_WEIGHT_DEFAULTS.items()
        }
        weights.update(self.projection_distill_overrides.get(student_name, {}))
        return weights

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

                if projection_outputs is not None:
                    # Detached-student losses: the teacher registers are detached and
                    # the students' inputs were detached inside the encoder, so none
                    # of this reaches the encoder or the primary bottleneck. Each
                    # student is scored independently -- with its own weights when
                    # projection_distill_overrides names it -- so a multi-student run
                    # is N parallel distillation experiments over one teacher.
                    students = self.model.encoder.register_students
                    student_outputs = projection_outputs["projected_registers"]
                    per_student = {
                        name: self._student_distill_weights(name)
                        for name in student_outputs
                    }
                    # ONE subsample per microbatch, shared by every arm. Drawing it
                    # per student would make two arms differ by their Gram sampling
                    # noise on top of the thing under test, and would rebuild the
                    # identical teacher matrices once per arm.
                    teacher_registers = projection_outputs["registers"].detach().float()
                    teacher_gram_state = build_teacher_gram_state(
                        teacher_registers,
                        gram_max_tokens=int(
                            max(w["gram_max_tokens"] for w in per_student.values())
                        ),
                        gram_within_max_cells=int(
                            max(
                                w["gram_within_max_cells"] for w in per_student.values()
                            )
                        ),
                        build_flat=any(
                            w["gram_weight"] > 0 for w in per_student.values()
                        ),
                        build_within=any(
                            w["gram_within_weight"] > 0 for w in per_student.values()
                        ),
                    )
                    for name, student_out in student_outputs.items():
                        distill_loss, distill_metrics = compute_projection_distill_loss(
                            teacher=teacher_registers,
                            student=student_out,
                            back_projections=dict(students[name].back_projections),
                            teacher_gram_state=teacher_gram_state,
                            **per_student[name],
                        )
                        loss = loss + distill_loss
                        if extra_metrics is None:
                            extra_metrics = {}
                        extra_metrics.update(
                            _namespace_student_metrics(distill_metrics, name)
                        )
                    projection_supervision_preds = projection_outputs[
                        "supervision_preds"
                    ]
                    if (
                        projection_supervision_preds is not None
                        and self.model.projection_supervision_heads is not None
                    ):
                        # Heads are keyed by student, then by Matryoshka width. Each
                        # head already carries its student's supervision weight (see
                        # LatentMIMConfig.projection_supervision_weight_scales), so
                        # the weights differ per arm while the loss shape does not.
                        heads = self.model.projection_supervision_heads
                        for name, per_dim in projection_supervision_preds.items():
                            scope = (
                                ""
                                if name == DEFAULT_REGISTER_STUDENT_NAME
                                else f"{name}/"
                            )
                            for dim_str, dim_preds in per_dim.items():
                                (
                                    proj_sup_loss,
                                    proj_per_modality,
                                ) = compute_supervision_loss(
                                    dim_preds, batch, heads[name][dim_str]
                                )
                                loss = loss + proj_sup_loss
                                for mod_name, mod_loss in proj_per_modality.items():
                                    extra_metrics[
                                        f"supervision_projection_{scope}"
                                        f"d{dim_str}/{mod_name}"
                                    ] = mod_loss

            return loss, latent, decoded, target_output, extra_metrics
