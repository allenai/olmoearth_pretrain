"""Training and optimizer abstraction for OlmoEarth Pretrain."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.nn.latent_mim import LatentMIM
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.open_set.catalog import ClassRegistry, build_default_registry
from olmoearth_pretrain.open_set.catalog.registry import ClassEntry
from olmoearth_pretrain.open_set.data.sampler import (
    PerImageClassSelection,
    RandomNegativeSampler,
)
from olmoearth_pretrain.open_set.text.embedding_cache import TextEmbeddingCache
from olmoearth_pretrain.open_set.text.siglip_encoder import SigLIPTextEncoder
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


def _build_per_image_selections(
    selections: list[PerImageClassSelection],
    class_union: list[ClassEntry],
) -> list[list[tuple[int, ClassEntry, bool]]]:
    """Convert sampler output to the format NLPSupervisionDecoder expects.

    Returns a list (per image) of ``(class_index_in_union, entry, is_positive)``
    tuples.
    """
    union_index: dict[tuple[str, str], int] = {
        (e.source, e.text): i for i, e in enumerate(class_union)
    }
    result: list[list[tuple[int, ClassEntry, bool]]] = []
    for sel in selections:
        image_entries: list[tuple[int, ClassEntry, bool]] = []
        for entry in sel.positives:
            key = (entry.source, entry.text)
            if key in union_index:
                image_entries.append((union_index[key], entry, True))
        for entry in sel.negatives:
            key = (entry.source, entry.text)
            if key in union_index:
                image_entries.append((union_index[key], entry, False))
        result.append(image_entries)
    return result


def _deduplicate_class_union(
    selections: list[PerImageClassSelection],
) -> list[ClassEntry]:
    """Build a deduplicated union of all class entries across all images."""
    seen: set[tuple[str, str]] = set()
    union: list[ClassEntry] = []
    for sel in selections:
        for entry in sel.all_entries:
            key = (entry.source, entry.text)
            if key not in seen:
                seen.add(key)
                union.append(entry)
    return union


@dataclass
class NLPSupervisionTrainConfig:
    """Runtime configuration for NLP supervision in the training loop."""

    text_encoder_name: str = "google/siglip2-so400m-patch14-384"
    text_cache_dir: str = ""
    sampler_k_pos: int = 2
    sampler_k_neg: int = 2
    sampler_seed: int | None = None
    target_size_source: str = "openstreetmap_raster"
    catalog_sources: list[str] | None = None


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
    nlp_supervision_train_config: NLPSupervisionTrainConfig | None = None

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> ContrastiveLatentMIMTrainModuleConfig:
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
        nlp_supervision_train_config: NLPSupervisionTrainConfig | None = None,
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
            nlp_supervision_train_config: Runtime config for NLP supervision.
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

        # NLP supervision setup.
        self._has_nlp_supervision = self.model.nlp_supervision_decoder is not None
        self._nlp_sampler: RandomNegativeSampler | None = None
        self._nlp_text_cache: TextEmbeddingCache | None = None
        self._nlp_rng: random.Random | None = None
        self._nlp_target_size_source: str = "openstreetmap_raster"

        if self._has_nlp_supervision and nlp_supervision_train_config is not None:
            cfg = nlp_supervision_train_config
            self._nlp_target_size_source = cfg.target_size_source

            # Build catalog (optionally restricted to specific sources).
            registry = build_default_registry()
            if cfg.catalog_sources is not None:
                filtered = [e for e in registry if e.source in cfg.catalog_sources]
                registry = ClassRegistry(filtered)
            logger.info(
                f"NLP supervision: {len(registry)} classes across {registry.sources()}"
            )

            # Build text embedding cache.
            text_encoder = SigLIPTextEncoder(model_name=cfg.text_encoder_name)
            self._nlp_text_cache = TextEmbeddingCache(
                encoder=text_encoder,
                cache_path=cfg.text_cache_dir or None,
            )
            self._nlp_text_cache.populate(registry)

            # Build sampler.
            self._nlp_sampler = RandomNegativeSampler(
                registry=registry,
                k_pos=cfg.sampler_k_pos,
                k_neg=cfg.sampler_k_neg,
                seed=cfg.sampler_seed,
            )
            self._nlp_rng = random.Random(cfg.sampler_seed)
            self.total_loss_name = f"{self.total_loss_name}+nlp_supervision"

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

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
        # Set the model to train mode
        self.model.train()
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        total_batch_con = torch.zeros([], device=self.device)

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
                    contrastive_loss = self.contrastive_loss.compute(pooled_a, pooled_b)
                    loss += contrastive_loss
                    total_batch_con += (
                        get_local_tensor(contrastive_loss.detach()) / num_microbatches
                    )

                loss = loss / num_microbatches
                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                # Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    del latent_a, latent_b
                    break

                del latent_a, latent_b
                loss.backward()

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
        self.log_regularization(total_batch_reg)

        del batch  # In case this helps with memory utilization.
        del masked_batch_a, masked_batch_b

    def _determine_target_size(
        self, batch: MaskedOlmoEarthSample
    ) -> tuple[int, int] | None:
        """Determine the output spatial size from a GT modality on the batch."""
        source = self._nlp_target_size_source
        tensor = getattr(batch, source, None)
        if tensor is None:
            return None
        # tensor: [B, H, W, 1, C]
        return (tensor.shape[1], tensor.shape[2])

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
            if extra_metrics is None:
                extra_metrics = {}
            self.log_extra_metrics(extra_metrics)
            with torch.no_grad():
                logger.debug("Target Encoder forward pass...")
                output_dict = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
                target_output, _, _ = unpack_encoder_output(output_dict)
            loss = self.loss_fn(decoded, target_output)
            if self.mae_loss is not None and reconstructed is not None:
                loss += self.mae_loss.compute(reconstructed, batch)

            # NLP supervision.  Always called (even when this rank's local
            # class_union is empty) so that the decoder's FSDP collectives stay
            # in sync with other ranks that may have sampled classes.
            if (
                self._has_nlp_supervision
                and self._nlp_sampler is not None
                and self._nlp_text_cache is not None
                and self.model.nlp_supervision_decoder is not None
            ):
                selections = self._nlp_sampler.sample(batch, self._nlp_rng)
                class_union = _deduplicate_class_union(selections)
                per_image = _build_per_image_selections(selections, class_union)
                target_size = self._determine_target_size(batch)

                if class_union:
                    text_encoding = self._nlp_text_cache.get_many(class_union)
                    text_tokens = text_encoding.tokens.to(self.device)
                    text_attn_mask = text_encoding.attention_mask.to(self.device)
                else:
                    # Placeholder text tensors so the decoder can still run
                    # dummy forwards to match other ranks' FSDP collectives.
                    text_dim = self.model.nlp_supervision_decoder.text_dim
                    text_tokens = torch.zeros((0, 1, text_dim), device=self.device)
                    text_attn_mask = torch.zeros(
                        (0, 1), dtype=torch.long, device=self.device
                    )

                nlp_loss, nlp_metrics = self.model.nlp_supervision_decoder(
                    tokens_and_masks=latent,
                    batch=batch,
                    patch_size=patch_size,
                    text_tokens=text_tokens,
                    text_attn_mask=text_attn_mask,
                    class_entries=class_union,
                    per_image_selections=per_image,
                    target_size=target_size,
                )
                loss = loss + nlp_loss
                for k, v in nlp_metrics.items():
                    if isinstance(v, int | float):
                        extra_metrics[k] = v

            return loss, latent, decoded, target_output, latent_projected_and_pooled
