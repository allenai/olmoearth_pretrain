"""olmo-core training module for open-set text-conditioned segmentation.

Mirrors the structure of :class:`LatentMIMTrainModule` but with a much
simpler training step: the dataloader hands us a ``MaskedOlmoEarthSample``,
we sample positive/negative classes per image, run one encoder + decoder
forward, and accumulate per-(image, class) BCE.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from logging import getLogger

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import ReduceType

from olmoearth_pretrain.config import require_olmo_core
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.open_set.catalog import build_default_registry
from olmoearth_pretrain.open_set.catalog.registry import ClassEntry, ClassRegistry
from olmoearth_pretrain.open_set.data.modality_subsample import (
    ModalitySubsampleConfig,
    subsample_modalities_masked,
)
from olmoearth_pretrain.open_set.data.sampler import (
    ClassSampler,
    PerImageClassSelection,
    RandomNegativeSamplerConfig,
)
from olmoearth_pretrain.open_set.model.open_set_model import OpenSetSegmenter
from olmoearth_pretrain.open_set.text.embedding_cache import (
    TextEmbeddingCache,
    TextEncoderConfig,
)
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)

require_olmo_core("open_set training")

logger = getLogger(__name__)


@dataclass
class OpenSetTrainModuleConfig(OlmoEarthTrainModuleConfig):
    """Configuration for :class:`OpenSetTrainModule`.

    Inherits the optimizer / transform / FSDP / autocast knobs from
    :class:`OlmoEarthTrainModuleConfig` and adds the open-set specific
    pieces. Calling ``build(model)`` constructs the registry, text-embedding
    cache, and sampler — matching the contract of the standard
    ``train(config)`` flow in :mod:`olmoearth_pretrain.internal.experiment`.

    Attributes:
        text_encoder_config: SigLIP encoder + on-disk cache configuration.
        sampler_config: Random-negative sampler configuration. Replace with
            a hard-negative variant when one lands.
        modality_subsample_config: Optional config for stochastic input-modality
            dropping. If None, all modalities present on the batch are kept.
        seed: Seed used for class sampling and modality subsampling.
        target_size_source: Which label-source modality's height/width drives
            the output resolution. Defaults to "openstreetmap_raster" — the
            MVP source. If None, falls back to the first sampled positive's
            source on each batch.
    """

    text_encoder_config: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    sampler_config: RandomNegativeSamplerConfig = field(
        default_factory=RandomNegativeSamplerConfig
    )
    modality_subsample_config: ModalitySubsampleConfig | None = None
    seed: int | None = None
    target_size_source: str | None = "openstreetmap_raster"

    def build(  # type: ignore[override]
        self,
        model: OpenSetSegmenter,
        device: torch.device | None = None,
    ) -> OpenSetTrainModule:
        """Build the train module, including registry/text_cache/sampler.

        The registry is sourced from ``build_default_registry`` (currently
        OSM only). The text cache is populated lazily from ``self.text_encoder_config``
        — this is the slow step (loading SigLIP). The sampler is cheap.
        """
        registry = build_default_registry()
        text_cache = self.text_encoder_config.build(registry)
        sampler = self.sampler_config.build(registry)

        kwargs = self.prepare_kwargs()
        # ``build`` consumes these — they should not be forwarded to the
        # train module's __init__.
        kwargs.pop("text_encoder_config", None)
        kwargs.pop("sampler_config", None)

        return OpenSetTrainModule(
            model=model,
            registry=registry,
            text_cache=text_cache,
            sampler=sampler,
            device=device,
            **kwargs,
        )


def _expand_selection(
    selections: list[PerImageClassSelection],
) -> tuple[list[ClassEntry], list[list[tuple[int, ClassEntry, bool]]]]:
    """Build the deduped class union and per-image (idx, entry, is_positive) lists."""
    union: list[ClassEntry] = []
    union_index: dict[tuple[str, str], int] = {}
    per_image: list[list[tuple[int, ClassEntry, bool]]] = []
    for sel in selections:
        records: list[tuple[int, ClassEntry, bool]] = []
        for entry, is_positive in (
            *((e, True) for e in sel.positives),
            *((e, False) for e in sel.negatives),
        ):
            key = (entry.source, entry.text)
            if key not in union_index:
                union_index[key] = len(union)
                union.append(entry)
            records.append((union_index[key], entry, is_positive))
        per_image.append(records)
    return union, per_image


class OpenSetTrainModule(OlmoEarthTrainModule):
    """Olmo-core ``TrainModule`` implementation for open-set segmentation."""

    def __init__(
        self,
        model: OpenSetSegmenter,
        registry: ClassRegistry,
        text_cache: TextEmbeddingCache,
        sampler: ClassSampler,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        rank_microbatch_size: int,
        modality_subsample_config: ModalitySubsampleConfig | None = None,
        seed: int | None = None,
        target_size_source: str | None = "openstreetmap_raster",
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
        """Initialize the open-set train module."""
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
        self.registry = registry
        self.text_cache = text_cache
        self.sampler = sampler
        self.modality_subsample_config = modality_subsample_config
        self.target_size_source = target_size_source
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def loss_fn(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-pixel binary cross-entropy on logits."""
        return F.binary_cross_entropy_with_logits(pred, target)

    def _resolve_target_size(
        self,
        sample: MaskedOlmoEarthSample,
        union: list[ClassEntry],
    ) -> tuple[int, int]:
        """Return ``(H, W)`` of the label-source raster used as the output grid."""
        candidate_sources = []
        if self.target_size_source is not None:
            candidate_sources.append(self.target_size_source)
        # Fall back to whichever source the union actually uses.
        candidate_sources.extend(e.source for e in union)
        for source in candidate_sources:
            tensor = getattr(sample, source, None)
            if tensor is not None:
                # tensor: [B, H, W, 1, num_bands]
                return int(tensor.shape[1]), int(tensor.shape[2])
        raise RuntimeError(
            "Could not infer target output size — no usable label-source modality "
            f"present on batch (tried: {candidate_sources})."
        )

    def train_batch(
        self,
        batch: tuple[int, MaskedOlmoEarthSample],
        dry_run: bool = False,
    ) -> None:
        """Train one batch.

        Microbatching is intentionally omitted for the MVP — the open-set
        forward already replicates the encoder output across class queries,
        so an explicit microbatch loop is mostly redundant. We can revisit
        if memory pressure shows up in practice.
        """
        self.model.train()
        patch_size, masked = batch
        masked = masked.to_device(self.device)

        # Encoder is frozen; ensure it sees every spatial position.
        masked = masked.unmask()

        # Optional: drop a subset of input modalities so the model learns to
        # operate without every sensor present.
        if self.modality_subsample_config is not None:
            masked = subsample_modalities_masked(
                masked, self.modality_subsample_config, self._rng
            )

        # Sample classes per image.
        selections = self.sampler.sample(masked, rng=self._rng)
        union, per_image = _expand_selection(selections)
        if not union:
            logger.warning(
                "Batch produced no class assignments — skipping. "
                "Check that the registry has entries for at least one source on the batch."
            )
            return

        # Pull text embeddings (cache is CPU-resident; move to device here).
        text_encoding = self.text_cache.get_many(
            union, device=self.device, dtype=self._text_dtype()
        )

        target_size = self._resolve_target_size(masked, union)

        with self._model_forward_context():
            logits = self.model(
                sample=masked,
                patch_size=patch_size,
                text_tokens=text_encoding.tokens,
                text_pooled=text_encoding.pooled,
                text_attn_mask=text_encoding.attention_mask,
                target_size=target_size,
            )  # [C, B, H, W]

            total_loss, n_assignments = self._accumulate_loss(masked, logits, per_image)

        if n_assignments == 0:
            return

        loss = total_loss / n_assignments
        loss.backward()

        if dry_run:
            return

        self.trainer.record_metric("train/bce", loss.detach(), ReduceType.mean)
        self.trainer.record_metric(
            "train/num_class_queries",
            torch.tensor(float(len(union)), device=self.device),
            ReduceType.mean,
        )
        self.trainer.record_metric(
            "train/num_assignments",
            torch.tensor(float(n_assignments), device=self.device),
            ReduceType.mean,
        )

    def _text_dtype(self) -> torch.dtype | None:
        """Cast text embeddings to the autocast dtype if AMP is enabled."""
        return self.autocast_precision

    def _accumulate_loss(
        self,
        masked: MaskedOlmoEarthSample,
        logits: torch.Tensor,
        per_image: list[list[tuple[int, ClassEntry, bool]]],
    ) -> tuple[torch.Tensor, int]:
        """Sum BCE losses across all (image, class) assignments.

        ``logits`` has shape ``[C, B, H, W]`` where ``C`` is the size of the
        deduped class union. Per-image records list ``(class_index, entry,
        is_positive)`` triples — ``is_positive`` is unused for the loss
        itself (the binary mask carries that info) but kept available for
        diagnostics.
        """
        # Cache extracted GT masks per source so we don't re-run extractors.
        # Logits are at ``target_size`` (== the label-source raster's H, W),
        # so the GT slice should match natively. We assert rather than
        # silently interpolate — a mismatch indicates a config bug.
        gt_cache: dict[tuple[str, str], torch.Tensor] = {}
        total = torch.zeros([], device=logits.device, dtype=logits.dtype)
        n = 0
        h, w = logits.shape[-2], logits.shape[-1]
        for image_index, records in enumerate(per_image):
            for class_index, entry, _is_positive in records:
                key = (entry.source, entry.text)
                if key not in gt_cache:
                    source_tensor = getattr(masked, entry.source)
                    gt_cache[key] = entry.extractor(source_tensor)  # [B, H, W]
                gt = gt_cache[key][image_index].to(dtype=logits.dtype)
                if gt.shape != (h, w):
                    raise RuntimeError(
                        f"GT mask shape {tuple(gt.shape)} for "
                        f"({entry.source}, {entry.text!r}) does not match "
                        f"logits shape {(h, w)}. Check that target_size_source "
                        "is set to the label-source modality whose raster "
                        "drives target_size."
                    )
                pred = logits[class_index, image_index]
                total = total + self.loss_fn(pred, gt)
                n += 1
        return total, n
