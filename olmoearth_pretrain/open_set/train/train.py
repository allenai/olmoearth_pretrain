r"""Minimal olmo-core entrypoint for open-set training.

This mirrors the structure of ``scripts/official/script.py`` — build configs,
build components, hand them to an olmo-core ``Trainer`` — but is stripped
down to just the open-set specific pieces. Real launch scripts can pull in
the same wandb / Beaker callbacks that the pretraining scripts use.

Usage (single-rank smoke test on local data):

    python -m olmoearth_pretrain.open_set.train.train \
        --checkpoint /path/to/step370000 \
        --dataset /path/to/dataset.csv \
        --work-dir /tmp/open_set_work
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from olmo_core.optim import AdamWConfig

from olmoearth_pretrain.config import require_olmo_core
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.open_set.catalog import build_default_registry
from olmoearth_pretrain.open_set.data.modality_subsample import (
    ModalitySubsampleConfig,
)
from olmoearth_pretrain.open_set.data.sampler import RandomNegativeSampler
from olmoearth_pretrain.open_set.model.cross_attn_decoder import (
    CrossAttnDecoderConfig,
)
from olmoearth_pretrain.open_set.model.encoder_wrapper import (
    FrozenOlmoEarthEncoder,
    load_encoder_from_distributed_checkpoint,
)
from olmoearth_pretrain.open_set.model.open_set_model import (
    OpenSetSegmenter,
    OpenSetSegmenterConfig,
)
from olmoearth_pretrain.open_set.text.embedding_cache import (
    TextEmbeddingCache,
    default_cache_path,
)
from olmoearth_pretrain.open_set.text.siglip_encoder import (
    DEFAULT_MODEL_NAME as DEFAULT_SIGLIP_MODEL,
)
from olmoearth_pretrain.open_set.text.siglip_encoder import SigLIPTextEncoder
from olmoearth_pretrain.open_set.train.train_module import (
    OpenSetTrainModuleConfig,
)

require_olmo_core("open_set training entrypoint")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a distributed olmo-core checkpoint directory (step{N}/).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Working directory for trainer state and the text-embedding cache.",
    )
    parser.add_argument(
        "--text-encoder",
        type=str,
        default=DEFAULT_SIGLIP_MODEL,
        help="HF model name for the SigLIP text encoder.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help="Patch size used for tokenization. Must be in the encoder's range.",
    )
    parser.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--k-pos",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--k-neg",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--subsample-modalities",
        action="store_true",
        help="Drop a random subset of input modalities each batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    return parser.parse_args()


def build_open_set_segmenter(
    checkpoint_path: Path,
    text_dim: int,
    decoder_config: CrossAttnDecoderConfig | None = None,
    device: torch.device | str = "cpu",
) -> OpenSetSegmenter:
    """Load the frozen encoder and wire it into an :class:`OpenSetSegmenter`."""
    encoder = load_encoder_from_distributed_checkpoint(checkpoint_path, device=device)
    frozen = FrozenOlmoEarthEncoder(encoder, trainable=False)
    seg_config = OpenSetSegmenterConfig(
        decoder_config=decoder_config or CrossAttnDecoderConfig(),
        text_dim=text_dim,
    )
    return seg_config.build(frozen)


def main() -> None:
    """Build everything and start training."""
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Catalog of classes we will train against.
    registry = build_default_registry()
    logger.info("Built registry with %d classes", len(registry))

    # 2. Text encoder + cache.
    text_encoder = SigLIPTextEncoder(model_name=args.text_encoder, device="cpu")
    cache_path = default_cache_path(args.work_dir, text_encoder)
    text_cache = TextEmbeddingCache(text_encoder, cache_path=cache_path)
    text_cache.populate(registry)

    # 3. Sampler.
    sampler = RandomNegativeSampler(
        registry=registry, k_pos=args.k_pos, k_neg=args.k_neg, seed=args.seed
    )

    # 4. Frozen encoder + segmenter.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmenter = build_open_set_segmenter(
        checkpoint_path=args.checkpoint,
        text_dim=text_encoder.embedding_dim,
        device=device,
    )

    # 5. Train module config.
    modality_subsample_config = (
        ModalitySubsampleConfig() if args.subsample_modalities else None
    )
    train_module_config = OpenSetTrainModuleConfig(
        optim_config=AdamWConfig(lr=args.lr, weight_decay=0.01, fused=False),
        rank_microbatch_size=args.rank_microbatch_size,
        transform_config=TransformConfig(transform_type="flip_and_rotate"),
        modality_subsample_config=modality_subsample_config,
        seed=args.seed,
    )

    # All the pieces below are intentionally referenced once so that the
    # entrypoint linter doesn't flag them as unused while the trainer/
    # dataloader wiring is left as a TODO.
    _ = (sampler, segmenter, train_module_config)

    # The dataloader build is left to the caller — open-set training reuses
    # the standard ``OlmoEarthDataLoader`` configured with whichever dataset
    # the user wants to train on. We deliberately do NOT wire up beaker /
    # wandb here so this script stays runnable in a smoke test; production
    # launch scripts should add the standard callbacks.
    #
    # The intended flow once a dataloader is wired:
    #     train_module = train_module_config.build(
    #         model=segmenter,
    #         registry=registry,
    #         text_cache=text_cache,
    #         sampler=sampler,
    #         device=device,
    #     )
    #     data_loader = build_dataloader(...)
    #     trainer_config = TrainerConfig(
    #         save_folder=str(args.work_dir / "trainer"),
    #         max_duration=Duration.steps(args.max_steps),
    #         checkpointer=CheckpointerConfig(),
    #     )
    #     trainer = trainer_config.build(train_module, data_loader)
    #     trainer.fit()
    raise NotImplementedError(
        "Wire up your OlmoEarthDataLoader + TrainerConfig and call "
        "trainer.fit(train_module). See scripts/official/script.py for the "
        "canonical pretraining setup."
    )


if __name__ == "__main__":
    main()
