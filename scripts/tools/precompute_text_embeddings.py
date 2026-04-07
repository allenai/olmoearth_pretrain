"""Pre-compute text embeddings for map modality class names.

Produces a .pt file with dict[modality_name, Tensor[num_classes, 768]].
Run once before training with text embedding targets.

Usage:
    python scripts/tools/precompute_text_embeddings.py \
        --output olmoearth_pretrain/data/text_embeddings.pt \
        --model BAAI/bge-base-en-v1.5
"""

import argparse
import logging

import torch
from sentence_transformers import SentenceTransformer

from olmoearth_pretrain.data.class_names import (
    OSM_RASTER_CLASSES,
    WORLDCEREAL_CLASSES,
    WORLDCOVER_CLASSES,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Pre-compute and save text embeddings for map modality class names."""
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings")
    parser.add_argument(
        "--output",
        type=str,
        default="olmoearth_pretrain/data/text_embeddings.pt",
        help="Output .pt file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace sentence-transformers model name",
    )
    args = parser.parse_args()

    logger.info(f"Loading model {args.model}")
    model = SentenceTransformer(args.model)

    embeddings: dict[str, dict] = {}

    # WorldCover: keyed by raw pixel value (int)
    wc_ids = sorted(WORLDCOVER_CLASSES.keys())
    wc_texts = [WORLDCOVER_CLASSES[k] for k in wc_ids]
    wc_embs = model.encode(wc_texts, convert_to_tensor=True, normalize_embeddings=True)
    embeddings["worldcover"] = {
        "class_ids": torch.tensor(wc_ids, dtype=torch.long),
        "embeddings": wc_embs.cpu(),
    }
    logger.info(f"WorldCover: {len(wc_ids)} classes, embedding dim={wc_embs.shape[-1]}")

    # OSM raster: keyed by channel index (order matches band_order)
    osm_embs = model.encode(
        OSM_RASTER_CLASSES, convert_to_tensor=True, normalize_embeddings=True
    )
    embeddings["openstreetmap_raster"] = {"embeddings": osm_embs.cpu()}
    logger.info(f"OSM raster: {len(OSM_RASTER_CLASSES)} classes")

    # WorldCereal: keyed by channel index
    wc_crop_embs = model.encode(
        WORLDCEREAL_CLASSES, convert_to_tensor=True, normalize_embeddings=True
    )
    embeddings["worldcereal"] = {"embeddings": wc_crop_embs.cpu()}
    logger.info(f"WorldCereal: {len(WORLDCEREAL_CLASSES)} classes")

    torch.save(embeddings, args.output)
    logger.info(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
