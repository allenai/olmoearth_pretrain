r"""Fit fixed embedding-normalization constants for one checkpoint.

The fitted normalizations in the embedding evals (center / center_l2 / zscore)
default to taking their statistics from each task's own train split. That
answers whether an embedding space needs normalizing, but it is not a shippable
transform: a global embedding run has no per-dataset train split to fit on, and
per-dataset constants would make two tiles' embeddings incomparable.

This script produces the deployable form instead -- ONE mean (and, for zscore,
one std) per checkpoint, fitted over a sample of eval windows and applied
unchanged everywhere. Point ``--embedding_norm_stats_path`` (embedding_eval_sweep)
or the ``embedding_norm_stats_path`` task field at the output.

The constants describe one model's embedding space, so refit them per
checkpoint, per head (``--eval_on_projected_registers`` for the distilled
student), and per embedding width.

Usage:
    python scripts/tools/fit_embedding_norm_stats.py \\
        --checkpoint /weka/.../some_run/step370000 \\
        --datasets m-eurosat,pastis_rslearn \\
        --mode zscore \\
        --eval_on_projected_registers \\
        --out /weka/.../some_run/emb_norm_zscore_proj.pt
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch.utils.data import DataLoader, IterableDataset

from olmoearth_pretrain.evals.datasets import get_eval_dataset
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn_variable_time
from olmoearth_pretrain.evals.embedding_diagnostics import (
    compute_geometry_diagnostics,
    flatten_rows,
    sample_row_indices,
)
from olmoearth_pretrain.evals.embedding_transforms import (
    FITTED_NORMALIZATIONS,
    EmbeddingNormalization,
    EmbeddingNormalizer,
)
from olmoearth_pretrain.evals.embeddings import get_embeddings
from olmoearth_pretrain.evals.eval_wrapper import get_eval_wrapper
from olmoearth_pretrain.model_loader import load_pretrain_checkpoint
from olmoearth_pretrain.nn.pooling import PoolingType

logger = logging.getLogger(__name__)


def collect_embeddings(
    model: torch.nn.Module,
    dataset_name: str,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    """Extract raw (un-normalized, un-quantized) embeddings for one dataset."""
    cfg = dataset_to_config(dataset_name)
    eval_ds = get_eval_dataset(
        eval_dataset=dataset_name,
        split=split,
        norm_stats_from_pretrained=True,
        input_modalities=cfg.supported_modalities,
    )
    is_iterable = isinstance(eval_ds, IterableDataset)
    loader = DataLoader(
        eval_ds,
        collate_fn=eval_collate_fn_variable_time,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False if is_iterable else True,
    )

    encoder = model.encoder if hasattr(model, "encoder") else model
    if hasattr(encoder, "disable_band_dropout"):
        encoder.disable_band_dropout()

    wrapper = get_eval_wrapper(
        encoder,
        task_type=cfg.task_type,
        patch_size=args.patch_size,
        pooling_type=PoolingType.MEAN,
        concat_features=False,
        use_pooled_tokens=False,
        eval_on_projected_registers=args.eval_on_projected_registers,
        eval_projection_dim=args.projection_dim,
    )
    embeddings, _ = get_embeddings(
        data_loader=loader, model=wrapper, is_train=False, quantize=False
    )
    rows = flatten_rows(embeddings)
    if rows.shape[0] > args.max_rows_per_dataset:
        # Cap per dataset so one large dense set cannot dominate constants that
        # every dataset will be normalized by.
        idx = sample_row_indices(rows.shape[0], max_rows=args.max_rows_per_dataset)
        assert idx is not None
        rows = rows[idx]
    logger.info(f"{dataset_name}: collected {rows.shape[0]} embedding rows")
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Pretrain checkpoint dir (contains config.json)",
    )
    parser.add_argument(
        "--datasets",
        default="m-eurosat",
        help=(
            "Comma-separated eval datasets to pool the statistics over. More "
            "datasets = constants less tied to any one region/sensor mix."
        ),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--mode",
        default=EmbeddingNormalization.ZSCORE.value,
        choices=[m.value for m in FITTED_NORMALIZATIONS],
        help="Which fitted normalization's constants to write",
    )
    parser.add_argument("--out", required=True, help="Output path for the constants")
    parser.add_argument(
        "--eval_on_projected_registers",
        action="store_true",
        help="Fit on the distilled student's output instead of the register grid",
    )
    parser.add_argument(
        "--projection_dim",
        type=int,
        default=None,
        help="Matryoshka prefix width to fit on (with --eval_on_projected_registers)",
    )
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_rows_per_dataset", type=int, default=65536)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrain_checkpoint(args.checkpoint, device)

    rows = torch.cat(
        [
            collect_embeddings(model, name, args.split, args, device)
            for name in args.datasets.split(",")
        ],
        dim=0,
    )
    mode = EmbeddingNormalization(args.mode)
    normalizer = EmbeddingNormalizer.fit(mode, rows)
    normalizer.save(args.out)

    before = compute_geometry_diagnostics(rows)
    after = compute_geometry_diagnostics(flatten_rows(normalizer(rows)))
    for key in sorted(before):
        logger.info(f"{key}: {before[key]:.4g} -> {after.get(key, float('nan')):.4g}")


if __name__ == "__main__":
    main()
