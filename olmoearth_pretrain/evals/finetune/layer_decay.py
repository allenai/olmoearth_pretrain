"""Layer-wise learning rate decay optimizer for fine-tuning."""

from __future__ import annotations

from collections import defaultdict
from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)

BACKBONE_PREFIX = "backbone"


def get_layer_id(name: str, num_layers: int) -> int:
    """Map a parameter name to its layer index for LR scaling.

    Returns 0..num_layers-1 for encoder blocks, 0 for patch_embeddings,
    and num_layers for everything else (head, wrapper = full LR).
    """
    if not name.startswith(BACKBONE_PREFIX):
        return num_layers
    relative = name[len(BACKBONE_PREFIX) + 1 :]
    if relative.startswith("blocks."):
        return int(relative.split(".")[1])
    if relative.startswith("patch_embeddings"):
        return 0
    return num_layers


def build_layer_decay_optimizer(
    model: nn.Module,
    lr: float,
    layer_decay_rate: float,
    num_layers: int,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Build AdamW with per-layer learning rate decay.

    LR for layer i = lr * layer_decay_rate ** (num_layers - i).
    Layer 0 is the shallowest (patch embeddings / first block),
    layer num_layers is the head (full LR).
    """
    groups: dict[int, list] = defaultdict(list)
    for name, param in model.named_parameters():
        layer_id = get_layer_id(name, num_layers)
        groups[layer_id].append(param)

    param_groups = []
    for layer_id in sorted(groups.keys()):
        scale = layer_decay_rate ** (num_layers - layer_id)
        group_lr = lr * scale
        param_groups.append({"params": groups[layer_id], "lr": group_lr})
        logger.info(
            f"layer_decay group layer={layer_id} lr={group_lr:.2e} "
            f"params={len(groups[layer_id])}"
        )

    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
