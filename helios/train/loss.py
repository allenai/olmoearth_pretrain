"""Loss functions for training."""

import torch
import torch.nn.functional as F


def patch_disc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float = 0.2,
    pred2unit: bool = True,
) -> torch.Tensor:
    """Patch discriminator loss.

    Args:
        pred: Predicted patches.
        target: Target patches.
        tau: Temperature parameter for the loss.
        pred2unit: Whether to normalize the predicted patches to unit length.

    Returns:
        Loss tensor.
    """
    # Input shape: (B, N, C)
    # Target shape: (B, N, C)
    # B is batch size
    # N is number of patches
    # C is embedding dimension
    #
    B, N, C = pred.shape

    if pred2unit:
        pred_mu = pred.mean(1, keepdims=True)
        pred_std = pred.std(1, keepdims=True)
        pred = (pred - pred_mu) / (pred_std + 1e-4)

    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    # n is batch dimension p is patch index from pred q is patch index from target, d is embedding dimension
    scores = torch.einsum("npd,nqd->npq", pred, target) / tau

    labels = torch.arange(N, dtype=torch.long, device=pred.device)[None].repeat(B, 1)
    # Target is the index of the patch in the target so we are aiming to make the scores from the same patch similar and different patches dissimilar
    loss = F.cross_entropy(
        scores.flatten(0, 1),
        labels.flatten(0, 1),
    ) * (tau * 2)

    return loss
