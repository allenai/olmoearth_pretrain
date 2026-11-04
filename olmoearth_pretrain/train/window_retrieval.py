"""Window-retrieval loss: retrieve a sub-range's student embedding from the full one.

For each microbatch, one window length ``L`` is drawn, and each sample gets its own
start inside its valid timesteps. The target ``z_t`` comes from a separate,
no-grad, unmasked encoder pass on the input sliced to the window (every modality,
mask and timestamp), so no target token attended to steps outside the window,
and it is exactly what fresh inference on the window returns. The input ``z_T``
is the student applied to the main pass's location tokens WITHOUT the detach the
distillation path uses, so the retrieval loss trains the retriever, the student,
and whatever upstream is trainable (the pooler; the encoder unless frozen).

With ``centered_weight`` > 0 a second MSE term compares prediction and target
after subtracting each sample's mean over its location tokens, so the gradient
is not dominated by the per-window offset and the tile mean (which a constant
shift already recovers) but by what differs between locations in a tile.

When a microbatch has too few valid steps for any window, a zero-valued term over
the retriever's parameters is returned instead, so every rank produces retriever
gradients and the manual gradient all-reduce stays aligned.
"""

from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.nn.window_retriever import WindowFields


def _is_spacetime(key: str) -> bool:
    """Whether a sample field has a time axis at dim 3 ([B, H, W, T, C])."""
    return Modality.get(key.removesuffix("_mask")).is_spacetime_varying


def valid_steps(batch: MaskedOlmoEarthSample) -> Tensor:
    """``[B]`` number of leading timesteps up to the last one with any data."""
    present = None
    for key, val in batch.as_dict(include_nones=False).items():
        if key.endswith("_mask") and _is_spacetime(key):
            p = (
                (val != MaskValue.MISSING.value).any(dim=4).any(dim=2).any(dim=1)
            )  # [B, T]
            present = p if present is None else present | p
    if present is None:
        raise ValueError("window retrieval needs a spacetime-varying modality")
    T = present.shape[1]
    steps = torch.arange(1, T + 1, device=present.device)
    return (present * steps).amax(dim=1)


def slice_window(
    batch: MaskedOlmoEarthSample, starts: Tensor, length: int
) -> MaskedOlmoEarthSample:
    """Steps ``[starts[b], starts[b] + length)`` of every spacetime field, per sample."""
    idx = starts[:, None] + torch.arange(length, device=starts.device)[None]  # [B, L]
    updates = {}
    for key, val in batch.as_dict(include_nones=False).items():
        if key == "timestamps":
            updates[key] = val.gather(1, idx[:, :, None].expand(-1, -1, val.shape[2]))
        elif _is_spacetime(key):
            B, H, W, _, C = val.shape
            updates[key] = val.gather(
                3, idx[:, None, None, :, None].expand(B, H, W, length, C)
            )
    return batch._replace(**updates)


def window_retrieval_loss(
    model: Any,
    batch: MaskedOlmoEarthSample,
    patch_size: int,
    location_query_patch_size: int | None,
    location_tokens: Tensor,
    min_length: int,
    max_length: int,
    centered_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """MSE between the retrieved and the fresh window student embeddings.

    Args:
        model: the LatentMIM, with ``window_retriever`` and an AttentionPooler student.
        batch: the main pass's (masked) microbatch.
        patch_size: the main pass's patch size.
        location_query_patch_size: the main pass's location-query patch size.
        location_tokens: the main pass's ``[B, N, D]`` location tokens, with grad.
        min_length: shortest window drawn, in steps.
        max_length: longest window drawn, in steps (always shorter than the input).
        centered_weight: weight on the MSE after removing each sample's mean over
            location tokens (within-tile structure).

    Returns:
        the loss and metrics.
    """
    retriever = model.window_retriever
    n = valid_steps(batch)
    hi = min(max_length, int(n.min().item()) - 1)
    if hi < min_length:
        zero = sum(p.sum() for p in retriever.parameters()) * 0.0
        return zero, {"window_retrieval/skipped": torch.ones((), device=n.device)}

    length = int(torch.randint(min_length, hi + 1, ()).item())
    starts = (torch.rand(n.shape, device=n.device) * (n - length + 1)).floor().long()
    window = slice_window(batch, starts, length)
    with torch.no_grad():
        out = model.encoder(
            window.unmask(),
            patch_size=patch_size,
            location_query_patch_size=location_query_patch_size,
        )
        _, _, decoder_kwargs = unpack_encoder_output(out)
        z_t = rearrange(
            decoder_kwargs["student_location_tokens"], "b h w d -> b (h w) d"
        )

    z_T = model.encoder.attention_pooler.student(location_tokens)
    months = batch.timestamps[..., 1].long()
    rows = torch.arange(len(starts), device=starts.device)
    fields = WindowFields(
        input_first_month=months[:, 0],
        input_length=n,
        window_first_month=months[rows, starts],
        window_last_month=months[rows, starts + length - 1],
        window_length=torch.full_like(n, length),
    )
    pred = retriever(z_T, fields)
    z_t, pred, z_T = z_t.float(), pred.float(), z_T.float()
    mse = F.mse_loss(pred, z_t)
    target_c = z_t - z_t.mean(dim=1, keepdim=True)
    centered = F.mse_loss(pred - pred.mean(dim=1, keepdim=True), target_c)
    loss = mse + centered_weight * centered if centered_weight > 0 else mse
    with torch.no_grad():
        floor_c = F.mse_loss(z_T - z_T.mean(dim=1, keepdim=True), target_c)
        metrics = {
            "window_retrieval/mse": mse.detach(),
            "window_retrieval/centered_mse": centered.detach(),
            "window_retrieval/within_tile_closed": 1
            - centered / floor_c.clamp_min(1e-12),
            "window_retrieval/cos": F.cosine_similarity(pred, z_t, dim=-1).mean(),
            "window_retrieval/cos_floor": F.cosine_similarity(z_T, z_t, dim=-1).mean(),
            "window_retrieval/length": torch.tensor(float(length), device=loss.device),
            "window_retrieval/skipped": torch.zeros((), device=loss.device),
        }
    return loss, metrics
