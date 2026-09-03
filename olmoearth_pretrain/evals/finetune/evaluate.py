"""Evaluation functions for finetuning."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

from olmoearth_pretrain.evals.finetune.model import BackboneWithHead, to_device
from olmoearth_pretrain.evals.metrics import (
    EvalMetric,
    EvalResult,
    classification_metrics,
    regression_metrics,
    segmentation_metrics,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_cls(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    is_multilabel: bool,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
) -> EvalResult:
    """Evaluate classification metrics."""
    module.eval()
    logits_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)  # (B, C)
        logits_all.append(logits.float().cpu())
        labels_all.append(label.cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    if is_multilabel:
        scores = torch.sigmoid(logits)
        preds = scores.gt(0.5).int()
    else:
        scores = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
    return classification_metrics(
        preds,
        labels,
        scores=scores,
        is_multilabel=is_multilabel,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )


def _reg_logits_to_pixel(
    logits: torch.Tensor,
    label: torch.Tensor,
    pixel_space_output: bool,
) -> torch.Tensor:
    """Convert regression head output to predictions matching the label shape.

    Handles both the linear head, which emits one value per patch
    (B, H//P, W//P, 1), and the UNet head, which emits per-pixel values
    (B, 1, H, W). Patch-space predictions are bilinearly upsampled to the
    label resolution; per-pixel predictions are returned as-is (resized only
    if they still differ from the label).

    NOTE: Only dense (per-pixel) regression is supported. Scalar-target
    regression (one value per sample) is NOT wired up: the eval wrapper forces
    spatial pooling for all REGRESSION tasks (see OlmoEarthEvalWrapper), so the
    head always produces a spatial map rather than a pooled (B, 1) vector. The
    (B, 1) -> (B,) squeeze branch below is therefore currently unreachable; a
    scalar-target task would need spatial pooling disabled in the wrapper first.
    """
    if pixel_space_output:
        preds = logits.squeeze(1).float()  # (B, 1, H, W) -> (B, H, W)
    else:
        preds = logits.squeeze(
            -1
        ).float()  # (B, H//P, W//P, 1) -> (B, H//P, W//P) or (B,)
    if preds.dim() == 3 and preds.shape[-2:] != label.shape[-2:]:
        preds = F.interpolate(
            preds.unsqueeze(1),
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)
    return preds


@torch.no_grad()
def eval_reg(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    primary_metric: EvalMetric | None = None,
) -> EvalResult:
    """Evaluate regression metrics (per-pixel or scalar targets)."""
    module.eval()
    preds_all, labels_all = [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)
            preds = _reg_logits_to_pixel(logits, label, module.pixel_space_output)
        preds_all.append(preds.cpu())
        labels_all.append(label.float().cpu())
    preds = torch.cat(preds_all, 0)
    labels = torch.cat(labels_all, 0)
    return regression_metrics(preds, labels, primary_metric=primary_metric)


def _seg_logits_to_pixel(
    logits: torch.Tensor,
    label: torch.Tensor,
    pixel_space_output: bool,
    num_classes: int,
    patch_size: int,
) -> torch.Tensor:
    """Pixel-shuffle patch-space logits and resize to label resolution."""
    if not pixel_space_output:
        H, W = logits.shape[1], logits.shape[2]
        logits = rearrange(
            logits,
            "b h w (c i j) -> b c (h i) (w j)",
            h=H,
            w=W,
            c=num_classes,
            i=patch_size,
            j=patch_size,
        )
    if logits.shape[-2:] != label.shape[-2:]:
        logits = F.interpolate(
            logits.float(),
            size=label.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    return logits


@torch.no_grad()
def eval_seg(
    module: BackboneWithHead,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    patch_size: int,
    primary_metric: EvalMetric | None = None,
    primary_metric_class: int | None = None,
    dump_tag: str | None = None,
    distributed: bool = False,
) -> EvalResult:
    """Evaluate segmentation metrics.

    ``distributed`` says the loader handed in is a SHARD of the split (one slice
    per rank). The shards are gathered onto rank 0, scored there, and the result
    broadcast back, so every rank returns the same metrics over the whole split --
    identical to the unsharded computation, since every metric here is
    order-invariant.

    Gathering is required rather than reducing a confusion matrix: auroc and prauc
    are computed from the raw per-pixel scores and cannot be recovered from
    counts, so a confusion-matrix all-reduce would leave those two silently wrong.
    Rank 0 ends up holding exactly what a single process held before, so peak
    memory is unchanged.
    """
    module.eval()
    preds_all, labels_all, scores_all = [], [], []
    for masked, label in loader:
        label = label.to(device=device)
        masked = to_device(masked, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = module(masked, label, is_train=False)
            logits = _seg_logits_to_pixel(
                logits, label, module.pixel_space_output, num_classes, patch_size
            )
        preds_all.append(torch.argmax(logits, dim=1).cpu())
        labels_all.append(label.cpu())
        scores_all.append(torch.softmax(logits.float(), dim=1).cpu())
    if preds_all:
        preds = torch.cat(preds_all, 0)
        labels = torch.cat(labels_all, 0)
        scores = torch.cat(scores_all, 0)
    else:
        # A shard can legitimately be empty: the strided split gives rank r nothing
        # when len(split) < world_size. torch.cat([]) raises, so stand in zero-length
        # tensors -- they contribute nothing to the gather and keep this rank at the
        # collectives, which it must reach or the job deadlocks.
        preds = torch.empty(0, dtype=torch.long)
        labels = torch.empty(0, dtype=torch.long)
        scores = torch.empty(0, num_classes, dtype=torch.float32)

    world = torch.distributed.get_world_size() if (
        distributed and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ) else 1
    if world > 1:
        rank = torch.distributed.get_rank()
        # every rank must reach both collectives, so gather unconditionally
        bucket: list[Any] | None = [None] * world if rank == 0 else None
        torch.distributed.gather_object((preds, labels, scores), bucket, dst=0)
        if rank == 0:
            assert bucket is not None
            parts = [b for b in bucket if b[0].numel() > 0]
            preds = torch.cat([b[0] for b in parts], 0)
            labels = torch.cat([b[1] for b in parts], 0)
            scores = torch.cat([b[2] for b in parts], 0)
            logger.info(
                "gathered %d validation samples from %d shards", preds.shape[0], world
            )

    # Optional dump of the finetuned test predictions (+ labels) for offline
    # visualization, mirroring the linear-probe OE_PRED_DUMP. No effect unless
    # OE_PRED_DIR is set or the weka marker file exists, and dump_tag is set
    # (test split only). dump_tag here is just the task name - it carries neither a
    # model identifier nor an LR, so two checkpoints finetuned on the same task
    # write the same filename and overwrite each other. Set OE_PRED_DIR per run to
    # keep them apart, as the linear probe does.
    import os as _os

    _pdir = _os.environ.get("OE_PRED_DIR")
    if not _pdir:
        _pmark = "/weka/dfive-default/piperw/dev/rslearn_projects/pastis2/oe_pred_dir.txt"
        if _os.path.exists(_pmark):
            with open(_pmark) as _mf:
                _pdir = _mf.read().strip()
    if dump_tag is not None and _pdir:
        _os.makedirs(_pdir, exist_ok=True)
        _pp = _os.path.join(_pdir, f"{dump_tag}_preds.pt")
        torch.save({"preds": preds, "labels": labels, "dump_tag": dump_tag}, _pp)
        print(f"[FT_PRED_DUMP] wrote {_pp} preds={tuple(preds.shape)}", flush=True)

    if world > 1:
        # Only rank 0 holds the full split, so only rank 0 can score it. Every rank
        # needs the same numbers back: the value drives scheduler.step() and the
        # best-checkpoint test, and ranks disagreeing there would diverge the LR
        # schedule and disagree about which epoch was best.
        payload: list[Any] = [None]
        if torch.distributed.get_rank() == 0:
            payload = [
                segmentation_metrics(
                    preds,
                    labels,
                    num_classes=num_classes,
                    scores=scores,
                    ignore_label=-1,
                    primary_metric=primary_metric,
                    primary_metric_class=primary_metric_class,
                )
            ]
        torch.distributed.broadcast_object_list(payload, src=0)
        result = payload[0]
        assert result is not None, "rank 0 failed to broadcast the eval result"
        return result

    return segmentation_metrics(
        preds,
        labels,
        num_classes=num_classes,
        scores=scores,
        ignore_label=-1,
        primary_metric=primary_metric,
        primary_metric_class=primary_metric_class,
    )
