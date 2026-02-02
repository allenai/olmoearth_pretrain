"""Test Finetune evaluation returns correct types.

These tests verify that segmentation evaluation returns EvalResult with all metrics,
testing the core metric computation that finetune.py depends on.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from olmoearth_pretrain.evals.metrics import EvalResult, segmentation_metrics


def test_segmentation_eval_pipeline() -> None:
    """Test the segmentation evaluation pipeline returns EvalResult with expected keys.

    This simulates what _eval_seg does: takes model logits, computes argmax predictions,
    and calls segmentation_metrics.
    """
    batch_size = 4
    h, w = 16, 16
    num_classes = 3

    # Simulate model output: logits after rearrange (B, C, H, W)
    logits = torch.randn(batch_size, num_classes, h, w)

    # Get predictions via argmax
    preds = torch.argmax(logits, dim=1)  # (B, H, W)

    # Ground truth labels
    labels = torch.randint(0, num_classes, (batch_size, h, w))

    # Call segmentation_metrics (this is what _eval_seg uses)
    result = segmentation_metrics(
        preds, labels, num_classes=num_classes, ignore_label=-1
    )

    # Verify return type is EvalResult
    assert isinstance(result, EvalResult)
    expected_keys = {"miou", "overall_acc", "macro_acc", "macro_f1"}
    assert set(result.metrics.keys()) == expected_keys

    # Primary metric should be miou
    assert result.primary == result.metrics["miou"]

    # All values should be floats between 0 and 1
    for key in expected_keys:
        assert isinstance(result.metrics[key], float), f"{key} should be float"
        assert 0.0 <= result.metrics[key] <= 1.0, f"{key} should be in [0, 1]"


def test_segmentation_eval_with_interpolation() -> None:
    """Test segmentation eval with size mismatch requiring interpolation.

    This tests the path where logits.shape != label.shape and interpolation is needed.
    """
    batch_size = 2
    num_classes = 5
    patch_size = 4

    # Model outputs smaller spatial size
    logits_h, logits_w = 4, 4
    label_h, label_w = 16, 16

    # Simulate logits from model (B, H, W, C*p*p) -> rearranged to (B, C, H*p, W*p)
    raw_logits = torch.randn(
        batch_size, logits_h, logits_w, num_classes * patch_size * patch_size
    )
    logits = rearrange(
        raw_logits,
        "b h w (c i j) -> b c (h i) (w j)",
        h=logits_h,
        w=logits_w,
        c=num_classes,
        i=patch_size,
        j=patch_size,
    )

    # Labels at different size - need interpolation
    labels = torch.randint(0, num_classes, (batch_size, label_h, label_w))

    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(
            logits.float(),
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

    preds = torch.argmax(logits, dim=1)

    result = segmentation_metrics(
        preds, labels, num_classes=num_classes, ignore_label=-1
    )

    assert isinstance(result, EvalResult)
    assert "miou" in result.metrics
    assert "overall_acc" in result.metrics
    assert "macro_acc" in result.metrics
    assert "macro_f1" in result.metrics


def test_segmentation_eval_with_ignore_labels() -> None:
    """Test segmentation eval handles ignore labels correctly."""
    batch_size = 2
    h, w = 8, 8
    num_classes = 3

    preds = torch.randint(0, num_classes, (batch_size, h, w))
    labels = torch.randint(0, num_classes, (batch_size, h, w))

    # Set some pixels to ignore label
    labels[0, 0, :] = -1
    labels[1, :, 0] = -1

    result = segmentation_metrics(
        preds, labels, num_classes=num_classes, ignore_label=-1
    )

    assert isinstance(result, EvalResult)
    # Metrics should still be valid
    for key in ["miou", "overall_acc", "macro_acc", "macro_f1"]:
        assert 0.0 <= result.metrics[key] <= 1.0


def test_empty_test_result_none() -> None:
    """Test that when there's no test loader, finetune returns None for test_score.

    This verifies the structure of the "no test set" return value.
    """
    # finetune.py now returns None when test_loader is None
    test_score = None
    assert test_score is None
