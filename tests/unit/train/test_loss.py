"""Test losses."""

import logging

import torch
import torch.nn.functional as F

from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.train.loss import (
    AdjustedPatchDiscriminationLoss,
    CrossEntropyLoss,
    InfoNCELoss,
    L1Loss,
    L2Loss,
    ModalityPatchDiscriminationLossNew,
    PatchDiscriminationLoss,
    PatchDiscriminationLossNew,
)
from olmoearth_pretrain.train.masking import MaskValue

logger = logging.getLogger(__name__)


def test_patch_disc_loss() -> None:
    """Just test that it runs as expected."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchDiscriminationLoss()
    loss_value = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 0.5


def test_adjusted_patch_disc_loss_comparison() -> None:
    """Compare loss under different mu/sigma configs."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )

    # Loss hard is very sharp focus on the hard negatives, expect higher loss
    loss_easy = AdjustedPatchDiscriminationLoss(mu=0.3, sigma=1.0).compute(
        preds, targets
    )
    loss_hard = AdjustedPatchDiscriminationLoss(mu=0.9, sigma=0.1).compute(
        preds, targets
    )

    assert loss_hard >= loss_easy or abs(loss_hard - loss_easy) < 1e-3


def test_if_old_and_new_loss_are_the_same() -> None:
    """Test that the old and new patch discrimination loss are the same."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 2
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    loss_old = PatchDiscriminationLoss()
    loss_new = PatchDiscriminationLossNew()
    old_loss = loss_old.compute(preds, targets)
    new_loss = loss_new.compute(preds, targets)
    logger.info(f"old_loss: {old_loss}, new_loss: {new_loss}")
    assert torch.isclose(old_loss, new_loss)


def test_if_old_and_new_loss_are_the_same_uneven_number_of_decoder_tokens() -> None:
    """Test that the old and new patch discrimination loss are the same."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 2

    s2_preds_mask = torch.randint(0, 3, (b, t_h, t_w, t))

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_preds_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss_old = PatchDiscriminationLoss()
    loss_new = PatchDiscriminationLossNew()
    old_loss = loss_old.compute(preds, targets)
    new_loss = loss_new.compute(preds, targets)
    logger.info(f"old_loss: {old_loss}, new_loss: {new_loss}")
    assert torch.isclose(old_loss, new_loss)


def test_patch_disc_loss_averaged_over_batch_size() -> None:
    """Test it doesn't scale with batch size."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchDiscriminationLoss()
    loss_value = loss.compute(preds, targets)

    # now, use a larger batch size
    b, t_h, t_w, t, d = 8, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss_value_8 = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert torch.isclose(loss_value, loss_value_8)


def test_l1_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2_l2a=torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = L1Loss()
    loss_value = loss.compute(preds, targets)
    # MAE should be 1 since preds are 1, targets are 0
    assert loss_value == 1


def test_l2_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2_l2a=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = L2Loss()
    loss_value = loss.compute(preds, targets)
    # MSE should be 4 since preds are 2, targets are 0
    assert loss_value == 4


def test_cross_entropy_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2_l2a=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_l2a_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.zeros((b, t, t_h, t_w, 1), dtype=torch.long),
        sentinel2_l2a_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, 1), dtype=torch.long),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = CrossEntropyLoss()
    loss_value = loss.compute(preds, targets)
    # loss for BCE, prediction of .5 for both classes
    assert torch.isclose(loss_value, -torch.log(torch.tensor(0.5)), 0.0001)


def test_infonce_loss() -> None:
    """Just test that it runs as expected."""
    b, d = 16, 128

    loss = InfoNCELoss()
    loss_value = loss.compute(torch.ones((b, d)), torch.zeros((b, d)))
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 0.5
    # check the weight
    loss = InfoNCELoss(weight=0.1)
    w_loss_value = loss.compute(torch.ones((b, d)), torch.zeros((b, d)))
    assert 0.1 * loss_value == w_loss_value


def _modality_patch_disc_reference(
    predictions: TokensAndMasks,
    targets: TokensAndMasks,
    tau: float = 0.1,
    pred2unit: bool = False,
) -> torch.Tensor:
    """Reference sequential implementation for testing parallelized version."""
    modality_preds, modality_masks = predictions.flatten_tokens_and_masks(
        return_lists=True
    )
    modality_targets = targets.flatten_tokens_and_masks(return_lists=True)[0]

    total_loss = 0
    for all_preds, all_masks, all_targets in zip(
        modality_preds, modality_masks, modality_targets
    ):
        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)

        if pred2unit:
            pred_mu = pred.mean(1, keepdims=True)
            pred_std = pred.std(1, keepdims=True)
            pred = (pred - pred_mu) / (pred_std + 1e-4)

        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)

        count = (all_masks == MaskValue.DECODER.value).sum(dim=-1)
        losses = []
        start = 0
        for c in count:
            end = start + c
            if c == 0:
                continue
            pred_sample = pred[:, start:end, :]
            target_sample = target[:, start:end, :]
            score_sample = (
                torch.einsum("npd,nqd->npq", pred_sample, target_sample) / tau
            )
            labels = torch.arange(c, dtype=torch.long, device=pred.device)[None]
            loss = F.cross_entropy(
                score_sample.flatten(0, 1),
                labels.flatten(0, 1),
                reduction="none",
            ) * (tau * 2)
            loss = loss.mean()
            losses.append(loss)
            start = end
        if len(losses) == 0:
            continue
        loss = torch.stack(losses).mean()
        total_loss += loss

    return total_loss


def test_modality_patch_disc_parallelized_matches_sequential() -> None:
    """Test that parallelized modality patch disc loss matches sequential."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(42)
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )

    loss_parallel = ModalityPatchDiscriminationLossNew()
    parallel_loss = loss_parallel.compute(preds, targets)
    reference_loss = _modality_patch_disc_reference(preds, targets)

    logger.info(f"parallel_loss: {parallel_loss}, reference_loss: {reference_loss}")
    assert torch.isclose(parallel_loss, reference_loss, rtol=1e-4, atol=1e-6)


def test_modality_patch_disc_parallelized_uneven_tokens() -> None:
    """Test parallelized loss with uneven number of decoder tokens per sample."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(123)
    # Random masks - some samples have more decoder tokens than others
    s2_mask = torch.randint(0, 3, (b, t_h, t_w, t))
    latlon_mask = torch.randint(0, 3, (b, 1))

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=latlon_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=latlon_mask,
    )

    loss_parallel = ModalityPatchDiscriminationLossNew()
    parallel_loss = loss_parallel.compute(preds, targets)
    reference_loss = _modality_patch_disc_reference(preds, targets)

    logger.info(f"parallel_loss: {parallel_loss}, reference_loss: {reference_loss}")
    assert torch.isclose(parallel_loss, reference_loss, rtol=1e-4, atol=1e-6)


def test_modality_patch_disc_parallelized_with_missing_samples() -> None:
    """Test parallelized loss when some samples have no decoder tokens."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(456)
    # Create masks where first sample has no decoder tokens
    s2_mask = torch.randint(0, 3, (b, t_h, t_w, t))
    s2_mask[0] = MaskValue.ONLINE_ENCODER.value  # No decoder tokens for first sample
    s2_mask[2] = MaskValue.MISSING.value  # All missing for third sample

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )

    loss_parallel = ModalityPatchDiscriminationLossNew()
    parallel_loss = loss_parallel.compute(preds, targets)
    reference_loss = _modality_patch_disc_reference(preds, targets)

    logger.info(f"parallel_loss: {parallel_loss}, reference_loss: {reference_loss}")
    assert torch.isclose(parallel_loss, reference_loss, rtol=1e-4, atol=1e-6)


def test_modality_patch_disc_parallelized_pred2unit() -> None:
    """Test parallelized loss with pred2unit=True."""
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(789)
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=torch.ones((b, t_h, t_w, t)) * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * MaskValue.DECODER.value,
    )

    loss_parallel = ModalityPatchDiscriminationLossNew(pred2unit=True)
    parallel_loss = loss_parallel.compute(preds, targets)
    reference_loss = _modality_patch_disc_reference(preds, targets, pred2unit=True)

    logger.info(f"parallel_loss: {parallel_loss}, reference_loss: {reference_loss}")
    assert torch.isclose(parallel_loss, reference_loss, rtol=1e-4, atol=1e-6)
