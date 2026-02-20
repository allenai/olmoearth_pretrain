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
    ModalityPatchDiscriminationLossVec,
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
            n = pred.shape[1]
            pred_mu = pred.mean(1, keepdims=True)
            # clamp denominator to match the Vec impl (avoids NaN when n=1)
            pred_var = ((pred - pred_mu) ** 2).sum(1, keepdim=True) / max(n - 1, 1)
            pred_std = pred_var.sqrt()
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

    loss_parallel = ModalityPatchDiscriminationLossVec()
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

    loss_parallel = ModalityPatchDiscriminationLossVec()
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

    loss_parallel = ModalityPatchDiscriminationLossVec()
    parallel_loss = loss_parallel.compute(preds, targets)
    reference_loss = _modality_patch_disc_reference(preds, targets)

    logger.info(f"parallel_loss: {parallel_loss}, reference_loss: {reference_loss}")
    assert torch.isclose(parallel_loss, reference_loss, rtol=1e-4, atol=1e-6)


def test_modality_patch_disc_parallelized_pred2unit() -> None:
    """Test parallelized loss with pred2unit=True and uniform masks.

    All tokens are decoder tokens so there is no masking to exercise, but this
    validates that the global-stats normalisation path runs correctly.
    """
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

    loss_parallel = ModalityPatchDiscriminationLossVec(pred2unit=True)
    parallel_loss = loss_parallel.compute(preds, targets)
    reference_loss = _modality_patch_disc_reference(preds, targets, pred2unit=True)

    logger.info(f"parallel_loss: {parallel_loss}, reference_loss: {reference_loss}")
    assert torch.isclose(parallel_loss, reference_loss, rtol=1e-4, atol=1e-6)


def test_modality_patch_disc_parallelized_pred2unit_uneven_tokens() -> None:
    """Test pred2unit=True combined with uneven decoder token counts.

    This exercises the masked-mean/variance path in the Vec implementation:
    the global statistics must be computed only over valid (decoder) tokens
    across the padded batch tensor, not over all positions.  Without masking,
    the padding values would corrupt the normalisation stats and the result
    would diverge from the sequential reference.
    """
    b, t_h, t_w, t, d = 5, 4, 4, 2, 8

    torch.manual_seed(101)
    s2_mask = torch.randint(0, 3, (b, t_h, t_w, t))
    # Force first sample to have no s2 decoder tokens so the zero-count branch is hit.
    s2_mask[0] = MaskValue.ONLINE_ENCODER.value
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

    loss_parallel = ModalityPatchDiscriminationLossVec(pred2unit=True)
    parallel_loss = loss_parallel.compute(preds, targets)
    reference_loss = _modality_patch_disc_reference(preds, targets, pred2unit=True)

    logger.info(f"parallel_loss: {parallel_loss}, reference_loss: {reference_loss}")
    assert torch.isclose(parallel_loss, reference_loss, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Extended vec loss tests: gradients, dtypes, edge cases, scale
# ---------------------------------------------------------------------------


def _make_tokens_and_masks(
    modalities: dict[str, tuple[tuple, tuple]],
    seed: int,
    requires_grad: bool = False,
) -> tuple[TokensAndMasks, TokensAndMasks]:
    """Helper to build pred/target TokensAndMasks from a modality spec dict.

    Each key maps to (data_shape, mask_shape).  Masks are random ints in [0, 3].
    """
    torch.manual_seed(seed)
    pred_kwargs: dict = {}
    target_kwargs: dict = {}
    for name, (data_shape, mask_shape) in modalities.items():
        pred_kwargs[name] = torch.randn(data_shape, requires_grad=requires_grad)
        target_kwargs[name] = torch.randn(data_shape)
        mask = torch.randint(0, 4, mask_shape)
        pred_kwargs[f"{name}_mask"] = mask
        target_kwargs[f"{name}_mask"] = mask
    return TokensAndMasks(**pred_kwargs), TokensAndMasks(**target_kwargs)


def test_vec_gradient_matches_sequential() -> None:
    """Verify gradients (not just loss) are equivalent between vec and sequential."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 16

    for seed in [0, 7, 42, 999]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))

        # --- vec path ---
        s2_pred_v = torch.randn((b, t_h, t_w, t, d), requires_grad=True)
        ll_pred_v = torch.randn((b, 1, d), requires_grad=True)
        preds_v = TokensAndMasks(
            sentinel2_l2a=s2_pred_v, sentinel2_l2a_mask=s2_mask,
            latlon=ll_pred_v, latlon_mask=ll_mask,
        )
        targets_v = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)), latlon_mask=ll_mask,
        )
        loss_v = ModalityPatchDiscriminationLossVec().compute(preds_v, targets_v)
        loss_v.backward()

        # --- sequential path with identical inputs ---
        torch.manual_seed(seed)
        s2_mask2 = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask2 = torch.randint(0, 4, (b, 1))
        s2_pred_s = s2_pred_v.detach().clone().requires_grad_(True)
        ll_pred_s = ll_pred_v.detach().clone().requires_grad_(True)
        preds_s = TokensAndMasks(
            sentinel2_l2a=s2_pred_s, sentinel2_l2a_mask=s2_mask2,
            latlon=ll_pred_s, latlon_mask=ll_mask2,
        )
        targets_s = TokensAndMasks(
            sentinel2_l2a=targets_v.sentinel2_l2a.clone(),
            sentinel2_l2a_mask=s2_mask2,
            latlon=targets_v.latlon.clone(), latlon_mask=ll_mask2,
        )
        loss_s = _modality_patch_disc_reference(preds_s, targets_s)
        loss_s.backward()

        assert torch.isclose(loss_v, loss_s, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: loss mismatch {loss_v.item()} vs {loss_s.item()}"
        )
        assert torch.allclose(s2_pred_v.grad, s2_pred_s.grad, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: s2 grad mismatch, "
            f"max diff={(s2_pred_v.grad - s2_pred_s.grad).abs().max().item()}"
        )
        # When no latlon decoder tokens exist, sequential gives None grad (not in graph)
        # while vec gives zero grad. Both mean "no gradient" — normalize for comparison.
        grad_v = ll_pred_v.grad if ll_pred_v.grad is not None else torch.zeros_like(ll_pred_v)
        grad_s = ll_pred_s.grad if ll_pred_s.grad is not None else torch.zeros_like(ll_pred_s)
        assert torch.allclose(grad_v, grad_s, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: latlon grad mismatch, "
            f"max diff={(grad_v - grad_s).abs().max().item()}"
        )


def test_vec_gradient_matches_sequential_bfloat16() -> None:
    """Same gradient test but in bfloat16 to match training autocast."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 16

    for seed in [0, 42]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))

        s2_data = torch.randn((b, t_h, t_w, t, d))
        ll_data = torch.randn((b, 1, d))
        s2_tgt = torch.randn((b, t_h, t_w, t, d))
        ll_tgt = torch.randn((b, 1, d))

        # vec
        s2_v = s2_data.bfloat16().requires_grad_(True)
        ll_v = ll_data.bfloat16().requires_grad_(True)
        preds_v = TokensAndMasks(
            sentinel2_l2a=s2_v, sentinel2_l2a_mask=s2_mask,
            latlon=ll_v, latlon_mask=ll_mask,
        )
        targets_v = TokensAndMasks(
            sentinel2_l2a=s2_tgt.bfloat16(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.bfloat16(), latlon_mask=ll_mask,
        )
        loss_v = ModalityPatchDiscriminationLossVec().compute(preds_v, targets_v)
        loss_v.backward()

        # sequential
        s2_s = s2_data.bfloat16().requires_grad_(True)
        ll_s = ll_data.bfloat16().requires_grad_(True)
        preds_s = TokensAndMasks(
            sentinel2_l2a=s2_s, sentinel2_l2a_mask=s2_mask,
            latlon=ll_s, latlon_mask=ll_mask,
        )
        targets_s = TokensAndMasks(
            sentinel2_l2a=s2_tgt.bfloat16(), sentinel2_l2a_mask=s2_mask,
            latlon=ll_tgt.bfloat16(), latlon_mask=ll_mask,
        )
        loss_s = _modality_patch_disc_reference(preds_s, targets_s)
        loss_s.backward()

        assert torch.isclose(loss_v.float(), loss_s.float(), rtol=5e-3, atol=1e-4), (
            f"seed={seed} bf16 loss mismatch {loss_v.item()} vs {loss_s.item()}"
        )
        assert torch.allclose(
            s2_v.grad.float(), s2_s.grad.float(), rtol=5e-3, atol=1e-4
        ), (
            f"seed={seed} bf16 s2 grad max diff="
            f"{(s2_v.grad - s2_s.grad).float().abs().max().item()}"
        )


def test_vec_multiple_seeds_forward() -> None:
    """Sweep 20 random seeds to catch any seed-dependent mismatch."""
    b, t_h, t_w, t, d = 6, 4, 4, 2, 8
    for seed in range(20):
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))
        preds = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        targets = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        par = ModalityPatchDiscriminationLossVec().compute(preds, targets)
        ref = _modality_patch_disc_reference(preds, targets)
        assert torch.isclose(par, ref, rtol=1e-4, atol=1e-6), (
            f"seed={seed}: {par.item()} vs {ref.item()}"
        )


def test_vec_single_decoder_token_per_sample() -> None:
    """Edge case: every sample has exactly 1 decoder token."""
    b, d = 8, 16
    torch.manual_seed(77)
    # 1 spatial position, 1 timestep → 1 token per sample
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, 1, 1, 1, d)),
        sentinel2_l2a_mask=torch.ones((b, 1, 1, 1), dtype=torch.long)
        * MaskValue.DECODER.value,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.ones((b, 1), dtype=torch.long)
        * MaskValue.ONLINE_ENCODER.value,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, 1, 1, 1, d)),
        sentinel2_l2a_mask=preds.sentinel2_l2a_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=preds.latlon_mask,
    )
    par = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    ref = _modality_patch_disc_reference(preds, targets)
    assert torch.isclose(par, ref, rtol=1e-4, atol=1e-6), (
        f"single-token: {par.item()} vs {ref.item()}"
    )


def test_vec_all_samples_zero_decoder_tokens() -> None:
    """Edge case: no decoder tokens in any sample → loss should be 0."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 8
    torch.manual_seed(88)
    zero_mask = torch.zeros((b, t_h, t_w, t), dtype=torch.long)  # all ONLINE_ENCODER
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=zero_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.zeros((b, 1), dtype=torch.long),
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=zero_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=torch.zeros((b, 1), dtype=torch.long),
    )
    par = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    assert par.item() == 0.0, f"expected 0.0 loss, got {par.item()}"


def test_vec_large_batch() -> None:
    """Larger batch closer to training microbatch size."""
    b, t_h, t_w, t, d = 32, 4, 4, 2, 64
    torch.manual_seed(2024)
    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    ll_mask = torch.randint(0, 4, (b, 1))
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=ll_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        latlon=torch.randn((b, 1, d)),
        latlon_mask=ll_mask,
    )
    par = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    ref = _modality_patch_disc_reference(preds, targets)
    assert torch.isclose(par, ref, rtol=1e-4, atol=1e-6), (
        f"large batch: {par.item()} vs {ref.item()}"
    )


def test_vec_multiple_modalities() -> None:
    """Test with 3 modalities having different mask patterns."""
    b, t_h, t_w, t, d = 6, 3, 3, 2, 16
    torch.manual_seed(555)

    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    s1_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    # worldcover: only decode modality — all decoder
    wc_mask = (
        torch.ones((b, t_h, t_w, 1), dtype=torch.long) * MaskValue.DECODER.value
    )

    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn((b, t_h, t_w, t, d)),
        sentinel1_mask=s1_mask,
        worldcover=torch.randn((b, t_h, t_w, 1, d)),
        worldcover_mask=wc_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
        sentinel1=torch.randn((b, t_h, t_w, t, d)),
        sentinel1_mask=s1_mask,
        worldcover=torch.randn((b, t_h, t_w, 1, d)),
        worldcover_mask=wc_mask,
    )
    par = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    ref = _modality_patch_disc_reference(preds, targets)
    assert torch.isclose(par, ref, rtol=1e-4, atol=1e-6), (
        f"multi-modality: {par.item()} vs {ref.item()}"
    )


def test_vec_modality_weights() -> None:
    """Test that modality_weights are applied identically."""
    b, t_h, t_w, t, d = 5, 3, 3, 2, 16
    weights = {"sentinel2_l2a": 2.0, "latlon": 0.5}

    for seed in [0, 42, 99]:
        torch.manual_seed(seed)
        s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
        ll_mask = torch.randint(0, 4, (b, 1))
        preds = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        targets = TokensAndMasks(
            sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
            sentinel2_l2a_mask=s2_mask,
            latlon=torch.randn((b, 1, d)),
            latlon_mask=ll_mask,
        )
        par = ModalityPatchDiscriminationLossVec(
            modality_weights=weights
        ).compute(preds, targets)
        ref = _modality_patch_disc_reference_weighted(
            preds, targets, modality_weights=weights
        )
        assert torch.isclose(par, ref, rtol=1e-4, atol=1e-6), (
            f"seed={seed} weighted: {par.item()} vs {ref.item()}"
        )


def _modality_patch_disc_reference_weighted(
    predictions: TokensAndMasks,
    targets: TokensAndMasks,
    tau: float = 0.1,
    modality_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Reference sequential implementation with modality weight support."""
    modality_preds, modality_masks = predictions.flatten_tokens_and_masks(
        return_lists=True
    )
    modality_targets = targets.flatten_tokens_and_masks(return_lists=True)[0]

    total_loss = 0
    for all_preds, all_masks, all_targets, modality in zip(
        modality_preds, modality_masks, modality_targets, targets.modalities
    ):
        pred = all_preds[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)
        target = all_targets[all_masks == MaskValue.DECODER.value].unsqueeze(dim=0)

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
        if modality_weights is not None:
            loss = loss * modality_weights[modality]
        total_loss += loss

    return total_loss


def test_vec_high_dim_large_tokens() -> None:
    """Higher dimension and larger spatial size closer to real training."""
    b, t_h, t_w, t, d = 8, 8, 8, 3, 128
    torch.manual_seed(314)
    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    preds = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
    )
    par = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    ref = _modality_patch_disc_reference(preds, targets)
    assert torch.isclose(par, ref, rtol=1e-3, atol=1e-5), (
        f"high-dim: {par.item()} vs {ref.item()}, "
        f"diff={abs(par.item() - ref.item())}"
    )


def test_vec_gradient_no_leak_to_non_decoder() -> None:
    """Non-decoder tokens should receive zero gradient from the loss."""
    b, t_h, t_w, t, d = 4, 3, 3, 2, 16
    torch.manual_seed(12)

    s2_mask = torch.randint(0, 4, (b, t_h, t_w, t))
    # Ensure at least some non-decoder tokens exist
    s2_mask[0, 0, 0, 0] = MaskValue.ONLINE_ENCODER.value
    s2_mask[1, 0, 0, 0] = MaskValue.MISSING.value

    s2_pred = torch.randn((b, t_h, t_w, t, d), requires_grad=True)
    preds = TokensAndMasks(
        sentinel2_l2a=s2_pred,
        sentinel2_l2a_mask=s2_mask,
    )
    targets = TokensAndMasks(
        sentinel2_l2a=torch.randn((b, t_h, t_w, t, d)),
        sentinel2_l2a_mask=s2_mask,
    )
    loss = ModalityPatchDiscriminationLossVec().compute(preds, targets)
    loss.backward()

    non_decoder = s2_mask != MaskValue.DECODER.value
    non_decoder_grad = s2_pred.grad[non_decoder]
    assert (non_decoder_grad == 0).all(), (
        f"non-decoder tokens got non-zero gradients: "
        f"max={non_decoder_grad.abs().max().item()}"
    )
