"""truncate_to_decoder leaves ModalityPatchDiscriminationMaskedNegativesVec unchanged."""

import torch

from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks
from olmoearth_pretrain.train.loss import ModalityPatchDiscriminationMaskedNegativesVec


def _sample(seed: int, decode_prob: float) -> tuple[dict, dict]:
    g = torch.Generator().manual_seed(seed)
    b, h, w, t, d = 5, 6, 6, 3, 16
    preds, targets = {}, {}
    for name, shape in [("sentinel2_l2a", (b, h, w, t)), ("worldcover", (b, h, w, 1))]:
        mask = torch.full(shape, MaskValue.TARGET_ENCODER_ONLY.value)
        mask[torch.rand(shape, generator=g) < decode_prob] = MaskValue.DECODER.value
        mask[torch.rand(shape, generator=g) < 0.1] = MaskValue.MISSING.value
        mask[0] = MaskValue.MISSING.value  # a sample with no decoder tokens
        tgt = torch.randn((*shape, d), generator=g)
        if name == "worldcover":
            # Repeated targets exercise the same-target negative masking.
            tgt = tgt.round()
        preds[name] = torch.randn((*shape, d), generator=g)
        preds[f"{name}_mask"] = mask
        targets[name] = tgt
        targets[f"{name}_mask"] = mask
    return preds, targets


def test_truncate_to_decoder_matches_loss_and_grads() -> None:
    """Same loss and prediction gradients with and without truncation."""
    for seed in range(8):
        for decode_prob in [0.0, 0.05, 0.3, 0.9]:
            preds, targets = _sample(seed, decode_prob)
            results = []
            for truncate in [False, True]:
                loss_fn = ModalityPatchDiscriminationMaskedNegativesVec(
                    tau=0.1,
                    same_target_threshold=0.999,
                    mask_negatives_for_modalities=["worldcover"],
                    truncate_to_decoder=truncate,
                )
                p = {
                    k: (v.clone().requires_grad_(True) if not k.endswith("_mask") else v)
                    for k, v in preds.items()
                }
                loss = loss_fn.compute(TokensAndMasks(**p), TokensAndMasks(**targets))
                loss.backward()
                results.append(
                    (loss.detach(), p["sentinel2_l2a"].grad, p["worldcover"].grad)
                )
            (l0, g0, w0), (l1, g1, w1) = results
            assert torch.allclose(l0, l1, atol=1e-6), (seed, decode_prob, l0, l1)
            for a, b in [(g0, g1), (w0, w1)]:
                if a is None:
                    assert b is None or not b.any()
                else:
                    assert torch.allclose(a, b, atol=1e-6), (seed, decode_prob)
