"""compact_decode (decoder-only queries, no per-token encoder outputs) = the full path.

A small D'-shaped model -- slot merge with a narrow front end, a Perceiver, a 2D-RoPE
decoder over the registers, pixel targets and the projection-only target -- gives the
same loss and the same gradients with and without ``compact_decode``.
"""

from unittest.mock import patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.flexi_vit import (
    CompactDecoderTokens,
    EncoderConfig,
    PerceiverConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.slot_merge import SlotMergeConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskingConfig,
    MaskValue,
)
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
S2_ONE_BANDSET = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
        )
    }
)
B, T, PX = 3, 6, 8


def _sample() -> MaskedOlmoEarthSample:
    """S2 + S1 with per-token ENCODER / DECODER masks varying per cell and time."""
    g = torch.Generator().manual_seed(1)
    fields: dict = {}
    for name in MODALITIES:
        bands = Modality.get(name).num_bands
        tok = torch.rand(B, PX, PX, T, generator=g)
        mask = torch.where(
            tok < 0.4, MaskValue.DECODER.value, MaskValue.ONLINE_ENCODER.value
        )
        fields[name] = torch.randn(B, PX, PX, T, bands, generator=g)
        fields[f"{name}_mask"] = mask[..., None].expand(-1, -1, -1, -1, bands).clone()
    fields["timestamps"] = torch.tensor([[[1, t % 12, 2020] for t in range(T)]]).expand(
        B, -1, -1
    )
    return MaskedOlmoEarthSample(**fields)


def _model():  # type: ignore[no-untyped-def]
    torch.manual_seed(0)
    encoder = EncoderConfig(
        supported_modality_names=MODALITIES,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        tokenization_config=S2_ONE_BANDSET,
        perceiver_config=PerceiverConfig(
            register_dim=16,
            latent_depth=1,
            pixel_latents=True,
            random_latent_stride=True,
            max_latents=32,
            eval_latent_stride=1,
        ),
        slot_merge_config=SlotMergeConfig(
            merge_after=0,
            narrow_depth=1,
            narrow_dim=64,
            max_slots_per_cell=4,
            narrow_front_end=True,
        ),
    )
    decoder = PredictorConfig(
        supported_modality_names=MODALITIES,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        drop_path=0.0,
        position_encoding="rope",
        use_perceiver=True,
        register_dim=16,
        tokenization_config=S2_ONE_BANDSET,
    )
    return LatentMIMConfig(
        encoder_config=encoder, decoder_config=decoder, projection_only_target=True
    ).build()


def _forward(compact: bool, patch_size: int):  # type: ignore[no-untyped-def]
    model = _model()
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=B,
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
        compact_decode=compact,
        target_chunk_size=2,
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        train_module = config.build(model, device=torch.device("cpu"))
    torch.manual_seed(7)  # the pixel draw
    loss, _latent, decoded, _targets, _metrics = train_module.model_forward(
        _sample(), patch_size, train_module.token_exit_cfg
    )
    loss.backward()
    grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach(), decoded, grads


@pytest.mark.parametrize("patch_size", [1, 2])
def test_compact_decode_matches_the_full_path(patch_size: int) -> None:
    """Same loss and gradients; the compact path returns compact predictions."""
    full_loss, full_decoded, full_grads = _forward(False, patch_size)
    loss, decoded, grads = _forward(True, patch_size)
    assert isinstance(decoded, CompactDecoderTokens)
    assert not isinstance(full_decoded, CompactDecoderTokens)
    torch.testing.assert_close(loss, full_loss, atol=1e-5, rtol=1e-5)
    # The decoder's input norm / embed only feed positions the full path discards or
    # overwrites with the mask token: zero gradient there, untouched here.
    assert set(grads) <= set(full_grads)
    for name in set(full_grads) - set(grads):
        assert name.startswith("decoder."), name
        assert torch.all(full_grads[name] == 0), name
    for name, grad in grads.items():
        torch.testing.assert_close(grad, full_grads[name], atol=5e-5, rtol=1e-3)
