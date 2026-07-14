"""Per-sample TRAINING compute: perceiver base_2 vs v1.2 baseline.

Both runs share the dataloader (token_budget=2250, random hw/T windows,
random_time_with_decode masking), so per-sample token counts are identical;
this measures the compute applied to one budget-shaped masked sample by the
online encoder + decoder of each architecture (the target encoder is a
frozen patch-embed-only pass, identical for both; the x2 masked views and
backward scale both equally).

Run: PYTHONPATH=. python scripts/vnext/perceiver/bench_train_sample.py
"""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.perceiver import (
    PerceiverEncoderConfig,
    PerceiverPredictorConfig,
)
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

OBS = ["sentinel2_l2a", "sentinel1", "landsat"]
MAPS = [
    "worldcover",
    "srtm",
    "openstreetmap_raster",
    "wri_canopy_height_map",
    "cdl",
    "worldcereal",
]
MODALITIES = OBS + MAPS
TOKENIZATION = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[
                [
                    "B02",
                    "B03",
                    "B04",
                    "B08",
                    "B05",
                    "B06",
                    "B07",
                    "B8A",
                    "B11",
                    "B12",
                    "B01",
                    "B09",
                ]
            ]
        ),
        "landsat": ModalityTokenization(
            band_groups=[
                ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
            ]
        ),
    }
)
COMMON = dict(
    supported_modality_names=MODALITIES,
    max_sequence_length=12,
    tokenization_config=TOKENIZATION,
    position_encoding="rope_3d_mixed",
    rope_mixed_base=10000.0,
    rope_temporal_coordinate_scale=1.0 / 30.0,
)
ENC_COMMON = dict(
    embedding_size=768,
    num_heads=12,
    depth=12,
    mlp_ratio=4.0,
    max_patch_size=8,
    min_patch_size=1,
    drop_path=0.0,
    patch_embed_hidden_sizes=[64],
    **COMMON,
)


def build_models():
    """(baseline LatentMIM, perceiver LatentMIM) at base size."""
    baseline = LatentMIMConfig(
        encoder_config=EncoderConfig(**ENC_COMMON),
        decoder_config=PredictorConfig(
            encoder_embedding_size=768,
            decoder_embedding_size=768,
            depth=4,
            mlp_ratio=4.0,
            num_heads=12,
            **COMMON,
        ),
    ).build()
    perceiver = LatentMIMConfig(
        encoder_config=PerceiverEncoderConfig(
            **ENC_COMMON,
            latent_stride_hw=2,
            latent_stride_t=2,
            num_reads=2,
            readout_depth=2,
        ),
        decoder_config=PerceiverPredictorConfig(
            encoder_embedding_size=768,
            decoder_embedding_size=768,
            depth=0,
            head_depth=2,
            mlp_ratio=4.0,
            num_heads=12,
            **COMMON,
        ),
    ).build()
    return baseline, perceiver


def make_masked_sample(patches: int, timesteps: int, patch_size: int):
    """Budget-shaped sample: obs half time-masked, maps decode-only."""
    px = patches * patch_size
    fields: dict = {}
    enc_t = timesteps // 2  # first half visible, second half decode

    def add(name: str, bands: int, temporal: bool, decode_only: bool) -> None:
        shape = (1, px, px, timesteps, bands) if temporal else (1, px, px, bands)
        mask = torch.full(shape, MaskValue.DECODER.value, dtype=torch.long)
        if not decode_only and temporal:
            mask[:, :, :, :enc_t] = MaskValue.ONLINE_ENCODER.value
        fields[name] = torch.randn(*shape)
        fields[f"{name}_mask"] = mask

    for name in OBS:
        add(name, Modality.get(name).num_bands, temporal=True, decode_only=False)
    for name in MAPS:
        add(name, Modality.get(name).num_bands, temporal=False, decode_only=True)
    fields["timestamps"] = (
        torch.stack([torch.tensor([15, m % 12, 2023]) for m in range(timesteps)])
        .unsqueeze(0)
        .long()
    )
    sample = MaskedOlmoEarthSample(**fields)

    per_cell = 3 * timesteps + len(MAPS)
    total = patches * patches * per_cell
    visible = patches * patches * 3 * enc_t
    n_lat = ((patches + 1) // 2) ** 2 * ((timesteps + 1) // 2)
    print(
        f"  tokens/sample: total {total} (budget 2250), encoder-visible "
        f"{visible}, decode-targets {total - visible}; perceiver latents {n_lat}, "
        f"dense queries {patches * patches * timesteps}"
    )
    return sample


def flops_of(model, sample, patch_size: int) -> float:
    """Online fwd GMACs: encoder + decoder on one masked sample."""
    model = model.cuda().train()
    sample = sample.to_device(torch.device("cuda"))
    with torch.no_grad():
        counter = FlopCounterMode(display=False)
        with counter:
            model.forward(sample, patch_size=patch_size)
        gmacs = counter.get_total_flops() / 2 / 1e9
    model.cpu()
    torch.cuda.empty_cache()
    return gmacs


def main() -> None:
    """Measure per-sample training GMACs for both architectures."""
    torch.manual_seed(0)
    baseline, perceiver = build_models()
    for patches, timesteps, ps in [(8, 9, 4), (12, 3, 4)]:
        print(
            f"\n== window: {patches}x{patches} patches x {timesteps}t, "
            f"patch_size {ps} =="
        )
        sample = make_masked_sample(patches, timesteps, ps)
        g_base = flops_of(baseline, sample, ps)
        g_perc = flops_of(perceiver, sample, ps)
        print(f"  v1.2 baseline (enc+dec): {g_base:8.1f} GMACs/sample")
        print(
            f"  perceiver base_2      : {g_perc:8.1f} GMACs/sample "
            f"({g_base / g_perc:.2f}x fewer)"
        )


if __name__ == "__main__":
    main()
