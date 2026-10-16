"""Measure encoder MACs / wall-clock / memory: perceiver vs v1.2 baseline.

Standalone local benchmark (no trainer, random weights — cost is
weight-independent). Both encoders are built at base size with the v1.2
single-bandset tokenization and fed the same all-visible synthetic batch
(S2 + S1 + Landsat observations; maps are decode-only and never encoded at
inference). FLOPs counted with torch.utils.flop_counter (matmul-accurate,
SDPA included), then reported as MACs = FLOPs / 2.

Run: PYTHONPATH=. python scripts/vnext/perceiver/bench_perceiver.py
"""

from __future__ import annotations

import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.perceiver import PerceiverEncoderConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

MODALITIES = ["sentinel2_l2a", "sentinel1", "landsat"]
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
                ],
            ]
        ),
        "landsat": ModalityTokenization(
            band_groups=[
                ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
            ]
        ),
    }
)

COMMON = dict(
    embedding_size=768,
    num_heads=12,
    depth=12,
    mlp_ratio=4.0,
    supported_modality_names=MODALITIES,
    max_patch_size=8,
    min_patch_size=1,
    drop_path=0.0,
    max_sequence_length=12,
    tokenization_config=TOKENIZATION,
    patch_embed_hidden_sizes=[64],
    position_encoding="rope_3d_mixed",
    rope_mixed_base=10000.0,
    rope_temporal_coordinate_scale=1.0 / 30.0,
)


def make_sample(batch: int, px: int, timesteps: int) -> MaskedOlmoEarthSample:
    """All-visible S2+S1+Landsat sample of px x px pixels."""

    def obs(bands: int, t: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, px, px, t, bands) if t else (batch, px, px, bands)
        return (
            torch.randn(*shape),
            torch.full(shape, MaskValue.ONLINE_ENCODER.value, dtype=torch.long),
        )

    s2, s2_m = obs(Modality.SENTINEL2_L2A.num_bands, timesteps)
    s1, s1_m = obs(Modality.SENTINEL1.num_bands, timesteps)
    ls, ls_m = obs(Modality.LANDSAT.num_bands, timesteps)
    timestamps = (
        torch.stack(
            [torch.tensor([15, m % 12, 2023 + m // 12]) for m in range(timesteps)]
        )
        .unsqueeze(0)
        .repeat(batch, 1, 1)
        .long()
    )
    return MaskedOlmoEarthSample(
        sentinel2_l2a=s2,
        sentinel2_l2a_mask=s2_m,
        sentinel1=s1,
        sentinel1_mask=s1_m,
        landsat=ls,
        landsat_mask=ls_m,
        timestamps=timestamps,
    )


def bench(name: str, encoder: torch.nn.Module, sample, patch_size: int) -> dict:
    """FLOPs (single fwd) + median wall-clock + peak memory."""
    encoder = encoder.cuda().eval()
    sample = sample.to_device(torch.device("cuda"))

    with torch.no_grad():
        flop_counter = FlopCounterMode(display=False)
        with flop_counter:
            encoder.forward(sample, patch_size=patch_size, fast_pass=True)
        flops = flop_counter.get_total_flops()

        for _ in range(3):  # warmup
            encoder.forward(sample, patch_size=patch_size, fast_pass=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            encoder.forward(sample, patch_size=patch_size, fast_pass=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
    times.sort()
    med_ms = times[len(times) // 2] * 1e3
    params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print(
        f"{name:>28}: {flops / 2 / 1e9:9.1f} GMACs  {med_ms:8.1f} ms/fwd  "
        f"peak {peak_gib:5.2f} GiB  params {params:6.1f}M"
    )
    encoder.cpu()
    torch.cuda.empty_cache()
    return {"gmacs": flops / 2 / 1e9, "ms": med_ms, "gib": peak_gib}


def main() -> None:
    """Benchmark baseline vs perceiver encoders across input sizes."""
    torch.manual_seed(0)
    baseline = EncoderConfig(**COMMON).build()
    perceiver = PerceiverEncoderConfig(
        **COMMON, latent_stride_hw=2, latent_stride_t=2, num_reads=2, readout_depth=2
    ).build()
    perceiver_s3 = PerceiverEncoderConfig(
        **COMMON, latent_stride_hw=3, latent_stride_t=3, num_reads=2, readout_depth=2
    ).build()

    for label, px, t, ps, b in [
        ("96px x 12t (12x12 grid)", 96, 12, 8, 8),
        ("64px x 4t (16x16 grid)", 64, 4, 4, 8),
        ("32px x 12t (8x8 grid)", 32, 12, 4, 8),
    ]:
        print(f"\n== {label}, batch {b} ==")
        sample = make_sample(b, px, t)
        r_base = bench("v1.2 baseline Encoder", baseline, sample, ps)
        r_perc = bench("PerceiverEncoder (s2/2)", perceiver, sample, ps)
        r_s3 = bench("PerceiverEncoder (s3/3)", perceiver_s3, sample, ps)
        print(
            f"{'ratios vs baseline':>28}: perceiver "
            f"{r_base['gmacs'] / r_perc['gmacs']:.2f}x MACs, "
            f"{r_base['ms'] / r_perc['ms']:.2f}x speed | s3 "
            f"{r_base['gmacs'] / r_s3['gmacs']:.2f}x MACs, "
            f"{r_base['ms'] / r_s3['ms']:.2f}x speed"
        )


if __name__ == "__main__":
    main()
