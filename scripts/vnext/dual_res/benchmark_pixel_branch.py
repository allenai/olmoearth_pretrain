"""Benchmark the dual-resolution pixel-branch variants against a coarse-only model.

Times a realistic training step -- model forward (encoder + predictor + pixel heads),
target-encoder forward (``token_exit_cfg`` all-zero, as in training), the real
patch-discrimination loss, and backward -- on synthetic batches matching the training
distribution of ``base.py``: per batch one ``(patch_size, sampled_hw_p)`` pair is drawn
(patch size 1..8, hw_p 1..12), the timestep count is capped by the 2250-token budget,
and ``random_time_with_decode`` masking is applied. All models see identical batches.

Usage (single GPU)::

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/vnext/dual_res/benchmark_pixel_branch.py \
        --variants coarse_only,joint_d128,conv_d64 --cases 12 --batch-size 64

Reports per-variant mean/median step time, the slowdown ratio vs ``coarse_only``, and
peak GPU memory.
"""

import argparse
import gc
import json
import logging
import time

import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import _get_max_t_within_token_budget
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.nn.dual_res_encoder import DualResEncoderConfig
from olmoearth_pretrain.nn.dual_res_model import DualResLatentMIMConfig
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---- Mirrors scripts/vnext/dual_res/base.py ------------------------------------
MAX_PATCH_SIZE = 8
MAX_T = 12
TOKEN_BUDGET = 2250
SAMPLED_HW_P = list(range(1, 13))
PATCH_SIZES = list(range(1, 9))

S2 = Modality.SENTINEL2_L2A.name
S1 = Modality.SENTINEL1.name
LANDSAT = Modality.LANDSAT.name
SPACETIME_MODALITIES = [S2, S1, LANDSAT]
MAP_MODALITIES = [
    Modality.WORLDCOVER_ONEHOT.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]
TRAINING_MODALITIES = SPACETIME_MODALITIES + MAP_MODALITIES

MAP_TARGETS = {
    Modality.WORLDCOVER_ONEHOT.name: "ce",
    Modality.OPENSTREETMAP_RASTER.name: "bce",
    Modality.WORLDCEREAL.name: "bce",
    Modality.SRTM.name: "mse",
    Modality.WRI_CANOPY_HEIGHT_MAP.name: "mse",
}

S2_SINGLE_BANDSET = ModalityTokenization(
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
)
LANDSAT_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]]
)


def tokenization_config() -> TokenizationConfig:
    """Tokenization config matching ``base.py``."""
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": S2_SINGLE_BANDSET,
            "landsat": LANDSAT_SINGLE_BANDSET,
        }
    )


def masking_config(tok: TokenizationConfig) -> MaskingConfig:
    """Masking config matching ``base.py``."""
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": MAP_MODALITIES,
        },
        tokenization_config=tok,
    )


# ---- Model variants -------------------------------------------------------------
COARSE = dict(
    embedding_size=512,
    num_heads=8,
    depth=12,
    mlp_ratio=4.0,
    decoder_embedding_size=512,
    decoder_depth=4,
    decoder_num_heads=8,
)

# name -> dict of DualResEncoderConfig pixel-branch overrides (plus "pixel_mlp_ratio").
VARIANTS: dict[str, dict] = {
    # The original joint-attention pixel branch at its base.py settings.
    "joint_d128": dict(
        pixel_branch_type="joint",
        pixel_embedding_size=128,
        pixel_num_heads=4,
        pixel_mlp_ratio=4.0,
    ),
    "joint_d96": dict(
        pixel_branch_type="joint",
        pixel_embedding_size=96,
        pixel_num_heads=3,
        pixel_mlp_ratio=4.0,
    ),
    # Alternative 1: convolutional detail branch (ConvNeXt-style, expansion 2).
    "conv_d64": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
    ),
    "conv_d32": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=32,
        pixel_num_heads=2,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
    ),
    "conv_d48_k5": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=48,
        pixel_num_heads=3,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=5,
    ),
    "conv_d64_every2": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
        pixel_every_k_blocks=2,
    ),
    # Alternative 2: per-patch window attention with a coarse register token.
    "window_d64": dict(
        pixel_branch_type="window",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
    ),
    "window_d32": dict(
        pixel_branch_type="window",
        pixel_embedding_size=32,
        pixel_num_heads=2,
        pixel_mlp_ratio=2.0,
    ),
    "window_d64_every2": dict(
        pixel_branch_type="window",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=2,
    ),
    "window_d48_every2": dict(
        pixel_branch_type="window",
        pixel_embedding_size=48,
        pixel_num_heads=3,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=2,
    ),
    "window_d64_every3": dict(
        pixel_branch_type="window",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=3,
    ),
    "conv_d64_every3": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
        pixel_every_k_blocks=3,
    ),
    "conv_d32_every2": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=32,
        pixel_num_heads=2,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
        pixel_every_k_blocks=2,
    ),
    "perceiver_d64_every3": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=3,
    ),
    "perceiver_d128_every2": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=128,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=2,
    ),
    "conv_d64_nockpt": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
        pixel_grad_checkpointing=False,
    ),
    "window_d64_nockpt": dict(
        pixel_branch_type="window",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_grad_checkpointing=False,
    ),
    "perceiver_d64_nockpt": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_grad_checkpointing=False,
    ),
    # Alternative 3: cross-attention-only (Perceiver-style) pixel tokens.
    "perceiver_d64": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
    ),
    "perceiver_d128": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=128,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
    ),
    "perceiver_d64_read3": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_coarse_read_interval=3,
    ),
    "perceiver_d64_every2": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=2,
    ),
    "conv_d64_every4": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
        pixel_every_k_blocks=4,
    ),
    "conv_d32_every3": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=32,
        pixel_num_heads=2,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
        pixel_every_k_blocks=3,
    ),
    "window_d64_every4": dict(
        pixel_branch_type="window",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=4,
    ),
    "perceiver_d64_every4": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=4,
    ),
    "perceiver_d128_every3": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=128,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=3,
    ),
    "conv_d64_every6": dict(
        pixel_branch_type="conv",
        pixel_embedding_size=64,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_conv_kernel=3,
        pixel_every_k_blocks=6,
    ),
    "perceiver_d128_every4": dict(
        pixel_branch_type="perceiver",
        pixel_embedding_size=128,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_every_k_blocks=4,
    ),
    # PixelDiT-style post-trunk pathway (pixel-wise AdaLN + token compaction).
    "pixeldit_d16_m2": dict(
        pixel_branch_type="pixeldit",
        pixel_embedding_size=16,
        pixel_num_heads=2,
        pixel_mlp_ratio=4.0,
        pixel_dit_depth=2,
    ),
    "pixeldit_d16_m4": dict(
        pixel_branch_type="pixeldit",
        pixel_embedding_size=16,
        pixel_num_heads=2,
        pixel_mlp_ratio=4.0,
        pixel_dit_depth=4,
    ),
    "pixeldit_d32_m2": dict(
        pixel_branch_type="pixeldit",
        pixel_embedding_size=32,
        pixel_num_heads=2,
        pixel_mlp_ratio=4.0,
        pixel_dit_depth=2,
    ),
}


def build_model(variant: str, pixel_heads: bool = True) -> torch.nn.Module:
    """Build the full model (encoder + predictor [+ pixel heads]) for a variant."""
    tok = tokenization_config()
    coarse_kwargs = dict(
        embedding_size=COARSE["embedding_size"],
        num_heads=COARSE["num_heads"],
        depth=COARSE["depth"],
        mlp_ratio=COARSE["mlp_ratio"],
        supported_modality_names=TRAINING_MODALITIES,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=MAX_T,
        tokenization_config=tok,
        patch_embed_hidden_sizes=[64],
        position_encoding="rope_3d_mixed",
        rope_mixed_base=10000.0,
        rope_temporal_coordinate_scale=1.0 / 30.0,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=COARSE["embedding_size"],
        decoder_embedding_size=COARSE["decoder_embedding_size"],
        depth=COARSE["decoder_depth"],
        mlp_ratio=COARSE["mlp_ratio"],
        num_heads=COARSE["decoder_num_heads"],
        supported_modality_names=TRAINING_MODALITIES,
        max_sequence_length=MAX_T,
        tokenization_config=tok,
        position_encoding="rope_3d_mixed",
        rope_mixed_base=10000.0,
        rope_temporal_coordinate_scale=1.0 / 30.0,
    )
    if variant == "coarse_only":
        config = LatentMIMConfig(
            encoder_config=EncoderConfig(**coarse_kwargs),
            decoder_config=decoder_config,
        )
        return config.build()

    pixel_kwargs = dict(VARIANTS[variant])
    encoder_config = DualResEncoderConfig(**coarse_kwargs, **pixel_kwargs)
    config = DualResLatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        pixel_reconstruction=pixel_heads,
        pixel_recon_depth=2,
        pixel_recon_num_heads=pixel_kwargs.get("pixel_num_heads", 4),
        pixel_recon_mlp_ratio=pixel_kwargs.get("pixel_mlp_ratio", 4.0),
        pixel_recon_weight=1.0,
        map_targets=MAP_TARGETS if pixel_heads else {},
        pixel_map_weight=1.0,
    )
    return config.build()


# ---- Synthetic batches ----------------------------------------------------------
def make_batch(
    rng: np.random.Generator, batch_size: int, tok: TokenizationConfig
) -> tuple[int, MaskedOlmoEarthSample]:
    """Draw one (patch_size, hw_p) case and build a masked batch like the dataloader.

    Follows the training sampling: patch size uniform in 1..8, hw_p uniform in 1..12
    (one pair per rank microbatch), timestep count capped by the token budget, then
    ``random_time_with_decode`` masking at the sampled patch size.
    """
    patch_size = int(rng.choice(PATCH_SIZES))
    hw_p = int(rng.choice(SAMPLED_HW_P))
    hw = patch_size * hw_p
    g = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    data: dict[str, torch.Tensor] = {}
    for name in SPACETIME_MODALITIES:
        data[name] = rand(batch_size, hw, hw, MAX_T, Modality.get(name).num_bands)
    for name in MAP_MODALITIES:
        bands = Modality.get(name).num_bands
        if name == Modality.WORLDCOVER_ONEHOT.name:
            classes = torch.randint(0, bands, (batch_size, hw, hw, 1), generator=g)
            data[name] = torch.nn.functional.one_hot(classes, bands).float()
        else:
            data[name] = rand(batch_size, hw, hw, 1, bands)
    timestamps = torch.stack(
        [
            torch.randint(1, 28, (batch_size, MAX_T), generator=g),
            torch.randint(0, 12, (batch_size, MAX_T), generator=g),
            torch.full((batch_size, MAX_T), 2022),
        ],
        dim=-1,
    )
    sample = OlmoEarthSample(timestamps=timestamps, **data)

    # Token budget -> max timesteps, as subset_sample_default does.
    max_t = min(MAX_T, _get_max_t_within_token_budget(sample, hw_p, TOKEN_BUDGET, tok))
    if max_t < MAX_T:
        for name in SPACETIME_MODALITIES:
            data[name] = data[name][:, :, :, :max_t]
        sample = OlmoEarthSample(timestamps=timestamps[:, :max_t], **data)

    strategy = masking_config(tok).build()
    masked = strategy.apply_mask(sample, patch_size=patch_size)
    return patch_size, masked


def to_device(x: MaskedOlmoEarthSample, device: torch.device) -> MaskedOlmoEarthSample:
    """Move every tensor field of the masked sample to ``device``."""
    moved = {
        k: v.to(device, non_blocking=True)
        for k, v in x.as_dict().items()
        if torch.is_tensor(v)
    }
    return x._replace(**moved)


# ---- Benchmark loop -------------------------------------------------------------
def run_variant(
    variant: str,
    batches: list[tuple[int, MaskedOlmoEarthSample]],
    device: torch.device,
    warmup: int,
    pixel_heads: bool = True,
) -> dict:
    """Time fwd+loss+bwd for one variant over the given batches."""
    torch.manual_seed(0)
    model = build_model(variant, pixel_heads=pixel_heads).to(device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_fn = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_masked_negatives_vec",
            "tau": 0.1,
            "same_target_threshold": 0.999,
            "mask_negatives_for_modalities": MAP_MODALITIES,
        }
    ).build()
    token_exit_cfg = {m: 0 for m in TRAINING_MODALITIES}

    def step(patch_size: int, batch: MaskedOlmoEarthSample) -> float:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(batch, patch_size)
            decoded = outputs[1]
            aux_loss = outputs[5] if len(outputs) > 5 else None
            with torch.no_grad():
                target_dict = model.target_encoder.forward(
                    batch.unmask(), patch_size=patch_size, token_exit_cfg=token_exit_cfg
                )
                target = target_dict["tokens_and_masks"]
            loss = loss_fn.compute(decoded, target)
            if aux_loss is not None:
                loss = loss + aux_loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        return float(loss.detach())

    # Warmup.
    for patch_size, batch in batches[:warmup]:
        step(patch_size, to_device(batch, device))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    times = []
    for patch_size, batch in batches:
        batch = to_device(batch, device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(patch_size, batch)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    return dict(
        variant=variant,
        params_m=num_params / 1e6,
        mean_s=float(np.mean(times)),
        median_s=float(np.median(times)),
        total_s=float(np.sum(times)),
        per_case_s=[round(t, 4) for t in times],
        peak_gb=peak_gb,
    )


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        type=str,
        default="coarse_only,joint_d128,conv_d64,window_d64,perceiver_d64",
        help="Comma-separated variant names ('coarse_only' plus keys of VARIANTS).",
    )
    parser.add_argument("--cases", type=int, default=12, help="Batches per variant.")
    parser.add_argument(
        "--no-pixel-heads",
        action="store_true",
        help="Disable the pixel reconstruction decoder and map probe (isolates the "
        "encoder pixel-branch cost from the shared head cost).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output.")
    args = parser.parse_args()

    device = torch.device("cuda")
    tok = tokenization_config()
    rng = np.random.default_rng(args.seed)
    print("Generating batches...", flush=True)
    batches = [make_batch(rng, args.batch_size, tok) for _ in range(args.cases)]
    for i, (p, b) in enumerate(batches):
        hw = b.sentinel2_l2a.shape[1]
        t = b.sentinel2_l2a.shape[3]
        print(f"  case {i}: patch_size={p} hw={hw} ({hw // p}x{hw // p} tokens) T={t}")

    results = []
    for variant in args.variants.split(","):
        print(f"Running {variant}...", flush=True)
        res = run_variant(
            variant, batches, device, args.warmup, pixel_heads=not args.no_pixel_heads
        )
        gc.collect()
        torch.cuda.empty_cache()
        results.append(res)
        print(
            f"  {variant}: mean {res['mean_s'] * 1e3:.1f} ms  "
            f"median {res['median_s'] * 1e3:.1f} ms  peak {res['peak_gb']:.1f} GB  "
            f"params {res['params_m']:.1f}M"
        )

    base = next((r for r in results if r["variant"] == "coarse_only"), None)
    print("\n=== Summary ===")
    header = (
        f"{'variant':<22}{'mean ms':>9}{'median ms':>11}{'peak GB':>9}{'params M':>10}"
    )
    if base:
        header += f"{'x coarse':>10}"
    print(header)
    for r in results:
        line = (
            f"{r['variant']:<22}{r['mean_s'] * 1e3:>9.1f}{r['median_s'] * 1e3:>11.1f}"
            f"{r['peak_gb']:>9.1f}{r['params_m']:>10.1f}"
        )
        if base:
            line += f"{r['total_s'] / base['total_s']:>10.2f}"
        print(line)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
