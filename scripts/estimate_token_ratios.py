"""Estimate the percentage of tokens encoded and decoded for a given masking strategy.

This script samples from all possible input sizes (patch_size, sampled_hw_p) and
passes the data through a masking strategy to compute what ratio of tokens are
encoded vs decoded.

Example usage:
    python scripts/estimate_token_ratios.py --num_samples 1000
"""

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from olmoearth_pretrain.data.constants import (
    IMAGE_TILE_SIZE,
    MISSING_VALUE,
    Modality,
    ModalitySpec,
)
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig

logger = logging.getLogger(__name__)

# Default config from scripts/official/script.py
DEFAULT_TRAINING_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

DEFAULT_MASKING_CONFIG = {
    "type": "modality_cross_random",
    "encode_ratio": 0.5,
    "decode_ratio": 0.5,
    "allow_encoding_decoding_same_bandset": True,
    "only_decode_modalities": [
        Modality.WORLDCOVER.name,
        Modality.SRTM.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.WRI_CANOPY_HEIGHT_MAP.name,
        Modality.CDL.name,
        Modality.WORLDCEREAL.name,
    ],
}

# Single bandset tokenization configs (from single_bandset_band_dropout experiments)
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
    band_groups=[
        ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    ]
)


def build_single_bandset_tokenization_config() -> TokenizationConfig:
    """Build a TokenizationConfig with single bandset for S2 and Landsat."""
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": S2_SINGLE_BANDSET,
            "landsat": LANDSAT_SINGLE_BANDSET,
        }
    )


@dataclass
class TokenRatioResult:
    """Results from a single sample."""

    total_tokens: int
    encoded_tokens: int
    decoded_tokens: int
    target_only_tokens: int
    missing_tokens: int
    # Per-modality breakdown
    per_modality_stats: dict[str, dict[str, int]] | None = None

    @property
    def encoded_ratio(self) -> float:
        """Ratio of encoded tokens to total tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.encoded_tokens / self.total_tokens

    @property
    def decoded_ratio(self) -> float:
        """Ratio of decoded tokens to total tokens."""
        if self.total_tokens == 0:
            return 0.0
        return self.decoded_tokens / self.total_tokens

    @property
    def non_missing_tokens(self) -> int:
        """Total tokens excluding missing tokens."""
        return self.total_tokens - self.missing_tokens


def create_synthetic_sample(
    training_modalities: list[str],
    sampled_hw_p: int,
    patch_size: int,
    max_t: int,
    missing_prob: float = 0.1,
    rng: np.random.Generator | None = None,
) -> OlmoEarthSample:
    """Create a synthetic OlmoEarthSample with the given spatial/temporal dimensions.

    Args:
        training_modalities: List of modality names to include.
        sampled_hw_p: Number of patches in height and width.
        patch_size: Patch size in pixels (at 10m resolution).
        max_t: Number of timesteps.
        missing_prob: Probability of a modality being missing for a timestep.
        rng: Random number generator.

    Returns:
        A synthetic OlmoEarthSample.
    """
    if rng is None:
        rng = np.random.default_rng()

    sampled_hw = sampled_hw_p * patch_size  # pixels at 10m resolution
    sample_dict: dict[str, torch.Tensor] = {}

    # Always include timestamps
    sample_dict["timestamps"] = torch.rand(max_t, 3)  # [T, 3]

    for modality_name in training_modalities:
        modality_spec = Modality.get(modality_name)

        # Compute spatial dimensions for this modality
        h = sampled_hw * modality_spec.image_tile_size_factor
        w = sampled_hw * modality_spec.image_tile_size_factor
        num_bands = modality_spec.num_bands

        if modality_spec.is_spacetime_varying:
            # [H, W, T, C]
            data = torch.rand(h, w, max_t, num_bands)
            # Randomly mark some timesteps as missing
            for t in range(max_t):
                if rng.random() < missing_prob:
                    data[:, :, t, :] = MISSING_VALUE
        elif modality_spec.is_space_only_varying:
            # [H, W, C]
            data = torch.rand(h, w, num_bands)
            # Randomly mark entire modality as missing
            if rng.random() < missing_prob:
                data[:] = MISSING_VALUE
        elif modality_spec.is_time_only_varying:
            # [T, C]
            data = torch.rand(max_t, num_bands)
            for t in range(max_t):
                if rng.random() < missing_prob:
                    data[t, :] = MISSING_VALUE
        elif modality_spec.is_static_in_space_and_time:
            # [C]
            data = torch.rand(num_bands)
            if rng.random() < missing_prob:
                data[:] = MISSING_VALUE
        else:
            raise ValueError(f"Unknown modality type: {modality_name}")

        sample_dict[modality_name] = data

    return OlmoEarthSample(**sample_dict)


def count_tokens_in_mask(
    mask: torch.Tensor,
    modality_spec: ModalitySpec,
    patch_size: int,
) -> dict[str, int]:
    """Count tokens by mask value for a single modality.

    Args:
        mask: Mask tensor for the modality.
        modality_spec: The modality specification.
        patch_size: Patch size at 10m resolution.

    Returns:
        Dictionary with counts for each mask value.
    """
    # Masks are at patch resolution for spatial modalities
    if modality_spec.is_spatial:
        actual_patch_size = patch_size * modality_spec.image_tile_size_factor
        # Subsample to patch resolution
        mask = mask[0::actual_patch_size, 0::actual_patch_size]

    # Flatten to count
    flat_mask = mask.flatten()

    return {
        "encoded": (flat_mask == MaskValue.ONLINE_ENCODER.value).sum().item(),
        "decoded": (flat_mask == MaskValue.DECODER.value).sum().item(),
        "target_only": (flat_mask == MaskValue.TARGET_ENCODER_ONLY.value).sum().item(),
        "missing": (flat_mask == MaskValue.MISSING.value).sum().item(),
    }


def analyze_masked_sample(
    masked_sample: MaskedOlmoEarthSample,
    patch_size: int,
    track_per_modality: bool = False,
) -> TokenRatioResult:
    """Analyze a masked sample to count tokens by mask type.

    Args:
        masked_sample: The masked sample.
        patch_size: Patch size at 10m resolution.
        track_per_modality: Whether to track per-modality statistics.

    Returns:
        TokenRatioResult with token counts.
    """
    total_encoded = 0
    total_decoded = 0
    total_target_only = 0
    total_missing = 0
    total_tokens = 0
    per_modality_stats: dict[str, dict[str, int]] = {}

    for modality_name in masked_sample.modalities:
        mask_name = MaskedOlmoEarthSample.get_masked_modality_name(modality_name)
        mask = getattr(masked_sample, mask_name)
        if mask is None:
            continue

        modality_spec = Modality.get(modality_name)
        counts = count_tokens_in_mask(mask, modality_spec, patch_size)

        total_encoded += counts["encoded"]
        total_decoded += counts["decoded"]
        total_target_only += counts["target_only"]
        total_missing += counts["missing"]
        total_tokens += sum(counts.values())

        if track_per_modality:
            per_modality_stats[modality_name] = counts

    return TokenRatioResult(
        total_tokens=total_tokens,
        encoded_tokens=total_encoded,
        decoded_tokens=total_decoded,
        target_only_tokens=total_target_only,
        missing_tokens=total_missing,
        per_modality_stats=per_modality_stats if track_per_modality else None,
    )


def estimate_token_ratios(
    num_samples: int,
    training_modalities: list[str],
    masking_config: dict,
    min_patch_size: int = 1,
    max_patch_size: int = 8,
    sampled_hw_p_list: list[int] | None = None,
    token_budget: int = 2250,
    missing_prob: float = 0.1,
    seed: int = 42,
    track_per_modality: bool = False,
    tokenization_config: TokenizationConfig | None = None,
) -> list[TokenRatioResult]:
    """Estimate token encode/decode ratios by sampling many configurations.

    Args:
        num_samples: Number of samples to generate.
        training_modalities: List of modality names.
        masking_config: Configuration dict for the masking strategy.
        min_patch_size: Minimum patch size.
        max_patch_size: Maximum patch size.
        sampled_hw_p_list: List of possible hw_p values (patches per side).
        token_budget: Maximum tokens per instance.
        missing_prob: Probability of a modality/timestep being missing.
        seed: Random seed.
        track_per_modality: Whether to track per-modality statistics.
        tokenization_config: tokenization_config

    Returns:
        List of TokenRatioResult for each sample.
    """
    if sampled_hw_p_list is None:
        sampled_hw_p_list = list(range(1, 13))

    rng = np.random.default_rng(seed)
    patch_sizes = list(range(min_patch_size, max_patch_size + 1))
    results = []

    # Build the masking strategy
    config_copy = masking_config.copy()
    masking_strategy = MaskingConfig(
        strategy_config=config_copy,
        tokenization_config=tokenization_config,
    ).build()

    for _ in tqdm(range(num_samples), desc="Sampling"):
        # Sample patch_size and hw_p
        patch_size = int(rng.choice(patch_sizes))
        max_hw_p = int(IMAGE_TILE_SIZE / patch_size)
        valid_hw_p_list = [hp for hp in sampled_hw_p_list if 0 < hp <= max_hw_p]
        sampled_hw_p = int(rng.choice(valid_hw_p_list))

        # Estimate max_t based on token budget (simplified version)
        # This mimics OlmoEarthSample._get_max_t_within_token_budget
        tokens_per_timestep = estimate_tokens_per_timestep(
            training_modalities, sampled_hw_p, tokenization_config
        )
        static_tokens = estimate_static_tokens(
            training_modalities, sampled_hw_p, tokenization_config
        )
        available_budget = token_budget - static_tokens
        max_t = (
            max(1, int(available_budget / tokens_per_timestep))
            if tokens_per_timestep > 0
            else 12
        )
        max_t = min(max_t, 12)  # Cap at MAX_SEQUENCE_LENGTH
        # Create synthetic sample
        sample = create_synthetic_sample(
            training_modalities=training_modalities,
            sampled_hw_p=sampled_hw_p,
            patch_size=patch_size,
            max_t=max_t,
            missing_prob=missing_prob,
            rng=rng,
        )

        # Add batch dimension for masking
        sample_batched = add_batch_dimension(sample)

        # Apply masking
        try:
            masked_sample = masking_strategy.apply_mask(
                sample_batched, patch_size=patch_size
            )
        except Exception as e:
            logger.warning(f"Masking failed: {e}")
            continue

        # Remove batch dimension for analysis
        masked_sample_unbatched = remove_batch_dimension(masked_sample)

        # Analyze
        result = analyze_masked_sample(
            masked_sample_unbatched, patch_size, track_per_modality
        )
        results.append(result)

    return results


def _get_num_bandsets(
    modality_name: str,
    modality_spec: ModalitySpec,
    tokenization_config: TokenizationConfig | None,
) -> int:
    """Get number of bandsets, respecting tokenization config overrides."""
    if tokenization_config is not None:
        return tokenization_config.get_num_bandsets(modality_name)
    return modality_spec.num_band_sets


def estimate_tokens_per_timestep(
    training_modalities: list[str],
    sampled_hw_p: int,
    tokenization_config: TokenizationConfig | None = None,
) -> int:
    """Estimate tokens per timestep for spatiotemporal modalities."""
    tokens = 0
    for modality_name in training_modalities:
        modality_spec = Modality.get(modality_name)
        if modality_spec.is_spacetime_varying:
            num_bandsets = _get_num_bandsets(
                modality_name, modality_spec, tokenization_config
            )
            tokens += sampled_hw_p * sampled_hw_p * num_bandsets
    return tokens


def estimate_static_tokens(
    training_modalities: list[str],
    sampled_hw_p: int,
    tokenization_config: TokenizationConfig | None = None,
) -> int:
    """Estimate tokens for static/space-only modalities."""
    tokens = 0
    for modality_name in training_modalities:
        modality_spec = Modality.get(modality_name)
        num_bandsets = _get_num_bandsets(
            modality_name, modality_spec, tokenization_config
        )
        if modality_spec.is_space_only_varying:
            tokens += sampled_hw_p * sampled_hw_p * num_bandsets
        elif modality_spec.is_static_in_space_and_time:
            tokens += num_bandsets
    return tokens


def add_batch_dimension(sample: OlmoEarthSample) -> OlmoEarthSample:
    """Add a batch dimension to a sample."""
    sample_dict = {}
    for key, val in sample.as_dict().items():
        if val is not None:
            sample_dict[key] = val.unsqueeze(0)
    return OlmoEarthSample(**sample_dict)


def remove_batch_dimension(
    masked_sample: MaskedOlmoEarthSample,
) -> MaskedOlmoEarthSample:
    """Remove the batch dimension from a masked sample."""
    sample_dict = {}
    for key, val in masked_sample.as_dict().items():
        if val is not None:
            sample_dict[key] = val.squeeze(0)
    return MaskedOlmoEarthSample(**sample_dict)


def print_statistics(
    results: list[TokenRatioResult], show_per_modality: bool = False
) -> None:
    """Print statistics from the results."""
    if not results:
        print("No results to analyze.")
        return

    encoded_ratios = [r.encoded_ratio for r in results]
    decoded_ratios = [r.decoded_ratio for r in results]
    total_tokens = [r.total_tokens for r in results]

    # Compute ratios excluding missing tokens
    encoded_ratios_excl_missing = [
        r.encoded_tokens / r.non_missing_tokens if r.non_missing_tokens > 0 else 0
        for r in results
    ]
    decoded_ratios_excl_missing = [
        r.decoded_tokens / r.non_missing_tokens if r.non_missing_tokens > 0 else 0
        for r in results
    ]

    print("\n" + "=" * 60)
    print("TOKEN RATIO STATISTICS")
    print("=" * 60)

    print(f"\nNumber of samples: {len(results)}")

    print("\n--- Total Token Counts ---")
    print(f"  Mean total tokens:       {np.mean(total_tokens):.1f}")
    print(f"  Std total tokens:        {np.std(total_tokens):.1f}")
    print(f"  Min total tokens:        {np.min(total_tokens)}")
    print(f"  Max total tokens:        {np.max(total_tokens)}")

    print("\n--- Encoded Ratio (including missing as denominator) ---")
    print(
        f"  Mean:   {np.mean(encoded_ratios):.4f} ({np.mean(encoded_ratios) * 100:.2f}%)"
    )
    print(f"  Std:    {np.std(encoded_ratios):.4f}")
    print(
        f"  Min:    {np.min(encoded_ratios):.4f} ({np.min(encoded_ratios) * 100:.2f}%)"
    )
    print(
        f"  Max:    {np.max(encoded_ratios):.4f} ({np.max(encoded_ratios) * 100:.2f}%)"
    )
    print(
        f"  Median: {np.median(encoded_ratios):.4f} ({np.median(encoded_ratios) * 100:.2f}%)"
    )

    print("\n--- Decoded Ratio (including missing as denominator) ---")
    print(
        f"  Mean:   {np.mean(decoded_ratios):.4f} ({np.mean(decoded_ratios) * 100:.2f}%)"
    )
    print(f"  Std:    {np.std(decoded_ratios):.4f}")
    print(
        f"  Min:    {np.min(decoded_ratios):.4f} ({np.min(decoded_ratios) * 100:.2f}%)"
    )
    print(
        f"  Max:    {np.max(decoded_ratios):.4f} ({np.max(decoded_ratios) * 100:.2f}%)"
    )
    print(
        f"  Median: {np.median(decoded_ratios):.4f} ({np.median(decoded_ratios) * 100:.2f}%)"
    )

    print("\n--- Encoded Ratio (excluding missing tokens) ---")
    print(
        f"  Mean:   {np.mean(encoded_ratios_excl_missing):.4f} ({np.mean(encoded_ratios_excl_missing) * 100:.2f}%)"
    )
    print(f"  Std:    {np.std(encoded_ratios_excl_missing):.4f}")
    print(
        f"  Min:    {np.min(encoded_ratios_excl_missing):.4f} ({np.min(encoded_ratios_excl_missing) * 100:.2f}%)"
    )
    print(
        f"  Max:    {np.max(encoded_ratios_excl_missing):.4f} ({np.max(encoded_ratios_excl_missing) * 100:.2f}%)"
    )

    print("\n--- Decoded Ratio (excluding missing tokens) ---")
    print(
        f"  Mean:   {np.mean(decoded_ratios_excl_missing):.4f} ({np.mean(decoded_ratios_excl_missing) * 100:.2f}%)"
    )
    print(f"  Std:    {np.std(decoded_ratios_excl_missing):.4f}")
    print(
        f"  Min:    {np.min(decoded_ratios_excl_missing):.4f} ({np.min(decoded_ratios_excl_missing) * 100:.2f}%)"
    )
    print(
        f"  Max:    {np.max(decoded_ratios_excl_missing):.4f} ({np.max(decoded_ratios_excl_missing) * 100:.2f}%)"
    )

    print("\n--- Percentiles (Encoded, excluding missing) ---")
    for p in [5, 25, 50, 75, 95]:
        val = np.percentile(encoded_ratios_excl_missing, p)
        print(f"  {p}th percentile: {val:.4f} ({val * 100:.2f}%)")

    print("\n--- Percentiles (Decoded, excluding missing) ---")
    for p in [5, 25, 50, 75, 95]:
        val = np.percentile(decoded_ratios_excl_missing, p)
        print(f"  {p}th percentile: {val:.4f} ({val * 100:.2f}%)")

    # Per-modality breakdown
    if show_per_modality and results[0].per_modality_stats is not None:
        print("\n" + "=" * 60)
        print("PER-MODALITY BREAKDOWN (averaged across samples)")
        print("=" * 60)

        # Collect all modality names
        all_modalities = set()
        for r in results:
            if r.per_modality_stats:
                all_modalities.update(r.per_modality_stats.keys())

        for modality in sorted(all_modalities):
            encoded_list = []
            decoded_list = []
            target_only_list = []
            total_list = []

            for r in results:
                if r.per_modality_stats and modality in r.per_modality_stats:
                    stats = r.per_modality_stats[modality]
                    total = sum(stats.values())
                    non_missing = total - stats.get("missing", 0)
                    if non_missing > 0:
                        encoded_list.append(stats["encoded"] / non_missing)
                        decoded_list.append(stats["decoded"] / non_missing)
                        target_only_list.append(stats["target_only"] / non_missing)
                        total_list.append(total)

            if encoded_list:
                print(f"\n  {modality}:")
                print(f"    Mean tokens: {np.mean(total_list):.1f}")
                print(
                    f"    Encoded:     {np.mean(encoded_list) * 100:.1f}% (std: {np.std(encoded_list) * 100:.1f}%)"
                )
                print(
                    f"    Decoded:     {np.mean(decoded_list) * 100:.1f}% (std: {np.std(decoded_list) * 100:.1f}%)"
                )
                print(
                    f"    Target-only: {np.mean(target_only_list) * 100:.1f}% (std: {np.std(target_only_list) * 100:.1f}%)"
                )

    print("=" * 60)


def summarize_encode_modality_counts(results: list[TokenRatioResult]) -> None:
    """Distribution over samples of how many distinct modalities the encoder sees.

    "Seen by the encoder" = the modality has at least one ONLINE_ENCODER token in the
    sample. Decode-only modalities never get encoded tokens, so they are naturally
    excluded. A modality that is entirely missing in a sample also doesn't count. This
    directly answers "how often does the model encode only a single modality?".
    """
    counts = []
    per_modality_present: dict[str, int] = defaultdict(int)
    for r in results:
        if not r.per_modality_stats:
            continue
        encoded_mods = [
            m for m, s in r.per_modality_stats.items() if s.get("encoded", 0) > 0
        ]
        counts.append(len(encoded_mods))
        for m in encoded_mods:
            per_modality_present[m] += 1
    if not counts:
        print("\nNo per-modality stats available for the encode-modality histogram.")
        return

    counts_arr = np.array(counts)
    n = len(counts_arr)
    print("\n" + "=" * 60)
    print("NUMBER OF DISTINCT ENCODE MODALITIES SEEN PER SAMPLE")
    print("=" * 60)
    print(f"\n  Samples:                        {n}")
    print(f"  Mean encode modalities/sample:  {counts_arr.mean():.2f}")
    print(f"  Median:                         {int(np.median(counts_arr))}")
    print("\n  Distribution:")
    for k in range(0, int(counts_arr.max()) + 1):
        mask = counts_arr == k
        print(
            f"    {k} encode modality(ies): {mask.mean() * 100:5.1f}%  ({mask.sum()})"
        )
    print(f"\n  == 1 encode modality:  {(counts_arr == 1).mean() * 100:.1f}%")
    print(f"  <= 1 encode modality:  {(counts_arr <= 1).mean() * 100:.1f}%")
    print(f"  >= 2 encode modalities:{(counts_arr >= 2).mean() * 100:.1f}%")
    print("\n  Fraction of samples where each modality is among the encoded:")
    for m in sorted(per_modality_present):
        print(f"    {m:<28} {per_modality_present[m] / n * 100:5.1f}%")
    print("=" * 60)


def main() -> None:
    """Main entry point for the token ratio estimation script."""
    parser = argparse.ArgumentParser(
        description="Estimate token encode/decode ratios for masking strategies"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--min_patch_size",
        type=int,
        default=1,
        help="Minimum patch size",
    )
    parser.add_argument(
        "--max_patch_size",
        type=int,
        default=8,
        help="Maximum patch size",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=2250,
        help="Token budget per instance",
    )
    parser.add_argument(
        "--missing_prob",
        type=float,
        default=0.1,
        help="Probability of modality/timestep being missing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--masking_type",
        type=str,
        default="random_time_with_decode",
        help="Type of masking strategy",
    )
    parser.add_argument(
        "--encode_ratio",
        type=float,
        default=0.5,
        help="Encode ratio for masking strategy",
    )
    parser.add_argument(
        "--decode_ratio",
        type=float,
        default=0.5,
        help="Decode ratio for masking strategy",
    )
    parser.add_argument(
        "--per_modality",
        action="store_true",
        help="Show per-modality breakdown statistics",
    )
    parser.add_argument(
        "--single_bandset",
        action="store_true",
        help="Use single bandset tokenization for S2 and Landsat",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    # Build masking config based on args
    masking_config = {
        "type": args.masking_type,
        "encode_ratio": args.encode_ratio,
        "decode_ratio": args.decode_ratio,
    }

    # Add additional params for modality_cross strategies
    if "modality_cross" in args.masking_type:
        masking_config["allow_encoding_decoding_same_bandset"] = True
        masking_config["only_decode_modalities"] = [
            Modality.WORLDCOVER.name,
            Modality.SRTM.name,
            Modality.OPENSTREETMAP_RASTER.name,
            Modality.WRI_CANOPY_HEIGHT_MAP.name,
            Modality.CDL.name,
            Modality.WORLDCEREAL.name,
        ]
    elif args.masking_type in ("random_with_decode", "random_time_with_decode"):
        masking_config["only_decode_modalities"] = [
            Modality.WORLDCOVER.name,
            Modality.SRTM.name,
            Modality.OPENSTREETMAP_RASTER.name,
            Modality.WRI_CANOPY_HEIGHT_MAP.name,
            Modality.CDL.name,
            Modality.WORLDCEREAL.name,
        ]

    # Build tokenization config
    tokenization_config = None
    if args.single_bandset:
        tokenization_config = build_single_bandset_tokenization_config()

    print(f"Running with masking config: {masking_config}")
    print(f"Training modalities: {DEFAULT_TRAINING_MODALITIES}")
    if tokenization_config is not None:
        print(f"Tokenization overrides: {list(tokenization_config.overrides.keys())}")
    print(f"Sampling {args.num_samples} configurations...")

    results = estimate_token_ratios(
        num_samples=args.num_samples,
        training_modalities=DEFAULT_TRAINING_MODALITIES,
        masking_config=masking_config,
        min_patch_size=args.min_patch_size,
        max_patch_size=args.max_patch_size,
        sampled_hw_p_list=list(range(1, 13)),
        token_budget=args.token_budget,
        missing_prob=args.missing_prob,
        seed=args.seed,
        track_per_modality=True,  # needed for the encode-modality histogram (cheap)
        tokenization_config=tokenization_config,
    )

    print_statistics(results, show_per_modality=args.per_modality)
    summarize_encode_modality_counts(results)


if __name__ == "__main__":
    main()
