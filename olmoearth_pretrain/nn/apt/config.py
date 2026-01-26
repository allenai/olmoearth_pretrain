"""Configuration for APT adaptive patchification.

Provides dataclass-based configuration for APT components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from olmoearth_pretrain.config import Config


class ScorerType(Enum):
    """Type of scorer for patch complexity."""

    ENTROPY = "entropy"
    LAPLACIAN = "laplacian"


@dataclass
class APTScorerConfig(Config):
    """Configuration for APT patch scorers.

    Attributes:
        scorer_type: Type of scorer to use
        num_bins: Number of bins for entropy histogram (entropy scorer only)
        bands: Band indices to use for scoring
        normalize: Whether to normalize values before scoring
    """

    scorer_type: ScorerType = ScorerType.ENTROPY
    num_bins: int = 32
    bands: tuple[int, ...] = (0, 1, 2)  # RGB-equivalent for S2
    normalize: bool = True

    def build(self):
        """Build the scorer instance."""
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        from olmoearth_pretrain.nn.apt.scorers import EntropyScorer, LaplacianScorer

        normalizer = Normalizer(Strategy.PREDEFINED) if self.normalize else None

        if self.scorer_type == ScorerType.ENTROPY:
            return EntropyScorer(
                num_bins=self.num_bins,
                bands=self.bands,
                normalizer=normalizer,
                modality=None,  # Set per-modality at runtime
            )
        elif self.scorer_type == ScorerType.LAPLACIAN:
            return LaplacianScorer(
                bands=self.bands,
                normalizer=normalizer,
                modality=None,
            )
        else:
            raise ValueError(f"Unknown scorer type: {self.scorer_type}")


@dataclass
class APTPartitionerConfig(Config):
    """Configuration for APT quadtree partitioner.

    Attributes:
        base_patch_size: Smallest patch size in pixels
        num_scales: Number of patch size scales (1 = base only, 2 = base + 2x, etc.)
        thresholds: Entropy thresholds per scale (from coarsest to base).
                   Length should be num_scales - 1.
    """

    base_patch_size: int = 16
    num_scales: int = 3
    thresholds: list[float] = field(default_factory=lambda: [5.5, 4.0])

    def validate(self) -> None:
        """Validate configuration."""
        if len(self.thresholds) != self.num_scales - 1:
            raise ValueError(
                f"Expected {self.num_scales - 1} thresholds, got {len(self.thresholds)}"
            )
        if self.base_patch_size <= 0:
            raise ValueError("base_patch_size must be positive")
        if self.num_scales < 1:
            raise ValueError("num_scales must be at least 1")

    def build(self, scorer):
        """Build the partitioner instance.

        Args:
            scorer: Scorer instance for computing patch complexity
        """
        from olmoearth_pretrain.nn.apt.partitioner import QuadtreePartitioner

        self.validate()
        return QuadtreePartitioner(
            scorer=scorer,
            base_patch_size=self.base_patch_size,
            num_scales=self.num_scales,
            thresholds=self.thresholds,
        )


@dataclass
class APTEmbedConfig(Config):
    """Configuration for APT adaptive patch embedding.

    Attributes:
        embedding_size: Size of output embeddings
        base_patch_size: Base patch size in pixels
        num_scales: Number of patch size scales
    """

    embedding_size: int = 768
    base_patch_size: int = 16
    num_scales: int = 3

    def build(self, base_patch_embed):
        """Build the adaptive embed instance.

        Args:
            base_patch_embed: Pretrained FlexiPatchEmbed for base patches
        """
        from olmoearth_pretrain.nn.apt.adaptive_patch_embed import AdaptivePatchEmbed

        return AdaptivePatchEmbed(
            base_patch_embed=base_patch_embed,
            num_scales=self.num_scales,
            embedding_size=self.embedding_size,
            base_patch_size=self.base_patch_size,
        )


@dataclass
class APTTransformConfig(Config):
    """Configuration for APT score transform (dataloader integration).

    Attributes:
        modality_name: Modality to apply APT to
        scorer_config: Scorer configuration
        partitioner_config: Partitioner configuration
    """

    modality_name: str = "sentinel2_l2a"
    scorer_config: APTScorerConfig = field(default_factory=APTScorerConfig)
    partitioner_config: APTPartitionerConfig = field(
        default_factory=APTPartitionerConfig
    )

    def build(self):
        """Build the APT score transform."""
        from olmoearth_pretrain.data.apt_transform import APTScoreTransform

        return APTScoreTransform(
            modality_name=self.modality_name,
            base_patch_size=self.partitioner_config.base_patch_size,
            num_scales=self.partitioner_config.num_scales,
            thresholds=self.partitioner_config.thresholds,
            num_bins=self.scorer_config.num_bins,
            bands=self.scorer_config.bands,
            normalize=self.scorer_config.normalize,
        )


@dataclass
class APTMaskingConfig(Config):
    """Configuration for APT-aware masking strategy.

    Attributes:
        base_strategy_type: Type of base masking strategy
        apt_modalities: Modalities to apply APT to
        encode_ratio: Ratio of tokens to encode
        decode_ratio: Ratio of tokens to decode
        base_patch_size: Base patch size
    """

    base_strategy_type: str = "random"
    apt_modalities: list[str] = field(default_factory=lambda: ["sentinel2_l2a"])
    encode_ratio: float = 0.5
    decode_ratio: float = 0.5
    base_patch_size: int = 16
    base_strategy_kwargs: dict = field(default_factory=dict)

    def build(self):
        """Build the adaptive masking strategy."""
        from olmoearth_pretrain.nn.apt.masking import APTAwareMaskingConfig

        config = APTAwareMaskingConfig(
            base_strategy_type=self.base_strategy_type,
            apt_modalities=self.apt_modalities,
            base_patch_size=self.base_patch_size,
            encode_ratio=self.encode_ratio,
            decode_ratio=self.decode_ratio,
            base_strategy_kwargs=self.base_strategy_kwargs,
        )
        return config.build()


@dataclass
class APTConfig(Config):
    """Complete configuration for APT system.

    Combines scorer, partitioner, embedding, and masking configurations.

    Attributes:
        scorer: Scorer configuration
        partitioner: Partitioner configuration
        embed: Embedding configuration
        transform: Transform configuration
        masking: Masking configuration
        apt_modalities: List of modalities to apply APT to
        enabled: Whether APT is enabled
    """

    scorer: APTScorerConfig = field(default_factory=APTScorerConfig)
    partitioner: APTPartitionerConfig = field(default_factory=APTPartitionerConfig)
    embed: APTEmbedConfig = field(default_factory=APTEmbedConfig)
    transform: APTTransformConfig = field(default_factory=APTTransformConfig)
    masking: APTMaskingConfig = field(default_factory=APTMaskingConfig)
    apt_modalities: list[str] = field(default_factory=lambda: ["sentinel2_l2a"])
    enabled: bool = True

    def validate(self) -> None:
        """Validate the complete configuration."""
        self.partitioner.validate()

        # Ensure consistency
        if self.partitioner.base_patch_size != self.embed.base_patch_size:
            raise ValueError(
                "partitioner.base_patch_size must equal embed.base_patch_size"
            )
        if self.partitioner.num_scales != self.embed.num_scales:
            raise ValueError("partitioner.num_scales must equal embed.num_scales")

    def build_components(self, base_patch_embed=None) -> dict[str, Any]:
        """Build all APT components.

        Args:
            base_patch_embed: Optional pretrained FlexiPatchEmbed

        Returns:
            Dict with scorer, partitioner, embed (if base_patch_embed provided),
            transform, and masking components
        """
        self.validate()

        scorer = self.scorer.build()
        partitioner = self.partitioner.build(scorer)
        transform = self.transform.build()
        masking = self.masking.build()

        components = {
            "scorer": scorer,
            "partitioner": partitioner,
            "transform": transform,
            "masking": masking,
        }

        if base_patch_embed is not None:
            components["embed"] = self.embed.build(base_patch_embed)

        return components

    @classmethod
    def default_s2_config(cls) -> "APTConfig":
        """Get default configuration for Sentinel-2.

        Returns:
            APTConfig configured for S2 optical imagery
        """
        return cls(
            scorer=APTScorerConfig(
                scorer_type=ScorerType.ENTROPY,
                num_bins=32,
                bands=(0, 1, 2),  # B02, B03, B04 (RGB-equivalent)
                normalize=True,
            ),
            partitioner=APTPartitionerConfig(
                base_patch_size=16,
                num_scales=3,  # 16, 32, 64
                thresholds=[5.5, 4.0],
            ),
            embed=APTEmbedConfig(
                embedding_size=768,
                base_patch_size=16,
                num_scales=3,
            ),
            transform=APTTransformConfig(modality_name="sentinel2_l2a"),
            masking=APTMaskingConfig(
                apt_modalities=["sentinel2_l2a"],
            ),
            apt_modalities=["sentinel2_l2a"],
            enabled=True,
        )
