"""Quadtree partitioner for APT adaptive patchification.

Implements hierarchical subdivision of images into patches of varying sizes
based on content complexity.
"""

import logging
from dataclasses import dataclass

import numpy as np

from olmoearth_pretrain.nn.apt.scorers import Scorer

logger = logging.getLogger(__name__)


@dataclass
class PatchDescriptor:
    """Descriptor for an adaptive patch.

    Attributes:
        x: Top-left x coordinate in base patch units
        y: Top-left y coordinate in base patch units
        scale: Scale index (0 = base, 1 = 2x, 2 = 4x, etc.)
        size: Patch size in pixels
        score: Complexity score from scorer
        timestep: Timestep index for temporal data (None for static)
    """

    x: int
    y: int
    scale: int
    size: int
    score: float
    timestep: int | None = None

    @property
    def x_pixels(self) -> int:
        """Get x coordinate in pixels."""
        return self.x * self.size

    @property
    def y_pixels(self) -> int:
        """Get y coordinate in pixels."""
        return self.y * self.size

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "scale": self.scale,
            "size": self.size,
            "score": self.score,
            "timestep": self.timestep,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PatchDescriptor":
        """Create from dictionary."""
        return cls(**d)


class QuadtreePartitioner:
    """Quadtree-based adaptive image partitioner.

    Partitions an image into patches of varying sizes using a quadtree structure.
    Complex regions get smaller patches, homogeneous regions get larger patches.
    """

    def __init__(
        self,
        scorer: Scorer,
        base_patch_size: int = 1,
        num_scales: int = 4,
        thresholds: list[float]
        | None = None,  # How do I appropriately set these thresholds? Need to do some experimentation and visualization to determine the best thresholds.
    ):
        """Initialize the partitioner.

        Args:
            scorer: Scorer instance for computing patch complexity
            base_patch_size: Smallest patch size (base of quadtree)
            num_scales: Number of scales (1 = base only, 2 = base + 2x, etc.)
            thresholds: Entropy thresholds per scale (from coarsest to base).
                       If score < threshold, keep large patch.
                       Length should be num_scales - 1 (no threshold for base).
                       Default: [5.5, 4.0] for 3 scales.
        """
        self.scorer = scorer
        self.base_patch_size = base_patch_size
        self.num_scales = num_scales

        if thresholds is None:
            raise ValueError("Thresholds must be provided")
        if len(thresholds) != num_scales - 1:
            raise ValueError(
                f"Expected {num_scales - 1} thresholds, got {len(thresholds)}"
            )
        self.thresholds = thresholds

        # Compute patch sizes for each scale
        self.patch_sizes = [base_patch_size * (2**i) for i in range(num_scales)]

    def partition(
        self,
        image: np.ndarray,
        timestep: int | None = None,
        log_stats: bool = False,
    ) -> list[PatchDescriptor]:
        """Partition an image into adaptive patches.

        Args:
            image: Image with shape [H, W, C]
            timestep: Optional timestep index for temporal data
            log_stats: Whether to log token count and reduction ratio

        Returns:
            List of PatchDescriptor objects describing the partition
        """
        h, w, c = image.shape
        max_patch_size = self.patch_sizes[-1]

        # Ensure image dimensions are divisible by max patch size
        if h % max_patch_size != 0 or w % max_patch_size != 0:
            logger.warning(
                f"Image size ({h}, {w}) not divisible by max patch size {max_patch_size}. "
                f"Some edge regions may be excluded."
            )

        patches: list[PatchDescriptor] = []

        # Start recursive partitioning from coarsest scale
        h_coarse = h // max_patch_size
        w_coarse = w // max_patch_size

        for yi in range(h_coarse):
            for xi in range(w_coarse):
                self._partition_recursive(
                    image=image,
                    x=xi * max_patch_size,
                    y=yi * max_patch_size,
                    scale=self.num_scales - 1,
                    patches=patches,
                    timestep=timestep,
                )

        if log_stats:
            self._log_partition_stats(patches, (h, w))

        return patches

    def _log_partition_stats(
        self, patches: list[PatchDescriptor], image_shape: tuple[int, int]
    ) -> None:
        """Log token count and reduction statistics.

        Args:
            patches: List of patch descriptors
            image_shape: (H, W) of the image
        """
        token_count = self.get_token_count(patches)
        reduction_ratio = self.get_reduction_ratio(patches, image_shape)
        h, w = image_shape
        uniform_tokens = (h // self.base_patch_size) * (w // self.base_patch_size)

        # Count patches by scale
        scale_counts = {}
        for p in patches:
            scale_counts[p.scale] = scale_counts.get(p.scale, 0) + 1

        logger.info(
            f"APT Partition Stats: "
            f"tokens={token_count} (uniform would be {uniform_tokens}), "
            f"reduction={1 - reduction_ratio:.1%}, "
            f"by_scale={scale_counts}"
        )

    def _partition_recursive(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        scale: int,
        patches: list[PatchDescriptor],
        timestep: int | None,
    ) -> None:
        """Recursively partition a region.

        Args:
            image: Full image
            x: Top-left x coordinate in pixels
            y: Top-left y coordinate in pixels
            scale: Current scale index
            patches: List to append patches to
            timestep: Optional timestep index
        """
        patch_size = self.patch_sizes[scale]

        # Extract patch region
        patch = image[y : y + patch_size, x : x + patch_size, :]

        # Compute complexity score
        score = self.scorer.compute_score(patch)

        # Base case: at finest scale, always accept
        if scale == 0:
            patches.append(
                PatchDescriptor(
                    x=x // self.base_patch_size,
                    y=y // self.base_patch_size,
                    scale=scale,
                    size=patch_size,
                    score=score,
                    timestep=timestep,
                )
            )
            return

        # Decision: keep large patch or subdivide?
        threshold = self.thresholds[scale - 1]

        if score < threshold:
            # Low complexity: keep as large patch
            patches.append(
                PatchDescriptor(
                    x=x // self.base_patch_size,
                    y=y // self.base_patch_size,
                    scale=scale,
                    size=patch_size,
                    score=score,
                    timestep=timestep,
                )
            )
        else:
            # High complexity: subdivide into 4 children
            child_size = self.patch_sizes[scale - 1]
            for dy in [0, child_size]:
                for dx in [0, child_size]:
                    self._partition_recursive(
                        image=image,
                        x=x + dx,
                        y=y + dy,
                        scale=scale - 1,
                        patches=patches,
                        timestep=timestep,
                    )

    def partition_temporal(
        self,
        image: np.ndarray,
        log_stats: bool = False,
    ) -> list[list[PatchDescriptor]]:
        """Partition a temporal image stack.

        Args:
            image: Image with shape [H, W, T, C]
            log_stats: Whether to log token count and reduction ratio

        Returns:
            List of patch descriptor lists, one per timestep
        """
        h, w, t, c = image.shape
        all_patches = []

        for ti in range(t):
            frame = image[:, :, ti, :]
            patches = self.partition(frame, timestep=ti, log_stats=False)
            all_patches.append(patches)

        if log_stats:
            self._log_temporal_partition_stats(all_patches, (h, w), t)

        return all_patches

    def _log_temporal_partition_stats(
        self,
        all_patches: list[list[PatchDescriptor]],
        image_shape: tuple[int, int],
        num_timesteps: int,
    ) -> None:
        """Log token count and reduction statistics for temporal data.

        Args:
            all_patches: List of patch descriptor lists per timestep
            image_shape: (H, W) of each frame
            num_timesteps: Number of timesteps
        """
        total_tokens = sum(len(patches) for patches in all_patches)
        h, w = image_shape
        uniform_tokens_per_frame = (h // self.base_patch_size) * (
            w // self.base_patch_size
        )
        uniform_tokens_total = uniform_tokens_per_frame * num_timesteps
        reduction_ratio = total_tokens / uniform_tokens_total if uniform_tokens_total > 0 else 1.0

        # Aggregate scale counts across all timesteps
        scale_counts = {}
        for patches in all_patches:
            for p in patches:
                scale_counts[p.scale] = scale_counts.get(p.scale, 0) + 1

        avg_tokens_per_frame = total_tokens / num_timesteps if num_timesteps > 0 else 0

        logger.info(
            f"APT Temporal Partition Stats: "
            f"total_tokens={total_tokens}, "
            f"avg_per_frame={avg_tokens_per_frame:.1f} (uniform would be {uniform_tokens_per_frame}), "
            f"reduction={1 - reduction_ratio:.1%}, "
            f"timesteps={num_timesteps}, "
            f"by_scale={scale_counts}"
        )

    def get_token_count(self, patches: list[PatchDescriptor]) -> int:
        """Get total token count from patch descriptors.

        Each patch becomes one token regardless of size.
        """
        return len(patches)

    def get_reduction_ratio(
        self, patches: list[PatchDescriptor], image_shape: tuple[int, int]
    ) -> float:
        """Compute token reduction ratio compared to uniform base patches.

        Args:
            patches: List of patch descriptors
            image_shape: (H, W) of the image

        Returns:
            Reduction ratio (1.0 = no reduction, 0.5 = 50% reduction)
        """
        h, w = image_shape
        uniform_tokens = (h // self.base_patch_size) * (w // self.base_patch_size)
        adaptive_tokens = len(patches)

        if uniform_tokens == 0:
            return 1.0

        return adaptive_tokens / uniform_tokens

    def patches_to_base_grid(
        self,
        patches: list[PatchDescriptor],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Create a base-grid mapping from patches.

        Useful for dense prediction tasks that need rectangular feature maps.

        Args:
            patches: List of patch descriptors
            image_shape: (H, W) of the image

        Returns:
            Array of shape [H//base_patch_size, W//base_patch_size] where
            each cell contains the index of the patch covering it.
        """
        h, w = image_shape
        grid_h = h // self.base_patch_size
        grid_w = w // self.base_patch_size

        grid = np.zeros((grid_h, grid_w), dtype=np.int32)

        for idx, patch in enumerate(patches):
            # Compute coverage in base patch units
            size_in_base = patch.size // self.base_patch_size
            for dy in range(size_in_base):
                for dx in range(size_in_base):
                    gy = patch.y + dy
                    gx = patch.x + dx
                    if 0 <= gy < grid_h and 0 <= gx < grid_w:
                        grid[gy, gx] = idx

        return grid
