"""Scorers for APT adaptive patchification.

These scorers compute complexity/compressibility measures for image patches,
used to decide whether a patch should be kept large or subdivided.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np

from olmoearth_pretrain.data.constants import ModalitySpec
from olmoearth_pretrain.data.normalize import Normalizer

logger = logging.getLogger(__name__)


class Scorer(ABC):
    """Abstract base class for patch complexity scorers."""

    @abstractmethod
    def compute_score(self, patch: np.ndarray) -> float:
        """Compute complexity score for a patch.

        Higher score = more complex = should use smaller patches.
        Lower score = more homogeneous = can use larger patches.

        Args:
            patch: Image patch with shape [H, W, C]

        Returns:
            Complexity score (higher = more complex)
        """
        raise NotImplementedError

    def compute_scores_grid(
        self,
        image: np.ndarray,
        patch_size: int,
    ) -> np.ndarray:
        """Compute scores for a grid of patches.

        Args:
            image: Full image with shape [H, W, C] or [H, W, T, C]
            patch_size: Size of patches to score

        Returns:
            Grid of scores with shape [H//patch_size, W//patch_size]
            or [H//patch_size, W//patch_size, T] for temporal data
        """
        has_time = image.ndim == 4
        if has_time:
            h, w, t, c = image.shape
        else:
            h, w, c = image.shape
            t = 1
            image = image[:, :, np.newaxis, :]

        h_patches = h // patch_size
        w_patches = w // patch_size

        scores = np.zeros((h_patches, w_patches, t), dtype=np.float32)

        for ti in range(t):
            for hi in range(h_patches):
                for wi in range(w_patches):
                    patch = image[
                        hi * patch_size : (hi + 1) * patch_size,
                        wi * patch_size : (wi + 1) * patch_size,
                        ti,
                        :,
                    ]
                    scores[hi, wi, ti] = self.compute_score(patch)

        if not has_time or t == 1:
            return scores[:, :, 0]
        return scores


class EntropyScorer(Scorer):
    """Compute patch complexity using Shannon entropy on RGB bands.

    Uses the helios Normalizer to normalize values to [0, 1] before
    binning for entropy computation.
    """

    def __init__(
        self,
        num_bins: int = 32,  # TODO: where did this come from?
        bands: tuple[int, ...] = (
            0,
            1,
            2,
        ),  # TODO: I think this might be wrong and we should get the band order using constants instead of hardcoding
        normalizer: Normalizer | None = None,
        modality: ModalitySpec | None = None,
    ):
        """Initialize the entropy scorer.

        Args:
            num_bins: Number of bins for histogram computation
            bands: Indices of bands to use for entropy (default: RGB-equivalent B02, B03, B04)
            normalizer: Helios Normalizer for preprocessing
            modality: ModalitySpec for the modality being scored
        """
        self.num_bins = num_bins
        self.bands = bands
        self.normalizer = normalizer
        self.modality = modality

    def compute_score(self, patch: np.ndarray) -> float:
        """Compute entropy score for a patch.

        Args:
            patch: Image patch with shape [H, W, C]

        Returns:
            Average entropy across selected bands (higher = more complex)
        """
        # Step 1: Normalize using helios Normalizer (maps to [0, 1])
        if self.normalizer is not None and self.modality is not None:
            patch = self.normalizer.normalize(self.modality, patch)

        # Step 2: Extract selected bands (default: RGB-equivalent)
        selected = patch[..., list(self.bands)]  # [H, W, len(bands)]
        flat = selected.reshape(-1, len(self.bands))

        # Step 3: Per-channel entropy on normalized values
        entropies = []
        for c in range(len(self.bands)):
            # Clip to [0, 1] after normalization (handles outliers)
            channel = np.clip(flat[:, c], 0, 1)
            hist, _ = np.histogram(channel, bins=self.num_bins, range=(0, 1))

            # Avoid division by zero
            total = hist.sum()
            if total == 0:
                entropies.append(0.0)
                continue

            p = hist / total
            p = p[p > 0]  # avoid log(0)
            entropy = -(p * np.log2(p)).sum()
            entropies.append(entropy)

        return float(np.mean(entropies))


# TODO: Very experimental, may not be useful
class LaplacianScorer(Scorer):
    """Compute patch complexity using Laplacian edge detection.

    Higher Laplacian magnitude indicates more edges/detail.
    """

    def __init__(
        self,
        bands: tuple[int, ...] = (0, 1, 2),
        normalizer: Normalizer | None = None,
        modality: ModalitySpec | None = None,
    ):
        """Initialize the Laplacian scorer.

        Args:
            bands: Indices of bands to use for edge detection
            normalizer: Helios Normalizer for preprocessing
            modality: ModalitySpec for the modality being scored
        """
        self.bands = bands
        self.normalizer = normalizer
        self.modality = modality

        # Laplacian kernel
        self.kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution without scipy dependency."""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        h, w = image.shape

        # Pad image
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")

        # Convolve
        output = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                output[i, j] = np.sum(padded[i : i + kh, j : j + kw] * kernel)

        return output

    def compute_score(self, patch: np.ndarray) -> float:
        """Compute Laplacian score for a patch.

        Args:
            patch: Image patch with shape [H, W, C]

        Returns:
            Mean absolute Laplacian magnitude (higher = more edges)
        """
        # Step 1: Normalize using helios Normalizer
        if self.normalizer is not None and self.modality is not None:
            patch = self.normalizer.normalize(self.modality, patch)

        # Step 2: Extract selected bands
        selected = patch[..., list(self.bands)]  # [H, W, len(bands)]

        # Step 3: Compute Laplacian for each band
        magnitudes = []
        for c in range(len(self.bands)):
            channel = np.clip(selected[..., c], 0, 1)
            laplacian = self._convolve2d(channel, self.kernel)
            magnitudes.append(np.mean(np.abs(laplacian)))

        return float(np.mean(magnitudes))
