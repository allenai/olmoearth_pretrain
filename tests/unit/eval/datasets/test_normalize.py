"""Unit tests for normalization functions."""

import numpy as np
import pytest

from olmoearth_pretrain.evals.datasets.normalize import (
    NormMethod,
    impute_normalization_stats,
    normalize_bands,
)


class TestImputeNormalizationStats:
    """Test impute_normalization_stats function."""

    def test_empty_imputes_returns_original(self) -> None:
        """Test that empty imputes list returns original band_info."""
        band_info = {"band1": {"mean": 0.5, "std": 0.1}}
        result = impute_normalization_stats(band_info, [])
        assert result == band_info

    def test_impute_band_already_present(self) -> None:
        """Test that imputing a band that is already present in the band info raises an error."""
        band_info = {
            "01 - Coastal aerosol": {"mean": 0.1, "std": 0.02},
            "02 - Blue": {"mean": 0.2, "std": 0.03},
        }
        imputes = [("02 - Blue", "01 - Coastal aerosol")]
        all_bands = ["01 - Coastal aerosol", "02 - Blue"]
        with pytest.raises(ValueError):
            impute_normalization_stats(band_info, imputes, all_bands)

    def test_impute_missing_band(self) -> None:
        """Test imputing a missing band from another band."""
        band_info = {
            "01 - Coastal aerosol": {"mean": 0.1, "std": 0.02},
        }
        all_bands = ["01 - Coastal aerosol", "02 - Blue"]
        imputes = [("01 - Coastal aerosol", "02 - Blue")]

        result = impute_normalization_stats(band_info, imputes, all_bands)

        assert len(result) == 2
        assert result["01 - Coastal aerosol"] == {"mean": 0.1, "std": 0.02}
        assert result["02 - Blue"] == {"mean": 0.1, "std": 0.02}  # imputed

    def test_multiple_imputes(self) -> None:
        """Test multiple impute mappings."""
        band_info = {
            "01 - Coastal aerosol": {"mean": 0.1, "std": 0.02},
            "03 - Green": {"mean": 0.3, "std": 0.04},
        }
        all_bands = ["01 - Coastal aerosol", "02 - Blue", "03 - Green", "04 - Red"]
        imputes = [
            ("01 - Coastal aerosol", "02 - Blue"),
            ("03 - Green", "04 - Red"),
        ]

        result = impute_normalization_stats(band_info, imputes, all_bands)

        assert len(result) == 4
        assert result["01 - Coastal aerosol"] == {"mean": 0.1, "std": 0.02}
        assert result["02 - Blue"] == {"mean": 0.1, "std": 0.02}  # imputed from 01
        assert result["03 - Green"] == {"mean": 0.3, "std": 0.04}
        assert result["04 - Red"] == {"mean": 0.3, "std": 0.04}  # imputed from 03

    def test_impute_order_matters(self) -> None:
        """Test that first matching impute is used when multiple imputes target same band."""
        band_info = {
            "01 - Coastal aerosol": {"mean": 0.1, "std": 0.02},
            "03 - Green": {"mean": 0.3, "std": 0.04},
        }
        all_bands = ["01 - Coastal aerosol", "02 - Blue", "03 - Green"]
        imputes = [
            ("01 - Coastal aerosol", "02 - Blue"),
            ("03 - Green", "02 - Blue"),  # This should not be used
        ]

        result = impute_normalization_stats(band_info, imputes, all_bands)

        # Should use first impute (01 -> 02)
        assert result["02 - Blue"] == {"mean": 0.1, "std": 0.02}


class TestNormalizeBands:
    """Test normalize_bands function."""

    rtol = 1e-6
    atol = 1e-6

    def setup_method(self) -> None:
        """Set up test data."""
        # Create a simple 2-channel image: shape (channels, height, width)
        self.image = np.array(
            [[[0.0, 0.5], [1.0, 0.3]], [[0.2, 0.8], [1.2, 0.4]]], dtype=np.float32
        )
        self.means = np.array([0.5, 0.6])
        self.stds = np.array([0.1, 0.2])
        self.mins = np.array([0.0, 0.0])
        self.maxs = np.array([1.0, 1.5])

    def assert_outputs_equal(self, result: np.ndarray, expected: np.ndarray) -> None:
        """Test that the result is equal to the expected output."""
        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_no_norm_returns_unchanged(self) -> None:
        """Test NO_NORM method returns image unchanged."""
        result = normalize_bands(
            self.image, self.means, self.stds, method=NormMethod.NO_NORM
        )
        np.testing.assert_array_equal(result, self.image)

    def test_standardize_method(self) -> None:
        """Test STANDARDIZE method applies z-score normalization."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(self.image, means, stds, method=NormMethod.STANDARDIZE)

        # Pre-computed: (image - means) / stds
        expected = np.array(
            [
                [[-5.0, 0.0], [5.0, -2.0]],  # Channel 0: (val - 0.5) / 0.1
                [[-2.0, 1.0], [3.0, -1.0]],  # Channel 1: (val - 0.6) / 0.2
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_no_clip_method(self) -> None:
        """Test NORM_NO_CLIP method normalizes but doesn't clip."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            self.image, means, stds, method=NormMethod.NORM_NO_CLIP
        )

        expected = np.array(
            [
                [[-2.0, 0.5], [3.0, -0.5]],  # Channel 0: (val - 0.4) / 0.2
                [[-0.5, 1.0], [2.0, 0.0]],  # Channel 1: (val - 0.4) / 0.4
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_norm_yes_clip_method(self) -> None:
        """Test NORM_YES_CLIP method normalizes and clips to [0,1]."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            self.image, means, stds, method=NormMethod.NORM_YES_CLIP
        )

        # Pre-computed: same as norm_no_clip but clipped to [0, 1]
        expected = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],  # Channel 0 clipped
                [[0.0, 1.0], [1.0, 0.0]],  # Channel 1 clipped
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_yes_clip_3_std_method(self) -> None:
        """Test NORM_YES_CLIP_3_STD method uses 3 std bounds."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            self.image, means, stds, method=NormMethod.NORM_YES_CLIP_3_STD
        )

        # Pre-computed: (image - (means - 3*stds)) / (6 * stds), clipped
        # Channel 0: bounds [0.2, 0.8], Channel 1: bounds [0.0, 1.2]
        expected = np.array(
            [
                [[0.0, 0.5], [1.0, 0.16666667]],  # Channel 0
                [[0.16666667, 0.6666667], [1.0, 0.33333333]],  # Channel 1
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_yes_clip_2_std_method(self) -> None:
        """Test NORM_YES_CLIP_2_STD method uses 2 std bounds."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            self.image, means, stds, method=NormMethod.NORM_YES_CLIP_2_STD
        )

        # Pre-computed: (image - (means - 2*stds)) / (4 * stds), clipped
        # Channel 0: bounds [0.3, 0.7], Channel 1: bounds [0.2, 1.0]
        expected = np.array(
            [
                [[0.0, 0.5], [1.0, 0.0]],  # Channel 0
                [[0.0, 0.75], [1.0, 0.25]],  # Channel 1
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_yes_clip_int_method(self) -> None:
        """Test NORM_YES_CLIP_INT method quantizes to 8-bit integers."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            self.image, means, stds, method=NormMethod.NORM_YES_CLIP_INT
        )

        # Pre-computed: normalize -> scale to 255 -> clip to uint8 -> scale back
        expected = np.array(
            [
                [[0.0, 0.49803922], [1.0, 0.0]],  # Channel 0: 127/255 ≈ 0.498
                [[0.0, 1.0], [1.0, 0.0]],  # Channel 1
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_yes_clip_3_std_int_method(self) -> None:
        """Test NORM_YES_CLIP_3_STD_INT method uses 3 std bounds with 8-bit quantization."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            self.image, means, stds, method=NormMethod.NORM_YES_CLIP_3_STD_INT
        )

        expected = np.array(
            [
                [[0.0, 0.49803922], [1.0, 0.16470588]],  # Channel 0
                [[0.16470588, 0.6666667], [1.0, 0.33333334]],  # Channel 1
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_yes_clip_2_std_int_method(self) -> None:
        """Test NORM_YES_CLIP_2_STD_INT method uses 2 std bounds with 8-bit quantization."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            self.image, means, stds, method=NormMethod.NORM_YES_CLIP_2_STD_INT
        )

        expected = np.array(
            [
                [[0.0, 0.49803922], [1.0, 0.0]],  # Channel 0
                [[0.0, 0.7490196], [1.0, 0.24705882]],  # Channel 1: 191/255, 63/255
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_yes_clip_min_max_int_method(self) -> None:
        """Test NORM_YES_CLIP_MIN_MAX_INT method uses provided mins/maxs."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        mins = self.mins.reshape(2, 1, 1)
        maxs = self.maxs.reshape(2, 1, 1)

        result = normalize_bands(
            self.image,
            means,
            stds,
            mins=mins,
            maxs=maxs,
            method=NormMethod.NORM_YES_CLIP_MIN_MAX_INT,
        )

        expected = np.array(
            [
                [[0.0, 0.49803922], [1.0, 0.29803923]],  # Channel 0: 127/255, 76/255
                [
                    [0.13333334, 0.53333336],
                    [0.8, 0.26666668],
                ],  # Channel 1: 34/255, 136/255, 204/255, 68/255
            ],
            dtype=np.float32,
        )

        self.assert_outputs_equal(result, expected)

    def test_norm_min_max_int_raises_error_when_mins_none(self) -> None:
        """Test NORM_YES_CLIP_MIN_MAX_INT raises an error when mins is None."""
        # THIS IS PRIMARILY USED by DINOV3 evals
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        maxs = self.maxs.reshape(2, 1, 1)

        with pytest.raises(ValueError):
            normalize_bands(
                self.image,
                means,
                stds,
                mins=None,
                maxs=maxs,
                method=NormMethod.NORM_YES_CLIP_MIN_MAX_INT,
            )

    def test_string_method_parameter(self) -> None:
        """Test that string method parameter is properly converted to enum."""
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result1 = normalize_bands(self.image, means, stds, method="standardize")
        result2 = normalize_bands(
            self.image, means, stds, method=NormMethod.STANDARDIZE
        )

        np.testing.assert_allclose(result1, result2)

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid NormMethod"):
            normalize_bands(self.image, self.means, self.stds, method="invalid_method")

    def test_different_image_shapes(self) -> None:
        """Test function works with different image shapes."""
        # 1D case - single channel
        image_1d = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        means_1d = np.array([0.5])
        stds_1d = np.array([0.1])

        result = normalize_bands(
            image_1d, means_1d, stds_1d, method=NormMethod.STANDARDIZE
        )
        expected = (image_1d - 0.5) / 0.1
        self.assert_outputs_equal(result, expected)

        # 3D case with multiple channels
        image_3d = np.random.rand(3, 64, 64).astype(np.float32)
        means_3d = np.array([0.5, 0.4, 0.6]).reshape(3, 1, 1)
        stds_3d = np.array([0.1, 0.15, 0.12]).reshape(3, 1, 1)

        result = normalize_bands(
            image_3d, means_3d, stds_3d, method=NormMethod.STANDARDIZE
        )
        expected = (image_3d - means_3d) / stds_3d
        self.assert_outputs_equal(result, expected)

    def test_boundary_values(self) -> None:
        """Test function handles boundary values correctly."""
        # Test with zeros
        image_zeros = np.zeros((2, 1, 1), dtype=np.float32)
        means = self.means.reshape(2, 1, 1)
        stds = self.stds.reshape(2, 1, 1)
        result = normalize_bands(
            image_zeros, means, stds, method=NormMethod.STANDARDIZE
        )
        expected = (0.0 - means) / stds
        self.assert_outputs_equal(result, expected)

    def test_clipping_behavior(self) -> None:
        """Test that clipping methods properly constrain values."""
        # Create image with extreme values
        extreme_image = np.array([[[-10.0, 10.0]]], dtype=np.float32)
        means = np.array([0.0]).reshape(1, 1, 1)
        stds = np.array([1.0]).reshape(1, 1, 1)

        # Test clipping methods constrain to [0, 1]
        for method in [
            NormMethod.NORM_YES_CLIP,
            NormMethod.NORM_YES_CLIP_3_STD,
            NormMethod.NORM_YES_CLIP_2_STD,
        ]:
            result = normalize_bands(extreme_image, means, stds, method=method)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

        # Test non-clipping method doesn't constrain
        result = normalize_bands(
            extreme_image, means, stds, method=NormMethod.NORM_NO_CLIP
        )
        assert np.any(result < 0.0) or np.any(result > 1.0)
