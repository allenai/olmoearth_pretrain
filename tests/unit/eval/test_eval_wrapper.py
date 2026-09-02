"""Unit tests for eval wrapper."""

from types import SimpleNamespace

import pytest
import torch

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.eval_wrapper import EvalWrapper, OlmoEarthEvalWrapper
from olmoearth_pretrain.nn.pooling import PoolingType


class TestExtractCenterToken:
    """Tests for _extract_center_token static method."""

    def test_odd_spatial_dims(self) -> None:
        """Get center token for odd dimensions."""
        B, H, W, D = 2, 7, 7, 64
        x = torch.randn(B, H, W, D)
        result = EvalWrapper._extract_center_token(x)
        assert result.shape == (B, D)
        assert torch.equal(result, x[:, 3, 3, :])

    def test_even_spatial_dims(self) -> None:
        """Get bottom-right of center for even dimensions."""
        B, H, W, D = 2, 8, 8, 64
        x = torch.randn(B, H, W, D)
        result = EvalWrapper._extract_center_token(x)
        assert result.shape == (B, D)
        assert torch.equal(result, x[:, 4, 4, :])

    def test_non_square(self) -> None:
        """Correct center for non-square dimensions."""
        B, H, W, D = 3, 4, 6, 32
        x = torch.randn(B, H, W, D)
        result = EvalWrapper._extract_center_token(x)
        assert result.shape == (B, D)
        assert torch.equal(result, x[:, 2, 3, :])


class TestPoolRegisters:
    """Tests for the register-bottleneck eval embedding."""

    GRID = (4, 4)
    DIM = 8

    def _wrapper(
        self, task_type: TaskType, use_center_token: bool
    ) -> OlmoEarthEvalWrapper:
        return OlmoEarthEvalWrapper(
            model=SimpleNamespace(),  # type: ignore[arg-type]
            task_type=task_type,
            patch_size=1,
            pooling_type=PoolingType.MEAN,
            use_center_token=use_center_token,
        )

    def _registers(self, batch: int = 2) -> torch.Tensor:
        n_h, n_w = self.GRID
        return torch.randn(batch, n_h, n_w, self.DIM)

    def test_center_token_takes_center_cell(self) -> None:
        """Center-pixel classification keeps only the center register, not the mean."""
        wrapper = self._wrapper(TaskType.CLASSIFICATION, use_center_token=True)
        registers = self._registers()
        out = wrapper._pool_registers({"registers": registers})
        n_h, n_w = self.GRID
        expected = registers[:, n_h // 2, n_w // 2, :]
        assert out.shape == (registers.shape[0], self.DIM)
        assert torch.equal(out, expected)

    def test_classification_without_center_token_pools(self) -> None:
        """Window-level classification still averages the whole grid."""
        wrapper = self._wrapper(TaskType.CLASSIFICATION, use_center_token=False)
        registers = self._registers()
        out = wrapper._pool_registers({"registers": registers})
        assert out.shape == (registers.shape[0], self.DIM)
        assert torch.allclose(out, registers.mean(dim=(1, 2)))

    def test_segmentation_keeps_grid(self) -> None:
        """Dense tasks get the coarse spatial map."""
        wrapper = self._wrapper(TaskType.SEGMENTATION, use_center_token=False)
        registers = self._registers()
        out = wrapper._pool_registers({"registers": registers})
        assert out.shape == (registers.shape[0], *self.GRID, self.DIM)

    def test_center_token_rejected_for_segmentation(self) -> None:
        """The existing guard still forbids center-token dense tasks."""
        with pytest.raises(ValueError, match="use_center_token"):
            self._wrapper(TaskType.SEGMENTATION, use_center_token=True)


class TestPoolProjectedRegisters:
    """Tests for probing the detached low-dim register projection."""

    GRID = (4, 4)
    DIM = 16
    PROJ_DIM = 8

    def _wrapper(
        self,
        task_type: TaskType = TaskType.SEGMENTATION,
        eval_projection_dim: int | None = None,
        eval_on_projected_registers: bool = True,
        eval_on_encoder_tokens: bool = False,
    ) -> OlmoEarthEvalWrapper:
        return OlmoEarthEvalWrapper(
            model=SimpleNamespace(),  # type: ignore[arg-type]
            task_type=task_type,
            patch_size=1,
            pooling_type=PoolingType.MEAN,
            eval_on_projected_registers=eval_on_projected_registers,
            eval_on_encoder_tokens=eval_on_encoder_tokens,
            eval_projection_dim=eval_projection_dim,
        )

    def _encoder_output(self, batch: int = 2) -> dict[str, torch.Tensor]:
        n_h, n_w = self.GRID
        return {
            "registers": torch.randn(batch, n_h, n_w, self.DIM),
            "projected_registers": torch.randn(batch, n_h, n_w, self.PROJ_DIM),
        }

    def test_projected_grid_for_segmentation(self) -> None:
        """Dense tasks get the projected grid at the student width."""
        wrapper = self._wrapper()
        out = wrapper._pool_registers(self._encoder_output())
        assert out.shape == (2, *self.GRID, self.PROJ_DIM)

    def test_projection_dim_takes_matryoshka_prefix(self) -> None:
        """eval_projection_dim slices the first d dims of the student."""
        wrapper = self._wrapper(eval_projection_dim=4)
        encoder_output = self._encoder_output()
        out = wrapper._pool_registers(encoder_output)
        assert out.shape == (2, *self.GRID, 4)
        expected = encoder_output["projected_registers"][..., :4]
        assert torch.equal(out, expected)

    def test_missing_projection_raises(self) -> None:
        """A model without the student cannot be probed on it."""
        wrapper = self._wrapper()
        with pytest.raises(ValueError, match="register_projection_dims"):
            wrapper._pool_registers({"registers": torch.randn(2, 16, self.DIM)})

    def test_projected_and_encoder_tokens_mutually_exclusive(self) -> None:
        """Both opt-outs at once are contradictory."""
        with pytest.raises(ValueError, match="mutually"):
            self._wrapper(eval_on_encoder_tokens=True)

    def test_projection_dim_requires_projected_flag(self) -> None:
        """A prefix width without the projected-registers flag is rejected."""
        with pytest.raises(ValueError, match="eval_on_projected_registers"):
            self._wrapper(eval_projection_dim=4, eval_on_projected_registers=False)
