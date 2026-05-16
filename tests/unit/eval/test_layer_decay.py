"""Tests for layer-wise learning rate decay optimizer."""

import torch
import torch.nn as nn

from olmoearth_pretrain.evals.finetune.layer_decay import (
    build_layer_decay_optimizer,
    get_layer_id,
)


class TestGetLayerId:
    """Tests for get_layer_id parameter name mapping."""

    def test_head_params_get_full_lr_layer(self) -> None:
        """Head params should map to num_layers (full LR)."""
        assert get_layer_id("_head.weight", num_layers=12) == 12
        assert get_layer_id("_head.bias", num_layers=12) == 12

    def test_wrapper_params_get_full_lr_layer(self) -> None:
        """Wrapper params should map to num_layers (full LR)."""
        assert get_layer_id("wrapper.some.param", num_layers=12) == 12

    def test_patch_embeddings_map_to_layer_0(self) -> None:
        """Patch embeddings should map to layer 0."""
        assert get_layer_id("backbone.patch_embeddings.weight", num_layers=12) == 0
        assert (
            get_layer_id(
                "backbone.patch_embeddings.per_modality.sentinel2.weight",
                num_layers=12,
            )
            == 0
        )

    def test_composite_encodings_map_to_layer_0(self) -> None:
        """Composite encodings should map to layer 0 (low LR)."""
        assert (
            get_layer_id(
                "backbone.composite_encodings.per_modality_channel_embeddings.sentinel2_l2a",
                num_layers=12,
            )
            == 0
        )
        assert (
            get_layer_id(
                "backbone.composite_encodings.pos_embed",
                num_layers=12,
            )
            == 0
        )

    def test_blocks_map_to_their_index(self) -> None:
        """Encoder blocks should map to their block index."""
        assert get_layer_id("backbone.blocks.0.attn.qkv.weight", num_layers=12) == 0
        assert get_layer_id("backbone.blocks.5.mlp.fc1.weight", num_layers=12) == 5
        assert get_layer_id("backbone.blocks.11.norm.weight", num_layers=12) == 11

    def test_other_backbone_params_get_full_lr(self) -> None:
        """Non-block, non-patch backbone params should get full LR."""
        assert get_layer_id("backbone.norm.weight", num_layers=12) == 12

    def test_num_layers_4(self) -> None:
        """Layer mapping should work with num_layers=4 (nano)."""
        assert get_layer_id("backbone.blocks.3.attn.qkv.weight", num_layers=4) == 3
        assert get_layer_id("_head.weight", num_layers=4) == 4


class _FakeBackbone(nn.Module):
    def __init__(self, num_blocks: int = 4):
        super().__init__()
        self.patch_embeddings = nn.Linear(3, 8)
        self.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(8)


class _FakeModel(nn.Module):
    def __init__(self, num_blocks: int = 4):
        super().__init__()
        self.backbone = _FakeBackbone(num_blocks)
        self._head = nn.Linear(8, 2)


class TestBuildLayerDecayOptimizer:
    """Tests for build_layer_decay_optimizer."""

    def test_creates_correct_number_of_groups(self) -> None:
        """Should create num_layers+1 param groups."""
        model = _FakeModel(num_blocks=4)
        opt = build_layer_decay_optimizer(
            model, lr=1e-3, layer_decay_rate=0.5, num_layers=4
        )
        # layers 0..4 = 5 groups
        assert len(opt.param_groups) == 5

    def test_deepest_layer_has_base_lr(self) -> None:
        """Last group (head) should have the full base LR."""
        model = _FakeModel(num_blocks=4)
        opt = build_layer_decay_optimizer(
            model, lr=1e-3, layer_decay_rate=0.5, num_layers=4
        )
        last_group = opt.param_groups[-1]
        assert abs(last_group["lr"] - 1e-3) < 1e-10

    def test_shallowest_layer_has_decayed_lr(self) -> None:
        """First group (layer 0) should have lr * decay^num_layers."""
        model = _FakeModel(num_blocks=4)
        opt = build_layer_decay_optimizer(
            model, lr=1e-3, layer_decay_rate=0.5, num_layers=4
        )
        first_group = opt.param_groups[0]
        expected_lr = 1e-3 * (0.5**4)
        assert abs(first_group["lr"] - expected_lr) < 1e-10

    def test_all_params_included(self) -> None:
        """Every model parameter should appear in exactly one group."""
        model = _FakeModel(num_blocks=4)
        opt = build_layer_decay_optimizer(
            model, lr=1e-3, layer_decay_rate=0.5, num_layers=4
        )
        total_opt_params = sum(len(g["params"]) for g in opt.param_groups)
        total_model_params = sum(1 for _ in model.parameters())
        assert total_opt_params == total_model_params

    def test_lr_monotonically_increases(self) -> None:
        """LR should increase from shallow to deep layers."""
        model = _FakeModel(num_blocks=4)
        opt = build_layer_decay_optimizer(
            model, lr=1e-3, layer_decay_rate=0.65, num_layers=4
        )
        lrs = [g["lr"] for g in opt.param_groups]
        for i in range(len(lrs) - 1):
            assert lrs[i] <= lrs[i + 1]

    def test_can_step(self) -> None:
        """Optimizer should be able to complete a step without errors."""
        model = _FakeModel(num_blocks=4)
        opt = build_layer_decay_optimizer(
            model, lr=1e-3, layer_decay_rate=0.65, num_layers=4
        )
        x = torch.randn(2, 3)
        out = model._head(model.backbone.blocks[0](model.backbone.patch_embeddings(x)))
        out.sum().backward()
        opt.step()
