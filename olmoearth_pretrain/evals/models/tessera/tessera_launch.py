"""Tessera model launch script for evaluation."""

from olmoearth_pretrain.evals.models.registry import (
    BaselineModelName,
    make_registered_build_model_config,
)

build_model_config = make_registered_build_model_config(BaselineModelName.TESSERA)
