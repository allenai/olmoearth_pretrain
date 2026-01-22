"""Benchmark all-bands-in-single-token tokenization.

This script benchmarks inference throughput using a custom tokenization
strategy where all Sentinel-2 bands are grouped into a single token.

Configuration:
- Image size: 16x16
- Timestamps: 12
- Modality: Sentinel-2 L2A only
- Custom tokenization: all 12 bands in a single token (vs default 3 bandsets)

Usage:
    python benchmark_all_bands_tokenization.py benchmark run_name local
"""

import sys
from pathlib import Path
from typing import Any

# Add official directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "official"))

from script import build_common_components

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.inference_benchmarking.data_models import RunParams
from olmoearth_pretrain.inference_benchmarking.run_throughput_benchmark import (
    ThroughputBenchmarkRunnerConfig,
)
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import (
    ModalityTokenization,
    TokenizationBandSet,
    TokenizationConfig,
)

# All Sentinel-2 L2A bands in a single token
SENTINEL2_ALL_BANDS_SINGLE_TOKEN = ModalityTokenization(
    band_groups=[
        TokenizationBandSet(
            bands=[
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
            ]
        ),
    ]
)


def build_model_config_with_all_bands_tokenization(
    run_params: RunParams, training_modalities: list[str]
) -> Any:
    """Build model config with all Sentinel-2 bands in a single token.

    Args:
        run_params: The run parameters containing model_size.
        training_modalities: List of modality names to support.

    Returns:
        A LatentMIMConfig with custom tokenization.
    """
    model_size = MODEL_SIZE_ARGS[run_params.model_size]

    tokenization_config = TokenizationConfig(
        overrides={
            Modality.SENTINEL2_L2A.name: SENTINEL2_ALL_BANDS_SINGLE_TOKEN,
        }
    )

    encoder_config = EncoderConfig(
        embedding_size=int(model_size["encoder_embedding_size"]),
        num_heads=int(model_size["encoder_num_heads"]),
        depth=int(model_size["encoder_depth"]),
        mlp_ratio=float(model_size["mlp_ratio"]),
        supported_modality_names=training_modalities,
        tokenization_config=tokenization_config,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=int(model_size["encoder_embedding_size"]),
        decoder_embedding_size=int(model_size["decoder_embedding_size"]),
        depth=int(model_size["decoder_depth"]),
        mlp_ratio=float(model_size["mlp_ratio"]),
        num_heads=int(model_size["decoder_num_heads"]),
        supported_modality_names=training_modalities,
        max_sequence_length=12,
        tokenization_config=tokenization_config,
    )

    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_inference_benchmarking_config(
    common: CommonComponents,
) -> ThroughputBenchmarkRunnerConfig:
    """Build benchmark config with all-bands tokenization."""
    default_run_params = RunParams(
        model_size="base_shallow_decoder",
        use_s1=False,
        use_s2=True,
        use_landsat=False,
        image_size=16,
        patch_size=4,
        num_timesteps=12,
        batch_size=128,
        bf16=True,
        wandb_enabled=False,  # Disable wandb for local testing
        profiler_enabled=False,
        benchmark_interval_s=30,  # Shorter interval for testing
        min_batches_per_interval=5,
    )

    return ThroughputBenchmarkRunnerConfig(
        default_run_params=default_run_params,
        training_modalities=[Modality.SENTINEL2_L2A.name],
        sweep_keys=["batch_size"],
        sweep_group_name="all_bands_tokenization_benchmark",
        work_dir=Path("./benchmark_work_dir"),
        # model_config_builder is passed separately to main()
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        inference_benchmarking_config_builder=build_inference_benchmarking_config,
        benchmark_model_config_builder=build_model_config_with_all_bands_tokenization,
    )
