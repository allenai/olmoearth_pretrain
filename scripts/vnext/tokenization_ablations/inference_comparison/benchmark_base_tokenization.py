"""Benchmark base tokenization (default model config).

This script benchmarks inference throughput using the default tokenization
strategy, which uses the standard bandset groupings for each modality.

Configuration:
- Image size: 16x16
- Timestamps: 12
- Modality: Sentinel-2 L2A only
- Uses default model config builder (no custom tokenization)

Usage:
    python benchmark_base_tokenization.py benchmark run_name local
"""

import sys
from pathlib import Path

# Add official directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "official"))

from script import build_common_components

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.inference_benchmarking.data_models import RunParams
from olmoearth_pretrain.inference_benchmarking.run_throughput_benchmark import (
    ThroughputBenchmarkRunnerConfig,
)
from olmoearth_pretrain.internal.experiment import CommonComponents, main


def build_inference_benchmarking_config(
    common: CommonComponents,
) -> ThroughputBenchmarkRunnerConfig:
    """Build benchmark config using default model config builder."""
    default_run_params = RunParams(
        model_size="base_shallow_decoder",
        use_s1=False,
        use_s2=True,
        use_landsat=False,
        image_size=16,
        patch_size=1,
        num_timesteps=12,
        batch_size=128,
        bf16=True,
        wandb_enabled=True,
        profiler_enabled=False,
        benchmark_interval_s=180,
        min_batches_per_interval=10,
    )

    return ThroughputBenchmarkRunnerConfig(
        default_run_params=default_run_params,
        training_modalities=[Modality.SENTINEL2_L2A.name],
        sweep_dict={"batch_size": [8, 16, 32, 64, 128]},
        sweep_group_name="base_tokenization_benchmark",
        work_dir=Path("./benchmark_work_dir"),
        # Uses default model_config_builder (None)
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        inference_benchmarking_config_builder=build_inference_benchmarking_config,
    )
