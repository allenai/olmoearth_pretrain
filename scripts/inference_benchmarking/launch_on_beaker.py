"""Script to run locally to launch throughput benchmarking task in Beaker."""

from helios.inference_benchmarking.constants import SWEEPS
from helios.inference_benchmarking.run_throughput_benchmark import (
    ThroughputBenchmarkRunnerConfig,
)
from helios.internal.common import build_common_components
from helios.internal.experiment import main


def inference_benchmarking_config_builder() -> ThroughputBenchmarkRunnerConfig:
    """Build the inference benchmarking configuration."""
    return ThroughputBenchmarkRunnerConfig(
        sweep_dict=SWEEPS["batch"],
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        inference_benchmarking_config_builder=inference_benchmarking_config_builder,
    )
