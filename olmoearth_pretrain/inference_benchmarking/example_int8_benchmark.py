"""Example script demonstrating INT8 quantization for inference benchmarking.

This script shows several common INT8 benchmarking scenarios:
1. Basic INT8 vs BF16 comparison
2. W8A8 vs W8 weight-only comparison
3. Batch size sweep with INT8 enabled
4. Compile mode comparison

Usage:
    python -m olmoearth_pretrain.inference_benchmarking.example_int8_benchmark
"""

from pathlib import Path

from olmoearth_pretrain.inference_benchmarking.data_models import RunParams
from olmoearth_pretrain.inference_benchmarking.run_throughput_benchmark import (
    ThroughputBenchmarkRunnerConfig,
)


def example_int8_vs_bf16():
    """Compare INT8 quantized inference vs standard BF16 inference."""
    print("=" * 80)
    print("Example 1: INT8 vs BF16 Comparison")
    print("=" * 80)

    config = ThroughputBenchmarkRunnerConfig(
        default_run_params=RunParams(
            model_size="base",
            batch_size=128,
            image_size=64,
            patch_size=4,
            num_timesteps=12,
            use_s2=True,
            bf16=True,
        ),
        sweep_keys=["int8_enabled"],
        sweep_group_name="int8_vs_bf16_comparison",
        work_dir=Path("./benchmark_work_dir"),
    )

    runner = config.build()
    print(
        f"\nRunning {len(runner.build_sweep_run_params())} benchmark configurations..."
    )
    runner.run()


def example_int8_mode_comparison():
    """Compare W8A8 (weight+activation) vs W8 (weight-only) quantization."""
    print("=" * 80)
    print("Example 2: W8A8 vs W8 Weight-Only Comparison")
    print("=" * 80)

    config = ThroughputBenchmarkRunnerConfig(
        default_run_params=RunParams(
            model_size="base",
            batch_size=128,
            int8_enabled=True,  # Enable INT8
            bf16=True,
        ),
        sweep_keys=["int8_mode"],  # Sweeps over ["w8a8", "w8"]
        sweep_group_name="w8a8_vs_w8_comparison",
        work_dir=Path("./benchmark_work_dir"),
    )

    runner = config.build()
    print(
        f"\nRunning {len(runner.build_sweep_run_params())} benchmark configurations..."
    )
    runner.run()


def example_batch_size_sweep_int8():
    """Sweep batch sizes with INT8 enabled to find optimal throughput."""
    print("=" * 80)
    print("Example 3: Batch Size Sweep with INT8")
    print("=" * 80)

    config = ThroughputBenchmarkRunnerConfig(
        default_run_params=RunParams(
            model_size="base",
            int8_enabled=True,
            int8_mode="w8a8",
            compile_mode="max-autotune",
            bf16=True,
        ),
        sweep_keys=["batch"],  # Sweeps batch_size
        sweep_group_name="int8_batch_sweep",
        work_dir=Path("./benchmark_work_dir"),
    )

    runner = config.build()
    print(
        f"\nRunning {len(runner.build_sweep_run_params())} benchmark configurations..."
    )
    print("Note: This will test many batch sizes to find optimal throughput")
    runner.run()


def example_compile_mode_comparison():
    """Compare different TorchInductor compile modes."""
    print("=" * 80)
    print("Example 4: Compile Mode Comparison")
    print("=" * 80)

    config = ThroughputBenchmarkRunnerConfig(
        default_run_params=RunParams(
            model_size="base",
            batch_size=128,
            int8_enabled=True,
            int8_mode="w8a8",
            bf16=True,
        ),
        sweep_keys=["compile_mode"],  # Sweeps over compile strategies
        sweep_group_name="compile_mode_comparison",
        work_dir=Path("./benchmark_work_dir"),
    )

    runner = config.build()
    print(
        f"\nRunning {len(runner.build_sweep_run_params())} benchmark configurations..."
    )
    print("Comparing: default, reduce-overhead, max-autotune")
    runner.run()


def example_cross_product_sweep():
    """Run a cross-product sweep over multiple dimensions."""
    print("=" * 80)
    print("Example 5: Cross-Product Sweep (INT8 × Batch Size)")
    print("=" * 80)

    config = ThroughputBenchmarkRunnerConfig(
        default_run_params=RunParams(
            model_size="base",
            bf16=True,
        ),
        sweep_dict={
            "int8_enabled": [True, False],
            "batch_size": [64, 128, 256],
        },
        cross_product_sweep=True,  # All combinations: 2 × 3 = 6 runs
        sweep_group_name="int8_batch_cross_product",
        work_dir=Path("./benchmark_work_dir"),
    )

    runner = config.build()
    print(
        f"\nRunning {len(runner.build_sweep_run_params())} benchmark configurations..."
    )
    print("Testing all combinations of INT8 on/off × 3 batch sizes")
    runner.run()


def example_with_smoothquant():
    """Example using SmoothQuant for models with activation outliers."""
    print("=" * 80)
    print("Example 6: INT8 with SmoothQuant")
    print("=" * 80)

    config = ThroughputBenchmarkRunnerConfig(
        default_run_params=RunParams(
            model_size="base",
            batch_size=128,
            int8_enabled=True,
            int8_mode="w8a8",
            int8_smoothquant=True,  # Enable SmoothQuant outlier smoothing
            compile_mode="max-autotune",
            bf16=True,
        ),
        sweep_keys=["int8_smoothquant"],  # Compare with/without SmoothQuant
        sweep_group_name="smoothquant_comparison",
        work_dir=Path("./benchmark_work_dir"),
    )

    runner = config.build()
    print(
        f"\nRunning {len(runner.build_sweep_run_params())} benchmark configurations..."
    )
    print("Note: Use SmoothQuant if standard INT8 causes >1% accuracy drop")
    runner.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run INT8 benchmarking examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Which example to run (1-6)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples (WARNING: This will take a long time!)",
    )

    args = parser.parse_args()

    examples = {
        1: ("INT8 vs BF16", example_int8_vs_bf16),
        2: ("W8A8 vs W8", example_int8_mode_comparison),
        3: ("Batch Size Sweep", example_batch_size_sweep_int8),
        4: ("Compile Modes", example_compile_mode_comparison),
        5: ("Cross-Product Sweep", example_cross_product_sweep),
        6: ("SmoothQuant", example_with_smoothquant),
    }

    if args.all:
        print("\n🚀 Running ALL examples (this will take a while)...\n")
        for i, (name, func) in examples.items():
            print(f"\n{'=' * 80}")
            print(f"Running Example {i}: {name}")
            print(f"{'=' * 80}\n")
            func()
    elif args.example:
        name, func = examples[args.example]
        print(f"\n🚀 Running Example {args.example}: {name}\n")
        func()
    else:
        print("\n📚 Available INT8 Benchmarking Examples:\n")
        for i, (name, _) in examples.items():
            print(f"  {i}. {name}")
        print("\nUsage:")
        print(f"  python -m {__name__} --example N")
        print(f"  python -m {__name__} --all")
        print("\nTo run a specific example:")
        print(f"  python -m {__name__} --example 1")
