"""Sweep script to generate and run experiments for all shape configurations."""

import subprocess  # noqa: S404 # nosec
import sys
from pathlib import Path

from base import BASE_PREDICTOR_CONFIG, SHAPE_SWEEP_OPTIONS


def generate_shape_overrides(shape_name: str) -> list[str]:
    """Generate override arguments for a given shape configuration.

    Args:
        shape_name: Name of the shape from SHAPE_SWEEP_OPTIONS

    Returns:
        List of override strings (e.g., ["--model.encoder_config.embedding_size=384", ...])
    """
    if shape_name not in SHAPE_SWEEP_OPTIONS:
        available = ", ".join(SHAPE_SWEEP_OPTIONS.keys())
        raise ValueError(
            f"Invalid shape name: {shape_name}. Available shapes: {available}"
        )

    shape_config = SHAPE_SWEEP_OPTIONS[shape_name]

    # Encoder overrides from shape config
    overrides = [
        f"--model.encoder_config.embedding_size={shape_config['encoder_embedding_size']}",
        f"--model.encoder_config.num_heads={shape_config['encoder_num_heads']}",
        f"--model.encoder_config.depth={shape_config['encoder_depth']}",
        f"--model.encoder_config.mlp_ratio={shape_config['mlp_ratio']}",
    ]

    # Predictor overrides (fixed to BASE_PREDICTOR_CONFIG)
    overrides.extend(
        [
            f"--model.decoder_config.encoder_embedding_size={BASE_PREDICTOR_CONFIG['encoder_embedding_size']}",
            f"--model.decoder_config.decoder_embedding_size={BASE_PREDICTOR_CONFIG['decoder_embedding_size']}",
            f"--model.decoder_config.depth={BASE_PREDICTOR_CONFIG['decoder_depth']}",
            f"--model.decoder_config.num_heads={BASE_PREDICTOR_CONFIG['decoder_num_heads']}",
            f"--model.decoder_config.mlp_ratio={BASE_PREDICTOR_CONFIG['mlp_ratio']}",
        ]
    )

    return overrides


def run_sweep(
    base_script: str = "base.py",
    subcommand: str = "train",
    cluster: str = "local",
    filter_pattern: str | None = None,
    shapes: list[str] | None = None,
    extra_overrides: list[str] | None = None,
):
    """Run experiments for all shape configurations.

    Args:
        base_script: Path to the base script (e.g., "base.py")
        subcommand: Subcommand to run (e.g., "train", "launch", "dry_run")
        cluster: Cluster name (e.g., "local", "jupiter")
        filter_pattern: Optional pattern to filter shapes (e.g., "C" to only run C variants)
        shapes: Optional explicit list of shape names to run (overrides filter_pattern)
        extra_overrides: Additional override arguments to pass to all runs
    """
    script_path = Path(__file__).parent / base_script

    if shapes is None:
        shapes = list(SHAPE_SWEEP_OPTIONS.keys())
        if filter_pattern:
            shapes = [shape for shape in shapes if filter_pattern in shape]

    print(f"Running sweep over {len(shapes)} shape configurations:")
    for shape in shapes:
        print(f"  - {shape}")
    if extra_overrides:
        print("\nExtra overrides (applied to all runs):")
        for override in extra_overrides:
            print(f"  {override}")
    print()

    for shape_name in shapes:
        run_name = f"inference_shapes_{shape_name}"

        print(f"\n{'=' * 60}")
        print(f"Running: {shape_name}")
        print(f"Run name: {run_name}")
        print(f"{'=' * 60}")

        # Generate overrides for this shape
        overrides = generate_shape_overrides(shape_name)

        # Add extra overrides if provided
        if extra_overrides:
            overrides.extend(extra_overrides)

        cmd = [
            sys.executable,
            str(script_path),
            subcommand,
            run_name,
            cluster,
            *overrides,
        ]

        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd)  # noqa: S603 # nosec

        if result.returncode != 0:
            print(f"❌ Failed for {shape_name} (exit code: {result.returncode})")
            # Optionally: ask user if they want to continue
            response = input("Continue with remaining shapes? (y/n): ")
            if response.lower() != "y":
                break
        else:
            print(f"✅ Successfully completed {shape_name}")


def print_overrides(shape_name: str) -> None:
    """Print the overrides for a given shape (useful for debugging)."""
    overrides = generate_shape_overrides(shape_name)
    print(f"Overrides for {shape_name}:")
    for override in overrides:
        print(f"  {override}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep over shape configurations by generating size overrides"
    )
    parser.add_argument(
        "--script",
        default="base.py",
        help="Base script to run (default: base.py)",
    )
    parser.add_argument(
        "--subcommand",
        default="train",
        choices=["train", "launch", "dry_run"],
        help="Subcommand to run (default: train)",
    )
    parser.add_argument(
        "--cluster",
        default="local",
        help="Cluster name (default: local)",
    )
    parser.add_argument(
        "--filter",
        help="Filter shapes by pattern (e.g., 'C' to only run C variants)",
    )
    parser.add_argument(
        "--shapes",
        help="Specific shape name to run (e.g., --shapes C1_mid_depth_narrow_MLPlean). Defaults to all shapes if not specified.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print the overrides for the specified shape(s), don't run",
    )

    # Parse known args and collect remaining args as extra overrides
    args, extra_overrides = parser.parse_known_args()

    # Convert single shape to list format for internal use
    shapes_list = [args.shapes] if args.shapes else None

    if args.print_only:
        # Default to all shapes if none specified
        if shapes_list is None:
            if args.filter:
                shape_names = [
                    shape
                    for shape in SHAPE_SWEEP_OPTIONS.keys()
                    if args.filter in shape
                ]
            else:
                shape_names = list(SHAPE_SWEEP_OPTIONS.keys())
        else:
            shape_names = shapes_list

        for shape in shape_names:
            print_overrides(shape)
            if extra_overrides:
                print("Extra overrides:")
                for override in extra_overrides:
                    print(f"  {override}")
            print()
    else:
        run_sweep(
            base_script=args.script,
            subcommand=args.subcommand,
            cluster=args.cluster,
            filter_pattern=args.filter,
            shapes=shapes_list,
            extra_overrides=extra_overrides if extra_overrides else None,
        )
