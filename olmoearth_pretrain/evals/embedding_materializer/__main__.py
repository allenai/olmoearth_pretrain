r"""CLI for baking precomputed embedding products into rslearn eval datasets.

Usage:
    python -m olmoearth_pretrain.evals.embedding_materializer \\
        --dataset_path /weka/dfive-default/olmoearth/eval_datasets/<name>/ \\
        --products aef [--overwrite] [--workers 8]
"""

import argparse
import logging
import sys

from upath import UPath

from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    DEFAULT_METADATA_CACHE_DIR,
    PRODUCTS,
    materialize_product,
    write_manifest,
)

logger = logging.getLogger(__name__)

PRODUCT_NAMES = sorted(PRODUCTS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: argument list; defaults to sys.argv[1:].

    Returns:
        the parsed argparse Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Materialize precomputed embedding products (AlphaEarth/GSE) "
        "as raster layers in an rslearn eval dataset."
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to the rslearn eval dataset root.",
    )
    parser.add_argument(
        "--products",
        required=True,
        help=f"Comma-separated product names to materialize ({PRODUCT_NAMES}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite layers that already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of threads materializing windows concurrently.",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Optional comma-separated window groups to restrict to.",
    )
    parser.add_argument(
        "--metadata_cache_dir",
        default=DEFAULT_METADATA_CACHE_DIR,
        help="Where the data source caches its spatial index; relative paths "
        "resolve against the dataset root.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=25,
        help="Log progress every N windows.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the embedding materializer CLI.

    Args:
        argv: argument list; defaults to sys.argv[1:].

    Returns:
        process exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    product_names = [p.strip() for p in args.products.split(",") if p.strip()]
    if not product_names:
        logger.error("No products specified.")
        return 1

    dataset_path = UPath(args.dataset_path)
    groups = args.groups.split(",") if args.groups else None

    unknown = sorted(set(product_names) - set(PRODUCTS))
    if unknown:
        logger.error(f"Unknown product(s) {unknown}; expected one of {PRODUCT_NAMES}")
        return 1

    for product_name in product_names:
        manifest = materialize_product(
            dataset_path=dataset_path,
            product=PRODUCTS[product_name],
            overwrite=args.overwrite,
            workers=args.workers,
            groups=groups,
            log_every=args.log_every,
            metadata_cache_dir=args.metadata_cache_dir,
            cli_args=vars(args),
        )
        write_manifest(dataset_path, product_name, manifest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
