r"""CLI for baking precomputed embedding products into rslearn eval datasets.

Usage:
    python -m olmoearth_pretrain.evals.embedding_materializer \\
        --dataset_path /weka/dfive-default/olmoearth/eval_datasets/<name>/ \\
        --products aef,tessera [--year 2019] [--overwrite] [--workers 8]
"""

import argparse
import logging
import sys

from upath import UPath

from olmoearth_pretrain.evals.embedding_materializer.fetchers import (
    AEFFetcher,
    EmbeddingFetcher,
    TesseraFetcher,
)
from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    materialize_product,
    write_manifest,
)

logger = logging.getLogger(__name__)

PRODUCT_NAMES = ["aef", "tessera"]


def build_fetcher(product_name: str, args: argparse.Namespace) -> EmbeddingFetcher:
    """Construct the fetcher for a product name.

    Args:
        product_name: one of PRODUCT_NAMES.
        args: parsed CLI arguments.

    Returns:
        the configured EmbeddingFetcher.

    Raises:
        ValueError: if the product name is unknown.
    """
    if product_name == "aef":
        return AEFFetcher(metadata_cache_dir=args.aef_cache_dir)
    if product_name == "tessera":
        return TesseraFetcher()
    raise ValueError(
        f"Unknown product '{product_name}'; expected one of {PRODUCT_NAMES}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: argument list; defaults to sys.argv[1:].

    Returns:
        the parsed argparse Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Materialize precomputed embedding products (AlphaEarth/GSE, "
        "Tessera) as raster layers in an rslearn eval dataset."
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
        "--year",
        type=int,
        default=None,
        help="Fixed product year for all windows. If unset, each window uses "
        "the year of its time-range midpoint.",
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
        help="Number of threads fetching/writing windows concurrently.",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Optional comma-separated window groups to restrict to.",
    )
    parser.add_argument(
        "--aef_cache_dir",
        default=None,
        help="Directory for caching the AEF spatial index CSV.",
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

    for product_name in product_names:
        fetcher = build_fetcher(product_name, args)
        manifest = materialize_product(
            dataset_path=dataset_path,
            fetcher=fetcher,
            product_name=product_name,
            year=args.year,
            overwrite=args.overwrite,
            workers=args.workers,
            groups=groups,
            log_every=args.log_every,
            cli_args=vars(args),
        )
        write_manifest(dataset_path, product_name, manifest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
