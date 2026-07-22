"""Materialize AEF + Tessera embedding layers for the AEF supplemental eval datasets.

Runs the embedding materializer over each of the AlphaEarth-supplemental
evaluation datasets registered in studio_ingest's registry.json (paths are
resolved from the registry, not hardcoded). For every window this bakes the
product's embedding raster in as a layer (gse / tessera), skipping windows
whose layer already exists, so the script is resumable and safe to re-run.

Coverage expectations: AEF is global for 2018-2024, so most windows should
succeed. Tessera is global for 2024 only (US+EU back to 2017), so the African
datasets will record coverage gaps for other label years — gaps are logged and
listed in each dataset's embedding_materializer_manifest_<product>.json, and
the corresponding windows simply lack the layer.

After materializing, to make the datasets AEF/Tessera-evaluable you still need
to (a) add the gse/tessera raster layer entry to each dataset's model.yaml and
(b) list the modality in the dataset's registry modalities; see
docs/PrecomputedEmbeddingEvals.md.

Example:
    python scripts/tools/materialize_aef_supplemental_embeddings.py
    python scripts/tools/materialize_aef_supplemental_embeddings.py \
        --datasets descals,glance --products aef --workers 16
"""

import argparse
import logging

from olmoearth_pretrain.evals.embedding_materializer.fetchers import (
    AEFFetcher,
    EmbeddingFetcher,
    TesseraFetcher,
)
from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    materialize_product,
    write_manifest,
)
from olmoearth_pretrain.evals.studio_ingest.registry import get_dataset_entry

logger = logging.getLogger(__name__)

# The AlphaEarth-supplemental evaluation datasets (arXiv:2507.22291), as
# ingested in evals/studio_ingest/registry.json.
AEF_SUPPLEMENTAL_DATASETS = [
    "africa_crop_mask",
    "canada_crops_coarse",
    "canada_crops_fine",
    "descals",
    "ethiopia_crops",
    "glance",
    "lcmap_lu",
    "us_trees",
]


def build_fetcher(product_name: str) -> EmbeddingFetcher:
    """Build the fetcher for a product name ("aef" or "tessera")."""
    if product_name == "aef":
        return AEFFetcher()
    if product_name == "tessera":
        return TesseraFetcher()
    raise ValueError(f"Unknown embedding product '{product_name}'")


def main() -> None:
    """Materialize the requested products into each supplemental dataset."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(AEF_SUPPLEMENTAL_DATASETS),
        help="Comma-separated registry dataset names to materialize.",
    )
    parser.add_argument(
        "--products",
        type=str,
        default="aef,tessera",
        help="Comma-separated products to materialize (aef, tessera).",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent fetch threads."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite layers that already exist.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Fixed annual product layer to fetch. Default: derive per window "
            "from its time-range midpoint (usually what you want, since the "
            "supplemental datasets span multiple label years)."
        ),
    )
    args = parser.parse_args()

    dataset_names = args.datasets.split(",")
    product_names = args.products.split(",")
    # Fetchers are reused across datasets so e.g. the AEF tile index is only
    # loaded once.
    fetchers = {name: build_fetcher(name) for name in product_names}

    summaries = []
    for dataset_name in dataset_names:
        entry = get_dataset_entry(dataset_name)
        if not entry.weka_path:
            raise ValueError(f"Registry entry '{dataset_name}' has no weka_path.")
        for product_name, fetcher in fetchers.items():
            logger.info(f"=== {dataset_name} / {product_name} ===")
            manifest = materialize_product(
                entry.weka_path,
                fetcher,
                product_name=product_name,
                year=args.year,
                overwrite=args.overwrite,
                workers=args.workers,
                cli_args=vars(args),
            )
            write_manifest(entry.weka_path, product_name, manifest)
            summaries.append((dataset_name, product_name, manifest))

    logger.info("=== Summary ===")
    for dataset_name, product_name, manifest in summaries:
        logger.info(
            f"{dataset_name} / {product_name}: "
            f"written={manifest.get('num_windows_written')} "
            f"skipped_existing={manifest.get('num_windows_skipped_existing')} "
            f"coverage_gaps={manifest.get('num_coverage_gaps')}"
        )


if __name__ == "__main__":
    main()
