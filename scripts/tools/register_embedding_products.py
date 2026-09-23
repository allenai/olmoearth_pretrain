"""Register baked embedding products in the eval registry so evals can use them.

Run on a Weka-mounted machine after a bake (the embedding materializer for
AEF, tessera_v2_export.py for Tessera v2), then commit registry.json:

    python scripts/tools/register_embedding_products.py --dry_run
    python scripts/tools/register_embedding_products.py \
        --datasets lcmap_lu --products aef

For each (dataset, product) pair whose bake manifest shows a finished bake with
no failures and at least --min_coverage of the windows, this:

1. makes sure the dataset's config.json declares the layer (the materializer
   already declares AEF's; Tessera v2's is written directly, so it is added
   here);
2. records the product under the registry entry's ``embedding_products``.
   Eval jobs read the git-tracked registry, never the manifest: the
   precomputed baseline gets its model.yaml input added at load time, and
   every model on the dataset skips windows whose product layer is not
   completed on disk, so all models are scored on the same windows. Because
   that check reads the dataset at load time, a layer deleted after
   registration (e.g. by scan_embedding_nans.py) is excluded without
   re-registering;
3. re-stamps ``config_json_sha256``, since step 1 or the bake itself changed
   config.json and eval jobs verify it.

Datasets registered by the older wire_embedding_modalities.py carry the
product in ``modalities`` and as a model.yaml input. Registering them moves
the product to ``embedding_products`` and deletes the model.yaml input from
the copy eval jobs read. Commit that model.yaml along with the registry.

Coverage matters as much as completion: a product scored on part of a dataset
cannot be compared with models scored on all of it, and excluding the
uncovered windows would shrink every model's test set. A product below
--min_coverage is left unregistered; Tessera covers 8% of ethiopia_crops.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import yaml
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.evals.embedding_materializer.materialize import (
    CONFIG_BACKUP_NAME,
    PRODUCTS,
)
from olmoearth_pretrain.evals.studio_ingest.provenance import sha256_of_file
from olmoearth_pretrain.evals.studio_ingest.registry import Registry
from olmoearth_pretrain.evals.studio_ingest.schema import (
    EmbeddingProductRecord,
    EvalDatasetEntry,
)
from olmoearth_pretrain.internal.all_evals import AEF_SUPPLEMENTAL_DATASETS

logger = logging.getLogger(__name__)

# Product name -> the modality whose name is the layer name. The materializer's
# products, plus tessera_v2: no v2 product is published, so our own inference
# run (evals/datasets/tessera_v2_export.py) bakes it and writes a manifest in
# the materializer's shape.
PRODUCT_TO_MODALITY: dict[str, ModalitySpec] = {
    **{name: product.modality for name, product in PRODUCTS.items()},
    "tessera_v2": Modality.TESSERA_V2,
}

# Fraction of a dataset's windows that must carry the layer before the product
# is registered there.
DEFAULT_MIN_COVERAGE = 0.99


def manifest_path(weka_path: str, product: str) -> UPath:
    """Return the bake manifest path for one (dataset, product)."""
    return UPath(weka_path) / f"embedding_materializer_manifest_{product}.json"


def load_manifest(weka_path: str, product: str) -> dict[str, Any] | None:
    """Load a bake manifest, or None if the bake has not finished."""
    path = manifest_path(weka_path, product)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def bake_is_complete(
    manifest: dict[str, Any] | None, min_coverage: float
) -> tuple[bool, str]:
    """Report whether a bake is finished, failure-free, and well-covered.

    Args:
        manifest: the bake manifest, or None if there is none yet.
        min_coverage: minimum fraction of windows that must carry the layer.

    Returns:
        (ready, reason) — reason describes the manifest state either way.
    """
    if manifest is None:
        return False, "no manifest yet (the bake has not finished)"
    written = manifest.get("num_windows_written", 0)
    skipped = manifest.get("num_windows_skipped_existing", 0)
    failed = manifest.get("num_windows_failed", 0)
    gaps = manifest.get("num_coverage_gaps", 0)
    no_year = manifest.get("num_windows_without_year", 0)
    have = written + skipped
    total = have + gaps + failed + no_year
    coverage = have / total if total else 0.0
    state = (
        f"written={written} skipped={skipped} gaps={gaps} failed={failed} "
        f"coverage={coverage:.1%}"
    )
    if failed:
        return False, f"{state} — re-run the bake to retry failures"
    if have == 0:
        return False, f"{state} — nothing baked"
    if coverage < min_coverage:
        return False, (
            f"{state} — below --min_coverage {min_coverage:.1%}; excluding the "
            "uncovered windows would shrink every model's test set"
        )
    return True, state


def config_layer_block(modality: ModalitySpec) -> dict[str, Any]:
    """Build the config.json entry for a layer written directly (no data source)."""
    return {
        "type": "raster",
        "band_sets": [
            {"bands": list(modality.band_order), "dtype": "float32"},
        ],
    }


def add_config_layer(config: dict[str, Any], modality: ModalitySpec) -> bool:
    """Declare the modality's raster layer in a parsed config.json, in place.

    Args:
        config: the parsed rslearn dataset config.
        modality: the modality whose name is the layer name.

    Returns:
        True if the config was modified, False if the layer already existed.

    Raises:
        ValueError: if the config has no "layers" mapping.
    """
    layers = config.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("config.json has no 'layers' mapping")
    if modality.name in layers:
        return False
    layers[modality.name] = config_layer_block(modality)
    return True


def patch_config_json(weka_path: str, modality: ModalitySpec, dry_run: bool) -> None:
    """Declare the modality's layer in the dataset folder's config.json.

    Backs up the pre-edit file once. The backup matters because ingest
    overwrites config.json from the source dataset (_try_copy_config_json),
    which would silently drop the layer.
    """
    config_json = UPath(weka_path) / "config.json"
    with config_json.open() as f:
        config = json.load(f)
    if not add_config_layer(config, modality):
        return
    if dry_run:
        logger.info("    config.json: would add '%s' layer", modality.name)
        return
    backup = UPath(weka_path) / CONFIG_BACKUP_NAME
    if not backup.exists():
        with config_json.open("rb") as src, backup.open("wb") as dst:
            dst.write(src.read())
    with config_json.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    logger.info("    config.json: added '%s' layer", modality.name)


def remove_model_yaml_input(text: str, modality: ModalitySpec) -> str | None:
    """Delete an input from a model.yaml, preserving the rest of the file.

    The edit is textual so the file's comments and layout survive, which a
    parse/re-dump round trip would not; the result is parsed back and checked.

    Args:
        text: the model.yaml contents.
        modality: the modality whose input to delete.

    Returns:
        the updated contents, or None if the file has no such input.

    Raises:
        ValueError: if the input cannot be located unambiguously, or deleting
            it changes anything but that input.
    """
    before = yaml.safe_load(text)["data"]["init_args"]["inputs"]
    if modality.name not in before:
        return None

    lines = text.splitlines()
    starts = [
        idx for idx, line in enumerate(lines) if line.strip() == f"{modality.name}:"
    ]
    if len(starts) != 1:
        raise ValueError(
            f"expected exactly one '{modality.name}:' line, found {len(starts)}; "
            "edit this model.yaml by hand"
        )
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent
    ):
        end += 1
    result = "\n".join(lines[:start] + lines[end:])
    result += "\n" if text.endswith("\n") else ""

    after = yaml.safe_load(result)["data"]["init_args"]["inputs"]
    if after != {k: v for k, v in before.items() if k != modality.name}:
        raise ValueError(
            f"deleting the '{modality.name}' input changed other inputs; edit "
            "this model.yaml by hand"
        )
    return result


def strip_model_yaml(
    entry: EvalDatasetEntry, modality: ModalitySpec, dry_run: bool
) -> None:
    """Delete a legacy product input from the model.yaml eval jobs read."""
    path = UPath(entry.model_yaml_path)
    updated = remove_model_yaml_input(path.read_text(), modality)
    if updated is None:
        return
    if dry_run:
        logger.info(
            "    model.yaml: would delete '%s' input in %s", modality.name, path
        )
        return
    path.write_text(updated)
    logger.info("    model.yaml: deleted '%s' input in %s", modality.name, path)


def register_product(entry: EvalDatasetEntry, product: str, dry_run: bool) -> None:
    """Record a live product on a registry entry (steps 1 and 2), in place."""
    modality = PRODUCT_TO_MODALITY[product]
    patch_config_json(entry.weka_path, modality, dry_run)
    strip_model_yaml(entry, modality, dry_run)

    record = EmbeddingProductRecord(product=product)
    if entry.embedding_products.get(modality.name) == record and (
        modality.name not in entry.modalities
    ):
        logger.info("    registry: '%s' already registered", modality.name)
        return
    logger.info(
        "    registry: %s '%s'",
        "would register" if dry_run else "registered",
        modality.name,
    )
    if not dry_run:
        entry.embedding_products[modality.name] = record
        entry.modalities = [m for m in entry.modalities if m != modality.name]


def restamp_config_json(entry: EvalDatasetEntry, dry_run: bool) -> None:
    """Set config_json_sha256 to the dataset folder's current config.json (step 3)."""
    sha = sha256_of_file(UPath(entry.weka_path) / "config.json")
    if entry.config_json_sha256 == sha:
        return
    logger.info("  config_json_sha256: %s %s", "would set" if dry_run else "set", sha)
    if not dry_run:
        entry.config_json_sha256 = sha


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(AEF_SUPPLEMENTAL_DATASETS),
        help="Comma-separated registry dataset names to register.",
    )
    parser.add_argument(
        "--products",
        default=",".join(PRODUCT_TO_MODALITY),
        help=f"Comma-separated products: {sorted(PRODUCT_TO_MODALITY)}.",
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=(
            "Minimum fraction of a dataset's windows that must carry the layer "
            "before the product is registered there. Default "
            f"{DEFAULT_MIN_COVERAGE:.0%}."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Register the requested products on the requested datasets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    products = [p.strip() for p in args.products.split(",") if p.strip()]
    unknown = sorted(set(products) - set(PRODUCT_TO_MODALITY))
    if unknown:
        raise SystemExit(
            f"Unknown product(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(sorted(PRODUCT_TO_MODALITY))}."
        )
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    registry = Registry.load()
    missing = sorted(set(dataset_names) - set(registry.datasets))
    if missing:
        raise SystemExit(f"Not in the registry: {', '.join(missing)}")

    not_live: list[str] = []
    for dataset in dataset_names:
        entry = registry.datasets[dataset]
        logger.info("=== %s ===", dataset)
        if not UPath(entry.weka_path).exists():
            logger.warning(
                "  %s not found — skipping (is Weka mounted?)", entry.weka_path
            )
            continue

        registered_any = False
        for product in products:
            manifest = load_manifest(entry.weka_path, product)
            ready, reason = bake_is_complete(manifest, args.min_coverage)
            logger.info("  %s: %s", product, reason)
            if ready:
                register_product(entry, product, args.dry_run)
                registered_any = True
            else:
                not_live.append(f"{dataset}/{product} ({reason})")
        # Only re-stamp where a bake is being registered: elsewhere a changed
        # config.json is drift that eval jobs should keep failing on.
        if registered_any:
            restamp_config_json(entry, args.dry_run)

    if args.dry_run:
        logger.info("[dry_run] not writing the registry.")
    else:
        registry.save()

    if not_live:
        logger.info("Not registered: %s", "; ".join(not_live))
        logger.info("Re-run this script for those once their bake finishes.")
    logger.info(
        "Commit registry.json and any model.yaml under data/rslearn_dataset_configs/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
