"""Wire a materialized embedding product into eval datasets so evals can read it.

The embedding materializer bakes rasters into windows and marks the layers
completed, but it does not touch any config. Three more things must be true
before a precomputed baseline (``--model=aef`` / ``--model=tessera_precomputed``)
can run on a dataset:

1. the dataset folder's ``config.json`` declares the raster layer (rslearn
   looks up ``dataset.layers[<name>]`` when reading, so a missing entry is a
   KeyError);
2. the dataset's ``model.yaml`` declares a matching input;
3. the registry entry lists the modality in ``modalities``, which is what
   ``supported_modalities`` — and therefore the sweep's task gating — reads.

This script does all three, idempotently, for each (dataset, product) pair.
It must run on a Weka-mounted machine: config.json can only live in the
dataset folder, and step 3 is gated on the materializer manifest.

The model.yaml input is written with ``required: false`` unless --required is
passed. That default is the safe one for a bake that is still in flight:
rslearn drops windows missing any *required* input's layers, so a premature
``required: true`` empties the dataset for every eval that reads the same
model.yaml, S2 baselines included. Once a product is fully baked, re-run with
--required to switch it over — that filters the product's coverage-gap windows
out cleanly instead of letting them reach the model, which would raise in
PrecomputedEmbedding.forward. Note that flipping it also drops those windows
from the other evals on that dataset, so previously recorded numbers on it are
no longer directly comparable.

Step 3 (and --required) only fire for products whose manifest reports a
finished, clean bake covering at least --min_coverage of the dataset's windows.
Datasets still materializing, carrying fetch failures, or only partly covered
by the product are reported and left alone, so a dataset goes live exactly when
its data is ready. Partial coverage is not merely a smaller sample: Tessera
covers 8% of ethiopia_crops, and requiring it there would drop the other 92% of
windows from every eval on that dataset. See
docs/PrecomputedEmbeddingCoverage.md for the measured numbers. Pass
--no_enable_modality to stage steps 1 and 2 without turning anything on.

After this script, run backfill_eval_registry_provenance.py to re-stamp
config_json_sha256 (step 1 changes config.json, and eval jobs verify it) and
to link config_repo_dir, then commit registry.json and the new model.yaml
copies under data/rslearn_dataset_configs/.

Example:
    python scripts/tools/wire_embedding_modalities.py --dry_run
    python scripts/tools/wire_embedding_modalities.py \
        --datasets lcmap_lu --products aef,tessera --required
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.evals.embedding_materializer.fetchers import TESSERA_PRODUCTS
from olmoearth_pretrain.evals.studio_ingest.provenance import (
    RSLEARN_DATASET_CONFIGS_DIR,
    find_repo_root,
)
from olmoearth_pretrain.evals.studio_ingest.registry import Registry
from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry
from olmoearth_pretrain.internal.all_evals import AEF_SUPPLEMENTAL_DATASETS

logger = logging.getLogger(__name__)

# Materializer product name -> the modality whose name is the layer/input name.
# Mirrors embedding_materializer.fetchers.build_fetcher so the two cannot drift.
# tessera_v2 is the exception: no v2 product is published, so its layer is
# baked by our own inference run (evals/datasets/tessera_v2_export.py), which
# writes a manifest in the materializer's shape so the gate below still works.
PRODUCT_TO_MODALITY: dict[str, ModalitySpec] = {
    "aef": Modality.GSE,
    **{name: product.modality for name, product in TESSERA_PRODUCTS.items()},
    "tessera_v2": Modality.TESSERA_V2,
}

# Backup of the pre-edit config.json, written once so re-runs keep the
# pristine copy. Useful because ingest overwrites the dataset folder's
# config.json from the source dataset (_try_copy_config_json), which would
# silently drop the layer added here.
CONFIG_BACKUP_NAME = "config.json.pre_embedding_layers.bak"

# Fraction of a dataset's windows that must carry the layer before the product
# goes live there. See bake_is_complete for why partial coverage is unsafe, and
# docs/PrecomputedEmbeddingCoverage.md for the measured per-dataset numbers.
DEFAULT_MIN_COVERAGE = 0.99


def manifest_path(weka_path: str, product: str) -> UPath:
    """Return the materializer manifest path for one (dataset, product)."""
    return UPath(weka_path) / f"embedding_materializer_manifest_{product}.json"


def bake_is_complete(
    weka_path: str, product: str, min_coverage: float
) -> tuple[bool, str]:
    """Report whether a product's bake is finished, clean, and well-covered.

    Coverage matters as much as completion. Because rslearn builds the dataset
    from every input in model.yaml before the model selects which it reads, a
    required input filters coverage-gap windows out for *all* evals on the
    dataset, not just this product's — so wiring a sparsely covered product
    would quietly shrink the OlmoEarth baselines on the same dataset. A bake
    that finished but only covers a fraction of the windows is therefore not
    ready to go live.

    Args:
        weka_path: the dataset folder.
        product: materializer product name (e.g. "aef", "tessera").
        min_coverage: minimum fraction of windows that must carry the layer.

    Returns:
        (ready, reason) — reason describes the manifest state either way.
    """
    path = manifest_path(weka_path, product)
    if not path.exists():
        return False, "no manifest yet (materializer has not finished)"
    with path.open() as f:
        manifest = json.load(f)
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
        return False, f"{state} — re-run the materializer to retry failures"
    if have == 0:
        return False, f"{state} — nothing baked"
    if coverage < min_coverage:
        return False, (
            f"{state} — below --min_coverage {min_coverage:.1%}; requiring this "
            "layer would drop the uncovered windows from every eval on this "
            "dataset"
        )
    return True, state


def config_layer_block(modality: ModalitySpec) -> dict[str, Any]:
    """Build the rslearn config.json raster layer entry for a modality."""
    return {
        "type": "raster",
        "band_sets": [
            {"bands": list(modality.band_order), "dtype": "float32"},
        ],
    }


def add_config_layer(config: dict[str, Any], modality: ModalitySpec) -> bool:
    """Add the modality's raster layer to a parsed config.json, in place.

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


def model_yaml_input_block(
    modality: ModalitySpec, indent: str, required: bool
) -> list[str]:
    """Build the model.yaml input lines for a precomputed embedding modality.

    Args:
        modality: the modality whose name is the input and layer name.
        indent: leading whitespace for the input's own key.
        required: value for the input's ``required`` field.

    Returns:
        the block's lines, without trailing newlines.
    """
    return [
        f"{indent}{modality.name}:",
        f"{indent}  data_type: raster",
        f"{indent}  dtype: FLOAT32",
        f"{indent}  layers:",
        f"{indent}    - {modality.name}",
        f"{indent}  required: {str(required).lower()}",
        f"{indent}  use_all_bands_in_order_of_band_set_idx: 0",
        f"{indent}  passthrough: true",
    ]


def _inputs_line_index(lines: list[str]) -> int:
    """Return the index of the ``inputs:`` mapping key line.

    Raises:
        ValueError: if there is not exactly one such line — better to bail out
            than to guess which mapping to edit.
    """
    matches = [
        idx
        for idx, line in enumerate(lines)
        if line.strip() == "inputs:" and line != line.lstrip()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one indented 'inputs:' line, found {len(matches)}; "
            "edit this model.yaml by hand"
        )
    return matches[0]


def _child_indent(lines: list[str], inputs_idx: int) -> str:
    """Return the leading whitespace used by the entries under ``inputs:``."""
    parent_indent = len(lines[inputs_idx]) - len(lines[inputs_idx].lstrip())
    for line in lines[inputs_idx + 1 :]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent > parent_indent:
            return " " * indent
        break
    return " " * (parent_indent + 2)


def insert_model_yaml_input(
    text: str, modality: ModalitySpec, required: bool
) -> str | None:
    """Add an input for the modality to a model.yaml, preserving formatting.

    The block is inserted directly after the ``inputs:`` line — mapping order
    is not meaningful to YAML, and a targeted textual insert keeps the file's
    comments and layout intact, which a parse/re-dump round trip would not.

    Args:
        text: the model.yaml contents.
        modality: the modality to add an input for.
        required: value for the input's ``required`` field.

    Returns:
        the updated contents, or None if the input was already present.

    Raises:
        ValueError: if the file's structure is not what we expect, or if the
            result does not parse back with the input in place.
    """
    parsed = yaml.safe_load(text)
    try:
        inputs = parsed["data"]["init_args"]["inputs"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"model.yaml has no data.init_args.inputs mapping: {e}")
    if modality.name in inputs:
        return None

    lines = text.splitlines()
    inputs_idx = _inputs_line_index(lines)
    block = model_yaml_input_block(modality, _child_indent(lines, inputs_idx), required)
    updated = lines[: inputs_idx + 1] + block + lines[inputs_idx + 1 :]
    result = "\n".join(updated) + ("\n" if text.endswith("\n") else "")

    check = yaml.safe_load(result)
    added = check["data"]["init_args"]["inputs"].get(modality.name)
    if added is None or added.get("layers") != [modality.name]:
        raise ValueError(
            f"inserting the '{modality.name}' input did not produce the "
            "expected structure; edit this model.yaml by hand"
        )
    return result


def set_model_yaml_required(text: str, modality: ModalitySpec, required: bool) -> str:
    """Set an existing input's ``required`` field, preserving formatting.

    Args:
        text: the model.yaml contents.
        modality: the modality whose input to update.
        required: the value to set.

    Returns:
        the updated contents (unchanged if it already had this value).

    Raises:
        ValueError: if the input has no ``required`` line to update.
    """
    lines = text.splitlines()
    key = f"{modality.name}:"
    for idx, line in enumerate(lines):
        if line.strip() != key:
            continue
        indent = len(line) - len(line.lstrip())
        for offset in range(idx + 1, len(lines)):
            candidate = lines[offset]
            if not candidate.strip():
                continue
            if len(candidate) - len(candidate.lstrip()) <= indent:
                break
            if candidate.strip().startswith("required:"):
                child = candidate[: len(candidate) - len(candidate.lstrip())]
                lines[offset] = f"{child}required: {str(required).lower()}"
                return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        break
    raise ValueError(
        f"no 'required:' line found under the '{modality.name}' input; "
        "edit this model.yaml by hand"
    )


def repo_config_dir(repo_root: Path, dataset: str) -> Path:
    """Return the repo directory that holds a dataset's model.yaml."""
    return repo_root / RSLEARN_DATASET_CONFIGS_DIR / dataset


def model_yaml_source(entry: EvalDatasetEntry, repo_root: Path) -> UPath:
    """Return the model.yaml that eval jobs actually read for this entry.

    Entries with ``config_repo_dir`` read the git-tracked copy; the Weka copy
    is then only a snapshot. Patching the file eval jobs read (and mirroring
    to the other location) keeps the two in sync either way.
    """
    if entry.config_repo_dir is not None:
        return UPath(repo_root / entry.config_repo_dir / "model.yaml")
    return UPath(entry.weka_path) / "model.yaml"


def patch_config_json(weka_path: str, modality: ModalitySpec, dry_run: bool) -> bool:
    """Add the modality's layer to the dataset folder's config.json.

    Args:
        weka_path: the dataset folder.
        modality: the modality to declare.
        dry_run: if set, report but do not write.

    Returns:
        True if the file needed (or would need) changing.

    Raises:
        FileNotFoundError: if the dataset folder has no config.json.
    """
    config_json = UPath(weka_path) / "config.json"
    if not config_json.exists():
        raise FileNotFoundError(f"no config.json at {weka_path}")
    with config_json.open() as f:
        config = json.load(f)
    if not add_config_layer(config, modality):
        logger.info("    config.json: '%s' layer already declared", modality.name)
        return False
    if dry_run:
        logger.info("    config.json: would add '%s' layer", modality.name)
        return True

    backup = UPath(weka_path) / CONFIG_BACKUP_NAME
    if not backup.exists():
        with config_json.open("rb") as src, backup.open("wb") as dst:
            dst.write(src.read())
        logger.info("    config.json: backed up to %s", CONFIG_BACKUP_NAME)
    with config_json.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    logger.info("    config.json: added '%s' layer", modality.name)
    return True


def patch_model_yaml(
    entry: EvalDatasetEntry,
    repo_root: Path,
    modality: ModalitySpec,
    required: bool,
    dry_run: bool,
) -> bool:
    """Add or update the modality's input in the dataset's model.yaml.

    Writes the same bytes to the Weka copy and to
    data/rslearn_dataset_configs/<dataset>/model.yaml, which is the condition
    under which backfill_eval_registry_provenance.py will link
    ``config_repo_dir`` and move the dataset onto the git-pinned config.

    Args:
        entry: the registry entry being wired.
        repo_root: the repo checkout root.
        modality: the modality to add an input for.
        required: value for the input's ``required`` field.
        dry_run: if set, report but do not write.

    Returns:
        True if either copy needed (or would need) changing.
    """
    source = model_yaml_source(entry, repo_root)
    if not source.exists():
        raise FileNotFoundError(f"no model.yaml at {source}")
    original = source.read_text()

    updated = insert_model_yaml_input(original, modality, required)
    if updated is None:
        updated = set_model_yaml_required(original, modality, required)
        if updated == original:
            logger.info(
                "    model.yaml: '%s' input already present (required: %s)",
                modality.name,
                str(required).lower(),
            )
        else:
            logger.info(
                "    model.yaml: %s '%s' required -> %s",
                "would set" if dry_run else "set",
                modality.name,
                str(required).lower(),
            )
    else:
        logger.info(
            "    model.yaml: %s '%s' input (required: %s)",
            "would add" if dry_run else "added",
            modality.name,
            str(required).lower(),
        )

    targets = [UPath(entry.weka_path) / "model.yaml"]
    repo_yaml = UPath(repo_config_dir(repo_root, entry.name) / "model.yaml")
    targets.append(repo_yaml)
    stale = [t for t in targets if not t.exists() or t.read_text() != updated]
    if not stale:
        return False
    if dry_run:
        for target in stale:
            logger.info("    would write %s", target)
        return True
    for target in stale:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(updated)
        logger.info("    wrote %s", target)
    return True


def enable_modality(
    entry: EvalDatasetEntry, modality: ModalitySpec, dry_run: bool
) -> bool:
    """Add the modality to the registry entry's ``modalities``, in place.

    Args:
        entry: the registry entry to update.
        modality: the modality to list.
        dry_run: if set, report but do not modify the entry.

    Returns:
        True if the entry was (or would be) modified.
    """
    if modality.name in entry.modalities:
        logger.info("    registry: '%s' already listed", modality.name)
        return False
    if dry_run:
        logger.info("    registry: would add '%s' to modalities", modality.name)
        return True
    entry.modalities = sorted([*entry.modalities, modality.name])
    logger.info("    registry: added '%s' to modalities", modality.name)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(AEF_SUPPLEMENTAL_DATASETS),
        help="Comma-separated registry dataset names to wire.",
    )
    parser.add_argument(
        "--products",
        default="aef,tessera",
        help=f"Comma-separated products: {sorted(PRODUCT_TO_MODALITY)}.",
    )
    parser.add_argument(
        "--required",
        action="store_true",
        help=(
            "Write 'required: true' on the model.yaml input, so the product's "
            "coverage-gap windows are filtered out of the dataset (for every "
            "eval reading this model.yaml) instead of reaching the model and "
            "raising. Applied only where the manifest shows a finished, clean "
            "bake — a dataset still materializing stays 'required: false', "
            "since requiring a layer that is not there yet empties it."
        ),
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=(
            "Minimum fraction of a dataset's windows that must carry the layer "
            "before it goes live. A product that covers only part of a dataset "
            "(e.g. Tessera outside its 2024-global / US+EU-since-2017 "
            "footprint) would, once required, drop the uncovered windows from "
            "every eval on that dataset — including the OlmoEarth baselines. "
            f"Default {DEFAULT_MIN_COVERAGE:.0%}."
        ),
    )
    parser.add_argument(
        "--no_enable_modality",
        action="store_true",
        help=(
            "Patch config.json/model.yaml but never touch the registry, so no "
            "dataset goes live yet."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Wire the requested products into the requested datasets."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    repo_root = find_repo_root()
    if repo_root is None:
        raise SystemExit("Could not locate the repo root; run from a checkout.")

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

    registry_changed = False
    config_changed: list[str] = []
    not_live: list[str] = []

    for dataset in dataset_names:
        entry = registry.datasets[dataset]
        logger.info("=== %s ===", dataset)
        if not UPath(entry.weka_path).exists():
            logger.warning(
                "  %s not found — skipping (is Weka mounted?)", entry.weka_path
            )
            continue

        for product in products:
            modality = PRODUCT_TO_MODALITY[product]
            complete, reason = bake_is_complete(
                entry.weka_path, product, args.min_coverage
            )
            logger.info("  %s (%s): %s", product, modality.name, reason)

            if patch_config_json(entry.weka_path, modality, args.dry_run):
                config_changed.append(dataset)
            # Requiring a layer that is not fully baked would filter every
            # window still missing it out of the dataset — for the S2 evals
            # reading this model.yaml too, not just this product's.
            required = args.required and complete
            if args.required and not complete:
                logger.info(
                    "    model.yaml: keeping 'required: false' — not ready "
                    "here (see the manifest line above); re-run with "
                    "--required once it is"
                )
            patch_model_yaml(entry, repo_root, modality, required, args.dry_run)

            if args.no_enable_modality:
                not_live.append(f"{dataset}/{product} (--no_enable_modality)")
            elif complete:
                registry_changed |= enable_modality(entry, modality, args.dry_run)
            else:
                not_live.append(f"{dataset}/{product} ({reason})")

    if registry_changed and not args.dry_run:
        registry.save()
    elif registry_changed:
        logger.info("[dry_run] registry would change; not writing.")

    logger.info("=== Next steps ===")
    if config_changed:
        logger.info(
            "config.json changed for %s — run "
            "'python scripts/tools/backfill_eval_registry_provenance.py' to "
            "re-stamp config_json_sha256 (eval jobs verify it) and link "
            "config_repo_dir.",
            ", ".join(sorted(set(config_changed))),
        )
    if not_live:
        logger.info("Not yet enabled: %s", "; ".join(not_live))
        logger.info("Re-run this script for those once their bake finishes.")
    logger.info(
        "Then commit registry.json and data/rslearn_dataset_configs/*/model.yaml."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
