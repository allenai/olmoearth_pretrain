"""Assemble a global class-id space across all open-set datasets.

The open-set label bank stores per-dataset class ids (0..254, uint8). At pretraining
time we need a single global id space so a pixel in the combined ``open_set`` label
layer is globally unique across datasets. This module builds that mapping and writes
``data/open_set_segmentation_data/class_mapping.json``.

It also builds a separate registry for regression datasets, which are materialized into
the two-band ``open_set_regression`` layer rather than the classification class space.

Presence-only datasets (positive/foreground pixels only, no background class) are merged
into one synthetic training group so that other presence-only classes act as negatives
for each other at train time. See ``AGENT_SUMMARY.md`` and the plan for details.

Only datasets with registry status ``completed`` are included. Datasets in
``EXCLUDED_SLUGS`` (held-out evals) are dropped.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from upath import UPath

from .manifest import OUTPUT_ROOT, load_registry
from .pretrain_constants import (
    EXCLUDED_SLUGS,
    OPEN_SET_DTYPE,
    OPEN_SET_NODATA,
    OPEN_SET_REGRESSION_DTYPE,
    PRESENCE_ONLY_GROUP,
    REGRESSION_DATASET_ID_NODATA,
    REGRESSION_VALUE_MAX_OUT,
    REGRESSION_VALUE_MIN_OUT,
    REGRESSION_VALUE_NODATA,
)

# Repo (version-controlled) home for the generated class mapping. Computed relative to
# this file: .../olmoearth_pretrain/olmoearth_pretrain/open_set_segmentation_data/ ->
# .../olmoearth_pretrain/data/open_set_segmentation_data/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARIES_DIR = (
    _REPO_ROOT / "data" / "open_set_segmentation_data" / "dataset_summaries"
)
DEFAULT_OUTPUT_PATH = (
    _REPO_ROOT / "data" / "open_set_segmentation_data" / "class_mapping.json"
)

# Class-name tokens that indicate a background / negative / no-label class. A dataset
# that has such a class is NOT presence-only (it already provides negatives). Detection
# datasets fall here because they emit an explicit background(0) class.
_BACKGROUND_NAME_RE = re.compile(
    r"\b(background|negative|no[\s_-]?data|nodata|unlabel|unlabelled|unlabeled"
    r"|absence|absent|not\s|no\s+label)\b",
    re.IGNORECASE,
)

# Wording in a dataset summary that indicates a presence-/positive-only dataset.
_PRESENCE_TEXT_RE = re.compile(
    r"presence[\s_-]?only|positive[\s_-]?only|foreground[\s_-]?only|presence/absence",
    re.IGNORECASE,
)


@dataclass
class AssemblyResult:
    """Result of assembling the global class space."""

    mapping: dict[str, Any]
    # Datasets whose summary text calls them presence-only but which HAVE a
    # background/negative class (i.e. "marked differently"). These need a human decision.
    ambiguous_presence_only: list[str] = field(default_factory=list)


def _load_metadata(datasets_root: UPath, slug: str) -> dict[str, Any] | None:
    """Load ``datasets/{slug}/metadata.json``; return None if missing."""
    p = datasets_root / slug / "metadata.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _has_background_class(classes: list[dict[str, Any]]) -> bool:
    """Whether any class looks like a background/negative/no-label class."""
    for c in classes:
        name = c.get("name") or ""
        if _BACKGROUND_NAME_RE.search(name):
            return True
    return False


def _summary_says_presence_only(summaries_dir: Path | None, slug: str) -> bool:
    """Whether the dataset summary markdown mentions presence-/positive-only."""
    if summaries_dir is None:
        return False
    p = Path(summaries_dir) / f"{slug}.md"
    if not p.exists():
        return False
    return bool(_PRESENCE_TEXT_RE.search(p.read_text()))


def assemble_classes(
    datasets_root: UPath | None = None,
    summaries_dir: Path | None = DEFAULT_SUMMARIES_DIR,
    excluded_slugs: frozenset[str] = EXCLUDED_SLUGS,
    registry: dict[str, Any] | None = None,
) -> AssemblyResult:
    """Build the global class mapping across all completed open-set datasets.

    Args:
        datasets_root: root containing ``{slug}/metadata.json``. Defaults to
            ``OUTPUT_ROOT/datasets`` (weka).
        summaries_dir: directory of ``{slug}.md`` dataset summaries used to
            cross-check presence-only detection. None to skip the text check.
        excluded_slugs: slugs to drop entirely (held-out evals).
        registry: pre-loaded registry dict (for testing); loads from disk if None.

    Returns:
        the assembly result (mapping dict + ambiguous presence-only slugs).
    """
    if datasets_root is None:
        datasets_root = OUTPUT_ROOT / "datasets"
    if registry is None:
        registry = load_registry()

    # Deterministic order: sort completed, non-excluded datasets by slug.
    entries = sorted(
        (
            e
            for e in registry["datasets"]
            if e.get("status") == "completed" and e["slug"] not in excluded_slugs
        ),
        key=lambda e: e["slug"],
    )

    global_classes: list[dict[str, Any]] = []
    training_datasets: list[dict[str, Any]] = []
    presence_only_ids: list[int] = []
    regression_datasets: list[dict[str, Any]] = []
    ambiguous: list[str] = []

    next_global_id = 0
    next_regression_id = REGRESSION_DATASET_ID_NODATA + 1

    for entry in entries:
        slug = entry["slug"]
        metadata = _load_metadata(datasets_root, slug)
        if metadata is None:
            # Registry says completed but metadata.json is missing (e.g. not synced
            # locally). Skip rather than fail so partial runs work for verification.
            continue

        task_type = metadata.get("task_type") or entry.get("task_type")

        if task_type == "regression":
            reg = metadata.get("regression") or {}
            value_range = reg.get("value_range")
            regression_datasets.append(
                {
                    "dataset_id": next_regression_id,
                    "slug": slug,
                    "name": metadata.get("name", slug),
                    "target_name": reg.get("name"),
                    "unit": reg.get("unit"),
                    "source_dtype": reg.get("dtype"),
                    "value_range": value_range,
                    "source_nodata_value": reg.get("nodata_value"),
                }
            )
            next_regression_id += 1
            continue

        # Classification.
        classes = metadata.get("classes") or []
        if not classes:
            continue

        has_background = _has_background_class(classes)
        text_presence_only = _summary_says_presence_only(summaries_dir, slug)
        presence_only = not has_background
        if text_presence_only and has_background:
            # Summary calls it presence-only but it has a background class: needs a
            # human decision (the plan asked us to surface these).
            ambiguous.append(slug)

        first_global_id = next_global_id
        dataset_global_ids: list[int] = []
        for c in sorted(classes, key=lambda c: c["id"]):
            global_id = next_global_id
            next_global_id += 1
            dataset_global_ids.append(global_id)
            global_classes.append(
                {
                    "global_id": global_id,
                    "slug": slug,
                    "local_id": c["id"],
                    "name": c.get("name"),
                }
            )

        if presence_only:
            presence_only_ids.extend(dataset_global_ids)
        else:
            training_datasets.append(
                {
                    "name": slug,
                    "slug": slug,
                    "presence_only": False,
                    "global_ids": dataset_global_ids,
                    "global_id_range": [first_global_id, next_global_id],
                }
            )

    # All presence-only classes form a single synthetic training group.
    if presence_only_ids:
        training_datasets.append(
            {
                "name": PRESENCE_ONLY_GROUP,
                "slug": None,
                "presence_only": True,
                "global_ids": sorted(presence_only_ids),
            }
        )

    mapping = {
        "open_set": {
            "dtype": OPEN_SET_DTYPE,
            "nodata_value": OPEN_SET_NODATA,
            "num_classes": len(global_classes),
            "classes": global_classes,
            "training_datasets": training_datasets,
            "presence_only_group": PRESENCE_ONLY_GROUP,
        },
        "open_set_regression": {
            "dtype": OPEN_SET_REGRESSION_DTYPE,
            "num_bands": 2,
            "band0": "regression dataset id (1-based; 0 = no label at this pixel)",
            "band1": (
                f"value linearly remapped from value_range to "
                f"[{REGRESSION_VALUE_MIN_OUT}, {REGRESSION_VALUE_MAX_OUT}] "
                f"({REGRESSION_VALUE_NODATA} = nodata)"
            ),
            "dataset_id_nodata": REGRESSION_DATASET_ID_NODATA,
            "value_nodata": REGRESSION_VALUE_NODATA,
            "value_out_range": [REGRESSION_VALUE_MIN_OUT, REGRESSION_VALUE_MAX_OUT],
            "datasets": regression_datasets,
        },
        "excluded_slugs": sorted(excluded_slugs),
    }
    return AssemblyResult(mapping=mapping, ambiguous_presence_only=ambiguous)


def write_mapping(
    result: AssemblyResult, output_path: Path = DEFAULT_OUTPUT_PATH
) -> None:
    """Write the class mapping JSON atomically."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(result.mapping, f, indent=2)
    tmp.rename(output_path)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets_root",
        type=str,
        default=None,
        help="Root containing {slug}/metadata.json (default: OUTPUT_ROOT/datasets on weka)",
    )
    parser.add_argument(
        "--summaries_dir",
        type=str,
        default=str(DEFAULT_SUMMARIES_DIR),
        help="Directory of {slug}.md dataset summaries for presence-only cross-check",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write class_mapping.json",
    )
    args = parser.parse_args()

    datasets_root = UPath(args.datasets_root) if args.datasets_root else None
    summaries_dir = Path(args.summaries_dir) if args.summaries_dir else None
    result = assemble_classes(datasets_root=datasets_root, summaries_dir=summaries_dir)
    write_mapping(result, Path(args.output))

    open_set = result.mapping["open_set"]
    regression = result.mapping["open_set_regression"]
    print(f"Wrote {args.output}")
    print(
        f"  classification: {open_set['num_classes']} classes across "
        f"{len(open_set['training_datasets'])} training groups"
    )
    print(f"  regression: {len(regression['datasets'])} datasets")
    if result.ambiguous_presence_only:
        print(
            "  WARNING: these datasets are called presence-only in their summary but "
            "have a background/negative class (needs review): "
            + ", ".join(result.ambiguous_presence_only)
        )


if __name__ == "__main__":
    main()
