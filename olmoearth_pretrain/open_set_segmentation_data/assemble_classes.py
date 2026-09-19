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
import math
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
# Curated concept annotation for the presence-only pool (merge + overlap/conflict).
DEFAULT_CONCEPTS_PATH = (
    _REPO_ROOT / "data" / "open_set_segmentation_data" / "presence_only_concepts.json"
)

# A dataset is presence-only (pooled with cross-dataset negatives) iff it is
# foreground-only (no explicit negative/background class) AND has few classes
# (<= PRESENCE_ONLY_MAX_CLASSES). Otherwise it is a self-contained multiclass training
# group of its own (rich classification, or a dataset that already provides its own
# negatives). See AGENT_SUMMARY.md sec 5.
PRESENCE_ONLY_MAX_CLASSES = 3

# Class-name patterns that indicate a background / negative / no-label class (a paired
# negative such as "non_crop", "no water", "not-flooded", "no-change", "stable_forest",
# or an explicit background/nodata class). A dataset with such a class provides its own
# negatives and is NOT presence-only. NOTE: a bare catch-all "other"/"unknown" does NOT
# count as a disqualifying negative (e.g. GFW oil/wind/other stays foreground-only).
_NEGATIVE_NAME_RE = re.compile(
    r"^(background|negative|no[\s_-]?data|nodata|unlabell?ed|absence|absent|none|null"
    r"|unburned|unburnt)$"
    r"|^(non|no|not)[\s._-]"
    r"|^stable[\s._-]",
    re.IGNORECASE,
)


@dataclass
class AssemblyResult:
    """Result of assembling the global class space."""

    mapping: dict[str, Any]
    # Datasets whose summary text calls them presence-only but which HAVE a
    # background/negative class (i.e. "marked differently"). These need a human decision.
    ambiguous_presence_only: list[str] = field(default_factory=list)
    # Concept-file members (slug:class) that matched no presence-only class (typo / dataset
    # became own-group / not completed). Surfaced so the concept file can be cleaned up.
    unmatched_concept_keys: list[str] = field(default_factory=list)


def _load_metadata(datasets_root: UPath, slug: str) -> dict[str, Any] | None:
    """Load ``datasets/{slug}/metadata.json``; return None if missing."""
    p = datasets_root / slug / "metadata.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _has_negative_class(classes: list[dict[str, Any]]) -> bool:
    """Whether any class looks like an explicit negative/background/no-label class."""
    for c in classes:
        name = (c.get("name") or "").strip()
        if _NEGATIVE_NAME_RE.match(name):
            return True
    return False


def _load_concepts(
    concepts_path: Path | None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[tuple[str, str]]]:
    """Load the presence-only concept annotation.

    Returns (key_to_concept, concept_info, overlap_pairs) where key is ``slug:class_name``.
    A missing/None path yields empty structures (no merges, no conflicts).
    """
    if concepts_path is None or not Path(concepts_path).exists():
        return {}, {}, []
    with Path(concepts_path).open() as f:
        cfg = json.load(f)
    concept_info: dict[str, dict[str, Any]] = cfg.get("concepts", {})
    key_to_concept: dict[str, str] = {}
    for concept, info in concept_info.items():
        for member in info.get("members", []):
            key_to_concept[member] = concept
    overlaps = [(a, b) for a, b in cfg.get("overlaps", [])]
    return key_to_concept, concept_info, overlaps


def _is_presence_only(classes: list[dict[str, Any]]) -> bool:
    """Presence-only = foreground-only (no negative class) AND few classes.

    Such datasets are pooled into the shared presence-only group (cross-dataset negatives).
    Everything else (many-class rich classifications, or datasets carrying their own
    negative/background class) becomes its own self-contained multiclass training group.
    """
    return len(classes) <= PRESENCE_ONLY_MAX_CLASSES and not _has_negative_class(
        classes
    )


def assemble_classes(
    datasets_root: UPath | None = None,
    summaries_dir: Path | None = DEFAULT_SUMMARIES_DIR,
    excluded_slugs: frozenset[str] = EXCLUDED_SLUGS,
    registry: dict[str, Any] | None = None,
    concepts_path: Path | None = DEFAULT_CONCEPTS_PATH,
) -> AssemblyResult:
    """Build the global class mapping across all completed open-set datasets.

    Args:
        datasets_root: root containing ``{slug}/metadata.json``. Defaults to
            ``OUTPUT_ROOT/datasets`` (weka).
        summaries_dir: directory of ``{slug}.md`` dataset summaries used to
            cross-check presence-only detection. None to skip the text check.
        excluded_slugs: slugs to drop entirely (held-out evals).
        registry: pre-loaded registry dict (for testing); loads from disk if None.
        concepts_path: Path to the concept merge and overlap configuration. None
            disables concept merging.

    Returns:
        the assembly result (mapping dict + ambiguous presence-only slugs).
    """
    if datasets_root is None:
        datasets_root = OUTPUT_ROOT / "datasets"
    if registry is None:
        registry = load_registry()

    key_to_concept, concept_info, overlap_pairs = _load_concepts(concepts_path)
    merged_concept_gid: dict[str, int] = {}  # merge-concept -> its single global_id
    gid_to_classentry: dict[int, dict[str, Any]] = {}  # for appending merged provenance
    concept_to_gids: dict[str, list[int]] = {}  # concept -> presence-only global_ids
    gid_to_concept: dict[int, str] = {}
    matched_concept_keys: set[str] = set()

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
            if not (
                isinstance(value_range, list)
                and len(value_range) == 2
                and all(math.isfinite(float(value)) for value in value_range)
                and float(value_range[1]) > float(value_range[0])
            ):
                raise ValueError(
                    f"regression dataset {slug} has invalid value_range {value_range!r}"
                )
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

        presence_only = _is_presence_only(classes)

        if presence_only:
            # Pooled classes; apply concept merge (same real-world thing across datasets
            # collapses to one global_id) and record concept membership for conflicts.
            for c in sorted(classes, key=lambda c: c["id"]):
                key = f"{slug}:{c.get('name')}"
                concept = key_to_concept.get(key)
                if concept is not None:
                    matched_concept_keys.add(key)
                member = {"slug": slug, "local_id": c["id"], "name": c.get("name")}
                if concept is not None and concept_info[concept].get("merge"):
                    if concept in merged_concept_gid:
                        # Reuse the merged class's global_id; just record provenance.
                        gid_to_classentry[merged_concept_gid[concept]][
                            "members"
                        ].append(member)
                        continue
                    global_id = next_global_id
                    next_global_id += 1
                    merged_concept_gid[concept] = global_id
                    ce = {
                        "global_id": global_id,
                        "slug": None,
                        "local_id": None,
                        "name": concept,
                        "concept": concept,
                        "members": [member],
                    }
                else:
                    global_id = next_global_id
                    next_global_id += 1
                    ce = {
                        "global_id": global_id,
                        "slug": slug,
                        "local_id": c["id"],
                        "name": c.get("name"),
                    }
                    if concept is not None:
                        ce["concept"] = concept
                global_classes.append(ce)
                gid_to_classentry[global_id] = ce
                presence_only_ids.append(global_id)
                if concept is not None:
                    concept_to_gids.setdefault(concept, []).append(global_id)
                    gid_to_concept[global_id] = concept
        else:
            first_global_id = next_global_id
            dataset_global_ids = []
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
            training_datasets.append(
                {
                    "name": slug,
                    "slug": slug,
                    "presence_only": False,
                    "global_ids": dataset_global_ids,
                    "global_id_range": [first_global_id, next_global_id],
                }
            )

    # Conflicts among presence-only classes, from the concept overlaps graph: every class
    # of concept A conflicts with every class of concept B (and vice versa). Members of the
    # same concept do NOT conflict (merge already collapsed identical ones). At pretraining
    # time a presence-only class draws negatives only from non-conflicting pool classes.
    conflicts: dict[int, set[int]] = {}
    for a, b in overlap_pairs:
        for x in concept_to_gids.get(a, []):
            for y in concept_to_gids.get(b, []):
                if x != y:
                    conflicts.setdefault(x, set()).add(y)
                    conflicts.setdefault(y, set()).add(x)

    # Concept keys named in the concept file that never matched a presence-only class
    # (typo, or the dataset became own-group / isn't completed). Surface for cleanup.
    unmatched_concept_keys = sorted(set(key_to_concept) - matched_concept_keys)

    # All presence-only classes form a single synthetic training group.
    if presence_only_ids:
        training_datasets.append(
            {
                "name": PRESENCE_ONLY_GROUP,
                "slug": None,
                "presence_only": True,
                "global_ids": sorted(presence_only_ids),
                "concepts": {
                    str(gid): gid_to_concept[gid] for gid in sorted(gid_to_concept)
                },
                "conflicts": {
                    str(gid): sorted(conflicts[gid]) for gid in sorted(conflicts)
                },
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
    return AssemblyResult(
        mapping=mapping,
        ambiguous_presence_only=ambiguous,
        unmatched_concept_keys=unmatched_concept_keys,
    )


def write_mapping(
    result: AssemblyResult,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    overwrite: bool = False,
) -> None:
    """Write the class mapping JSON atomically without replacing a frozen mapping."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing class mapping {output_path}; write to a "
            "candidate path or pass overwrite=True for a deliberate dataset rebuild"
        )
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Deliberately replace an existing mapping (requires rebuilding encoded labels)"
        ),
    )
    args = parser.parse_args()

    datasets_root = UPath(args.datasets_root) if args.datasets_root else None
    summaries_dir = Path(args.summaries_dir) if args.summaries_dir else None
    result = assemble_classes(datasets_root=datasets_root, summaries_dir=summaries_dir)
    write_mapping(result, Path(args.output), overwrite=args.overwrite)

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
