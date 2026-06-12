"""Create balanced OpenStreetMap raster split candidates from cached Presto metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from olmoearth_pretrain.evals.class_support import write_class_support

CLASS_NAMES = [
    "aerialway_pylon",
    "aerodrome",
    "airstrip",
    "amenity_fuel",
    "building",
    "chimney",
    "communications_tower",
    "crane",
    "flagpole",
    "fountain",
    "generator_wind",
    "helipad",
    "highway",
    "leisure",
    "lighthouse",
    "obelisk",
    "observatory",
    "parking",
    "petroleum_well",
    "power_plant",
    "power_substation",
    "power_tower",
    "river",
    "runway",
    "satellite_dish",
    "silo",
    "storage_tank",
    "taxiway",
    "water_tower",
    "works",
]
POWER_TOWER_CLASS_ID = 21
RARE_FOCUS_CLASS_IDS = [9, 10, 26, 27]

SPLIT_SIZES = {
    "train": 6144,
    "valid": 3072,
    "test": 3072,
}


@dataclass(frozen=True)
class VariantConfig:
    """Selection rule for one OSM balanced split variant."""

    name: str
    min_anchor_tile_presence: int
    min_classes_present: int = 1
    min_normalized_entropy: float = 0.0
    presence_caps: dict[int, float] | None = None
    cap_exempt_max_anchor_presence: int | None = None
    required_presence_by_split: dict[int, dict[str, int]] | None = None
    allow_extra_required_presence: bool = False


VARIANTS = [
    VariantConfig(
        name="osm_base_balanced",
        min_anchor_tile_presence=500,
        required_presence_by_split={
            POWER_TOWER_CLASS_ID: {"train": 2000, "valid": 900, "test": 900},
        },
    ),
    VariantConfig(
        name="osm_diverse_context",
        min_anchor_tile_presence=500,
        min_classes_present=3,
        min_normalized_entropy=0.5,
    ),
    VariantConfig(
        name="osm_rare_class_focused",
        min_anchor_tile_presence=51,
        presence_caps={12: 0.80, 4: 0.70},
        cap_exempt_max_anchor_presence=499,
        required_presence_by_split={
            class_id: {"train": 70, "valid": 20, "test": 20}
            for class_id in RARE_FOCUS_CLASS_IDS
        },
        allow_extra_required_presence=True,
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "analysis/pretrain_eval_label_metadata/"
            "presto_openstreetmap_raster_label_metadata.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("olmoearth_pretrain/evals/datasets/splits/presto_osm_balanced"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class OsmMetadata:
    """Loaded sparse OSM label metadata."""

    def __init__(self, path: Path) -> None:
        """Load cached OSM metadata arrays."""
        data = np.load(path, allow_pickle=False)
        self.sample_indices = data["sample_indices"]
        self.row_offsets = data["row_offsets"]
        self.label_ids = data["label_ids"]
        self.counts = data["counts"]

    def row_labels_counts(self, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return present class ids and pixel counts for a row."""
        start = int(self.row_offsets[row_idx])
        end = int(self.row_offsets[row_idx + 1])
        return self.label_ids[start:end].astype(int), self.counts[start:end].astype(int)


def tile_presence_counts(metadata: OsmMetadata) -> np.ndarray:
    """Count how many tiles each class appears in."""
    presence = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for row_idx in range(len(metadata.sample_indices)):
        labels, _ = metadata.row_labels_counts(row_idx)
        presence[labels] += 1
    return presence


def tile_records(metadata: OsmMetadata, global_presence: np.ndarray) -> pd.DataFrame:
    """Build per-tile diversity and rare-class metadata."""
    rows = []
    rare_cutoff = np.quantile(global_presence[global_presence > 0], 0.25)
    rare_ids = set(
        np.where((global_presence > 0) & (global_presence <= rare_cutoff))[0]
    )

    for row_idx, sample_index in enumerate(metadata.sample_indices.tolist()):
        labels, counts = metadata.row_labels_counts(row_idx)
        valid_pixels = int(counts.sum())
        num_classes = int(len(labels))
        if valid_pixels:
            freqs = counts / valid_pixels
            entropy = float(-(freqs * np.log2(freqs + 1e-12)).sum())
            max_entropy = float(np.log2(num_classes)) if num_classes > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy else 0.0
            top_class_fraction = float(freqs.max())
            rare_fraction = float(
                counts[np.asarray([int(label) in rare_ids for label in labels])].sum()
                / valid_pixels
            )
        else:
            entropy = 0.0
            normalized_entropy = 0.0
            top_class_fraction = 0.0
            rare_fraction = 0.0

        rows.append(
            {
                "row_idx": row_idx,
                "sample_index": int(sample_index),
                "valid_osm_pixels": valid_pixels,
                "num_classes_present": num_classes,
                "entropy": entropy,
                "normalized_entropy": normalized_entropy,
                "top_class_fraction": top_class_fraction,
                "rare_fraction": rare_fraction,
                "labels": " ".join(str(int(label)) for label in labels.tolist()),
                "class_names": " | ".join(CLASS_NAMES[int(label)] for label in labels),
            }
        )
    return pd.DataFrame(rows)


def split_row_indices(num_rows: int, seed: int) -> dict[str, np.ndarray]:
    """Create deterministic 80/10/10 source pools."""
    shuffled = np.random.RandomState(seed).permutation(num_rows)
    train_end = int(num_rows * 0.8)
    valid_end = train_end + int(num_rows * 0.1)
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def eligible_anchor_classes(global_presence: np.ndarray, min_presence: int) -> set[int]:
    """Return classes with enough tile support to anchor a split."""
    return set(np.where(global_presence >= min_presence)[0].astype(int).tolist())


def cap_exempt_classes(global_presence: np.ndarray, variant: VariantConfig) -> set[int]:
    """Return anchor classes whose tiles are exempt from filler presence caps."""
    if variant.cap_exempt_max_anchor_presence is None:
        return set()
    return set(
        np.where(
            (global_presence >= variant.min_anchor_tile_presence)
            & (global_presence <= variant.cap_exempt_max_anchor_presence)
        )[0]
        .astype(int)
        .tolist()
    )


def choose_anchor(
    labels: np.ndarray, anchors: set[int], global_presence: np.ndarray
) -> int | None:
    """Assign a tile to the rarest eligible class present."""
    eligible = [int(label) for label in labels.tolist() if int(label) in anchors]
    if not eligible:
        return None
    return min(eligible, key=lambda label: (global_presence[label], label))


def selection_score(row: pd.Series, variant: VariantConfig) -> float:
    """Rank candidate fill tiles after round-robin anchors are exhausted."""
    score = (
        0.35 * float(row.normalized_entropy)
        + 0.25 * float(row.rare_fraction)
        + 0.25 * min(float(row.num_classes_present) / 6.0, 1.0)
        + 0.15 * (1.0 - float(row.top_class_fraction))
    )
    if variant.name == "osm_base_balanced":
        score += 0.2 if row.num_classes_present >= 2 else 0.0
    return score


def labels_for_row(metadata: OsmMetadata, row_idx: int) -> set[int]:
    """Return present class ids for one metadata row."""
    labels, _ = metadata.row_labels_counts(row_idx)
    return set(labels.astype(int).tolist())


def fits_presence_caps(
    row_labels: set[int],
    selected_presence: dict[int, int],
    presence_caps: dict[int, float] | None,
    exempt_classes: set[int],
    target_size: int,
) -> bool:
    """Return whether adding a tile respects soft class presence caps."""
    if not presence_caps:
        return True
    if row_labels & exempt_classes:
        return True
    for class_id, max_fraction in presence_caps.items():
        if class_id in row_labels and selected_presence.get(class_id, 0) >= int(
            target_size * max_fraction
        ):
            return False
    return True


def fits_required_presence_limits(
    row_labels: set[int],
    selected_presence: dict[int, int],
    variant: VariantConfig,
    split: str,
    required_presence_limits: dict[int, int],
) -> bool:
    """Return whether a row avoids over-consuming quota classes for val/test."""
    if split == "train" or not variant.required_presence_by_split:
        return True
    required_classes = set(variant.required_presence_by_split)
    row_required_classes = row_labels & required_classes
    if not row_required_classes:
        return True
    return any(
        selected_presence.get(class_id, 0)
        < required_presence_limits.get(
            class_id, variant.required_presence_by_split[class_id].get(split, 0)
        )
        for class_id in row_required_classes
    )


def required_presence_limits_for_split(
    *,
    variant: VariantConfig,
    split: str,
    pool: pd.DataFrame,
    row_labels_by_idx: dict[int, set[int]],
) -> dict[int, int]:
    """Return max quota-class counts this split may consume during filling."""
    if not variant.required_presence_by_split:
        return {}
    current_required = {
        class_id: split_targets.get(split, 0)
        for class_id, split_targets in variant.required_presence_by_split.items()
    }
    if split == "train" or not variant.allow_extra_required_presence:
        return current_required

    split_order = ("valid", "test", "train")
    future_splits = split_order[split_order.index(split) + 1 :]
    limits = {}
    for class_id, split_targets in variant.required_presence_by_split.items():
        available = sum(
            1
            for row in pool.itertuples()
            if class_id in row_labels_by_idx[int(row.row_idx)]
        )
        future_required = sum(
            split_targets.get(future_split, 0) for future_split in future_splits
        )
        limits[class_id] = max(current_required[class_id], available - future_required)
    return limits


def add_selected_row(
    row_idx: int,
    row_labels_by_idx: dict[int, set[int]],
    selected: list[int],
    selected_set: set[int],
    selected_presence: dict[int, int],
) -> None:
    """Add a row and update true class-presence counters."""
    selected.append(row_idx)
    selected_set.add(row_idx)
    for class_id in row_labels_by_idx[row_idx]:
        selected_presence[class_id] = selected_presence.get(class_id, 0) + 1


def add_required_presence_rows(
    *,
    split: str,
    variant: VariantConfig,
    pool: pd.DataFrame,
    row_labels_by_idx: dict[int, set[int]],
    score_by_row: dict[int, float],
    selected: list[int],
    selected_set: set[int],
    selected_presence: dict[int, int],
    presence_caps: dict[int, float] | None,
    exempt_classes: set[int],
    target_size: int,
    required_presence_limits: dict[int, int],
) -> None:
    """Add high-scoring rows until configured class-presence minima are met."""
    if not variant.required_presence_by_split:
        return

    rows_by_class: dict[int, list[int]] = {}
    for class_id in variant.required_presence_by_split:
        rows = [
            int(row.row_idx)
            for row in pool.itertuples()
            if class_id in row_labels_by_idx[int(row.row_idx)]
        ]
        rows.sort(key=lambda row_idx: -score_by_row[row_idx])
        rows_by_class[class_id] = rows

    made_progress = True
    while made_progress and len(selected) < target_size:
        made_progress = False
        for class_id, split_targets in variant.required_presence_by_split.items():
            required = split_targets.get(split, 0)
            if selected_presence.get(class_id, 0) >= required:
                continue
            for row_idx in rows_by_class[class_id]:
                if row_idx in selected_set:
                    continue
                row_required_classes = row_labels_by_idx[row_idx] & set(
                    variant.required_presence_by_split
                )
                over_limit = any(
                    selected_presence.get(other_class_id, 0)
                    >= required_presence_limits.get(other_class_id, 0)
                    and not (
                        other_class_id == class_id
                        and selected_presence.get(other_class_id, 0) < required
                    )
                    for other_class_id in row_required_classes
                )
                if over_limit:
                    continue
                if not fits_presence_caps(
                    row_labels=row_labels_by_idx[row_idx],
                    selected_presence=selected_presence,
                    presence_caps=presence_caps,
                    exempt_classes=exempt_classes,
                    target_size=target_size,
                ):
                    continue
                add_selected_row(
                    row_idx=row_idx,
                    row_labels_by_idx=row_labels_by_idx,
                    selected=selected,
                    selected_set=selected_set,
                    selected_presence=selected_presence,
                )
                made_progress = True
                break

    unmet = {
        class_id: split_targets[split]
        for class_id, split_targets in variant.required_presence_by_split.items()
        if selected_presence.get(class_id, 0) < split_targets.get(split, 0)
    }
    if unmet:
        details = ", ".join(
            f"{CLASS_NAMES[class_id]}={selected_presence.get(class_id, 0)}/{target}"
            for class_id, target in unmet.items()
        )
        raise ValueError(f"Could not satisfy required {split} support: {details}")


def trim_to_fraction_caps(
    selected: list[int],
    row_labels_by_idx: dict[int, set[int]],
    score_by_row: dict[int, float],
    presence_caps: dict[int, float] | None,
    exempt_classes: set[int],
) -> list[int]:
    """Trim low-scoring rows until capped classes are below final-set fractions."""
    if not presence_caps:
        return selected

    selected_set = set(selected)
    while selected_set:
        presence = {class_id: 0 for class_id in presence_caps}
        for row_idx in selected_set:
            for class_id in presence_caps:
                if class_id in row_labels_by_idx[row_idx]:
                    presence[class_id] += 1

        violating_class = None
        for class_id, max_fraction in presence_caps.items():
            if presence[class_id] / len(selected_set) > max_fraction:
                violating_class = class_id
                break
        if violating_class is None:
            break

        removable = [
            row_idx
            for row_idx in selected_set
            if violating_class in row_labels_by_idx[row_idx]
            and not (row_labels_by_idx[row_idx] & exempt_classes)
        ]
        if not removable:
            break
        row_to_remove = min(
            removable, key=lambda row_idx: score_by_row.get(row_idx, 0.0)
        )
        selected_set.remove(row_to_remove)

    return [row_idx for row_idx in selected if row_idx in selected_set]


def select_variant_split(
    metadata: OsmMetadata,
    records: pd.DataFrame,
    pool_rows: np.ndarray,
    global_presence: np.ndarray,
    variant: VariantConfig,
    target_size: int,
    split: str,
    seed: int,
) -> pd.DataFrame:
    """Select one split for one variant."""
    anchors = eligible_anchor_classes(global_presence, variant.min_anchor_tile_presence)
    exempt_classes = cap_exempt_classes(global_presence, variant)
    pool = records.loc[records["row_idx"].isin(set(pool_rows.tolist()))].copy()
    pool = pool[
        (pool["num_classes_present"] >= variant.min_classes_present)
        & (pool["normalized_entropy"] >= variant.min_normalized_entropy)
    ].copy()

    anchor_rows: dict[int, list[int]] = {anchor: [] for anchor in anchors}
    for row in pool.itertuples():
        labels = np.fromstring(row.labels, dtype=int, sep=" ")
        anchor = choose_anchor(labels, anchors, global_presence)
        if anchor is not None:
            anchor_rows[anchor].append(int(row.row_idx))

    rng = np.random.RandomState(seed)
    score_by_row = {
        int(row.row_idx): selection_score(row, variant) for row in pool.itertuples()
    }
    for rows in anchor_rows.values():
        rows.sort(key=lambda row_idx: (-score_by_row[row_idx], rng.random()))

    row_labels_by_idx = {
        int(row.row_idx): labels_for_row(metadata, int(row.row_idx))
        for row in pool.itertuples()
    }
    anchor_order = sorted(anchors, key=lambda anchor: (global_presence[anchor], anchor))
    cursors = {anchor: 0 for anchor in anchor_order}
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_presence: dict[int, int] = {}

    while len(selected) < target_size:
        made_progress = False
        for anchor in anchor_order:
            rows = anchor_rows[anchor]
            cursor = cursors[anchor]
            while cursor < len(rows) and (
                rows[cursor] in selected_set
                or not fits_presence_caps(
                    row_labels=row_labels_by_idx[rows[cursor]],
                    selected_presence=selected_presence,
                    presence_caps=variant.presence_caps,
                    exempt_classes=exempt_classes,
                    target_size=target_size,
                )
            ):
                cursor += 1
            cursors[anchor] = cursor
            if cursor >= len(rows):
                continue
            row_idx = rows[cursor]
            add_selected_row(
                row_idx=row_idx,
                row_labels_by_idx=row_labels_by_idx,
                selected=selected,
                selected_set=selected_set,
                selected_presence=selected_presence,
            )
            cursors[anchor] += 1
            made_progress = True
            if len(selected) == target_size:
                break
        if not made_progress:
            break

    if len(selected) < target_size:
        for row in sorted(
            pool.itertuples(),
            key=lambda row: (-selection_score(row, variant), rng.random()),
        ):
            row_idx = int(row.row_idx)
            if row_idx in selected_set or not fits_presence_caps(
                row_labels=row_labels_by_idx[row_idx],
                selected_presence=selected_presence,
                presence_caps=variant.presence_caps,
                exempt_classes=exempt_classes,
                target_size=target_size,
            ):
                continue
            add_selected_row(
                row_idx=row_idx,
                row_labels_by_idx=row_labels_by_idx,
                selected=selected,
                selected_set=selected_set,
                selected_presence=selected_presence,
            )
            if len(selected) == target_size:
                break

    selected = trim_to_fraction_caps(
        selected=selected,
        row_labels_by_idx=row_labels_by_idx,
        score_by_row=score_by_row,
        presence_caps=variant.presence_caps,
        exempt_classes=exempt_classes,
    )

    selected_df = records.loc[records["row_idx"].isin(selected)].copy()
    selected_df["anchor_class_id"] = selected_df["labels"].map(
        lambda value: choose_anchor(
            np.fromstring(value, dtype=int, sep=" "),
            anchors,
            global_presence,
        )
    )
    selected_df["anchor_class_name"] = selected_df["anchor_class_id"].map(
        lambda value: CLASS_NAMES[int(value)] if value is not None else ""
    )
    return selected_df.sort_values("sample_index").reset_index(drop=True)


def summarize_split(selected: pd.DataFrame, metadata: OsmMetadata) -> pd.DataFrame:
    """Summarize true class presence in a selected split."""
    selected_rows = set(selected["row_idx"].astype(int).tolist())
    presence = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    pixels = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for row_idx in selected_rows:
        labels, counts = metadata.row_labels_counts(row_idx)
        presence[labels] += 1
        pixels[labels] += counts
    anchor_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    if "anchor_class_id" in selected.columns:
        anchors = selected["anchor_class_id"].dropna().astype(int)
        anchor_counts += np.bincount(anchors, minlength=len(CLASS_NAMES))
    total_pixels = int(pixels.sum())
    return pd.DataFrame(
        {
            "class_id": np.arange(len(CLASS_NAMES)),
            "class_name": CLASS_NAMES,
            "anchor_class_count": anchor_counts,
            "tile_presence_count": presence,
            "tile_presence_fraction": presence / max(len(selected_rows), 1),
            "pixel_count": pixels,
            "pixel_fraction": pixels / max(total_pixels, 1),
        }
    )


def split_presence_counts(
    selected: pd.DataFrame,
    row_labels_by_idx: dict[int, set[int]],
) -> dict[int, int]:
    """Count tile presence per class in one selected split."""
    counts: dict[int, int] = {}
    for row_idx in selected["row_idx"].astype(int):
        for class_id in row_labels_by_idx[row_idx]:
            counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def row_can_be_removed(
    *,
    row_idx: int,
    split_counts: dict[int, int],
    row_labels_by_idx: dict[int, set[int]],
    required_targets: dict[int, int],
) -> bool:
    """Return whether removing a row preserves required class minima."""
    for class_id in row_labels_by_idx[row_idx]:
        required = required_targets.get(class_id)
        if required is not None and split_counts.get(class_id, 0) - 1 < required:
            return False
    return True


def ensure_required_presence_by_replacement(
    *,
    selected_by_split: dict[str, pd.DataFrame],
    records: pd.DataFrame,
    metadata: OsmMetadata,
    variant: VariantConfig,
) -> dict[str, pd.DataFrame]:
    """Replace low-priority rows so every split meets required class support."""
    if not variant.required_presence_by_split:
        return selected_by_split

    row_labels_by_idx = {
        int(row.row_idx): labels_for_row(metadata, int(row.row_idx))
        for row in records.itertuples()
    }
    score_by_row = {
        int(row.row_idx): selection_score(row, variant) for row in records.itertuples()
    }
    selected_rows = {
        int(row_idx)
        for selected in selected_by_split.values()
        for row_idx in selected["row_idx"].astype(int).tolist()
    }
    records_by_row = {
        int(row.row_idx): row._asdict() for row in records.itertuples(index=False)
    }

    for split in SPLIT_SIZES:
        selected = selected_by_split[split].copy()
        required_targets = {
            class_id: split_targets.get(split, 0)
            for class_id, split_targets in variant.required_presence_by_split.items()
        }
        for class_id, required in required_targets.items():
            split_counts_by_split = {
                split_name: split_presence_counts(split_df, row_labels_by_idx)
                for split_name, split_df in selected_by_split.items()
            }
            split_counts = split_counts_by_split[split]
            while split_counts.get(class_id, 0) < required:
                row_to_split = {
                    int(row_idx): split_name
                    for split_name, split_df in selected_by_split.items()
                    for row_idx in split_df["row_idx"].astype(int).tolist()
                }
                candidates = [
                    row_idx
                    for row_idx, labels in row_labels_by_idx.items()
                    if class_id in labels
                    and row_to_split.get(row_idx) != split
                    and (
                        row_idx not in selected_rows
                        or row_can_be_removed(
                            row_idx=row_idx,
                            split_counts=split_counts_by_split[row_to_split[row_idx]],
                            row_labels_by_idx=row_labels_by_idx,
                            required_targets={
                                required_class_id: split_targets.get(
                                    row_to_split[row_idx], 0
                                )
                                for required_class_id, split_targets in (
                                    variant.required_presence_by_split or {}
                                ).items()
                            },
                        )
                    )
                ]
                candidates.sort(key=lambda row_idx: -score_by_row[row_idx])
                removable = [
                    int(row_idx)
                    for row_idx in selected["row_idx"].astype(int).tolist()
                    if class_id not in row_labels_by_idx[int(row_idx)]
                    and row_can_be_removed(
                        row_idx=int(row_idx),
                        split_counts=split_counts,
                        row_labels_by_idx=row_labels_by_idx,
                        required_targets=required_targets,
                    )
                ]
                removable.sort(key=lambda row_idx: score_by_row.get(row_idx, 0.0))
                if not candidates or not removable:
                    raise ValueError(
                        f"Could not meet {variant.name} {split} support for "
                        f"{CLASS_NAMES[class_id]}: "
                        f"{split_counts.get(class_id, 0)}/{required}"
                    )

                add_row_idx = candidates[0]
                remove_row_idx = removable[0]
                selected = selected[selected["row_idx"].astype(int) != remove_row_idx]
                selected = pd.concat(
                    [selected, pd.DataFrame([records_by_row[add_row_idx]])],
                    ignore_index=True,
                )
                donor_split = row_to_split.get(add_row_idx)
                if donor_split is not None:
                    selected_by_split[donor_split] = selected_by_split[donor_split][
                        selected_by_split[donor_split]["row_idx"].astype(int)
                        != add_row_idx
                    ]
                selected_rows.remove(remove_row_idx)
                selected_rows.add(add_row_idx)
                selected_by_split[split] = selected
                split_counts_by_split = {
                    split_name: split_presence_counts(split_df, row_labels_by_idx)
                    for split_name, split_df in selected_by_split.items()
                }
                split_counts = split_presence_counts(selected, row_labels_by_idx)

        selected_by_split[split] = selected.sort_values("sample_index").reset_index(
            drop=True
        )

    return selected_by_split


def main() -> None:
    """Generate OSM split candidate CSVs."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = OsmMetadata(args.metadata)
    global_presence = tile_presence_counts(metadata)
    records = tile_records(metadata, global_presence)
    source_pools = split_row_indices(len(metadata.sample_indices), args.seed)

    manifest = {"metadata": str(args.metadata), "variants": {}}
    for variant in VARIANTS:
        variant_dir = args.output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        manifest["variants"][variant.name] = {
            "config": {
                "min_anchor_tile_presence": variant.min_anchor_tile_presence,
                "min_classes_present": variant.min_classes_present,
                "min_normalized_entropy": variant.min_normalized_entropy,
                "presence_caps": {
                    CLASS_NAMES[class_id]: max_fraction
                    for class_id, max_fraction in (variant.presence_caps or {}).items()
                },
                "required_presence_by_split": {
                    CLASS_NAMES[class_id]: split_targets
                    for class_id, split_targets in (
                        variant.required_presence_by_split or {}
                    ).items()
                },
                "cap_exempt_classes": [
                    CLASS_NAMES[class_id]
                    for class_id in sorted(cap_exempt_classes(global_presence, variant))
                ],
            },
            "splits": {},
        }
        selected_by_split = {}
        for split, target_size in SPLIT_SIZES.items():
            target_size = SPLIT_SIZES[split]
            selected = select_variant_split(
                metadata=metadata,
                records=records,
                pool_rows=source_pools[split],
                global_presence=global_presence,
                variant=variant,
                target_size=target_size,
                split=split,
                seed=args.seed,
            )
            selected_by_split[split] = selected

        selected_by_split = ensure_required_presence_by_replacement(
            selected_by_split=selected_by_split,
            records=records,
            metadata=metadata,
            variant=variant,
        )

        for split, selected in selected_by_split.items():
            split_path = variant_dir / f"{split}.csv"
            selected.to_csv(split_path, index=False)
            summary_path = variant_dir / f"{split}_class_summary.csv"
            summarize_split(selected, metadata).to_csv(summary_path, index=False)
            manifest["variants"][variant.name]["splits"][split] = {
                "samples": int(len(selected)),
                "split_path": str(split_path),
                "class_summary_path": str(summary_path),
            }

        class_support_path = write_class_support(variant_dir)
        manifest["variants"][variant.name]["class_support_path"] = str(
            class_support_path
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
