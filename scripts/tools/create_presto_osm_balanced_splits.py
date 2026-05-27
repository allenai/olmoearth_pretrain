"""Create balanced OpenStreetMap raster split candidates from cached Presto metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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


VARIANTS = [
    VariantConfig(
        name="osm_base_balanced",
        min_anchor_tile_presence=500,
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
        default=Path("analysis/presto_osm_balanced_splits"),
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
    rare_ids = set(np.where((global_presence > 0) & (global_presence <= rare_cutoff))[0])

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


def choose_anchor(labels: np.ndarray, anchors: set[int], global_presence: np.ndarray) -> int | None:
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
        row_to_remove = min(removable, key=lambda row_idx: score_by_row.get(row_idx, 0.0))
        selected_set.remove(row_to_remove)

    return [row_idx for row_idx in selected if row_idx in selected_set]


def select_variant_split(
    metadata: OsmMetadata,
    records: pd.DataFrame,
    pool_rows: np.ndarray,
    global_presence: np.ndarray,
    variant: VariantConfig,
    target_size: int,
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
            while (
                cursor < len(rows)
                and (
                    rows[cursor] in selected_set
                    or not fits_presence_caps(
                        row_labels=row_labels_by_idx[rows[cursor]],
                        selected_presence=selected_presence,
                        presence_caps=variant.presence_caps,
                        exempt_classes=exempt_classes,
                        target_size=target_size,
                    )
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
    total_pixels = int(pixels.sum())
    return pd.DataFrame(
        {
            "class_id": np.arange(len(CLASS_NAMES)),
            "class_name": CLASS_NAMES,
            "tile_presence_count": presence,
            "tile_presence_fraction": presence / max(len(selected_rows), 1),
            "pixel_count": pixels,
            "pixel_fraction": pixels / max(total_pixels, 1),
        }
    )


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
                "cap_exempt_classes": [
                    CLASS_NAMES[class_id]
                    for class_id in sorted(cap_exempt_classes(global_presence, variant))
                ],
            },
            "splits": {},
        }
        for split, target_size in SPLIT_SIZES.items():
            selected = select_variant_split(
                metadata=metadata,
                records=records,
                pool_rows=source_pools[split],
                global_presence=global_presence,
                variant=variant,
                target_size=target_size,
                seed=args.seed,
            )
            split_path = variant_dir / f"{split}.csv"
            selected.to_csv(split_path, index=False)
            summary_path = variant_dir / f"{split}_class_summary.csv"
            summarize_split(selected, metadata).to_csv(summary_path, index=False)
            manifest["variants"][variant.name]["splits"][split] = {
                "samples": int(len(selected)),
                "split_path": str(split_path),
                "class_summary_path": str(summary_path),
            }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
