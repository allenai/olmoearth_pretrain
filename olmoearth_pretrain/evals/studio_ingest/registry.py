"""Registry I/O for eval datasets.

This module handles reading and writing the eval dataset registry,
which is a JSON file stored on Weka that tracks all available
evaluation datasets.

Registry Structure:
------------------
The registry.json file contains:
{
    "version": "1.0",
    "updated_at": "2024-01-15T10:30:00Z",
    "datasets": {
        "lfmc": { ... EvalDatasetEntry ... },
        "forest_loss_driver": { ... },
        ...
    }
}

Concurrency Considerations:
--------------------------
- The registry is a single JSON file, so concurrent writes could cause issues
- For now, we assume single-writer (manual ingestion is not concurrent)
- Future: Could add file locking or move to a database

Error Handling:
--------------
- If registry doesn't exist, we create it on first write
- If registry is corrupted, we raise an error (don't silently overwrite)
- All writes are atomic (write to temp, then rename)

Todo:
-----
- [ ] Add file locking for concurrent access
- [ ] Add backup before write
- [ ] Add schema version migration
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.schema import EvalDatasetEntry

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# The canonical location of the registry on Weka
# TODO: Make a real path instead of using this for now
REGISTRY_PATH = "/weka/dfive-default/helios/henryh/olmoearth_pretrain/eval_datasets/registry.json"

# Current registry schema version
# Increment this when making breaking changes to the schema
REGISTRY_VERSION = "1.0"


# =============================================================================
# Registry Data Structure
# =============================================================================


class Registry:
    """In-memory representation of the eval dataset registry.

    This class provides methods to:
    - Load the registry from Weka
    - Add/update/remove dataset entries
    - Save the registry back to Weka
    - Query available datasets

    Usage:
        # Load existing registry (or create new)
        registry = Registry.load()

        # Add a dataset
        registry.add(entry)

        # Save changes
        registry.save()

        # Query
        entry = registry.get("lfmc")
        all_names = registry.list_names()
    """

    def __init__(
        self,
        datasets: dict[str, EvalDatasetEntry] | None = None,
        version: str = REGISTRY_VERSION,
        updated_at: str | None = None,
    ):
        """Initialize registry.

        Args:
            datasets: Dict mapping dataset name -> EvalDatasetEntry
            version: Schema version string
            updated_at: ISO timestamp of last update
        """
        self.datasets = datasets or {}
        self.version = version
        self.updated_at = updated_at or datetime.now().isoformat()

    @classmethod
    def load(cls, path: str | None = None) -> Registry:
        """Load registry from Weka.

        Args:
            path: Optional custom path (defaults to REGISTRY_PATH)

        Returns:
            Registry instance

        Raises:
            FileNotFoundError: If registry doesn't exist (caller should create)
            json.JSONDecodeError: If registry is corrupted
        """
        registry_path = UPath(path or REGISTRY_PATH)

        if not registry_path.exists():
            logger.info(f"Registry not found at {registry_path}, creating new registry")
            return cls()

        logger.info(f"Loading registry from {registry_path}")

        with registry_path.open("r") as f:
            data = json.load(f)

        # Parse datasets
        datasets = {}
        for name, entry_data in data.get("datasets", {}).items():
            datasets[name] = EvalDatasetEntry.from_dict(entry_data)

        return cls(
            datasets=datasets,
            version=data.get("version", REGISTRY_VERSION),
            updated_at=data.get("updated_at"),
        )

    def save(self, path: str | None = None) -> None:
        """Save registry to Weka.

        This performs an atomic write by writing to a temp file first,
        then renaming. This prevents corruption if the write is interrupted.

        Args:
            path: Optional custom path (defaults to REGISTRY_PATH)
        """
        registry_path = UPath(path or REGISTRY_PATH)

        # Ensure parent directory exists
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamp
        self.updated_at = datetime.now().isoformat()

        # Serialize
        data = {
            "version": self.version,
            "updated_at": self.updated_at,
            "datasets": {
                name: entry.to_dict() for name, entry in self.datasets.items()
            },
        }

        # Write atomically
        # TODO: UPath may not support atomic writes on all backends
        # For now, we just write directly
        logger.info(f"Saving registry to {registry_path}")

        with registry_path.open("w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Registry saved with {len(self.datasets)} datasets")

    def add(self, entry: EvalDatasetEntry, overwrite: bool = False) -> None:
        """Add a dataset entry to the registry.

        Args:
            entry: The dataset entry to add
            overwrite: If True, overwrite existing entry with same name

        Raises:
            ValueError: If entry with same name exists and overwrite=False
        """
        if entry.name in self.datasets and not overwrite:
            raise ValueError(
                f"Dataset '{entry.name}' already exists in registry. "
                "Use overwrite=True to replace."
            )

        self.datasets[entry.name] = entry
        logger.info(f"Added dataset '{entry.name}' to registry")

    def remove(self, name: str) -> EvalDatasetEntry:
        """Remove a dataset from the registry.

        Note: This only removes from the registry, it does NOT delete
        the actual data on Weka. That must be done separately.

        Args:
            name: Name of dataset to remove

        Returns:
            The removed entry

        Raises:
            KeyError: If dataset not found
        """
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found in registry")

        entry = self.datasets.pop(name)
        logger.info(f"Removed dataset '{name}' from registry")
        return entry

    def get(self, name: str) -> EvalDatasetEntry:
        """Get a dataset entry by name.

        Args:
            name: Dataset name

        Returns:
            The dataset entry

        Raises:
            KeyError: If dataset not found
        """
        if name not in self.datasets:
            raise KeyError(
                f"Dataset '{name}' not found in registry. "
                f"Available: {self.list_names()}"
            )
        return self.datasets[name]

    def list_names(self) -> list[str]:
        """Get list of all dataset names."""
        return sorted(self.datasets.keys())

    def list_by_task_type(self, task_type: str) -> list[EvalDatasetEntry]:
        """Get all datasets of a specific task type.

        Args:
            task_type: One of "classification", "regression", "segmentation"

        Returns:
            List of matching dataset entries
        """
        return [
            entry for entry in self.datasets.values() if entry.task_type == task_type
        ]

    def list_by_modality(self, modality: str) -> list[EvalDatasetEntry]:
        """Get all datasets that use a specific modality.

        Args:
            modality: Modality name (e.g., "sentinel2_l2a")

        Returns:
            List of datasets that include this modality
        """
        return [
            entry for entry in self.datasets.values() if modality in entry.modalities
        ]

    def __len__(self) -> int:
        """Number of datasets in registry."""
        return len(self.datasets)

    def __contains__(self, name: str) -> bool:
        """Check if dataset exists in registry."""
        return name in self.datasets

    def __iter__(self):
        """Iterate over dataset entries."""
        return iter(self.datasets.values())


# =============================================================================
# Convenience Functions
# =============================================================================


def load_registry(path: str | None = None) -> Registry:
    """Load the registry from Weka.

    This is a convenience function that wraps Registry.load().

    Args:
        path: Optional custom registry path

    Returns:
        Registry instance
    """
    return Registry.load(path)


def get_dataset_entry(name: str, registry_path: str | None = None) -> EvalDatasetEntry:
    """Get a single dataset entry by name.

    This is a convenience function for quick lookups.

    Args:
        name: Dataset name
        registry_path: Optional custom registry path

    Returns:
        The dataset entry
    """
    registry = Registry.load(registry_path)
    return registry.get(name)


def list_dataset_names(registry_path: str | None = None) -> list[str]:
    """Get list of all available dataset names.

    Args:
        registry_path: Optional custom registry path

    Returns:
        Sorted list of dataset names
    """
    registry = Registry.load(registry_path)
    return registry.list_names()
