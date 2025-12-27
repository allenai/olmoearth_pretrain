"""Validate rslearn datasets before ingestion.

This module provides validation functions to ensure an rslearn dataset
is suitable for ingestion into the OlmoEarth eval system.

Validation Checks:
-----------------
1. Dataset structure: Verify the path exists and is a valid rslearn dataset
2. Modalities: Check that requested modalities exist in the dataset
3. Splits: Verify train/val/test splits are defined
4. Labels: Check that the target property exists
5. Data integrity: Optionally spot-check a few samples

Why Validate?
-------------
- Catch issues early before spending time on copying data
- Provide clear error messages about what's wrong
- Ensure consistency across all ingested datasets

Usage:
------
    from olmoearth_pretrain.evals.studio_ingest.validate import validate_dataset

    # Quick validation
    result = validate_dataset("gs://bucket/dataset", modalities=["sentinel2_l2a"])

    if result.is_valid:
        print("Dataset is valid!")
    else:
        for error in result.errors:
            print(f"Error: {error}")

Todo:
-----
- [ ] Add more detailed data integrity checks
- [ ] Check for consistent image sizes
- [ ] Check for temporal coverage
- [ ] Add warnings for potential issues (not just errors)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from upath import UPath

from olmoearth_pretrain.data.constants import Modality as DataModality

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class ValidationResult:
    """Result of dataset validation.

    Attributes:
        is_valid: True if all checks passed
        errors: List of error messages (blocking issues)
        warnings: List of warning messages (non-blocking issues)
        info: Dict of informational data collected during validation
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error and mark result as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(message)

    def __str__(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.is_valid:
            lines.append("✓ Dataset is valid")
        else:
            lines.append("✗ Dataset is INVALID")

        if self.errors:
            lines.append("\nErrors:")
            for e in self.errors:
                lines.append(f"  ✗ {e}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")

        if self.info:
            lines.append("\nInfo:")
            for k, v in self.info.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)


# =============================================================================
# Individual Validation Checks
# =============================================================================


def check_dataset_exists(path: UPath, result: ValidationResult) -> bool:
    """Check that the dataset path exists.

    Args:
        path: Dataset path
        result: ValidationResult to update

    Returns:
        True if exists, False otherwise
    """
    if not path.exists():
        result.add_error(f"Dataset path does not exist: {path}")
        return False

    result.info["path"] = str(path)
    return True


def check_rslearn_structure(path: UPath, result: ValidationResult) -> bool:
    """Check that the path has valid rslearn dataset structure.

    An rslearn dataset should have:
    - config.json or similar metadata
    - windows/ directory with data

    Args:
        path: Dataset path
        result: ValidationResult to update

    Returns:
        True if valid structure, False otherwise

    Todo:
        - This needs to be updated based on actual rslearn structure
        - May need to import rslearn and use its validation
    """
    # TODO: Implement actual rslearn structure validation
    # For now, just check it's a directory
    if not path.is_dir():
        result.add_error(f"Dataset path is not a directory: {path}")
        return False

    # Check for windows directory (rslearn convention)
    # TODO: Verify this is the correct structure
    windows_path = path / "windows"
    if not windows_path.exists():
        result.add_warning(
            f"No 'windows' directory found at {path}. "
            "This may not be a valid rslearn dataset."
        )

    return True


def check_modalities_exist(
    path: UPath,
    modalities: list[str],
    result: ValidationResult,
) -> bool:
    """Check that requested modalities exist in the dataset.

    Args:
        path: Dataset path
        modalities: List of modality names to check
        result: ValidationResult to update

    Returns:
        True if all modalities exist, False otherwise

    Todo:
        - This needs actual rslearn integration to check layers
        - For now, just validates modality names are valid OlmoEarth modalities
    """
    valid_modalities = {m.name for m in DataModality.values()}

    for modality in modalities:
        if modality not in valid_modalities:
            result.add_error(
                f"Unknown modality '{modality}'. "
                f"Valid modalities: {sorted(valid_modalities)}"
            )
            return False

    result.info["modalities"] = modalities

    # TODO: Actually check if these layers exist in the rslearn dataset
    # This requires loading the dataset and checking available layers

    return True


def check_splits_exist(
    path: UPath,
    result: ValidationResult,
) -> dict[str, int]:
    """Check that train/val/test splits are defined.

    Args:
        path: Dataset path
        result: ValidationResult to update

    Returns:
        Dict mapping split name -> sample count

    Todo:
        - This needs actual rslearn integration
        - For now, returns placeholder
    """
    # TODO: Implement actual split checking
    # rslearn datasets may define splits via tags or subdirectories

    # Placeholder - return empty and add warning
    result.add_warning(
        "Split validation not implemented. Assuming train/val/test splits exist."
    )

    # TODO: Return actual split counts
    return {}


def check_target_property(
    path: UPath,
    property_name: str,
    task_type: str,
    result: ValidationResult,
) -> bool:
    """Check that the target property exists in the dataset.

    Args:
        path: Dataset path
        property_name: Name of the property containing labels
        task_type: Type of task (classification, regression, etc.)
        result: ValidationResult to update

    Returns:
        True if property exists, False otherwise

    Todo:
        - This needs actual rslearn integration
        - Check that property values match expected type
    """
    result.info["target_property"] = property_name
    result.info["task_type"] = task_type

    # TODO: Actually verify the property exists
    result.add_warning(
        f"Target property '{property_name}' validation not implemented. "
        "Assuming it exists."
    )

    return True


def spot_check_samples(
    path: UPath,
    num_samples: int,
    result: ValidationResult,
) -> bool:
    """Spot check a few samples to verify data integrity.

    This loads a few random samples and checks:
    - Data can be loaded without errors
    - Data has expected shape
    - Data values are in reasonable range

    Args:
        path: Dataset path
        num_samples: Number of samples to check
        result: ValidationResult to update

    Returns:
        True if spot check passes, False otherwise

    Todo:
        - Implement actual sample loading and checking
    """
    result.add_warning(
        f"Spot check of {num_samples} samples not implemented. Skipping."
    )
    return True


# =============================================================================
# Main Validation Function
# =============================================================================


def validate_dataset(
    source_path: str | UPath,
    modalities: list[str],
    task_type: str = "classification",
    target_property: str = "category",
    spot_check: bool = True,
    num_spot_check_samples: int = 5,
) -> ValidationResult:
    """Validate an rslearn dataset for ingestion.

    This runs all validation checks and returns a ValidationResult.

    Args:
        source_path: Path to the rslearn dataset (GCS or local)
        modalities: List of modality names to validate
        task_type: Type of task (classification, regression, segmentation)
        target_property: Name of the property containing labels
        spot_check: Whether to spot-check sample data
        num_spot_check_samples: Number of samples to spot-check

    Returns:
        ValidationResult with all errors, warnings, and info

    Example:
        result = validate_dataset(
            "gs://bucket/dataset",
            modalities=["sentinel2_l2a"],
            task_type="classification",
            target_property="category",
        )

        if not result.is_valid:
            print(result)
            raise ValueError("Dataset validation failed")
    """
    result = ValidationResult()
    path = UPath(source_path)

    logger.info(f"Validating dataset at {path}")

    # Run checks in order - some depend on previous checks passing
    if not check_dataset_exists(path, result):
        return result

    if not check_rslearn_structure(path, result):
        return result

    if not check_modalities_exist(path, modalities, result):
        return result

    splits = check_splits_exist(path, result)
    result.info["splits"] = splits

    check_target_property(path, target_property, task_type, result)

    if spot_check:
        spot_check_samples(path, num_spot_check_samples, result)

    logger.info(f"Validation complete: {'VALID' if result.is_valid else 'INVALID'}")
    return result
