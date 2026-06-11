"""Registry of *direct* rslearn tasks for ERA5 supervised pretraining (v0).

Unlike the eval-dataset registry (``olmoearth_pretrain.evals.studio_ingest``),
entries here point straight at rslearn datasets that already live on Weka with
their own ``model.yaml`` (+ ``computed.json`` norm stats). No ingestion, copy,
or stat-computation step is required — an entry is pure metadata.

Each entry is referenced by a short nickname which is what you pass on the
command line, e.g.::

    common.tasks=[burnrisk_canada_nbac, lfmc_global]

The pipeline then resolves every nickname into a fully-specified
``Era5TaskSpec`` (weka path + model.yaml + task metadata), so multiple direct
tasks compose into one supervised objective without re-typing per-task paths.

Fields that can be derived from ``model.yaml`` at resolve time (the rslearn
task type, and hence whether a segmentation target must be reduced to a scalar
classification label) are derived in ``script.py`` rather than duplicated here;
the registry stays a thin, declarative pointer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from upath import UPath

logger = logging.getLogger(__name__)

# Git-tracked registry that ships next to this script (source of truth).
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "direct_registry.json"


class DirectRslearnTaskEntry(BaseModel):
    """Metadata for one direct-load rslearn task.

    Args:
        name: Nickname / task identifier (also used as the supervised head
            name). Injected from the registry key, so it does not need to be
            repeated inside each entry's JSON body.
        weka_path: Path to the rslearn dataset on Weka (the dir with
            ``config.json``). This overrides whatever ``data.init_args.path`` is
            baked into ``model.yaml``.
        model_yaml_path: Path to the data-only ``model.yaml`` describing the
            rslearn inputs / task / transforms.
        task_type: Supervised task type emitted by the head
            (``classification`` / ``regression``). Note this is the *olmoearth*
            task type, which may differ from the rslearn task type in
            ``model.yaml`` (e.g. an rslearn ``segmentation`` dataset consumed as
            scalar ``classification``).
        num_classes: Number of output classes (regression: number of outputs).
        is_multilabel: Multi-label classification flag.
        modality_layer_name: rslearn input-dict key holding the ERA5-daily
            tensor (must match ``data.init_args.inputs`` in ``model.yaml``).
        label_extractor_name: Optional explicit ``LabelExtractor`` name. Leave
            ``None`` to let the pipeline auto-select (e.g. ``segmentation_to_scalar``
            when an rslearn segmentation target feeds a classification head).
        weight: Default sampling weight relative to the other tasks. Can be
            overridden per run via ``common.task_weights``.
        groups: rslearn window groups to read for the train split.
        tags: Optional rslearn tag filter applied on top of ``groups``.
        val_groups: rslearn window groups for the val split. ``None`` defers to
            the ``val_config`` in ``model.yaml``.
        val_tags: Optional rslearn tag filter for the val split.
        test_groups: rslearn window groups for the test split. ``None`` defers
            to the ``test_config`` in ``model.yaml``.
        test_tags: Optional rslearn tag filter for the test split.
        norm_stats_from_pretrained: Use the pretrain ``computed.json`` stats.
        max_samples: Optional cap on the number of samples (debug / smoke runs).
        notes: Free-form human notes (ignored by the pipeline).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    weka_path: str
    model_yaml_path: str
    task_type: str = "classification"
    num_classes: int | None = 2
    is_multilabel: bool = False
    modality_layer_name: str = "era5_daily"
    label_extractor_name: str | None = None
    weight: float = 1.0
    groups: list[str] = Field(default_factory=lambda: ["train"])
    tags: dict[str, str] = Field(default_factory=dict)
    val_groups: list[str] | None = None
    val_tags: dict[str, str] = Field(default_factory=dict)
    test_groups: list[str] | None = None
    test_tags: dict[str, str] = Field(default_factory=dict)
    norm_stats_from_pretrained: bool = True
    max_samples: int | None = None
    notes: str | None = None


class DirectRslearnRegistry:
    """In-memory view of the direct-rslearn task registry (a JSON file)."""

    def __init__(self, tasks: dict[str, DirectRslearnTaskEntry]):
        """Initialize from a mapping of nickname -> entry."""
        self.tasks = tasks

    @classmethod
    def load(cls, path: str | None = None) -> DirectRslearnRegistry:
        """Load the registry from JSON.

        Args:
            path: Optional custom path (defaults to ``DEFAULT_REGISTRY_PATH``).
        """
        registry_path = (
            UPath(path) if path is not None else UPath(DEFAULT_REGISTRY_PATH)
        )
        if not registry_path.exists():
            raise FileNotFoundError(
                f"Direct rslearn registry not found at {registry_path}. "
                "Create it or pass common.direct_registry_path."
            )
        logger.info("Loading direct rslearn registry from %s", registry_path)
        with registry_path.open("r") as f:
            data = json.load(f)
        tasks = {
            name: DirectRslearnTaskEntry.model_validate({**entry, "name": name})
            for name, entry in data.get("tasks", {}).items()
        }
        return cls(tasks)

    def get(self, name: str) -> DirectRslearnTaskEntry:
        """Get an entry by nickname, with a helpful error listing options."""
        if name not in self.tasks:
            raise KeyError(
                f"Task {name!r} not found in direct rslearn registry. "
                f"Available: {self.list_names()}"
            )
        return self.tasks[name]

    def list_names(self) -> list[str]:
        """Return all registered nicknames, sorted."""
        return sorted(self.tasks.keys())

    def __contains__(self, name: str) -> bool:
        """Check whether a nickname is registered."""
        return name in self.tasks

    def __iter__(self) -> Iterator[DirectRslearnTaskEntry]:
        """Iterate over entries."""
        return iter(self.tasks.values())

    def __len__(self) -> int:
        """Number of registered tasks."""
        return len(self.tasks)
