"""Utilities for distributed corpus pipeline: shard splitting, Beaker job submission."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from upath import UPath

from olmoearth_pretrain.dataset_creation.create_windows.from_corpus import (
    CorpusEntry,
    read_corpus_csv,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardSpec:
    """Defines a shard: which slice of the corpus this worker handles."""

    shard_id: int
    num_shards: int
    entries: list[CorpusEntry]

    @property
    def window_names(self) -> list[str]:
        """Return the rslearn window names for this shard."""
        return [e.sample_id for e in self.entries]


def load_corpus(corpus_path: str) -> list[CorpusEntry]:
    """Load corpus from CSV or JSON (studio_corpus_lonlats.json format)."""
    path = UPath(corpus_path)
    if path.suffix == ".json":
        return _load_lonlats_json(path)
    return read_corpus_csv(path)


def _load_lonlats_json(path: UPath) -> list[CorpusEntry]:
    """Load the studio_corpus_lonlats.json format: list of [lon, lat] pairs.

    Generates deterministic sample_ids and default time ranges.
    """
    from datetime import UTC, datetime

    from olmoearth_pretrain.dataset_creation.constants import WINDOW_DURATION

    with path.open() as f:
        lonlats = json.load(f)

    entries = []
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = start_time + WINDOW_DURATION

    for i, (lon, lat) in enumerate(lonlats):
        entries.append(
            CorpusEntry(
                sample_id=f"corpus_{i:07d}",
                lon=float(lon),
                lat=float(lat),
                start_time=start_time,
                end_time=end_time,
            )
        )
    return entries


def split_into_shards(entries: list[CorpusEntry], num_shards: int) -> list[ShardSpec]:
    """Split corpus entries into num_shards disjoint shards."""
    shards = []
    for shard_id in range(num_shards):
        shard_entries = entries[shard_id::num_shards]
        shards.append(
            ShardSpec(shard_id=shard_id, num_shards=num_shards, entries=shard_entries)
        )
    return shards


def get_shard(entries: list[CorpusEntry], shard_id: int, num_shards: int) -> ShardSpec:
    """Get a single shard without materializing all shards."""
    shard_entries = entries[shard_id::num_shards]
    return ShardSpec(shard_id=shard_id, num_shards=num_shards, entries=shard_entries)


def launch_beaker_jobs(
    *,
    run_name: str,
    step_name: str,
    worker_cmd_template: list[str],
    num_shards: int,
    cluster: str,
    shard_ids: list[int] | None = None,
) -> list[str]:
    """Submit Beaker jobs for each shard.

    Args:
        run_name: identifier for this pipeline run (used as Beaker name prefix).
        step_name: human-readable name prefix (e.g. "rslearn", "convert").
        worker_cmd_template: command list with {shard_id} and {num_shards} placeholders.
        num_shards: total number of shards.
        cluster: Beaker cluster (e.g. "ai2/jupiter").
        shard_ids: if set, only launch these shard ids (for resume).

    Returns:
        list of experiment IDs.
    """
    from olmo_core.utils import generate_uuid

    from olmoearth_pretrain.internal.common import build_launch_config

    if shard_ids is None:
        shard_ids = list(range(num_shards))

    experiment_ids = []
    for sid in shard_ids:
        cmd = [
            c.format(shard_id=sid, num_shards=num_shards) for c in worker_cmd_template
        ]
        config = build_launch_config(
            name=f"{run_name}-{step_name}-shard{sid:05d}-{generate_uuid()[:6]}",
            cmd=cmd,
            clusters=cluster,
            task_name=f"{step_name}-worker",
        )
        config.num_gpus = 0
        config.num_nodes = 1
        config.preemptible = True
        config.retries = 2
        # Replace --all-extras (which conflicts) with only what data workers need
        config.setup_steps = [
            s.replace(
                "uv sync --locked --all-extras",
                "uv sync --locked --extra dataset-creation",
            )
            for s in config.setup_steps
        ]

        logger.info(f"Launching shard {sid}/{num_shards}: {' '.join(cmd)}")
        experiment_id = config.launch(torchrun=False, entrypoint="python")
        experiment_ids.append(experiment_id)

    logger.info(f"Launched {len(experiment_ids)} {step_name} jobs")
    return experiment_ids
