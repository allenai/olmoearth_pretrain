"""Utilities for distributed corpus pipeline: shard splitting, Beaker job submission."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

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
    """Load corpus from CSV, JSON, or JSONL.

    JSON accepts either the original studio lon/lat list format or a list of records
    with the same fields as the CSV schema.
    """
    path = UPath(corpus_path)
    if path.suffix == ".jsonl":
        return _load_records_jsonl(path)
    if path.suffix == ".json":
        return _load_lonlats_json(path)
    return read_corpus_csv(path)


def _load_lonlats_json(path: UPath) -> list[CorpusEntry]:
    """Load corpus JSON.

    Supports the original studio_corpus_lonlats.json format, a list of [lon, lat]
    pairs, plus record-style JSON matching the CSV schema.
    """
    from olmoearth_pretrain.dataset_creation.constants import WINDOW_DURATION

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "samples" not in data:
            raise ValueError(f"JSON corpus object must contain a 'samples' key: {path}")
        data = data["samples"]

    if not isinstance(data, list):
        raise ValueError(f"JSON corpus must be a list or object with samples: {path}")

    if not data:
        return []

    if isinstance(data[0], dict):
        return _records_to_entries(data)

    entries = []
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = start_time + WINDOW_DURATION

    for i, pair in enumerate(data):
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            raise ValueError(f"expected [lon, lat] pair at JSON row {i}: {pair!r}")
        lon, lat = pair
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


def _load_records_jsonl(path: UPath) -> list[CorpusEntry]:
    records = []
    with path.open() as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"expected JSON object at JSONL row {line_idx}")
            records.append(record)
    return _records_to_entries(records)


def _parse_time(raw: object) -> datetime:
    if not isinstance(raw, str):
        raise ValueError(f"timestamp must be a string, got {type(raw).__name__}")
    value = raw.strip()
    if not value:
        raise ValueError("empty timestamp")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _records_to_entries(records: list[dict]) -> list[CorpusEntry]:
    from olmoearth_pretrain.dataset_creation.constants import WINDOW_DURATION

    entries = []
    seen: set[str] = set()
    required = {"sample_id", "lon", "lat", "start_time"}
    for row_idx, record in enumerate(records):
        missing = required - set(record)
        if missing:
            raise ValueError(f"JSON corpus row {row_idx} missing columns {missing}")

        sample_id = str(record["sample_id"]).strip()
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id {sample_id!r} at row {row_idx}")
        seen.add(sample_id)

        start_time = _parse_time(record["start_time"])
        end_raw = record.get("end_time")
        end_time = _parse_time(end_raw) if end_raw else start_time + WINDOW_DURATION
        entries.append(
            CorpusEntry(
                sample_id=sample_id,
                lon=float(record["lon"]),
                lat=float(record["lat"]),
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
    clusters: list[str] | str,
    shard_ids: list[int] | None = None,
) -> list[str]:
    """Submit Beaker jobs for each shard.

    Args:
        run_name: identifier for this pipeline run (used as Beaker name prefix).
        step_name: human-readable name prefix (e.g. "rslearn", "convert").
        worker_cmd_template: command list with {shard_id} and {num_shards} placeholders.
        num_shards: total number of shards.
        clusters: Beaker cluster(s) (e.g. "ai2/jupiter" or ["ai2/jupiter", "ai2/saturn"]).
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
            clusters=clusters,
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
