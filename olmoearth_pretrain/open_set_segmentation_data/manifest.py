"""Load the dataset manifest and the on-disk registry, and update registry status.

The registry (``registry.json``) is the source of truth for the slug a dataset uses and
its completion status. Do not re-derive slugs; look them up here.

Layout: the central ``registry.json`` and the ``AGENT_SUMMARY.md`` task spec live in the
repo under ``data/open_set_segmentation_data/`` (version-controlled). The bulk label
outputs (``datasets/{slug}/`` with metadata/locations, each dataset's own
``registry_entry.json``, and ``raw/{slug}/``) live on weka under ``OUTPUT_ROOT``.
"""

import json
import re
from typing import Any

from upath import UPath

MANIFEST_PATH = UPath("data/open_set_segmentation_datasets.json")
# Repo (version-controlled) home for the registry + task spec.
REPO_DATA_ROOT = UPath("data/open_set_segmentation_data")
# Weka home for the bulk label outputs (datasets/, raw/, per-dataset registry_entry.json).
OUTPUT_ROOT = UPath("/weka/dfive-default/helios/dataset_creation/open_set_segmentation")
REGISTRY_PATH = REPO_DATA_ROOT / "registry.json"


def slugify(name: str) -> str:
    """Snake_case slug: lowercase, non-alphanumeric -> '_', collapse repeats."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return re.sub(r"_+", "_", s).strip("_")


def load_manifest() -> list[dict[str, Any]]:
    """Return the list of dataset entries from the manifest JSON."""
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def load_registry() -> dict[str, Any]:
    """Return the parsed registry."""
    with REGISTRY_PATH.open() as f:
        return json.load(f)


def get_entry(slug: str) -> dict[str, Any]:
    """Return the registry entry for a slug (raises if missing)."""
    reg = load_registry()
    for e in reg["datasets"]:
        if e["slug"] == slug:
            return e
    raise KeyError(f"slug {slug!r} not in registry")


def find_slug(name: str) -> str:
    """Return the registry slug for a manifest name."""
    reg = load_registry()
    for e in reg["datasets"]:
        if e["name"] == name:
            return e["slug"]
    raise KeyError(f"name {name!r} not in registry")


def registry_entry_path(slug: str) -> UPath:
    """Path of a dataset's own registry_entry.json (on weka, in its dataset dir)."""
    return OUTPUT_ROOT / "datasets" / slug / "registry_entry.json"


def write_registry_entry(
    slug: str,
    status: str,
    task_type: str | None = None,
    num_samples: int | None = None,
    notes: str | None = None,
) -> None:
    """Record a dataset's status in its OWN ``datasets/{slug}/registry_entry.json``.

    Dataset scripts call this. It NEVER touches the central ``registry.json`` — that file
    is owned solely by the orchestrator, which merges these per-dataset entries via
    ``aggregate_registry()``. Writing per-dataset avoids concurrent-write corruption of
    the shared central file.
    """
    entry = {
        "slug": slug,
        "status": status,
        "task_type": task_type,
        "num_samples": num_samples,
        "notes": notes or "",
    }
    p = registry_entry_path(slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / (p.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(entry, f, indent=2)
    tmp.rename(p)


# Back-compat alias: existing scripts call update_status; it now writes the per-dataset
# entry only (never the central registry). Prefer write_registry_entry in new code.
def update_status(
    slug: str,
    status: str,
    task_type: str | None = None,
    num_samples: int | None = None,
    notes: str | None = None,
) -> None:
    """Deprecated name for :func:`write_registry_entry` (per-dataset entry only)."""
    write_registry_entry(slug, status, task_type, num_samples, notes)


def aggregate_registry() -> dict[str, Any]:
    """ORCHESTRATOR ONLY: merge all datasets/*/registry_entry.json into central registry.json.

    Reads each dataset dir's registry_entry.json and copies status/task_type/num_samples/
    notes into the matching central entry, then writes registry.json atomically. Returns
    the updated registry dict. Dataset scripts must never call this.
    """
    reg = load_registry()
    by_slug = {e["slug"]: e for e in reg["datasets"]}
    datasets_dir = OUTPUT_ROOT / "datasets"
    if datasets_dir.exists():
        for d in datasets_dir.iterdir():
            ep = d / "registry_entry.json"
            if not ep.exists():
                continue
            with ep.open() as f:
                entry = json.load(f)
            slug = entry.get("slug", d.name)
            if slug in by_slug:
                for k in ("status", "task_type", "num_samples", "notes"):
                    if k in entry and entry[k] is not None:
                        by_slug[slug][k] = entry[k]
    tmp = REGISTRY_PATH.parent / (REGISTRY_PATH.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(reg, f, indent=2)
    tmp.rename(REGISTRY_PATH)
    return reg
