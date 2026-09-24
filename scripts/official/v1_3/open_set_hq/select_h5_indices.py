r"""Select the H5 sample indices of the high-quality open-set datasets.

The open-set H5s do not store the label-bank dataset a sample came from by name, but
every sample carries one of the two label layers and both encode it:

* ``open_set`` holds globally-unique class ids, and each class in
  ``class_mapping.json`` (``open_set.classes[*].slug``) names its source dataset;
* ``open_set_regression`` band 0 holds the 1-based regression dataset id, mapped to a
  slug by ``class_mapping.json`` (``open_set_regression.datasets``).

So the slug of an H5 sample is read straight off its labels -- the same labels the
probe trains on -- with no geometry matching. (Matching windows by location is
ambiguous: many label-bank datasets have several samples at the same window, e.g. one
per year, so a latlon does not identify an example.) The scan opens every H5 file once
and reads only the label layer(s), which is cheap next to the imagery.

A few classes are merged concepts shared by several datasets (``slug: null`` with a
``members`` list, e.g. ``wind_turbine``). Their pixels are credited to the window's
dataset when a dataset-specific class co-occurs; a window labeled only with such
classes cannot be attributed and is dropped (counted in the report).

Outputs (next to ``--output``): the sorted ``int64`` index array (``.npy``) of the
samples whose slug is in ``HIGH_QUALITY_SLUGS``, for
``OlmoEarthDatasetConfig.filter_idx_file``, and a ``.csv`` with
``sample_index, slug, labeled_pixels, other_slugs`` for EVERY H5 row, so later subsets
can be cut without re-scanning.

The indices are positions in ONE H5 build (``sample_<i>.h5``), so the output is
specific to ``--h5_dir``: the default output name embeds the build's sample count
(the H5 directory's basename) and the script must be re-run whenever
``open_set_base.OPEN_SET_H5_DIR`` changes. A stale file would be intersected with the
new build's indices silently, selecting the wrong windows.

Usage (from the repo root, weka mounted; the defaults follow ``OPEN_SET_H5_DIR``)::

    python scripts/official/v1_3/open_set_hq/select_h5_indices.py --limit 2000  # smoke test, no output
    python scripts/official/v1_3/open_set_hq/select_h5_indices.py
"""

import argparse
import csv
import json
import logging
import multiprocessing
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import tqdm
from upath import UPath

# The experiment scripts live one directory up (open_set_base.OPEN_SET_H5_DIR).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from open_set_base import CLASS_MAPPING_PATH, OPEN_SET_H5_DIR  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py  # noqa: E402
from olmoearth_pretrain.open_set_segmentation_data.manifest import (  # noqa: E402
    REGISTRY_PATH,
)
from olmoearth_pretrain.open_set_segmentation_data.pretrain_constants import (  # noqa: E402
    OPEN_SET_NODATA,
    REGRESSION_DATASET_ID_NODATA,
)

logger = logging.getLogger("select_h5_indices")

DEFAULT_H5_DIR = OPEN_SET_H5_DIR
FILTERS_DIR = "/weka/dfive-default/helios/dataset/open_set_dataset/filters"
OPEN_SET_KEY = Modality.OPEN_SET.name
OPEN_SET_REGRESSION_KEY = Modality.OPEN_SET_REGRESSION.name


def default_output_for_h5_dir(h5_dir: str) -> str:
    """The filter file for ``h5_dir``, named after the build's sample count.

    The H5 directory's basename is the number of samples in the build (see
    ``ConvertToH5py``), which changes with every regeneration, so embedding it keeps a
    filter from an older build from being picked up by mistake.
    """
    return f"{FILTERS_DIR}/open_set_hq_v1_{Path(h5_dir).name}.npy"


DEFAULT_OUTPUT = default_output_for_h5_dir(DEFAULT_H5_DIR)

# High-quality open-set datasets: labels made on (or at the resolution of) S2 / S1 /
# Landsat or from authoritative inventories with precise geometry, manual or
# expert-validated, Sentinel-era, and a target readable from a 10 m image stack.
# One dataset per concept, a few concepts with regionally complementary picks.
HIGH_QUALITY_SLUGS: tuple[str, ...] = (
    # Land cover / land use
    "dynamic_world_expert_training_labels",
    "lucas_land_use_cover_survey",
    # Agriculture
    "cropsight_us",
    "sen4agrinet",
    "lem_brazil",
    "worldcereal_reference_data_module_rdm",
    "olmoearth_fields_of_the_world",
    # Built / infrastructure / objects
    "spacenet_7_multi_temporal_buildings",
    "sentinelkilndb",
    "uspvdb_us_large_scale_solar_pv_database",
    "olmoearth_wind_turbine",
    "olmoearth_marine_infrastructure",
    "global_mining_footprint_tang_werner",
    "olmoearth_sentinel_2_vessels",
    "olmoearth_sentinel_1_vessels",
    "coastal_aquaculture_ponds_china_se_asia",
    # Change / disturbance
    "oscd_onera_satellite_change_detection",
    "dynamicearthnet",
    "deter_b_near_real_time_deforestation_degradation_alerts",
    "cems_wildfire_dataset",
    "sen1floods11",
    "olmoearth_landslide_sen12landslides",
    # Water / coast / wetlands
    "coast_train",
    "s1s2_water",
    "riverscope",
    "us_national_wetlands_inventory_nwi",
    "magicbathynet",
    "marida_marine_debris_archive",
    # Cryosphere
    "global_debris_covered_glaciers_herreid_pellicciotti",
    "snow_coverage_mapping_sentinel_2_manual",
    "ai4arctic_asip_sea_ice_dataset",
    # Vegetation / atmosphere
    "treesatai_benchmark_archive",
    "cloudsen12",
)


def load_registry_counts(registry_path: Path = REGISTRY_PATH) -> dict[str, int]:
    """``slug -> num_samples`` from the registry, for the per-slug count report."""
    with open(registry_path) as f:
        registry = json.load(f)
    return {
        entry["slug"]: int(entry.get("num_samples") or 0)
        for entry in registry["datasets"]
    }


# Key under which a merged-concept class (shared by several datasets) is counted.
CONCEPT_PREFIX = "concept:"


class SlugLookups(NamedTuple):
    """How label ids map back to label-bank dataset slugs."""

    # open_set global class id -> dataset slug, or ``concept:<name>`` for the merged
    # concept classes that several datasets share (``slug: null`` in the mapping).
    class_to_slug: dict[int, str]
    # open_set_regression band-0 dataset id -> dataset slug.
    regression_to_slug: dict[int, str]
    # ``concept:<name>`` -> the slugs of the datasets that contribute to it.
    concept_members: dict[str, frozenset[str]]


def slug_lookups_from_mapping(class_mapping: dict) -> SlugLookups:
    """Build the lookups from a ``class_mapping.json`` dict."""
    class_to_slug: dict[int, str] = {}
    concept_members: dict[str, frozenset[str]] = {}
    for entry in class_mapping["open_set"]["classes"]:
        global_id = int(entry["global_id"])
        if entry.get("slug") is not None:
            class_to_slug[global_id] = entry["slug"]
            continue
        concept = f"{CONCEPT_PREFIX}{entry['name']}"
        class_to_slug[global_id] = concept
        concept_members[concept] = frozenset(
            member["slug"] for member in entry.get("members", [])
        )
    regression_to_slug = {
        int(entry["dataset_id"]): entry["slug"]
        for entry in class_mapping["open_set_regression"]["datasets"]
    }
    return SlugLookups(class_to_slug, regression_to_slug, concept_members)


def load_slug_lookups(class_mapping_path: str = CLASS_MAPPING_PATH) -> SlugLookups:
    """Read the slug lookups from ``class_mapping.json``."""
    with open(class_mapping_path) as f:
        return slug_lookups_from_mapping(json.load(f))


def _count_slugs(
    ids: np.ndarray, nodata: int, id_to_slug: dict[int, str], kind: str
) -> Counter:
    """Labeled-pixel counts per slug for a raster of ``ids``.

    Ids missing from the mapping are reported as ``unknown_<kind>_<id>`` rather than
    dropped, so a class mapping that does not match the H5 build shows up in the
    report instead of silently emptying the selection.
    """
    values, counts = np.unique(ids[ids != nodata], return_counts=True)
    slug_counts: Counter = Counter()
    for value, count in zip(values.tolist(), counts.tolist()):
        slug_counts[id_to_slug.get(int(value), f"unknown_{kind}_{value}")] += count
    return slug_counts


def slug_counts_of_h5(h5_path: UPath, lookups: SlugLookups) -> Counter:
    """``slug -> labeled pixels`` of one H5 sample, from its label layer(s).

    Merged-concept classes are counted under their ``concept:<name>`` key; see
    :func:`resolve_slug` for how they are attributed to a dataset.
    """
    slug_counts: Counter = Counter()
    with h5_path.open("rb") as f:
        with h5py.File(f, "r") as h5file:
            if OPEN_SET_KEY in h5file:
                ids = np.asarray(h5file[OPEN_SET_KEY][()])
                slug_counts += _count_slugs(
                    ids, OPEN_SET_NODATA, lookups.class_to_slug, "class"
                )
            if OPEN_SET_REGRESSION_KEY in h5file:
                # (H, W, 1, 2): band 0 is the regression dataset id.
                ids = np.asarray(h5file[OPEN_SET_REGRESSION_KEY][()])[..., 0]
                slug_counts += _count_slugs(
                    ids,
                    REGRESSION_DATASET_ID_NODATA,
                    lookups.regression_to_slug,
                    "regression",
                )
    return slug_counts


def resolve_slug(
    slug_counts: Counter, concept_members: dict[str, frozenset[str]] | None = None
) -> tuple[str, int, str]:
    """``(slug, labeled_pixels, other_slugs)`` for a sample's slug pixel counts.

    A window's labels come from one example, so normally there is exactly one dataset
    slug. Pixels of a merged-concept class (``concept:<name>``, shared by several
    datasets) are credited to the window's dataset slug when that dataset is one of the
    concept's members; a window labeled ONLY with concept classes cannot be attributed
    and keeps the ``concept:<name>`` key as its slug (never in ``keep_slugs``, so it is
    dropped; the report counts these). If several slugs remain the most-labeled one
    wins and the rest are recorded (``|`` joined). No labeled pixel gives ``""``.
    """
    if not slug_counts:
        return "", 0, ""
    concept_members = concept_members or {}
    counts = Counter(slug_counts)
    datasets = [slug for slug in counts if not slug.startswith(CONCEPT_PREFIX)]
    if datasets:
        dominant = max(datasets, key=lambda slug: counts[slug])
        for concept in [slug for slug in counts if slug.startswith(CONCEPT_PREFIX)]:
            if dominant in concept_members.get(concept, ()):
                counts[dominant] += counts.pop(concept)
    ordered = counts.most_common()
    slug, pixels = ordered[0]
    other_slugs = "|".join(other for other, _ in ordered[1:])
    return slug, pixels, other_slugs


# Set once per worker by ``_init_worker`` so the lookups are not pickled per task.
_WORKER_LOOKUPS: SlugLookups | None = None


def _init_worker(lookups: SlugLookups):
    global _WORKER_LOOKUPS
    _WORKER_LOOKUPS = lookups


def _scan_one(args: tuple[int, str]) -> tuple[int, Counter]:
    index, h5_path = args
    assert _WORKER_LOOKUPS is not None
    return index, slug_counts_of_h5(UPath(h5_path), _WORKER_LOOKUPS)


def scan_h5_dir(
    h5_dir: UPath,
    sample_indices: Iterable[int],
    lookups: SlugLookups,
    workers: int,
) -> dict[int, Counter]:
    """``sample_index -> slug pixel counts`` for the given samples of ``h5_dir``."""
    jobs = [
        (
            index,
            str(h5_dir / ConvertToH5py.sample_file_pattern.format(index=index)),
        )
        for index in sample_indices
    ]
    results: dict[int, Counter] = {}
    if workers <= 1:
        _init_worker(lookups)
        iterator: Iterable[tuple[int, Counter]] = map(_scan_one, jobs)
    else:
        pool = multiprocessing.Pool(
            workers, initializer=_init_worker, initargs=(lookups,)
        )
        iterator = pool.imap_unordered(_scan_one, jobs, chunksize=256)
    for index, slug_counts in tqdm.tqdm(iterator, total=len(jobs), desc="scanning H5s"):
        results[index] = slug_counts
    if workers > 1:
        pool.close()
        pool.join()
    return results


def select_indices(
    slug_counts_by_index: dict[int, Counter],
    keep_slugs: set[str],
    concept_members: dict[str, frozenset[str]] | None = None,
) -> tuple[np.ndarray, list[tuple[int, str, int, str]]]:
    """Return (sorted kept indices, ``(sample_index, slug, labeled_pixels, other_slugs)`` rows)."""
    rows: list[tuple[int, str, int, str]] = []
    kept: list[int] = []
    for index in sorted(slug_counts_by_index):
        slug, pixels, other_slugs = resolve_slug(
            slug_counts_by_index[index], concept_members
        )
        rows.append((index, slug, pixels, other_slugs))
        if slug in keep_slugs:
            kept.append(index)
    return np.array(kept, dtype=np.int64), rows


def report(
    rows: list[tuple[int, str, int, str]],
    keep_slugs: set[str],
    registry_counts: dict[str, int],
) -> None:
    """Log per-slug matched counts vs the registry, plus data-quality guards."""
    counts = Counter(slug for _, slug, _, _ in rows)
    unlabeled = counts.pop("", 0)
    if unlabeled:
        logger.warning("%d H5 rows have no labeled pixel (slug unknown)", unlabeled)
    unknown = {slug: n for slug, n in counts.items() if slug.startswith("unknown_")}
    if unknown:
        logger.warning(
            "%d H5 rows carry label ids missing from the class mapping (%d ids); "
            "is CLASS_MAPPING_PATH the mapping this H5 build was made with? first: %s",
            sum(unknown.values()),
            len(unknown),
            sorted(unknown)[:10],
        )
    multi = sum(1 for _, _, _, other_slugs in rows if other_slugs)
    if multi:
        logger.warning("%d H5 rows carry labels from more than one dataset", multi)
    concepts = {
        slug: n for slug, n in counts.items() if slug.startswith(CONCEPT_PREFIX)
    }
    if concepts:
        logger.warning(
            "%d H5 rows are labeled only with merged-concept classes shared by several "
            "datasets and cannot be attributed to one (dropped): %s",
            sum(concepts.values()),
            dict(sorted(concepts.items())),
        )
    for slug in sorted(keep_slugs):
        if slug not in registry_counts:
            logger.warning("slug %s is not in registry.json (typo?)", slug)
        n = counts.get(slug, 0)
        if n == 0:
            logger.warning("slug %s matched ZERO H5 rows", slug)
        else:
            logger.info(
                "  %-60s %7d H5 rows (registry num_samples %d)",
                slug,
                n,
                registry_counts.get(slug, 0),
            )
    total = sum(counts.get(s, 0) for s in keep_slugs)
    logger.info("kept %d of %d H5 rows", total, len(rows))


def count_metadata_rows(h5_dir: UPath) -> int:
    """Number of samples listed in the build's ``sample_metadata.csv``."""
    with (h5_dir / ConvertToH5py.sample_metadata_fname).open() as f:
        return sum(1 for _ in f) - 1  # header


def main() -> None:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--h5_dir", type=str, default=DEFAULT_H5_DIR)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output .npy (default: the per-build file for --h5_dir, "
            f"{FILTERS_DIR}/open_set_hq_v1_<num_samples>.npy)"
        ),
    )
    parser.add_argument("--class_mapping", type=str, default=CLASS_MAPPING_PATH)
    parser.add_argument(
        "--slugs",
        type=str,
        default=None,
        help="Comma-separated slugs to keep (default: HIGH_QUALITY_SLUGS)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke test: scan only the first N samples and report, writing nothing",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    keep_slugs = set(args.slugs.split(",")) if args.slugs else set(HIGH_QUALITY_SLUGS)
    registry_counts = load_registry_counts()
    lookups = load_slug_lookups(args.class_mapping)

    h5_dir = UPath(args.h5_dir)
    num_samples = int(h5_dir.name)
    metadata_rows = count_metadata_rows(h5_dir)
    if metadata_rows != num_samples:
        raise RuntimeError(
            f"{args.h5_dir} names {num_samples} samples but its "
            f"{ConvertToH5py.sample_metadata_fname} has {metadata_rows} rows; the H5 "
            "build looks incomplete or the directory is wrong"
        )
    sample_indices = range(
        num_samples if args.limit is None else min(args.limit, num_samples)
    )
    logger.info(
        "scanning %d of %d samples in %s", len(sample_indices), num_samples, h5_dir
    )

    slug_counts_by_index = scan_h5_dir(h5_dir, sample_indices, lookups, args.workers)
    indices, rows = select_indices(
        slug_counts_by_index, keep_slugs, lookups.concept_members
    )
    report(rows, keep_slugs, registry_counts)

    if args.limit is not None:
        logger.info("--limit given: not writing an output file")
        return

    output = UPath(args.output or default_output_for_h5_dir(args.h5_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        np.save(f, indices)
    logger.info("wrote %d indices to %s", len(indices), output)
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "slug", "labeled_pixels", "other_slugs"])
        writer.writerows(rows)
    logger.info("wrote index -> slug table to %s", csv_output)


if __name__ == "__main__":
    main()
