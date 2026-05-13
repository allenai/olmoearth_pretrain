"""Candidate subset ablation: base osm_sampling + filtered candidates.

Usage:
    python scripts/candidate_ablations/run_candidate_ablation.py train <run_name> <cluster> \
        --candidate_columns in_top_combined in_top_solo_novelty \
        --candidate_parquet /path/to/scored_candidates.parquet

Score columns (pick any combination):
    in_top_combined, in_top_solo_novelty, in_top_solo_xglobal_bridge,
    in_top_solo_sparse_infill, in_top_solo_xlocal_bridge, in_top_solo_prototypes,
    in_top_drop_novelty, in_top_drop_xglobal_bridge, in_top_drop_sparse_infill,
    in_top_drop_xlocal_bridge, in_top_drop_prototypes
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from candidate_utils import SCORE_COLUMNS, save_candidate_sample_ids_file
from script_config import (
    BASE_H5PY_DIR,
    CANDIDATE_H5PY_DIR,
    DEFAULT_PARQUET_PATH,
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Parse candidate-specific args before olmo-core sees them
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--candidate_columns",
    nargs="+",
    default=SCORE_COLUMNS,
    help="Which parquet score columns to filter on (default: all). "
    "A candidate is included if any of these columns is 1.",
)
_parser.add_argument(
    "--candidate_parquet",
    default=DEFAULT_PARQUET_PATH,
    help="Path to the scored candidates parquet file.",
)
_known, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

CANDIDATE_COLUMNS: list[str] = _known.candidate_columns
CANDIDATE_PARQUET: str = _known.candidate_parquet


def _get_sample_ids_file(parquet_path: str, columns: list[str]) -> str:
    """Get a deterministic path for the cached sample IDs file.

    Written next to the parquet with a hash-based name so different
    column selections produce different files.
    """
    key = f"{parquet_path}:{','.join(sorted(columns))}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return str(Path(parquet_path).parent / f"_sample_ids_{digest}.txt")


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """Build a concat dataset: base osm_sampling + filtered candidate subset."""
    ids_file = _get_sample_ids_file(CANDIDATE_PARQUET, CANDIDATE_COLUMNS)
    print(f"Preparing candidate sample IDs -> {ids_file}", flush=True)
    save_candidate_sample_ids_file(CANDIDATE_PARQUET, CANDIDATE_COLUMNS, ids_file)
    logger.info(f"Candidate ablation: columns={CANDIDATE_COLUMNS}, ids_file={ids_file}")

    base_config = OlmoEarthDatasetConfig(
        h5py_dir=BASE_H5PY_DIR,
        training_modalities=common.training_modalities,
    )
    candidate_config = OlmoEarthDatasetConfig(
        h5py_dir=CANDIDATE_H5PY_DIR,
        training_modalities=common.training_modalities,
        filter_sample_ids_file=ids_file,
    )
    return OlmoEarthConcatDatasetConfig(
        dataset_configs=[base_config, candidate_config],
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
