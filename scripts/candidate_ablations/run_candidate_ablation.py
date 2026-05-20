"""Candidate subset ablation: base osm_sampling + filtered candidates.

Usage:
    python scripts/candidate_ablations/run_candidate_ablation.py train <run_name> <cluster> \
        --candidate_columns novelty xglobal_bridge sparse_infill \
        --select_top 50000 \
        --candidate_parquet /path/to/combined_acquisition_scores.parquet \
        --candidate_h5py_dir /path/to/candidate/h5py/dir

Strategies (pick any combination):
    novelty, xglobal_bridge, sparse_infill, xlocal_bridge, prototypes
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from candidate_utils import STRATEGY_NAMES, save_candidate_sample_ids_file
from script_config import (
    BASE_H5PY_DIR,
    build_dataloader_config,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from script_config import (
    build_common_components as _build_common_components,
)

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Parse candidate-specific args before olmo-core sees them.
# All args are optional at parse time so that eval workers (which re-import
# this module without the candidate flags) don't crash.  Actual validation
# happens in build_dataset_config() where the values are needed.
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--candidate_columns",
    nargs="+",
    default=STRATEGY_NAMES,
    help="Strategy names to select candidates from (default: all). "
    "Top --select_top samples per strategy are unioned.",
)
_parser.add_argument(
    "--select_top",
    type=int,
    default=None,
    help="Number of top-scoring candidates to select per strategy.",
)
_parser.add_argument(
    "--total_budget",
    type=int,
    default=None,
    help="Total candidate budget, distributed evenly across strategies. "
    "Overrides --select_top (computes select_top = total_budget // num_strategies).",
)
_parser.add_argument(
    "--candidate_parquet",
    default=None,
    help="Path to the combined_acquisition_scores parquet file.",
)
_parser.add_argument(
    "--candidate_h5py_dir",
    default=None,
    help="Path to the candidate h5py data directory.",
)
_parser.add_argument(
    "--score_suffix",
    default=None,
    help="Score column suffix (REQUIRED, e.g. 'normalized_score', 'diverse_score_p95'). "
    "The column looked up for each strategy is '{strategy}_{suffix}'.",
)
_known, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

CANDIDATE_COLUMNS: list[str] = _known.candidate_columns
CANDIDATE_PARQUET: str | None = _known.candidate_parquet
CANDIDATE_H5PY_DIR_RESOLVED: str | None = _known.candidate_h5py_dir
SCORE_SUFFIX: str | None = _known.score_suffix

if _known.total_budget is not None:
    SELECT_TOP: int | None = _known.total_budget // len(CANDIDATE_COLUMNS)
    logger.info(
        f"--total_budget={_known.total_budget} with {len(CANDIDATE_COLUMNS)} strategies "
        f"-> select_top={SELECT_TOP}"
    )
elif _known.select_top is not None:
    SELECT_TOP = _known.select_top
else:
    SELECT_TOP = None


def build_common_components(
    script: str, cmd: str, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Wrap the default builder to forward candidate args to the Beaker job.

    The module-level argparse strips candidate flags from sys.argv so they
    don't pollute the dotfield overrides that config.merge() expects.  But
    BeakerLaunchConfig.cmd is built from those same overrides, so the remote
    job would never see them.  We patch launch.cmd here to re-inject them.
    """
    common = _build_common_components(script, cmd, run_name, cluster, overrides)
    if (
        common.launch is not None
        and CANDIDATE_PARQUET is not None
        and CANDIDATE_H5PY_DIR_RESOLVED is not None
        and SELECT_TOP is not None
        and SCORE_SUFFIX is not None
    ):
        extra = (
            ["--candidate_columns"]
            + CANDIDATE_COLUMNS
            + ["--score_suffix", SCORE_SUFFIX]
            + ["--select_top", str(SELECT_TOP)]
            + ["--candidate_parquet", CANDIDATE_PARQUET]
            + ["--candidate_h5py_dir", CANDIDATE_H5PY_DIR_RESOLVED]
        )
        # Insert after the 4 positional args (script, cmd, run_name, cluster)
        common.launch.cmd = common.launch.cmd[:4] + extra + common.launch.cmd[4:]
    return common


def _get_sample_ids_file(
    parquet_path: str, strategies: list[str], select_top: int, score_suffix: str
) -> str:
    """Get a deterministic path for the cached sample IDs file.

    Written next to the parquet with a hash-based name so different
    strategy/top-N/suffix selections produce different files.
    """
    key = f"{parquet_path}:{','.join(sorted(strategies))}:{select_top}:{score_suffix}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    return str(Path(parquet_path).parent / f"_sample_ids_{digest}.txt")


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """Build a concat dataset: base osm_sampling + filtered candidate subset."""
    if CANDIDATE_PARQUET is None or CANDIDATE_H5PY_DIR_RESOLVED is None:
        raise RuntimeError(
            "Both --candidate_parquet and --candidate_h5py_dir are required "
            "for training. Re-run with these flags."
        )
    if SELECT_TOP is None:
        raise RuntimeError("--select_top (or --total_budget) is required for training.")
    if SCORE_SUFFIX is None:
        raise RuntimeError("--score_suffix is required for training.")
    ids_file = _get_sample_ids_file(
        CANDIDATE_PARQUET, CANDIDATE_COLUMNS, SELECT_TOP, SCORE_SUFFIX
    )
    print(f"Preparing candidate sample IDs -> {ids_file}", flush=True)
    save_candidate_sample_ids_file(
        CANDIDATE_PARQUET,
        CANDIDATE_COLUMNS,
        SELECT_TOP,
        CANDIDATE_H5PY_DIR_RESOLVED,
        ids_file,
        score_suffix=SCORE_SUFFIX,
    )
    logger.info(
        f"Candidate ablation: strategies={CANDIDATE_COLUMNS}, "
        f"select_top={SELECT_TOP}, score_suffix={SCORE_SUFFIX}, ids_file={ids_file}"
    )

    base_config = OlmoEarthDatasetConfig(
        h5py_dir=BASE_H5PY_DIR,
        training_modalities=common.training_modalities,
    )
    candidate_config = OlmoEarthDatasetConfig(
        h5py_dir=CANDIDATE_H5PY_DIR_RESOLVED,
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
