"""Metadata for built-in eval dataset adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DatasetFamily = Literal[
    "pretrain_subset",
    "geobench",
    "mados",
    "sen1floods11",
    "pastis",
    "breizhcrops",
]


@dataclass(frozen=True)
class BuiltinEvalDatasetSpec:
    """Registration metadata for a built-in eval dataset family."""

    family: DatasetFamily
    match: str
    adapter_class_name: str
    adapter_module: str
    config_names: tuple[str, ...]
    match_prefix: bool = False
    path_key: str | None = None
    pretrain_norm_warning: str | None = None

    def matches(self, dataset_name: str) -> bool:
        """Return whether this spec handles a dataset name."""
        if self.match_prefix:
            return dataset_name.startswith(self.match)
        return dataset_name == self.match


BUILTIN_EVAL_DATASET_SPECS: tuple[BuiltinEvalDatasetSpec, ...] = (
    BuiltinEvalDatasetSpec(
        family="pretrain_subset",
        match="pretrain_subset",
        match_prefix=True,
        adapter_class_name="PretrainSubsetDataset",
        adapter_module=".pretrain_subset",
        config_names=(
            "pretrain_subset",
            "pretrain_subset_worldcover",
            "pretrain_subset_osm",
            "pretrain_subset_srtm",
            "pretrain_subset_canopy",
            "pretrain_subset_cdl",
            "pretrain_subset_worldcereal",
        ),
    ),
    BuiltinEvalDatasetSpec(
        family="geobench",
        match="m-",
        match_prefix=True,
        adapter_class_name="GeobenchDataset",
        adapter_module=".geobench_dataset",
        config_names=(
            "m-eurosat",
            "m-bigearthnet",
            "m-so2sat",
            "m-brick-kiln",
            "m-sa-crop-type",
            "m-cashew-plant",
            "m-forestnet",
        ),
        path_key="GEOBENCH_DIR",
    ),
    BuiltinEvalDatasetSpec(
        family="mados",
        match="mados",
        adapter_class_name="MADOSDataset",
        adapter_module=".mados_dataset",
        config_names=("mados",),
        path_key="MADOS_DIR",
        pretrain_norm_warning=(
            "MADOS has very different norm stats than our pretraining dataset"
        ),
    ),
    BuiltinEvalDatasetSpec(
        family="sen1floods11",
        match="sen1floods11",
        adapter_class_name="Sen1Floods11Dataset",
        adapter_module=".floods_dataset",
        config_names=("sen1floods11",),
        path_key="FLOODS_DIR",
    ),
    BuiltinEvalDatasetSpec(
        family="pastis",
        match="pastis",
        match_prefix=True,
        adapter_class_name="PASTISRDataset",
        adapter_module=".pastis_dataset",
        config_names=("pastis", "pastis128"),
    ),
    BuiltinEvalDatasetSpec(
        family="breizhcrops",
        match="breizhcrops",
        adapter_class_name="BreizhCropsDataset",
        adapter_module=".breizhcrops",
        config_names=("breizhcrops",),
        path_key="BREIZHCROPS_DIR",
    ),
)

BUILTIN_EVAL_DATASET_SPECS_BY_FAMILY = {
    spec.family: spec for spec in BUILTIN_EVAL_DATASET_SPECS
}


def get_builtin_eval_dataset_spec(
    dataset_name: str,
) -> BuiltinEvalDatasetSpec | None:
    """Return the built-in dataset family that handles a dataset name."""
    return next(
        (spec for spec in BUILTIN_EVAL_DATASET_SPECS if spec.matches(dataset_name)),
        None,
    )


def get_lazy_dataset_exports() -> dict[str, tuple[str, str]]:
    """Return adapter exports for package-level lazy imports."""
    return {
        spec.adapter_class_name: (spec.adapter_module, spec.adapter_class_name)
        for spec in BUILTIN_EVAL_DATASET_SPECS
    }
