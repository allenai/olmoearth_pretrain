"""OlmoEarth Pretrain eval datasets."""

import logging
from importlib import import_module
from typing import Any

from torch.utils.data import Dataset

import olmoearth_pretrain.evals.datasets.paths as paths

from .normalize import NormMethod
from .registry import get_builtin_eval_dataset_spec, get_lazy_dataset_exports

logger = logging.getLogger(__name__)

_LAZY_IMPORTS = {
    **get_lazy_dataset_exports(),
    "from_registry_entry": (".rslearn_dataset", "from_registry_entry"),
    "get_dataset_entry": (
        "olmoearth_pretrain.evals.studio_ingest.registry",
        "get_dataset_entry",
    ),
}

__all__ = [
    "NormMethod",
    "get_eval_dataset",
    "paths",
    "scale_train_samples",
]


def __getattr__(name: str) -> Any:
    """Lazily expose dataset adapters imported from this package historically."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name, package=__name__)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def _load_dataset_adapter(name: str) -> Any:
    return globals().get(name) or __getattr__(name)


def scale_train_samples(train_samples: int, label_fraction: float) -> int:
    """Scale pretrain-probe train samples for low-label runs."""
    if not 0 < label_fraction <= 1:
        raise ValueError("label_fraction must be in (0, 1].")
    return max(1, int(train_samples * label_fraction))


def _build_registry_dataset(
    eval_dataset: str,
    split: str,
    norm_stats_from_pretrained: bool | None,
    input_modalities: list[str],
    label_fraction: float,
    norm_method: str,
) -> Dataset:
    eval_dataset_entry = _load_dataset_adapter("get_dataset_entry")(eval_dataset)
    return _load_dataset_adapter("from_registry_entry")(
        entry=eval_dataset_entry,
        split=split,
        norm_stats_from_pretrained=norm_stats_from_pretrained,
        norm_method=norm_method,
        input_modalities_override=input_modalities if input_modalities else None,
        label_fraction=label_fraction,
    )


def _build_pretrain_subset_dataset(
    dataset_cls: Any,
    split: str,
    input_modalities: list[str],
    label_fraction: float,
    kwargs: dict[str, Any],
) -> Dataset:
    return dataset_cls(
        h5py_dir=kwargs["h5py_dir"],
        training_modalities=kwargs.get("training_modalities", input_modalities),
        max_samples=kwargs.get("max_samples", 512),
        patch_size=kwargs.get("pretrain_patch_size", 4),
        hw_p=kwargs.get("pretrain_hw_p", 8),
        seed=kwargs.get("pretrain_seed", 42),
        split=kwargs.get("pretrain_split", split),
        target_modality=kwargs.get("target_modality"),
        label_seed=kwargs.get("pretrain_label_seed", 42),
        train_samples=scale_train_samples(
            kwargs.get("pretrain_train_samples", 512), label_fraction
        ),
        valid_samples=kwargs.get("pretrain_valid_samples", 512),
        test_samples=kwargs.get("pretrain_test_samples", 512),
        split_strategy=kwargs.get("pretrain_split_strategy", "random"),
        geographic_bin_size_deg=kwargs.get("pretrain_geographic_bin_size_deg", 5.0),
    )


def _build_geobench_dataset(
    dataset_cls: Any,
    eval_dataset: str,
    path_key: str,
    split: str,
    label_fraction: float,
    use_pretrain_norm: bool,
    norm_method: str,
) -> Dataset:
    # m- == "modified for geobench"
    return dataset_cls(
        geobench_dir=paths.get_path(path_key),
        dataset=eval_dataset,
        split=split,
        label_fraction=label_fraction,
        norm_stats_from_pretrained=use_pretrain_norm,
        norm_method=norm_method,
    )


def _build_path_backed_dataset(
    dataset_cls: Any,
    path_key: str,
    split: str,
    label_fraction: float,
    use_pretrain_norm: bool,
    norm_method: str,
) -> Dataset:
    return dataset_cls(
        path_to_splits=paths.get_path(path_key),
        split=split,
        label_fraction=label_fraction,
        norm_stats_from_pretrained=use_pretrain_norm,
        norm_method=norm_method,
    )


def _build_pastis_dataset(
    dataset_cls: Any,
    eval_dataset: str,
    split: str,
    input_modalities: list[str],
    label_fraction: float,
    use_pretrain_norm: bool,
    norm_method: str,
) -> Dataset:
    split_path_key = "PASTIS_DIR_ORIG" if "128" in eval_dataset else "PASTIS_DIR"
    return dataset_cls(
        path_to_splits=paths.get_path(split_path_key),
        dir_partition=paths.get_path("PASTIS_DIR_PARTITION"),
        split=split,
        label_fraction=label_fraction,
        norm_stats_from_pretrained=use_pretrain_norm,
        input_modalities=input_modalities,
        norm_method=norm_method,
    )


def get_eval_dataset(
    eval_dataset: str,
    split: str,
    norm_stats_from_pretrained: bool | None = None,
    input_modalities: list[str] | None = None,
    label_fraction: float = 1.0,
    # Default to 2std no clip - this matches what our model sees in pretraining,
    # so when using dataset stats (e.g. for MADOS) consistency is important.
    norm_method: str = NormMethod.NORM_NO_CLIP_2_STD,
    **kwargs: Any,
) -> Dataset:
    """Build the dataset wrapper for a downstream evaluation task.

    Args:
        eval_dataset: Registry name or built-in dataset key.
        split: Split to load: ``train``, ``valid``/``val``, or ``test``.
        norm_stats_from_pretrained: Whether to use pretraining normalization stats.
            ``None`` lets registry-backed datasets use their registry default; built-in
            datasets treat it as ``False`` for backward compatibility.
        input_modalities: Optional modality override for multimodal datasets.
        label_fraction: Fraction of training labels to use for low-label evals.
        norm_method: Dataset normalization strategy when not using pretrain stats.
        **kwargs: Dataset-family specific options, including pretrain probe target
            modality and split sizing.

    Returns:
        A PyTorch dataset that yields eval samples and labels.
    """
    input_modalities = list(input_modalities or [])
    spec = get_builtin_eval_dataset_spec(eval_dataset)

    if spec is None:
        return _build_registry_dataset(
            eval_dataset=eval_dataset,
            split=split,
            norm_stats_from_pretrained=norm_stats_from_pretrained,
            input_modalities=input_modalities,
            label_fraction=label_fraction,
            norm_method=norm_method,
        )

    dataset_cls = _load_dataset_adapter(spec.adapter_class_name)
    use_pretrain_norm = bool(norm_stats_from_pretrained)

    if spec.family == "pretrain_subset":
        return _build_pretrain_subset_dataset(
            dataset_cls=dataset_cls,
            split=split,
            input_modalities=input_modalities,
            label_fraction=label_fraction,
            kwargs=kwargs,
        )
    if spec.family == "geobench":
        return _build_geobench_dataset(
            dataset_cls=dataset_cls,
            eval_dataset=eval_dataset,
            path_key=spec.path_key or "GEOBENCH_DIR",
            split=split,
            label_fraction=label_fraction,
            use_pretrain_norm=use_pretrain_norm,
            norm_method=norm_method,
        )
    if spec.family == "pastis":
        return _build_pastis_dataset(
            dataset_cls=dataset_cls,
            eval_dataset=eval_dataset,
            split=split,
            input_modalities=input_modalities,
            label_fraction=label_fraction,
            use_pretrain_norm=use_pretrain_norm,
            norm_method=norm_method,
        )
    if spec.path_key is not None:
        if use_pretrain_norm and spec.pretrain_norm_warning is not None:
            logger.warning(spec.pretrain_norm_warning)
        return _build_path_backed_dataset(
            dataset_cls=dataset_cls,
            path_key=spec.path_key,
            split=split,
            label_fraction=label_fraction,
            use_pretrain_norm=use_pretrain_norm,
            norm_method=norm_method,
        )

    raise ValueError(f"Unsupported built-in eval dataset family: {spec.family}")
