"""Load the OlmoEarth models from Hugging Face.

This module works with or without olmo-core installed:
- Without olmo-core: inference-only mode (loading pre-trained models)
- With olmo-core: full functionality including training

The weights are converted to pth file from distributed checkpoint like this:

    import json
    from pathlib import Path

    import torch

    from olmo_core.config import Config
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    checkpoint_path = Path("/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000")
    with (checkpoint_path / "config.json").open() as f:
        config_dict = json.load(f)
        model_config = Config.from_dict(config_dict["model"])

    model = model_config.build()

    train_module_dir = checkpoint_path / "model_and_optim"
    load_model_and_optim_state(str(train_module_dir), model)
    torch.save(model.state_dict(), "OlmoEarth-v1-Nano.pth")
"""

import copy
import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from os import PathLike
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from upath import UPath

from olmoearth_pretrain.config import Config

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "weights.pth"


class ModelID(StrEnum):
    """OlmoEarth pre-trained model ID."""

    OLMOEARTH_V1_NANO = "OlmoEarth-v1-Nano"
    OLMOEARTH_V1_TINY = "OlmoEarth-v1-Tiny"
    OLMOEARTH_V1_BASE = "OlmoEarth-v1-Base"
    OLMOEARTH_V1_LARGE = "OlmoEarth-v1-Large"

    OLMOEARTH_V1_1_NANO = "OlmoEarth-v1_1-Nano"
    OLMOEARTH_V1_1_TINY = "OlmoEarth-v1_1-Tiny"
    OLMOEARTH_V1_1_BASE = "OlmoEarth-v1_1-Base"

    OLMOEARTH_V1_2_NANO = "OlmoEarth-v1_2-Nano"
    OLMOEARTH_V1_2_TINY = "OlmoEarth-v1_2-Tiny"
    OLMOEARTH_V1_2_SMALL = "OlmoEarth-v1_2-Small"
    OLMOEARTH_V1_2_BASE = "OlmoEarth-v1_2-Base"

    def repo_id(self) -> str:
        """Return the Hugging Face repo ID for this model."""
        return f"allenai/{self.value}"


def load_model_from_id(model_id: ModelID, load_weights: bool = True) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from Hugging Face.

    Args:
        model_id: the model ID to load.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
            the config.json will still be downloaded from Hugging Face.
    """
    config_fpath = _resolve_artifact_path(model_id, CONFIG_FILENAME)
    model = _load_model_from_config(config_fpath)

    if not load_weights:
        return model

    state_dict_fpath = _resolve_artifact_path(model_id, WEIGHTS_FILENAME)
    state_dict = _load_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def load_model_from_path(
    model_path: PathLike | str, load_weights: bool = True
) -> torch.nn.Module:
    """Initialize and load the weights for the specified model from a path.

    Args:
        model_path: the path to the model.
        load_weights: whether to load the weights. Set false to skip downloading the
            weights from Hugging Face and leave them randomly initialized. Note that
    """
    config_fpath = _resolve_artifact_path(model_path, CONFIG_FILENAME)
    model = _load_model_from_config(config_fpath)

    if not load_weights:
        return model

    state_dict_fpath = _resolve_artifact_path(model_path, WEIGHTS_FILENAME)
    state_dict = _load_state_dict(state_dict_fpath)
    model.load_state_dict(state_dict)
    return model


def load_pretrain_checkpoint(
    checkpoint_dir: PathLike | str, device: torch.device | None = None
) -> torch.nn.Module:
    """Load a raw pretraining checkpoint directory into a model, eval-ready.

    Unlike ``load_model_from_path`` (which expects a released ``weights.pth``),
    this reads a checkpoint as the trainer writes it: a ``config.json`` plus
    either a distributed ``model_and_optim/`` directory or an already-converted
    ``weights.pth``. Requires olmo-core for the distributed layout.
    """
    ckpt_path = UPath(checkpoint_dir)
    with (ckpt_path / CONFIG_FILENAME).open() as f:
        config_dict = json.load(f)
    config_dict = patch_legacy_encoder_config(config_dict)
    model = Config.from_dict(config_dict["model"]).build()

    train_module_dir = ckpt_path / "model_and_optim"
    weights_path = ckpt_path / WEIGHTS_FILENAME
    if train_module_dir.exists():
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        load_model_and_optim_state(str(train_module_dir), model)
    elif weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        raise FileNotFoundError(
            f"Neither {train_module_dir} nor {weights_path} found in {ckpt_path}"
        )

    if device is not None:
        model.to(device)
    model.eval()
    return model


def _resolve_artifact_path(
    model_id_or_path: ModelID | PathLike | str, filename: str
) -> UPath:
    """Resolve the artifact file path for the specified model ID or path, downloading it from Hugging Face if necessary."""
    if isinstance(model_id_or_path, ModelID):
        return UPath(
            hf_hub_download(repo_id=model_id_or_path.repo_id(), filename=filename)  # nosec
        )
    base = UPath(model_id_or_path)
    return base / filename


@dataclass(frozen=True)
class _RemovedField:
    """An EncoderConfig field this version no longer supports.

    Args:
        inert: Values the field took when its feature was OFF. A checkpoint carrying
            one of these rebuilds identically without the field, so it loads cleanly.
        feature: Human name of the removed feature, for the error message.
        note: What the user can do about it.
    """

    inert: tuple[Any, ...]
    feature: str
    note: str


#: EncoderConfig fields removed from this version, keyed by name.
#:
#: Old checkpoints still carry these keys, and neither deserializer copes on its own:
#: with olmo-core, ``Config.from_dict`` goes through omegaconf and RAISES on any unknown
#: key, so even an inert leftover blocks the load; without olmo-core, the standalone
#: path silently DROPS unknown keys, so a checkpoint that genuinely used one would load
#: as though the feature had been off and quietly build a different model.
#: :func:`patch_legacy_encoder_config` handles both: inert keys are stripped, active
#: ones raise.
REMOVED_ENCODER_FIELDS: dict[str, _RemovedField] = {
    "attn_window_size": _RemovedField(
        inert=(None,),
        feature="windowed (local) spatial attention",
        note="no run ever set it; re-add the window mask to load this checkpoint",
    ),
    "register_read_layers": _RemovedField(
        inert=(None, []),
        feature="multi-depth register reads (mdr)",
        note=(
            "the bottleneck now always re-reads the final encoder layer; this "
            "checkpoint read from several depths and cannot be rebuilt"
        ),
    ),
    "register_shared_read_kv": _RemovedField(
        inert=(False,),
        feature="shared K/V across the register reads",
        note="the reads own their key/value projections again, so the parameter set differs",
    ),
    "register_fused_read": _RemovedField(
        inert=(None,),
        feature="fused multi-depth read source",
        note="requires multi-depth reads, which were removed with it",
    ),
    "register_learned_read_weighting": _RemovedField(
        inert=(False,),
        feature="learned per-read residual gates",
        note="the read_gates parameter no longer exists",
    ),
    "register_latent_every_n": _RemovedField(
        inert=(1, None),
        feature="thinned latent self-attention (one LSA per N reads)",
        note="the schedule is now 1:1, so the block count differs",
    ),
    "register_output_dim": _RemovedField(
        inert=(None,),
        feature="the bottleneck's output projection",
        note="the grid now ships at register_dim; the output_proj weights have no home",
    ),
    "register_unit_norm": _RemovedField(
        inert=(False,),
        feature="unit-sphere (L2-normalized) registers",
        note="the served grid is no longer projected onto a sphere",
    ),
    # Inert whenever the sphere itself is off, which the entry above enforces.
    "register_unit_norm_scale": _RemovedField(
        inert=(),
        feature="the unit-sphere radius",
        note="only meaningful with register_unit_norm, which was removed",
    ),
    "band_dropout_groups": _RemovedField(
        inert=(None, {}),
        feature="grouped (resolution-group) band dropout",
        note="band dropout is per-band again; this changes the input distribution, not the weights",
    ),
    "patch_embed_linear_skip": _RemovedField(
        inert=(False,),
        feature="the patch-embed linear skip",
        note="the pixel_skip Linear no longer exists",
    ),
    "merge_bandsets": _RemovedField(
        inert=(False,),
        feature="bandset merging (multi-bandset tokens merged into one)",
        note="the merge step and the token count it produced are both gone",
    ),
    # Unused whenever merging is off, which the entry above enforces. Unlike
    # register_unit_norm_scale this field is int-defaulted, so it is always PRESENT in a
    # config from that window rather than dropped -- hence a real inert value, not ().
    "merge_after_layer": _RemovedField(
        inert=(-1,),
        feature="the bandset-merge depth",
        note="only meaningful with merge_bandsets, which was removed",
    ),
    "register_students": _RemovedField(
        inert=(None, []),
        feature="multi-student distillation from one backbone",
        note=(
            "the per-student projection heads were replaced by the three scalar knobs; "
            "this checkpoint's student weights have no home"
        ),
    ),
}


def _removed_encoder_fields_to_strip(encoder_config: dict) -> list[str]:
    """Find removed fields in a checkpoint config, refusing any that were USED.

    A removed field left at its feature-off value is inert: dropping it rebuilds the
    identical model, so it is reported for stripping. A field that was ACTIVE cannot be
    honoured -- the code implementing it is gone -- so loading would give a different
    model than the one trained, and that raises instead.

    Args:
        encoder_config: The ``model.encoder_config`` sub-dict of a checkpoint config.

    Returns:
        Names of inert removed fields, to delete before deserialization.

    Raises:
        ValueError: If any removed feature is active in this config.
    """
    inert_present: list[str] = []
    active: list[str] = []
    for name, removed in REMOVED_ENCODER_FIELDS.items():
        if name not in encoder_config:
            continue
        value = encoder_config[name]
        if any(value == inert for inert in removed.inert):
            inert_present.append(name)
        else:
            active.append(f"  {name}={value!r} -- {removed.feature}: {removed.note}")
    if active:
        raise ValueError(
            "this checkpoint uses model features that have since been removed, so it "
            "cannot be rebuilt by this version:\n"
            + "\n".join(active)
            + "\n\nCheck out a commit that still has them to load it."
        )
    return inert_present


def patch_legacy_encoder_config(config_dict: dict) -> dict:
    """Patch checkpoint config dicts saved by older code.

    Applied before passing the dict to ``Config.from_dict``. First it REFUSES configs
    that use a since-removed feature, and STRIPS the removed keys that were merely left
    at their feature-off values (see :func:`_removed_encoder_fields_to_strip`) --
    olmo-core's deserializer rejects any unknown key, so those leftovers would otherwise
    block the load outright. Then three fixups for configs that CAN still be rebuilt:

    1. ``use_linear_patch_embed``: old checkpoints used Conv2d for patch projection and
       have no such key. Without this patch they would incorrectly default to True
       (Linear) and fail to load.
    2. ``register_grid_size``: the dynamic-grid register bottleneck used ``None`` as its
       sentinel, but ``as_config_dict`` drops None values, so dynamic-grid checkpoints
       saved the key as absent (or null). On reload that fell back to a fixed grid and
       produced an incompatible model. When the bottleneck is enabled and the field is
       missing/null, restore the dynamic sentinel (0).
    3. ``register_dim``: it used to default to ``embedding_size // 2`` and is now
       required, so a bottleneck checkpoint that relied on the default (and therefore
       saved no key, since ``as_config_dict`` drops None) would now fail validation.
       Restore the width the old code would have built.

    Raises:
        ValueError: If the config uses a model feature this version has removed.
    """
    enc = config_dict.get("model", {}).get("encoder_config", {})
    if not isinstance(enc, dict):
        return config_dict
    strip = _removed_encoder_fields_to_strip(enc)
    bottleneck_dim_missing = (
        enc.get("use_register_bottleneck")
        and enc.get("register_dim") is None
        and enc.get("embedding_size") is not None
    )
    needs_patch = (
        bool(strip)
        or "use_linear_patch_embed" not in enc
        or (
            enc.get("use_register_bottleneck") and enc.get("register_grid_size") is None
        )
        or bottleneck_dim_missing
    )
    if not needs_patch:
        return config_dict
    config_dict = copy.deepcopy(config_dict)
    enc = config_dict["model"]["encoder_config"]
    for name in strip:
        logger.info(
            "dropping removed-but-inert legacy config field %r (was %r)",
            name,
            enc.pop(name),
        )
    if "use_linear_patch_embed" not in enc:
        enc["use_linear_patch_embed"] = False
    if enc.get("use_register_bottleneck") and enc.get("register_grid_size") is None:
        enc["register_grid_size"] = 0
    if bottleneck_dim_missing:
        enc["register_dim"] = enc["embedding_size"] // 2
        logger.info(
            "legacy checkpoint has no register_dim; restoring the old default "
            "embedding_size // 2 = %d",
            enc["register_dim"],
        )
    return config_dict


def _load_model_from_config(path: UPath) -> torch.nn.Module:
    """Load the model config from the specified path."""
    with path.open() as f:
        config_dict = json.load(f)
    config_dict = patch_legacy_encoder_config(config_dict)
    model_config = Config.from_dict(config_dict["model"])
    return model_config.build()


def _load_state_dict(path: UPath) -> dict[str, torch.Tensor]:
    """Load the model state dict from the specified path."""
    with path.open("rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    return state_dict
