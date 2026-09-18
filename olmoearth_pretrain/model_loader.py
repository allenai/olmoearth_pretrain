"""Load the OlmoEarth models from Hugging Face.

This module works with or without olmo-core installed:
- Without olmo-core: inference-only mode (loading pre-trained models)
- With olmo-core: full functionality including training

The weights are converted to a pth file from a distributed checkpoint like this
(``load_pretrain_checkpoint`` applies ``patch_legacy_encoder_config`` before building,
which is required for any config.json that still carries since-removed fields):

    import torch

    from olmoearth_pretrain.model_loader import load_pretrain_checkpoint

    model = load_pretrain_checkpoint(
        "/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000"
    )
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

        load_model_and_optim_state(
            str(train_module_dir),
            model,
            key_mapping=legacy_state_dict_key_mapping(model),
        )
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
    """A config field this version no longer supports.

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
    "register_grid_size": _RemovedField(
        inert=(0, None),
        feature="fixed register grid (register_grid_size > 0)",
        note=(
            "the bottleneck now always clones one latent across the patch grid; a "
            "checkpoint with distinct per-cell registers cannot be rebuilt"
        ),
    ),
    "register_contrastive_source": _RemovedField(
        inert=("registers", None),
        feature="a contrastive head reading the encoder tokens under the bottleneck",
        note="the contrastive head now always reads the register latents when the bottleneck is on",
    ),
    "register_projection_type": _RemovedField(
        inert=("linear", None),
        feature="the perceiver-type distillation student",
        note="the student is always a per-cell linear map now; a second-bottleneck student cannot be rebuilt",
    ),
    "register_read_depth": _RemovedField(
        inert=(1, None),
        feature="a read count decoupled from the latent depth (the legacy schedule)",
        note=(
            "the bottleneck now pairs exactly one read with each latent block, so the "
            "read count is register_latent_depth; other read counts cannot be rebuilt"
        ),
    ),
    "register_interleave": _RemovedField(
        inert=(True, None),
        feature="the legacy read schedule (all reads, then all self-attention)",
        note=(
            "the bottleneck now always interleaves one read with each latent block; a "
            "checkpoint trained with register_interleave=False cannot be rebuilt"
        ),
    ),
    "register_latent_self_attn": _RemovedField(
        inert=(True, None),
        feature="a bottleneck without latent self-attention (nolsa)",
        note=(
            "every read is now followed by a latent self-attention block; a checkpoint "
            "trained with register_latent_self_attn=False has no such blocks to load"
        ),
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
    "register_temporal_anchor": _RemovedField(
        inert=(None,),
        feature="the temporally-anchored register read (tanchor)",
        note=(
            "the read blocks rotate over (row, col) again; the anchored 3D read has the "
            "same parameters but computes a different read, so the weights would load "
            "into a model that is not the one trained"
        ),
    ),
}


#: ``LatentMIMConfig`` fields removed from this version (the ``model`` section itself).
REMOVED_MODEL_FIELDS: dict[str, _RemovedField] = {
    "supervision_source": _RemovedField(
        inert=("registers", None),
        feature="supervision heads on the distillation student",
        note="heads attach to the register grid only; the student heads' weights have no home",
    ),
    "projection_supervision_weight_scale": _RemovedField(
        inert=(None,),
        feature="a separate weight for the student supervision heads",
        note="only meaningful with supervision_source != registers, which was removed",
    ),
}


#: ``SupervisionHeadConfig`` fields removed from this version, applied to
#: ``model.supervision_head_config``.
REMOVED_SUPERVISION_HEAD_FIELDS: dict[str, _RemovedField] = {
    "register_supervision": _RemovedField(
        inert=(True,),
        feature="supervision heads on the decoder tokens (register_supervision=False)",
        note="the heads now always read the register grid; decoder-token heads have a different input width",
    ),
}


#: ``SupervisionModalityConfig`` fields removed from this version, keyed by name. Same
#: contract as :data:`REMOVED_ENCODER_FIELDS`, applied to every entry of
#: ``model.supervision_head_config.modality_configs``.
REMOVED_SUPERVISION_MODALITY_FIELDS: dict[str, _RemovedField] = {
    "time_conditioned": _RemovedField(
        inert=(False,),
        feature="time-conditioned (day-of-year MLP) supervision heads",
        note="the MLP over [register_cell ; phi(day_of_year)] no longer exists",
    ),
    # The two below are int-defaulted, so they are always PRESENT in a config from that
    # window; their defaults are the inert values (unused unless time_conditioned).
    "time_harmonics": _RemovedField(
        inert=(4,),
        feature="the day-of-year harmonic count",
        note="only meaningful with time_conditioned, which was removed",
    ),
    "time_mlp_hidden_dim": _RemovedField(
        inert=(64,),
        feature="the time-conditioned head's hidden width",
        note="only meaningful with time_conditioned, which was removed",
    ),
}


def _removed_fields_to_strip(
    section: dict, registry: dict[str, _RemovedField], where: str
) -> list[str]:
    """Find removed fields in one config section, refusing any that were USED.

    A removed field left at its feature-off value is inert: dropping it rebuilds the
    identical model, so it is reported for stripping. A field that was ACTIVE cannot be
    honoured -- the code implementing it is gone -- so loading would give a different
    model than the one trained, and that raises instead.

    Args:
        section: The config sub-dict to inspect (e.g. ``model.encoder_config``).
        registry: The removed-field registry that applies to this section.
        where: Dotted path of ``section``, for the error message.

    Returns:
        Names of inert removed fields, to delete before deserialization.

    Raises:
        ValueError: If any removed feature is active in this section.
    """
    inert_present: list[str] = []
    active: list[str] = []
    for name, removed in registry.items():
        if name not in section:
            continue
        value = section[name]
        if any(value == inert for inert in removed.inert):
            inert_present.append(name)
        else:
            active.append(
                f"  {where}.{name}={value!r} -- {removed.feature}: {removed.note}"
            )
    if active:
        raise ValueError(
            "this checkpoint uses model features that have since been removed, so it "
            "cannot be rebuilt by this version:\n"
            + "\n".join(active)
            + "\n\nCheck out a commit that still has them to load it."
        )
    return inert_present


def _supervision_modality_sections(model_config: dict) -> dict[str, dict]:
    """``modality name -> its config dict`` for the supervision head, if any."""
    head = model_config.get("supervision_head_config")
    if not isinstance(head, dict):
        return {}
    modality_configs = head.get("modality_configs")
    if not isinstance(modality_configs, dict):
        return {}
    return {
        name: cfg for name, cfg in modality_configs.items() if isinstance(cfg, dict)
    }


#: Flat ``register_*`` encoder fields written by checkpoints from before the Perceiver
#: settings moved into the nested ``perceiver_config``, mapped to the field they
#: became there.
LEGACY_FLAT_REGISTER_FIELDS: dict[str, str] = {
    "register_dim": "register_dim",
    "register_latent_depth": "latent_depth",
    "register_num_heads": "num_heads",
    "register_per_depth_read_proj": "per_depth_read_proj",
    "register_attn_dim": "attn_dim",
    "register_projection_dims": "projection_dims",
    "register_projection_output_norm": "projection_output_norm",
}
#: The old encoder field that configured the student's back-projection heads; it now
#: lives on ``LatentMIMConfig.register_distillation_head_config``.
LEGACY_BACK_PROJECTION_HIDDEN = "register_back_projection_hidden"
_DISTILLATION_HEAD_CONFIG_CLASS = (
    "olmoearth_pretrain.nn.register_distillation_head.RegisterDistillationHeadConfig"
)
_PERCEIVER_CONFIG_CLASS = "olmoearth_pretrain.nn.flexi_vit.PerceiverConfig"


LEGACY_DECODER_PERCEIVER_FLAG = "use_register_bottleneck"


def legacy_key_for_current(key: str) -> str:
    """Where an old checkpoint stores the parameter now called ``key``.

    Two moves are undone: the Perceiver was stored under ``register_bottleneck``, and
    the student's back-projection heads lived on the encoder
    (``encoder.register_back_projections``) before they moved into
    ``LatentMIM.register_distillation_head``. Keys untouched by either come back
    unchanged.
    """
    from olmoearth_pretrain.nn.flexi_vit import (
        LEGACY_BACK_PROJECTIONS_ATTR,
        LEGACY_PERCEIVER_ATTR,
    )

    heads = "register_distillation_head.back_projections."
    if key.startswith(heads):
        key = "encoder." + LEGACY_BACK_PROJECTIONS_ATTR + "." + key[len(heads) :]
    if key.startswith("perceiver."):
        key = LEGACY_PERCEIVER_ATTR + key[len("perceiver") :]
    elif ".perceiver." in key:
        key = key.replace(".perceiver.", f".{LEGACY_PERCEIVER_ATTR}.", 1)
    return key


def current_key_for_legacy(key: str) -> str:
    """Inverse of :func:`legacy_key_for_current`: the current name of an old key."""
    from olmoearth_pretrain.nn.flexi_vit import (
        LEGACY_BACK_PROJECTIONS_ATTR,
        LEGACY_PERCEIVER_ATTR,
    )

    old_heads = "encoder." + LEGACY_BACK_PROJECTIONS_ATTR + "."
    if key.startswith(old_heads):
        key = "register_distillation_head.back_projections." + key[len(old_heads) :]
    if key.startswith(LEGACY_PERCEIVER_ATTR + "."):
        key = "perceiver" + key[len(LEGACY_PERCEIVER_ATTR) :]
    elif f".{LEGACY_PERCEIVER_ATTR}." in key:
        key = key.replace(f".{LEGACY_PERCEIVER_ATTR}.", ".perceiver.", 1)
    return key


def legacy_state_dict_key_mapping(model: torch.nn.Module) -> dict[str, str]:
    """``{current key: key in an old checkpoint}`` for every parameter that moved.

    Pass the result as ``key_mapping`` to olmo-core's :func:`load_model_and_optim_state`
    (or :func:`swap_param_keys`); with checkpoint metadata available it is a no-op for
    checkpoints written after the moves. See :func:`legacy_key_for_current`.
    """
    mapping: dict[str, str] = {}
    for key in model.state_dict():
        old = legacy_key_for_current(key)
        if old != key:
            mapping[key] = old
    return mapping


def _has_flat_register_fields(enc: dict) -> bool:
    return (
        "use_register_bottleneck" in enc
        or LEGACY_BACK_PROJECTION_HIDDEN in enc
        or any(name in enc for name in LEGACY_FLAT_REGISTER_FIELDS)
    )


def _nest_legacy_perceiver_fields(model: dict, enc: dict) -> None:
    """Nest an old config's flat register fields into ``perceiver_config``.

    In place. With the Perceiver off the flat leftovers are simply dropped: they
    never built anything. A student (``register_projection_dims``) was always
    distilled by the old code, so it also gets a ``register_distillation_head_config``
    on the model, carrying the old ``register_back_projection_hidden``.
    """
    enabled = bool(enc.pop("use_register_bottleneck", False))
    hidden = enc.pop(LEGACY_BACK_PROJECTION_HIDDEN, None)
    flat = {name: enc.pop(name) for name in LEGACY_FLAT_REGISTER_FIELDS if name in enc}
    if not enabled:
        if flat:
            logger.info(
                "dropping register_* fields of a legacy config whose bottleneck is "
                "off: %s",
                sorted(flat),
            )
        return
    nested: dict[str, Any] = {"_CLASS_": _PERCEIVER_CONFIG_CLASS}
    nested.update(
        {LEGACY_FLAT_REGISTER_FIELDS[name]: value for name, value in flat.items()}
    )
    if nested.get("register_dim") is None and enc.get("embedding_size") is not None:
        # register_dim used to default to embedding_size // 2 and is now required, so
        # a checkpoint that relied on the default (and saved no key, since
        # as_config_dict drops None) must get the width the old code built.
        nested["register_dim"] = enc["embedding_size"] // 2
        logger.info(
            "legacy checkpoint has no register_dim; restoring the old default "
            "embedding_size // 2 = %d",
            nested["register_dim"],
        )
    enc["perceiver_config"] = nested
    logger.info(
        "nested legacy Perceiver fields into perceiver_config: %s",
        sorted(flat),
    )
    if nested.get("projection_dims"):
        head: dict[str, Any] = {"_CLASS_": _DISTILLATION_HEAD_CONFIG_CLASS}
        if hidden is not None:
            head["back_projection_hidden"] = hidden
        model["register_distillation_head_config"] = head
        logger.info(
            "legacy student found; adding register_distillation_head_config "
            "(back_projection_hidden=%r)",
            hidden,
        )


def patch_legacy_encoder_config(config_dict: dict) -> dict:
    """Patch checkpoint config dicts saved by older code.

    Applied before passing the dict to ``Config.from_dict``. First it REFUSES configs
    that use a since-removed feature, and STRIPS the removed keys that were merely left
    at their feature-off values (see :func:`_removed_fields_to_strip`; the registries
    are :data:`REMOVED_MODEL_FIELDS` for ``model``, :data:`REMOVED_ENCODER_FIELDS` for
    ``model.encoder_config``, :data:`REMOVED_SUPERVISION_HEAD_FIELDS` for
    ``model.supervision_head_config`` and :data:`REMOVED_SUPERVISION_MODALITY_FIELDS`
    for each supervision modality) --
    olmo-core's deserializer rejects any unknown key, so those leftovers would otherwise
    block the load outright. Then three fixups for configs that CAN still be rebuilt:

    1. ``use_linear_patch_embed``: old checkpoints used Conv2d for patch projection and
       have no such key. Without this patch they would incorrectly default to True
       (Linear) and fail to load.
    2. ``use_register_bottleneck`` / ``register_*``: the Perceiver settings used to be
       flat encoder fields and now live in the nested ``perceiver_config``; move them
       there (see :data:`LEGACY_FLAT_REGISTER_FIELDS`). ``register_dim`` also used to
       default to ``embedding_size // 2`` and is now required, so a checkpoint that
       saved no key gets the width the old code would have built. The decoder's
       ``use_register_bottleneck`` becomes ``use_perceiver``, and a student's
       ``register_back_projection_hidden`` becomes the model-level
       ``register_distillation_head_config``.

    Raises:
        ValueError: If the config uses a model feature this version has removed.
    """
    model = config_dict.get("model", {})
    enc = model.get("encoder_config", {}) if isinstance(model, dict) else {}
    if not isinstance(enc, dict):
        return config_dict
    strip = _removed_fields_to_strip(
        enc, REMOVED_ENCODER_FIELDS, "model.encoder_config"
    )
    strip_model = _removed_fields_to_strip(model, REMOVED_MODEL_FIELDS, "model")
    head = model.get("supervision_head_config")
    strip_head = (
        _removed_fields_to_strip(
            head, REMOVED_SUPERVISION_HEAD_FIELDS, "model.supervision_head_config"
        )
        if isinstance(head, dict)
        else []
    )
    strip_supervision = {
        name: fields
        for name, section in _supervision_modality_sections(model).items()
        if (
            fields := _removed_fields_to_strip(
                section,
                REMOVED_SUPERVISION_MODALITY_FIELDS,
                f"model.supervision_head_config.modality_configs.{name}",
            )
        )
    }
    dec = model.get("decoder_config")
    decoder_flag_legacy = isinstance(dec, dict) and LEGACY_DECODER_PERCEIVER_FLAG in dec
    needs_patch = (
        bool(strip)
        or bool(strip_model)
        or bool(strip_head)
        or bool(strip_supervision)
        or "use_linear_patch_embed" not in enc
        or _has_flat_register_fields(enc)
        or decoder_flag_legacy
    )
    if not needs_patch:
        return config_dict
    config_dict = copy.deepcopy(config_dict)
    model = config_dict["model"]
    enc = model["encoder_config"]
    for name in strip:
        logger.info(
            "dropping removed-but-inert legacy config field %r (was %r)",
            name,
            enc.pop(name),
        )
    for name in strip_model:
        logger.info(
            "dropping removed-but-inert legacy model field %r (was %r)",
            name,
            model.pop(name),
        )
    for name in strip_head:
        logger.info(
            "dropping removed-but-inert legacy supervision-head field %r (was %r)",
            name,
            model["supervision_head_config"].pop(name),
        )
    modality_sections = _supervision_modality_sections(model)
    for modality, fields in strip_supervision.items():
        for name in fields:
            logger.info(
                "dropping removed-but-inert legacy supervision field %s.%s (was %r)",
                modality,
                name,
                modality_sections[modality].pop(name),
            )
    if "use_linear_patch_embed" not in enc:
        enc["use_linear_patch_embed"] = False
    if _has_flat_register_fields(enc):
        _nest_legacy_perceiver_fields(model, enc)
    if decoder_flag_legacy:
        dec = model["decoder_config"]
        dec["use_perceiver"] = dec.pop(LEGACY_DECODER_PERCEIVER_FLAG)
        logger.info(
            "renamed legacy decoder field use_register_bottleneck -> use_perceiver"
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
