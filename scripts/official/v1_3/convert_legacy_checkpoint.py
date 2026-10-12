r"""Convert a v1.3 checkpoint trained on ``gabi/perceiver`` to the released layout.

The Perceiver code was cleaned up before release: config fields were renamed and
nested, and several parameters moved. It was later renamed to the ``AttentionPooler``.
A checkpoint written by the training branch, or under the pre-rename ``perceiver``
names, therefore neither deserializes nor loads under the current code. This tool
rewrites such a checkpoint ONCE, after which it loads with no compatibility code at
all::

    python scripts/official/v1_3/convert_legacy_checkpoint.py \
        /weka/.../regbtl_v1_2_gdyn_d768_..._mlpgram1/step667200  /weka/.../v1_3_release

The source directory must hold a ``config.json`` plus either the trainer's distributed
``model_and_optim/`` (needs olmo-core) or an already flat ``weights.pth``. The output
directory gets a ``config.json`` holding the ``model`` block in the current schema
plus the decode-only modalities (:func:`decode_only_section`), and a ``weights.pth``
under the current parameter names, and is loadable with
``olmoearth_pretrain.model_loader.load_pretrain_checkpoint`` (or
``load_model_from_path``). ``--config-only`` converts just the config.

What changes, and why:

* ``encoder_config.use_register_bottleneck`` + the flat ``register_*`` fields became
  the nested ``encoder_config.perceiver_config`` (``PerceiverConfig``); the decoder's
  ``use_register_bottleneck`` became ``use_perceiver``.
* ``register_back_projection_hidden`` configures the student's distillation heads,
  which now live on the model as ``register_distillation_head_config``.
* Fields whose feature was removed are dropped when they sit at the feature-off value
  (e.g. ``register_grid_size: 0``); at any other value the checkpoint describes a
  model this code cannot build, and conversion refuses.
* Parameters moved: ``encoder.register_bottleneck.*`` -> ``encoder.perceiver.*``;
  the student ``encoder.register_projection`` (+ ``_norm``) -> the
  ``encoder.perceiver.student`` Sequential (``.0`` Linear, ``.1`` LayerNorm), owned by
  the Perceiver whose grid it reads; ``encoder.register_back_projections.*`` ->
  ``register_distillation_head.back_projections.*``.
* The Perceiver was then renamed to the ``AttentionPooler``, and its learned latent to
  the location query (:func:`_rename_attention_pooler_config`,
  :func:`_rename_attention_pooler_key`): ``encoder_config.perceiver_config``
  (``PerceiverConfig``) -> ``encoder_config.attention_pooler_config``
  (``AttentionPoolerConfig``) with ``register_dim`` / ``latent_depth`` ->
  ``pooler_dim`` / ``pooler_depth``; the decoder's ``use_perceiver`` / ``register_dim``
  -> ``use_attention_pooler`` / ``pooler_dim``; and ``encoder.perceiver.*`` ->
  ``encoder.attention_pooler.*`` with ``.register`` -> ``.location_queries`` and
  ``.latent_blocks`` -> ``.self_attn_blocks``. Its outputs became location tokens:
  ``register_distillation_head_config`` (``RegisterDistillationHeadConfig``) ->
  ``location_token_distillation_head_config``
  (``LocationTokenDistillationHeadConfig``), ``register_distillation_head.*`` ->
  ``location_token_distillation_head.*`` and ``decoder.register_to_decoder_embed.*``
  -> ``decoder.location_tokens_to_decoder_embed.*``. A checkpoint already in the
  pre-rename release layout gets only this step.

Between the first release cut and the student's move onto the Perceiver, the student
briefly lived on the encoder as ``encoder.register_student``. A checkpoint written (or
converted) under that interim layout is converted as well: its config is already
current, and only those two parameters are renamed. The per-pixel-latent runs
(``v1_3_rc_*pix512``) were trained in that layout; their configs additionally carry
inert Perceiver fields (``REMOVED_PERCEIVER_FIELDS``) that are dropped, and latent-grid
settings (``DROPPED_PERCEIVER_RUNTIME_FIELDS``) that are dropped because the latent patch
size is now chosen per forward pass (their in-loop evals used one latent per pixel:
pass ``location_query_patch_size=1``).

The mapping is pinned by ``tests/unit/test_convert_legacy_checkpoint.py`` against the
release checkpoint's original config.
"""

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("convert_legacy_checkpoint")

# ---------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------

PERCEIVER_CONFIG_CLASS = "olmoearth_pretrain.nn.flexi_vit.PerceiverConfig"
ATTENTION_POOLER_CONFIG_CLASS = "olmoearth_pretrain.nn.flexi_vit.AttentionPoolerConfig"
LOCATION_TOKEN_DISTILLATION_HEAD_CONFIG_CLASS = (  # nosec B105 (a class path)
    "olmoearth_pretrain.nn.location_token_distillation_head."
    "LocationTokenDistillationHeadConfig"
)
DISTILLATION_HEAD_CONFIG_CLASS = (
    "olmoearth_pretrain.nn.register_distillation_head.RegisterDistillationHeadConfig"
)

#: Flat encoder fields -> their (pre-rename) name inside ``perceiver_config``.
FLAT_PERCEIVER_FIELDS: dict[str, str] = {
    "register_dim": "register_dim",
    "register_latent_depth": "latent_depth",
    "register_num_heads": "num_heads",
    "register_per_depth_read_proj": "per_depth_read_proj",
    "register_attn_dim": "attn_dim",
    "register_projection_dims": "student_dims",
    "register_projection_output_norm": "student_output_norm",
}
BACK_PROJECTION_HIDDEN = "register_back_projection_hidden"

#: Removed fields -> the values at which they were inert (feature off). Any other value
#: means the checkpoint used the feature and cannot be rebuilt.
REMOVED_ENCODER_FIELDS: dict[str, tuple[Any, ...]] = {
    "register_grid_size": (0, None),  # fixed register grid
    "register_contrastive_source": ("registers", None),
    "register_projection_type": ("linear", None),  # perceiver-type student
    "register_read_depth": (1, None),  # read count decoupled from latent depth
    "register_interleave": (True, None),  # legacy all-reads-then-self-attn schedule
    "register_latent_self_attn": (True, None),  # no-latent-self-attention (nolsa)
    "register_learned_read_weighting": (False,),  # learned per-read gates
}
#: Perceiver fields of the per-pixel-latent (``*pix512*``) training runs whose feature
#: was never released.
REMOVED_PERCEIVER_FIELDS: dict[str, tuple[Any, ...]] = {
    "read_time_range": (False, None),  # time-interval RoPE on the reads
    "read_time_rope": (False, None),  # temporal RoPE on the reads
    "share_read_kv": (False, None),  # one K/V projection shared by every read
}
#: Perceiver latent-grid settings of those runs. They are not part of the model: the
#: latent patch size is now a forward argument
#: (``Encoder.forward(location_query_patch_size=)``), drawn by the dataloader under its
#: ``max_location_queries`` in training and chosen by the caller at inference. Dropped at any value; the values are logged.
DROPPED_PERCEIVER_RUNTIME_FIELDS = (
    "pixel_latents",
    "random_latent_stride",
    "max_latents",
    "eval_latent_stride",
)
REMOVED_MODEL_FIELDS: dict[str, tuple[Any, ...]] = {
    "supervision_source": ("registers", None),  # heads on the student instead
}
REMOVED_SUPERVISION_HEAD_FIELDS: dict[str, tuple[Any, ...]] = {
    "register_supervision": (True,),  # heads on the decoder tokens instead
}
REMOVED_SUPERVISION_MODALITY_FIELDS: dict[str, tuple[Any, ...]] = {
    "time_conditioned": (False,),  # day-of-year MLP heads
    "time_harmonics": (4,),
    "time_mlp_hidden_dim": (64,),
}


class UnconvertibleCheckpoint(ValueError):
    """The checkpoint used a feature the released code no longer has."""


def _strip_removed(
    section: dict, registry: dict[str, tuple[Any, ...]], where: str
) -> None:
    active = []
    for name, inert_values in registry.items():
        if name not in section:
            continue
        value = section.pop(name)
        if not any(value == inert for inert in inert_values):
            active.append(f"{where}.{name}={value!r}")
    if active:
        raise UnconvertibleCheckpoint(
            "this checkpoint uses features the released code has removed, so it cannot "
            "be converted: " + ", ".join(active)
        )


def convert_model_config(model: dict) -> dict:
    """Return the ``model`` config subtree rewritten in the current schema.

    Idempotent: a subtree already in the current schema comes back unchanged.
    """
    model = copy.deepcopy(model)
    enc = model["encoder_config"]
    dec = model.get("decoder_config")

    _strip_removed(enc, REMOVED_ENCODER_FIELDS, "model.encoder_config")
    if isinstance(enc.get("perceiver_config"), dict):
        perceiver_section = enc["perceiver_config"]
        _strip_removed(
            perceiver_section,
            REMOVED_PERCEIVER_FIELDS,
            "model.encoder_config.perceiver_config",
        )
        dropped = {
            name: perceiver_section.pop(name)
            for name in DROPPED_PERCEIVER_RUNTIME_FIELDS
            if name in perceiver_section
        }
        if dropped:
            logger.info(
                "dropping latent-grid settings (now a forward argument, "
                "location_query_patch_size): %s",
                dropped,
            )
    _strip_removed(model, REMOVED_MODEL_FIELDS, "model")
    head = model.get("supervision_head_config")
    if isinstance(head, dict):
        _strip_removed(
            head, REMOVED_SUPERVISION_HEAD_FIELDS, "model.supervision_head_config"
        )
        for name, cfg in (head.get("modality_configs") or {}).items():
            if isinstance(cfg, dict):
                _strip_removed(
                    cfg,
                    REMOVED_SUPERVISION_MODALITY_FIELDS,
                    f"model.supervision_head_config.modality_configs.{name}",
                )

    hidden = enc.pop(BACK_PROJECTION_HIDDEN, None)
    enabled = bool(enc.pop("use_register_bottleneck", False))
    flat = {name: enc.pop(name) for name in FLAT_PERCEIVER_FIELDS if name in enc}
    if enabled:
        perceiver: dict[str, Any] = {"_CLASS_": PERCEIVER_CONFIG_CLASS}
        perceiver.update({FLAT_PERCEIVER_FIELDS[k]: v for k, v in flat.items()})
        if perceiver.get("register_dim") is None:
            # register_dim used to default to embedding_size // 2.
            perceiver["register_dim"] = enc["embedding_size"] // 2
        enc["perceiver_config"] = perceiver
        if perceiver.get("student_dims"):
            # The training branch always distilled a student.
            head_cfg: dict[str, Any] = {"_CLASS_": DISTILLATION_HEAD_CONFIG_CLASS}
            if hidden is not None:
                head_cfg["back_projection_hidden"] = hidden
            model["register_distillation_head_config"] = head_cfg
    elif flat:
        logger.info("dropping register_* fields of a config whose Perceiver was off")

    if isinstance(dec, dict) and "use_register_bottleneck" in dec:
        dec["use_perceiver"] = dec.pop("use_register_bottleneck")
    _rename_attention_pooler_config(model)
    return model


#: Pre-rename ``PerceiverConfig`` fields -> their ``AttentionPoolerConfig`` names.
RENAMED_POOLER_FIELDS = {"register_dim": "pooler_dim", "latent_depth": "pooler_depth"}
#: Pre-rename decoder (``PredictorConfig``) fields -> their current names.
RENAMED_DECODER_FIELDS = {
    "use_perceiver": "use_attention_pooler",
    "register_dim": "pooler_dim",
}


def _rename_attention_pooler_config(model: dict) -> None:
    """Rename the Perceiver config to the AttentionPooler one, in place."""
    enc = model["encoder_config"]
    if "perceiver_config" in enc:
        section = enc.pop("perceiver_config")
        if isinstance(section, dict):
            section = {RENAMED_POOLER_FIELDS.get(k, k): v for k, v in section.items()}
            section["_CLASS_"] = ATTENTION_POOLER_CONFIG_CLASS
        enc["attention_pooler_config"] = section
    dec = model.get("decoder_config")
    if isinstance(dec, dict):
        for old, new in RENAMED_DECODER_FIELDS.items():
            if old in dec:
                dec[new] = dec.pop(old)
    if "register_distillation_head_config" in model:
        head = model.pop("register_distillation_head_config")
        if isinstance(head, dict):
            head["_CLASS_"] = LOCATION_TOKEN_DISTILLATION_HEAD_CONFIG_CLASS
        model["location_token_distillation_head_config"] = head


# ---------------------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------------------

_HEADS_OLD = "encoder.register_back_projections."
_HEADS_NEW = "register_distillation_head.back_projections."


_STUDENT_NEW = "perceiver.student."


def _rename_attention_pooler_key(key: str) -> str:
    """The AttentionPooler name of a parameter saved under its pre-rename name."""
    key = re.sub(r"(^|\.)perceiver\.", r"\g<1>attention_pooler.", key, 1)
    key = re.sub(
        r"(^|\.)attention_pooler\.register$",
        r"\g<1>attention_pooler.location_queries",
        key,
    )
    key = re.sub(
        r"(^|\.)attention_pooler\.latent_blocks\.",
        r"\g<1>attention_pooler.self_attn_blocks.",
        key,
        1,
    )
    if key.startswith("register_distillation_head."):
        key = (
            "location_token_distillation_head."
            + key[len("register_distillation_head.") :]
        )
    return re.sub(
        r"(^|\.)register_to_decoder_embed\.",
        r"\g<1>location_tokens_to_decoder_embed.",
        key,
        1,
    )


def pre_rename_key(new: str) -> str:
    """Inverse of :func:`_rename_attention_pooler_key` (the pre-rename release name)."""
    if new.startswith("location_token_distillation_head."):
        new = (
            "register_distillation_head."
            + new[len("location_token_distillation_head.") :]
        )
    new = re.sub(
        r"(^|\.)location_tokens_to_decoder_embed\.",
        r"\g<1>register_to_decoder_embed.",
        new,
        1,
    )
    key = re.sub(
        r"(^|\.)attention_pooler\.location_queries$", r"\g<1>perceiver.register", new
    )
    key = re.sub(
        r"(^|\.)attention_pooler\.self_attn_blocks\.",
        r"\g<1>perceiver.latent_blocks.",
        key,
        1,
    )
    return re.sub(r"(^|\.)attention_pooler\.", r"\g<1>perceiver.", key, 1)


def convert_key(old: str) -> str:
    """The current name of a parameter saved under an earlier name.

    Accepts the ``gabi/perceiver`` name, the interim ``register_student.*`` name and the
    pre-rename release name (see the module docstring).
    """
    key = old
    if key.startswith(_HEADS_OLD):
        key = _HEADS_NEW + key[len(_HEADS_OLD) :]
    key = re.sub(r"(^|\.)register_projection_norm\.", rf"\g<1>{_STUDENT_NEW}1.", key, 1)
    key = re.sub(r"(^|\.)register_projection\.", rf"\g<1>{_STUDENT_NEW}0.", key, 1)
    key = re.sub(r"(^|\.)register_student\.", rf"\g<1>{_STUDENT_NEW}", key, 1)
    key = re.sub(r"(^|\.)register_bottleneck\.", r"\g<1>perceiver.", key, 1)
    return _rename_attention_pooler_key(key)


def legacy_key(new: str) -> str:
    """Inverse of :func:`convert_key` onto the ``gabi/perceiver`` names."""
    key = pre_rename_key(new)
    if key.startswith(_HEADS_NEW):
        key = _HEADS_OLD + key[len(_HEADS_NEW) :]
    # The student first: once it is ``register_projection*`` the ``perceiver.`` rename
    # below no longer sees it.
    key = re.sub(r"(^|\.)perceiver\.student\.0\.", r"\g<1>register_projection.", key, 1)
    key = re.sub(
        r"(^|\.)perceiver\.student\.1\.", r"\g<1>register_projection_norm.", key, 1
    )
    key = re.sub(r"(^|\.)perceiver\.", r"\g<1>register_bottleneck.", key, 1)
    return key


def interim_key(new: str) -> str:
    """The interim (``encoder.register_student``) name of a current parameter."""
    return re.sub(
        rf"(^|\.){re.escape(_STUDENT_NEW)}",
        r"\g<1>register_student.",
        pre_rename_key(new),
        1,
    )


def convert_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Rename every tensor to its current name.

    An encoder-only dump carries the training-only back-projection heads as
    ``register_back_projections.*`` (no ``encoder.`` prefix); they have no home in an
    encoder and are dropped.
    """
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("register_back_projections."):
            continue
        out[convert_key(key)] = value
    return out


# ---------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------


def _load_weights(src: Path, model: torch.nn.Module) -> None:
    """Load the source checkpoint's weights into the freshly built model."""
    dcp_dir = src / "model_and_optim"
    weights = src / "weights.pth"
    if dcp_dir.exists():
        from olmo_core.distributed.checkpoint import load_model_and_optim_state
        from torch.distributed.checkpoint import FileSystemReader

        # A parameter is looked up under whichever of its earlier names the checkpoint
        # stores: the training branch's, the interim encoder-level student's, or the
        # pre-rename release name.
        stored = set(FileSystemReader(str(dcp_dir)).read_metadata().state_dict_metadata)
        key_mapping: dict[str, str] = {}
        for k in model.state_dict():
            for candidate in (legacy_key(k), interim_key(k), pre_rename_key(k)):
                if candidate != k and f"model.{candidate}" in stored:
                    key_mapping[k] = candidate
                    break
        load_model_and_optim_state(str(dcp_dir), model, key_mapping=key_mapping)
    elif weights.exists():
        state_dict = torch.load(weights, map_location="cpu")
        model.load_state_dict(convert_state_dict(state_dict), strict=True)
    else:
        raise FileNotFoundError(f"neither {dcp_dir} nor {weights} exists")


def decode_only_section(source: dict) -> dict | None:
    """The ``only_decode_modalities`` record of a training config, as a ``train_module``.

    Loaders that serve only the encoder (``olmoearth_pretrain_minimal``) derive the
    encoder's input modalities as the supported ones minus these, read from
    ``train_module`` (else ``data_loader``) ``.masking_config.strategy_config``. None
    when the config records neither.
    """
    for section in ("train_module", "data_loader"):
        decode_only = (
            (source.get(section) or {})
            .get("masking_config", {})
            .get("strategy_config", {})
            .get("only_decode_modalities")
        )
        if decode_only is not None:
            return {
                "masking_config": {
                    "strategy_config": {"only_decode_modalities": decode_only}
                }
            }
    return None


def convert_checkpoint(
    src: Path, dst: Path, *, config_only: bool, verify: bool
) -> None:
    """Convert ``src`` into ``dst`` (created if needed)."""
    import olmoearth_pretrain.nn.latent_mim  # noqa: F401  (registers the model classes)
    from olmoearth_pretrain.config import Config

    source = json.loads((src / "config.json").read_text())
    model_config = Config.from_dict(convert_model_config(source["model"]))  # strict
    # Keep the ``model`` block (re-serialized from the parsed config, so it carries
    # exactly the current schema) and the decode-only modalities, which encoder-only
    # loaders need to know what the encoder accepts. The rest of the trainer, data
    # loader, eval and launch sections describe the training run, not the artifact.
    config: dict[str, Any] = {
        "model": model_config.as_config_dict(),
        "_converted_from": str(src),
    }
    decode_only = decode_only_section(source)
    if decode_only is None:
        logger.warning(
            "the source config records no only_decode_modalities; encoder-only "
            "loaders will not know which modalities the encoder accepts"
        )
    else:
        config["train_module"] = decode_only

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    logger.info("wrote %s", dst / "config.json")
    if config_only:
        return

    model = model_config.build()
    _load_weights(src, model)
    torch.save(model.state_dict(), dst / "weights.pth")
    logger.info("wrote %s (%d tensors)", dst / "weights.pth", len(model.state_dict()))

    if verify:
        from olmoearth_pretrain.model_loader import load_pretrain_checkpoint

        reloaded = load_pretrain_checkpoint(dst)
        for name, tensor in model.state_dict().items():
            if not torch.equal(tensor, reloaded.state_dict()[name]):
                raise RuntimeError(f"round-trip mismatch on {name}")
        logger.info("verified: %s reloads identically", dst)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "src",
        type=Path,
        help="checkpoint dir written on gabi/perceiver or before the AttentionPooler rename",
    )
    parser.add_argument(
        "dst", type=Path, help="output dir for config.json + weights.pth"
    )
    parser.add_argument(
        "--config-only", action="store_true", help="convert only config.json"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the reload check"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    convert_checkpoint(
        args.src, args.dst, config_only=args.config_only, verify=not args.no_verify
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
