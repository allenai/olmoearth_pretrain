r"""Convert a v1.3 checkpoint trained on ``gabi/perceiver`` to the released layout.

The Perceiver code was cleaned up before release: config fields were renamed and
nested, and several parameters moved. A checkpoint written by the training branch
therefore neither deserializes nor loads under the released code. This tool rewrites
such a checkpoint ONCE, after which it loads with no compatibility code at all::

    python scripts/official/v1_3/convert_legacy_checkpoint.py \
        /weka/.../regbtl_v1_2_gdyn_d768_..._mlpgram1/step667200  /weka/.../v1_3_release

The source directory must hold a ``config.json`` plus either the trainer's distributed
``model_and_optim/`` (needs olmo-core) or an already flat ``weights.pth``. The output
directory gets a ``config.json`` in the current schema and a ``weights.pth`` under the
current parameter names, and is loadable with
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
  ``encoder.register_student`` Sequential (``.0`` Linear, ``.1`` LayerNorm);
  ``encoder.register_back_projections.*`` ->
  ``register_distillation_head.back_projections.*``.

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
DISTILLATION_HEAD_CONFIG_CLASS = (
    "olmoearth_pretrain.nn.register_distillation_head.RegisterDistillationHeadConfig"
)

#: Flat encoder fields -> their name inside ``perceiver_config``.
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
    return model


# ---------------------------------------------------------------------------------------
# State dict
# ---------------------------------------------------------------------------------------

_HEADS_OLD = "encoder.register_back_projections."
_HEADS_NEW = "register_distillation_head.back_projections."


def convert_key(old: str) -> str:
    """The current name of a parameter saved under its ``gabi/perceiver`` name."""
    key = old
    if key.startswith(_HEADS_OLD):
        key = _HEADS_NEW + key[len(_HEADS_OLD) :]
    key = re.sub(
        r"(^|\.)register_projection_norm\.", r"\g<1>register_student.1.", key, 1
    )
    key = re.sub(r"(^|\.)register_projection\.", r"\g<1>register_student.0.", key, 1)
    key = re.sub(r"(^|\.)register_bottleneck\.", r"\g<1>perceiver.", key, 1)
    return key


def legacy_key(new: str) -> str:
    """Inverse of :func:`convert_key`."""
    key = new
    if key.startswith(_HEADS_NEW):
        key = _HEADS_OLD + key[len(_HEADS_NEW) :]
    key = re.sub(r"(^|\.)register_student\.0\.", r"\g<1>register_projection.", key, 1)
    key = re.sub(
        r"(^|\.)register_student\.1\.", r"\g<1>register_projection_norm.", key, 1
    )
    key = re.sub(r"(^|\.)perceiver\.", r"\g<1>register_bottleneck.", key, 1)
    return key


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

        key_mapping = {
            k: legacy_key(k) for k in model.state_dict() if legacy_key(k) != k
        }
        load_model_and_optim_state(str(dcp_dir), model, key_mapping=key_mapping)
    elif weights.exists():
        state_dict = torch.load(weights, map_location="cpu")
        model.load_state_dict(convert_state_dict(state_dict), strict=True)
    else:
        raise FileNotFoundError(f"neither {dcp_dir} nor {weights} exists")


def convert_checkpoint(
    src: Path, dst: Path, *, config_only: bool, verify: bool
) -> None:
    """Convert ``src`` into ``dst`` (created if needed)."""
    import olmoearth_pretrain.nn.latent_mim  # noqa: F401  (registers the model classes)
    from olmoearth_pretrain.config import Config

    config = json.loads((src / "config.json").read_text())
    config["model"] = convert_model_config(config["model"])
    model_config = Config.from_dict(config["model"])  # strict: proves the schema
    # Re-serialize from the parsed config so the file carries exactly the current schema.
    config["model"] = model_config.as_config_dict()
    config["_converted_from"] = str(src)

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
        "src", type=Path, help="checkpoint dir written on gabi/perceiver"
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
