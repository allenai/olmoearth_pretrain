#!/usr/bin/env python3
"""Diff the distilled student's weights across the frozen-teacher checkpoints.

Answers one question: in the frozen-teacher continuations, is the student
actually being updated? Their distillation losses sit within ~2e-5 of each other
despite a 10x LR difference, which is equally consistent with "converged, the
loss is genuinely flat" (a finding) and "barely moving" (a bug). The logged
metrics cannot separate those -- the grad norm is dominated by the still-
differentiated frozen encoder, and the supervision losses are logged in bf16, so
they hide sub-0.4% differences. The weights can.

Run it on a machine with /weka mounted:

    python scripts/tools/20260807_diff_frozen_student_weights.py

Reads only the student tensors (``*register_projection*``,
``*register_back_projections*``) out of each sharded checkpoint, so it does not
build a model and does not need a process group.

Reading the output:

* rewarm moved ~10x more than floor  -> both training normally; the flat loss is
  a real finding (the student is converged against this frozen teacher).
* both ~0                            -> the student is not updating; bug.
* both moved the SAME amount         -> the per-group LR is not being applied;
                                        bug, and a different one.
"""

import argparse
import dataclasses
import io
import sys
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader


def _backfill_dataclass(obj: object) -> None:
    """Give an unpickled dataclass any field its class gained since it was saved.

    The checkpoints' ``.metadata`` holds pickled ``_StorageInfo`` objects, and
    pickle restores ``__dict__`` verbatim -- so a field added to the class after
    the checkpoint was written is simply absent, and a newer torch reading an
    older checkpoint dies with AttributeError (``transform_descriptors`` is the
    one that bites here). Fill from the field's declared default, falling back to
    None for fields that have none.
    """
    cls = type(obj)
    if not dataclasses.is_dataclass(cls):
        return
    for field in dataclasses.fields(cls):
        if field.name in vars(obj):
            continue
        if field.default is not dataclasses.MISSING:
            value = field.default
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            value = field.default_factory()  # type: ignore[misc]
        else:
            # transform_descriptors is a sequence torch iterates over; None would
            # crash differently. An empty tuple means "no transforms applied".
            value = () if "descriptors" in field.name else None
        try:
            setattr(obj, field.name, value)
        except AttributeError:  # slotted dataclass; nothing we can do, keep going
            print(
                f"  warning: could not backfill {cls.__name__}.{field.name}",
                file=sys.stderr,
            )


def install_storage_info_compat() -> str:
    """Give ``_StorageInfo`` class-level defaults for fields old pickles lack.

    Patching the unpickled instances is not enough on its own: the loader keeps
    its own reference to the storage_data it read, and ``read_data`` resolves
    ``item_md`` from that, so an instance-level fix applied to our copy never
    reaches the objects it actually touches. A CLASS attribute does, because
    normal attribute lookup falls back to it for every instance whose __dict__
    lacks the field -- which is exactly what an older pickle produces.

    Returns a one-line description of what it did, for the startup banner.
    """
    from torch.distributed.checkpoint import filesystem as fs_module

    cls = getattr(fs_module, "_StorageInfo", None)
    if cls is None:
        return "compat: no _StorageInfo to patch"
    if getattr(cls, "__slots__", None):
        return "compat: _StorageInfo is slotted, cannot add class defaults"
    added = []
    for name, default in (("transform_descriptors", ()),):
        if not hasattr(cls, name):
            setattr(cls, name, default)
            added.append(name)
    return f"compat: added class defaults {added}" if added else "compat: not needed"


class CompatFileSystemReader(FileSystemReader):
    """FileSystemReader that also backfills the metadata it reads.

    Belt and braces alongside :func:`install_storage_info_compat` -- that one
    handles attribute lookup, this one fixes up any instance we can reach.
    """

    def read_metadata(self):  # noqa: D102 - inherited contract
        metadata = super().read_metadata()
        for storage_info in getattr(metadata, "storage_data", {}).values():
            _backfill_dataclass(storage_info)
        return metadata


ROOT = "/weka/dfive-default/olmoearth_pretrain/checkpoints/gabrielt"
PARENT_RUN = "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform"
PATTERNS = ("register_projection", "register_back_projections")


def find_dcp_dir(step_dir: Path) -> Path:
    """The directory holding the checkpoint's .metadata, searched recursively."""
    if (step_dir / ".metadata").exists():
        return step_dir
    hits = sorted(step_dir.glob("**/.metadata"))
    if not hits:
        raise SystemExit(f"no .metadata under {step_dir} -- is that a checkpoint?")
    return hits[0].parent


def load_student(step_dir: Path) -> dict[str, torch.Tensor]:
    """Load just the student tensors out of a sharded checkpoint.

    Reads the bytes directly rather than going through DCP's load planner. The
    planner path calls ``FileSystemReader.read_data``, which touches
    ``_StorageInfo.transform_descriptors`` -- a field ``__getstate__`` drops when
    it is None, so checkpoints written by one torch build raise AttributeError
    when read by another. Nothing here needs the planner: the metadata gives us
    (relative_path, offset, length) per tensor, and with no transforms recorded
    the slice at that offset is a plain ``torch.save``d tensor.
    """
    base = find_dcp_dir(step_dir)
    metadata = CompatFileSystemReader(base).read_metadata()
    all_keys = list(metadata.state_dict_metadata)
    # Model tensors are normally under a "model." prefix (the optimizer's live
    # under "optim."), but do not hard-depend on that -- just exclude optim.
    keys = [
        k
        for k in all_keys
        if any(p in k for p in PATTERNS) and not k.startswith("optim.")
    ]
    if not keys:
        raise SystemExit(
            f"no student tensors matched {PATTERNS} in {step_dir}.\n"
            f"first 20 keys present: {all_keys[:20]}"
        )

    shards: dict[str, list] = {k: [] for k in keys}
    for index, storage_info in metadata.storage_data.items():
        fqn = getattr(index, "fqn", None)
        if fqn in shards:
            shards[fqn].append((index, storage_info))

    out = {}
    for key in keys:
        entries = shards[key]
        if not entries:
            raise SystemExit(f"{key} has metadata but no storage entry in {base}")
        pieces = []
        for index, storage_info in entries:
            with open(base / storage_info.relative_path, "rb") as handle:
                handle.seek(storage_info.offset)
                raw = handle.read(storage_info.length)
            piece = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            pieces.append((index, piece))
        full_size = tuple(getattr(metadata.state_dict_metadata[key], "size", ()) or ())
        if len(pieces) == 1 and (
            not full_size or tuple(pieces[0][1].shape) == full_size
        ):
            tensor = pieces[0][1]
        else:  # sharded save: place each chunk at its recorded offset
            tensor = torch.zeros(full_size, dtype=pieces[0][1].dtype)
            for index, piece in pieces:
                offsets = tuple(getattr(index, "offset", None) or (0,) * piece.dim())
                tensor[tuple(slice(o, o + s) for o, s in zip(offsets, piece.shape))] = (
                    piece
                )
        out[key] = tensor.detach().to(torch.float32)
    return out


def compare(a: dict, b: dict, label: str) -> float:
    """Report per-tensor drift between two student state dicts; return overall."""
    print(f"\n=== {label}")
    print(f"{'tensor':52s} {'rel L2':>10s} {'max|d|':>10s} {'cos':>9s}")
    num_sq = den_sq = 0.0
    for k in sorted(a):
        if k not in b:
            print(f"{k:52s} {'MISSING in b':>31s}")
            continue
        x, y = a[k].flatten(), b[k].flatten()
        d = y - x
        rel = (d.norm() / x.norm().clamp_min(1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        num_sq += d.norm().item() ** 2
        den_sq += x.norm().item() ** 2
        short = k.replace("model.", "")
        print(f"{short:52s} {rel:10.3e} {d.abs().max():10.3e} {cos:9.6f}")
    overall = (num_sq**0.5) / max(den_sq**0.5, 1e-12)
    print(f"{'OVERALL relative L2 drift':52s} {overall:10.3e}")
    return overall


def main() -> None:
    """Load the three checkpoints and report student drift."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--step", type=int, default=5000, help="step of the two arms")
    p.add_argument("--parent-step", type=int, default=667200)
    args = p.parse_args()

    print(f"torch {torch.__version__} | {install_storage_info_compat()}")

    root = Path(args.root)
    paths = {
        "parent": root / PARENT_RUN / f"step{args.parent_step}",
        "rewarm(1e-4)": root
        / "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_frozen_rewarm"
        / f"step{args.step}",
        "floor(1e-5)": root
        / "regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_frozen_floor"
        / f"step{args.step}",
    }
    for name, path in paths.items():
        print(f"{name:14s} {path}  {'OK' if path.exists() else 'MISSING'}")
        if not path.exists():
            print("\nAvailable steps:", file=sys.stderr)
            for d in sorted(path.parent.glob("step*")):
                print("   ", d.name, file=sys.stderr)
            raise SystemExit(1)

    sd = {name: load_student(path) for name, path in paths.items()}
    print(f"\nstudent tensors found: {len(sd['parent'])}")

    drift_rewarm = compare(sd["parent"], sd["rewarm(1e-4)"], "parent -> rewarm (1e-4)")
    drift_floor = compare(sd["parent"], sd["floor(1e-5)"], "parent -> floor (1e-5)")
    drift_arms = compare(
        sd["floor(1e-5)"], sd["rewarm(1e-4)"], "floor -> rewarm (arm vs arm)"
    )

    print("\n=== verdict")
    print(f"drift(rewarm) = {drift_rewarm:.3e}   drift(floor) = {drift_floor:.3e}")
    print(f"arm vs arm    = {drift_arms:.3e}")
    # Deliberately NOT keyed on drift_rewarm/drift_floor ~ 10. Adam's per-step
    # displacement is ~lr regardless of gradient scale, so displacement grows
    # like lr*N only while the run is on a coherent descent trajectory. Once it
    # equilibrates in a basin, displacement saturates at a radius set by
    # curvature against gradient noise and scales like sqrt(lr) or weaker -- so a
    # ratio well under 10 is what a CONVERGED student looks like, not a bug. The
    # question this script can actually answer is simply: did the weights move,
    # and did the two arms end up in different places?
    if max(drift_rewarm, drift_floor) < 1e-4:
        print("  NOT TRAINING: neither student moved. The freeze is too aggressive.")
    elif drift_arms < 1e-4:
        print(
            "  LR NOT APPLIED: the arms are in the same place despite different "
            "configured LRs. Check the student param group reaches the optimizer."
        )
    else:
        print(
            f"  TRAINING, AND THE ARMS DIVERGED ({drift_arms:.1%} apart). The "
            "student and its per-group LR are both live.\n"
            "  Now compare against the loss: if the arms are this far apart in "
            "weights while their\n  distillation loss agrees to ~1e-5, the student "
            "is wandering in flat directions of\n  the objective -- i.e. converged, "
            "and more optimisation will not buy retention."
        )


if __name__ == "__main__":
    main()
