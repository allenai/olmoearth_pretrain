"""Short single-GPU PROFILING run for a joint-latent arm (not a training run).

Picks the arm from the run name (``PROFILE_TARGETS``), keeps its model, sampler, train
module and microbatch, and swaps the trainer for a profiling one: one GPU at the real
per-rank batch (global batch 64), no in-loop evals, no W&B, no checkpoints beyond the
ones olmo-core needs, and olmo-core's ``ProfilerCallback`` over ``PROFILE_ACTIVE`` steps
after ``PROFILE_SKIP`` steps of warm-up (so FlexAttention recompiles have settled). The
profiler logs the top kernels by GPU and by CPU time and saves a Chrome trace under
``<save_folder>/profiler``.

Launch with ``--launch.num_gpus=1``, e.g. ``python .../profile_joint_step.py launch
prof_tmlp1 ai2/jupiter --launch.num_gpus=1 --launch.priority=urgent``.
"""

import importlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from base import build_common_components as _base_common_components  # noqa: E402
from base import build_visualize_config  # noqa: E402
from olmo_core.train.callbacks import Callback, ProfilerCallback  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

# Run-name prefix -> arm module whose model/sampler/train module to profile.
PROFILE_TARGETS = {
    "prof_tmlp1": "pure_perceiver_joint_latentread_rstride_tmlp1",
    "prof_latentread": "pure_perceiver_joint_latentread",
    # Token-mixing window size: the same arm at 3x3 (as trained) and 5x5.
    "prof_mix6r1": "pure_perceiver_mix6_read6",
    "prof_mix6r2": "pure_perceiver_mix6_read6",
    "prof_mix6d128r1": "pure_perceiver_mix6_read6_d128",
    "prof_mix6d128r2": "pure_perceiver_mix6_read6_d128",
    # Latent budget memory check for the random-stride point arm: 512 (as trained) vs 1024.
    "prof_rstride512": "pure_perceiver_joint_latentread_rstride",
    "prof_rstride1024": "pure_perceiver_joint_latentread_rstride",
    # Slot merge vs the released v1.3 (step-time stalls seen on 8 GPUs, 2026-10-15).
    "prof_v13ld1": "rc_ld1_pixtgt_pix512",
    "prof_slot4r3": "rc_ld1_pixtgt_slot4r3_pix512",
    # AllCaptures + narrow stage arms (config D / E): memory + step time on 1 GPU.
    "prof_allcapD": "allcap_slot_n4d192",
    "prof_allcapE": "allcap_slot_n2d256",
    # Same, with per-module profiler regions (REGION_PREFIXES).
    "prof_regD": "allcap_slot_n4d192",
    "prof_regv13": "rc_ld1_pixtgt_pix512",
    # GPU memory snapshot (MemorySnapshotCallback) instead of the profiler.
    "prof_memD": "allcap_slot_n4d192",
    "prof_memv13": "rc_ld1_pixtgt_pix512",
    # D with the compiled mixed-RoPE rotation (encodings.use_compiled_mixed_rope).
    "prof_regDcr": "allcap_slot_n4d192",
    "prof_memDcr": "allcap_slot_n4d192",
    # D' (narrow front end, merged-slot budget, compiled RoPE built in).
    "prof_regDfe": "allcap_slot_n4d192_fe",
    "prof_memDfe": "allcap_slot_n4d192_fe",
    "prof_peakDfe": "allcap_slot_n4d192_fe",
    "prof_peakv13": "rc_ld1_pixtgt_pix512",
    "prof_peakFDfe": "allcap_slot_n4d192_fe",
    "prof_peakFv13": "rc_ld1_pixtgt_pix512",
    # Gradient / activation diagnostics from a trained checkpoint (GradDiagCallback;
    # launch with --trainer.load_path=<checkpoint dir>/stepN).
    "diag_allcap": "allcap_slot_n4d192_fe",
}
# Prefixes that turn on the compiled mixed-RoPE rotation (fused kernels; the eager
# rotation is dozens of small elementwise kernels per q/k per block).
COMPILED_ROPE_PREFIXES = ("prof_regDcr", "prof_memDcr")
MEMORY_PREFIXES = ("prof_memD", "prof_memv13", "prof_memDcr", "prof_memDfe")
# Peak / OOM capture (PeakMemoryCallback): what is live at each new peak step and at an
# OOM, plus that batch's shape.
PEAK_PREFIXES = ("prof_peakDfe", "prof_peakv13", "prof_peakFDfe", "prof_peakFv13")
PEAK_MAX_STEPS = 320
# Forced-shape peak runs (prof_peakF*): every batch at one (heavy) shape, so a few dozen
# steps suffice. The grid comes from a --data_loader.sampled_hw_p_list override on the
# launch (the patch size is still drawn per batch).
PEAK_FORCED_MAX_STEPS = 40
# Record allocations over steps [MEMORY_START, MEMORY_START + MEMORY_STEPS).
MEMORY_START = 40
MEMORY_STEPS = 4
# Prefixes whose profile also times each named module's FORWARD as a region (backward
# hooks break on modules returning dicts; backward is ~2x each region's forward).
REGION_PREFIXES = ("prof_regD", "prof_regv13", "prof_regDcr", "prof_regDfe")
# Region name -> submodule path(s) on the model (missing ones are skipped; a trailing
# ".*" means every child, aggregated under one name).
REGIONS = {
    "encoder": ["encoder"],
    "enc.patch_embed": ["encoder.patch_embeddings"],
    "enc.composite_enc": ["encoder.composite_encodings"],
    "enc.narrow_stage": ["encoder.narrow_stage"],
    "enc.narrow_up": ["encoder.narrow_up"],
    "enc.slot_merge": ["encoder.slot_merge"],
    "enc.trunk_blocks": ["encoder.blocks.*"],
    "enc.perceiver": ["encoder.perceiver"],
    "decoder": ["decoder"],
    "supervision_head": ["supervision_head"],
    "distill_head": ["register_distillation_head"],
    "target_encoder": ["target_encoder"],
}
# Perceiver-config overrides per prefix, applied on top of the arm's model.
PROFILE_PERCEIVER_OVERRIDES: dict[str, dict] = {
    "prof_mix6r1": {"token_mix_radius": 1},
    "prof_mix6r2": {"token_mix_radius": 2},
    "prof_mix6d128r1": {"token_mix_radius": 1},
    "prof_mix6d128r2": {"token_mix_radius": 2},
    "prof_rstride512": {"max_latents": 512},
    "prof_rstride1024": {"max_latents": 1024},
}
# Longer runs for memory checks: more batches drawn at the per-sample latent ceiling.
PROFILE_MAX_STEPS: dict[str, int] = {
    "prof_rstride512": 1000,
    "prof_rstride1024": 1000,
    "prof_v13ld1": 1000,
    "prof_slot4r3": 1000,
    "prof_allcapD": 600,
    "prof_allcapE": 600,
}
# One GPU at the real per-rank batch: v1.3's 512 global batch over 8 GPUs = 64.
GLOBAL_BATCH_SIZE = 64
PROFILE_SKIP = 300
PROFILE_WARMUP = 3
PROFILE_ACTIVE = 10
MAX_STEPS = PROFILE_SKIP + 1 + PROFILE_WARMUP + PROFILE_ACTIVE + 5


def _prefix(common: CommonComponents) -> str:
    # Longest match first, so no prefix can shadow a longer one.
    for prefix in sorted(PROFILE_TARGETS, key=len, reverse=True):
        if common.run_name.startswith(prefix):
            return prefix
    raise ValueError(
        f"run name {common.run_name!r} must start with one of {list(PROFILE_TARGETS)}"
    )


def build_common_components(script, cmd, run_name, cluster, overrides):  # noqa: ANN001, ANN201
    """The target arm's common components if it defines them, else v1.3's.

    AllCaptures arms swap Landsat L1 for L2, so v1.3's modality list would not match
    their corpus.
    """
    prefixes = sorted(PROFILE_TARGETS, key=len, reverse=True)
    prefix = next((p for p in prefixes if run_name.startswith(p)), None)
    builder = _base_common_components
    if prefix is not None:
        module = importlib.import_module(PROFILE_TARGETS[prefix])
        builder = getattr(module, "build_common_components", builder)
    return builder(script, cmd, run_name, cluster, overrides)


def _target(common: CommonComponents):
    return importlib.import_module(PROFILE_TARGETS[_prefix(common)])


def build_model_config(common: CommonComponents):
    """The target arm's model (plus any per-prefix Perceiver overrides)."""
    config = _target(common).build_model_config(common)
    if _prefix(common) in COMPILED_ROPE_PREFIXES:
        from olmoearth_pretrain.nn.encodings import use_compiled_mixed_rope

        use_compiled_mixed_rope(True)
    for field, value in PROFILE_PERCEIVER_OVERRIDES.get(_prefix(common), {}).items():
        setattr(config.encoder_config.perceiver_config, field, value)
    return config


def build_train_module_config(common: CommonComponents):
    """The target arm's train module (keeps its rank microbatch)."""
    return _target(common).build_train_module_config(common)


def build_dataset_config(common: CommonComponents):
    """The target arm's dataset."""
    return _target(common).build_dataset_config(common)


def build_dataloader_config(common: CommonComponents):
    """The target arm's sampler at the real per-rank batch on one GPU."""
    config = _target(common).build_dataloader_config(common)
    config.global_batch_size = GLOBAL_BATCH_SIZE
    return config


class RegionProfilerCallback(ProfilerCallback):
    """ProfilerCallback that also times named model regions (see ``REGIONS``)."""

    def pre_train(self):  # noqa: D102
        super().pre_train()
        if self._profiler is None:
            return
        import torch
        from torch.autograd.profiler import record_function

        model = self.trainer.train_module.model
        model = getattr(model, "module", model)

        def submodules(path: str) -> list:
            if path.endswith(".*"):
                try:
                    return list(model.get_submodule(path[:-2]).children())
                except AttributeError:
                    return []
            try:
                return [model.get_submodule(path)]
            except AttributeError:
                return []

        for name, paths in REGIONS.items():
            for path in paths:
                for module in submodules(path):
                    stack: dict = {}

                    def fwd_pre(_m, _a, name=name, stack=stack):  # type: ignore[no-untyped-def]
                        r = record_function(f"region:fwd:{name}")
                        r.__enter__()
                        stack.setdefault("fwd", []).append(r)

                    def fwd_post(_m, _a, _o, stack=stack):  # type: ignore[no-untyped-def]
                        stack["fwd"].pop().__exit__(None, None, None)

                    module.register_forward_pre_hook(fwd_pre)
                    module.register_forward_hook(fwd_post)
        # Padding: the share of real (valid) positions in the padded [B, L] inputs of
        # the first trunk block, the first decoder block and the narrow stage.
        self._pad_stats: dict = {}

        def valid_fraction(mask):  # type: ignore[no-untyped-def]
            if mask is None:
                return None
            m = mask
            while m.dim() > 2:
                m = m.any(dim=-2) if m.shape[-2] > 1 else m.squeeze(-2)
            return float(m.float().mean())

        def pad_hook(name):  # type: ignore[no-untyped-def]
            def hook(_m, args, kwargs):  # type: ignore[no-untyped-def]
                x = kwargs.get("x", args[0] if args else None)
                mask = kwargs.get("attn_mask", kwargs.get("valid"))
                if mask is None and len(args) > 2:
                    mask = args[2]
                entry = self._pad_stats.setdefault(name, [])
                if x is not None and x.dim() == 3:
                    entry.append((x.shape[0], x.shape[1], valid_fraction(mask)))

            return hook

        for name, path in (
            ("trunk_block0", "encoder.blocks.0"),
            ("decoder_block0", "decoder.blocks.0"),
            ("narrow_stage", "encoder.narrow_stage"),
        ):
            try:
                model.get_submodule(path).register_forward_pre_hook(
                    pad_hook(name), with_kwargs=True
                )
            except AttributeError:
                pass
        # Front-end padding: positions in the padded per-modality token grids the patch
        # embedding produces vs ONLINE_ENCODER (encoded) and MISSING positions.
        from olmoearth_pretrain.datatypes import MaskValue

        self._embed_stats = {"positions": 0, "online": 0, "missing": 0, "calls": 0}

        def embed_hook(_m, _args, out):  # type: ignore[no-untyped-def]
            for key, value in out.items():
                if not key.endswith("_mask"):
                    continue
                self._embed_stats["positions"] += value.numel()
                self._embed_stats["online"] += int(
                    (value == MaskValue.ONLINE_ENCODER.value).sum()
                )
                self._embed_stats["missing"] += int(
                    (value == MaskValue.MISSING.value).sum()
                )
            self._embed_stats["calls"] += 1

        try:
            model.get_submodule("encoder.patch_embeddings").register_forward_hook(
                embed_hook
            )
        except AttributeError:
            pass
        # Tokens per merged cell, as the narrow stage groups them (every step).
        self._cell_hist = torch.zeros(10, dtype=torch.long)
        self._cell_edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 10**9]

        def cell_hook(_m, args, kwargs):  # type: ignore[no-untyped-def]
            mask = kwargs.get("attn_mask")
            if mask is None or mask.dim() != 2:
                return
            n = mask.sum(dim=1).cpu()
            edges = torch.tensor(self._cell_edges)
            self._cell_hist += torch.bincount(
                torch.bucketize(n, edges, right=False), minlength=10
            )[:10]

        try:
            model.get_submodule(
                "encoder.narrow_stage.blocks.0"
            ).register_forward_pre_hook(cell_hook, with_kwargs=True)
        except AttributeError:
            pass
        logger.info(
            f"Region profiling on {len(REGIONS)} regions (cuda={torch.cuda.is_available()})"
        )

    def _on_trace_ready(self, prof):  # noqa: D102
        averages = self._profiler.key_averages()
        steps = PROFILE_ACTIVE
        rows = []
        for event in averages:
            if not event.key.startswith("region:"):
                continue
            device = getattr(event, "device_time_total", None)
            if device is None:
                device = getattr(event, "cuda_time_total", 0.0)
            rows.append((event.key, event.count / steps, event.cpu_time_total, device))
        lines = [
            f"{'region':40s} {'calls/step':>10s} {'CPU ms/step':>12s} {'GPU ms/step':>12s}"
        ]
        for key, calls, cpu, device in sorted(rows, key=lambda r: -r[3]):
            lines.append(
                f"{key:40s} {calls:10.1f} {cpu / 1e3 / steps:12.1f} {device / 1e3 / steps:12.1f}"
            )
        logger.info("Region profile (per step):\n" + "\n".join(lines))
        import torch

        for name, entries in self._pad_stats.items():
            recent = entries[-200:]
            n = max(1, len(recent))
            fracs = [f for _, _, f in recent if f is not None]
            logger.info(
                f"Padding {name}: calls {len(recent)}, "
                f"mean B {sum(b for b, _, _ in recent) / n:.1f}, "
                f"mean L {sum(seq for _, seq, _ in recent) / n:.0f}, "
                f"max L {max((seq for _, seq, _ in recent), default=0)}, "
                f"valid fraction {sum(fracs) / max(1, len(fracs)):.3f}"
            )
        e = self._embed_stats
        if e["calls"]:
            logger.info(
                f"Patch-embed grid over {e['calls']} calls: positions {e['positions']}, "
                f"online {e['online'] / max(1, e['positions']):.3f}, "
                f"missing {e['missing'] / max(1, e['positions']):.3f}"
            )
        hist = self._cell_hist.tolist()
        if sum(hist):
            labels = [
                "1",
                "2",
                "3-4",
                "5-8",
                "9-16",
                "17-32",
                "33-64",
                "65-128",
                "129-256",
                ">256",
            ]
            total = sum(hist)
            logger.info(
                "Tokens per cell at the narrow stage (share of cells): "
                + ", ".join(f"{lab}: {c / total:.3f}" for lab, c in zip(labels, hist))
            )
        if torch.cuda.is_available():
            logger.info(
                "Peak GPU memory: "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
                f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved"
            )
        super()._on_trace_ready(prof)


class MemorySnapshotCallback(Callback):
    """Record every CUDA allocation for a few steps; print what is live at the peak.

    Replays the allocator trace (alloc / free events with stack frames) to find the
    moment of peak allocated memory, then groups the blocks live at that moment by the
    innermost ``olmoearth_pretrain`` frame that allocated them. Also logs the narrow
    stage's grouped shape (cells G, padded length P, real-token fraction).
    """

    def pre_train(self):  # noqa: D102
        model = self.trainer.train_module.model
        model = getattr(model, "module", model)
        self._narrow: list = []

        def narrow_hook(_m, args, kwargs):  # type: ignore[no-untyped-def]
            x = kwargs.get("x", args[0] if args else None)
            mask = kwargs.get("attn_mask")
            if x is not None and mask is not None and mask.dim() == 2:
                self._narrow.append(
                    (x.shape[0], x.shape[1], float(mask.float().mean()))
                )

        try:
            model.get_submodule(
                "encoder.narrow_stage.blocks.0"
            ).register_forward_pre_hook(narrow_hook, with_kwargs=True)
        except AttributeError:
            pass

    def pre_step(self, batch):  # noqa: D102
        import torch

        if self.trainer.global_step + 1 == MEMORY_START:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.memory._record_memory_history(max_entries=2_000_000)

    def post_step(self):  # noqa: D102
        import torch

        if self.trainer.global_step != MEMORY_START + MEMORY_STEPS - 1:
            return
        snapshot = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        logger.info(
            f"Memory: peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
            f"peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB "
            f"over steps {MEMORY_START}..{MEMORY_START + MEMORY_STEPS - 1}"
        )
        if self._narrow:
            recent = self._narrow[-20:]
            logger.info(
                "Narrow stage grouped [G, P]: "
                + ", ".join(f"G={g} P={p} real={f:.3f}" for g, p, f in recent[:8])
                + f" ... mean real {sum(f for *_, f in recent) / len(recent):.3f}"
            )
        live: dict = {}
        current = peak = 0
        peak_live: dict = {}
        events = [e for trace in snapshot["device_traces"] for e in trace]
        for e in events:
            action, addr, size = e["action"], e["addr"], e["size"]
            if action == "alloc":
                live[addr] = e
                current += size
                if current > peak:
                    peak = current
                    peak_live = dict(live)
            elif action in ("free_completed",) and addr in live:
                current -= live.pop(addr)["size"]

        def site(e):  # type: ignore[no-untyped-def]
            frames = e.get("frames") or []
            ours = [
                f
                for f in frames
                if "olmoearth_pretrain" in f.get("filename", "")
                or "/ablations/" in f.get("filename", "")
            ]
            f = ours[0] if ours else (frames[0] if frames else {})
            name = f.get("filename", "?").split("olmoearth_pretrain/")[-1]
            return f"{name}:{f.get('line', '?')} {f.get('name', '')}"

        groups: dict = {}
        for e in peak_live.values():
            key = site(e)
            n, b = groups.get(key, (0, 0))
            groups[key] = (n + 1, b + e["size"])
        lines = [
            f"{b / 2**30:8.2f} GiB  {n:6d} blocks  {key}"
            for key, (n, b) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:30]
        ]
        logger.info(
            f"Live at the traced peak ({peak / 2**30:.1f} GiB over {len(events)} events,"
            " allocations made before tracing started are not seen):\n"
            + "\n".join(lines)
        )


def _live_by_site(snapshot: dict, top: int = 30) -> tuple[float, list[str]]:
    """Replay the allocator trace; return the peak and what is live at it, by site."""
    live: dict = {}
    current = peak = 0
    peak_live: dict = {}
    for trace in snapshot["device_traces"]:
        for e in trace:
            action, addr, size = e["action"], e["addr"], e["size"]
            if action == "alloc":
                live[addr] = e
                current += size
                if current > peak:
                    peak, peak_live = current, dict(live)
            elif action == "free_completed" and addr in live:
                current -= live.pop(addr)["size"]
    return peak / 2**30, _group_by_site(peak_live.values(), top)


def _peak_blocks(snapshot: dict) -> list:
    """The allocator-trace blocks live at the traced peak."""
    live: dict = {}
    current = peak = 0
    peak_live: dict = {}
    for trace in snapshot["device_traces"]:
        for e in trace:
            action, addr, size = e["action"], e["addr"], e["size"]
            if action == "alloc":
                live[addr] = e
                current += size
                if current > peak:
                    peak, peak_live = current, dict(live)
            elif action == "free_completed" and addr in live:
                current -= live.pop(addr)["size"]
    return list(peak_live.values())


def _group_by_site(blocks, top: int) -> list[str]:  # type: ignore[no-untyped-def]
    groups: dict = {}
    for e in blocks:
        frames = e.get("frames") or []
        ours = [
            f
            for f in frames
            if "olmoearth_pretrain" in f.get("filename", "")
            or "/ablations/" in f.get("filename", "")
        ]
        f = ours[0] if ours else (frames[0] if frames else {})
        key = (
            f"{f.get('filename', '?').split('olmoearth_pretrain/')[-1]}:"
            f"{f.get('line', '?')} {f.get('name', '')}"
        )
        n, b = groups.get(key, (0, 0))
        groups[key] = (n + 1, b + e["size"])
    return [
        f"{b / 2**30:8.2f} GiB  {n:6d} blocks  {key}"
        for key, (n, b) in sorted(groups.items(), key=lambda kv: -kv[1][1])[:top]
    ]


def _module_spans() -> list[tuple[str, str, int, int, set[int] | None]]:
    """(label, source file, first line, last line, trunk call lines) per component.

    A live block is assigned to the innermost of these on its allocation stack. Inside
    ``Encoder.apply_attn`` the lines that call a block are the trunk; the rest of that
    method is encoder bookkeeping (encodings, collapse, masks, the merge call).
    """
    import inspect

    from olmoearth_pretrain.nn import (
        flexi_vit,
        latent_mim,
        slot_merge,
        supervision_head,
    )

    def span(label, fn, trunk=False):  # type: ignore[no-untyped-def]
        lines, start = inspect.getsourcelines(fn)
        calls = (
            {start + i for i, line in enumerate(lines) if "blk(" in line}
            if trunk
            else None
        )
        return (label, inspect.getsourcefile(fn), start, start + len(lines) - 1, calls)

    spans = [
        span("narrow stage", slot_merge.NarrowStage.forward),
        span("slot merge", slot_merge.SlotMerge.forward),
        span("perceiver", flexi_vit.Perceiver.forward),
        span("decoder", flexi_vit.Predictor.forward),
        span("decoder", flexi_vit.Predictor.apply_attn),
        span("target projection", latent_mim.FrozenTargetProjection.forward),
        span("patch embedding", flexi_vit.MultiModalPatchEmbeddings.forward),
        span("supervision head", supervision_head.SupervisionHead.forward),
        span("encoder", flexi_vit.Encoder.apply_attn, trunk=True),
    ]
    if hasattr(flexi_vit.Predictor, "forward_compact"):
        spans.append(span("decoder", flexi_vit.Predictor.forward_compact))
    return spans


def _group_by_module(blocks, spans) -> list[str]:  # type: ignore[no-untyped-def]
    """Live bytes per model component (innermost component on the stack)."""
    totals: dict = {}
    for e in blocks:
        label = "other"
        for f in e.get("frames") or []:
            name, line = f.get("filename", ""), f.get("line", -1)
            if name.endswith("train/loss.py"):
                label = "loss"
                break
            hit = None
            for lab, src, first, last, calls in spans:
                if name == src and first <= line <= last:
                    hit = lab
                    if calls is not None:
                        hit = "trunk blocks" if line in calls else "encoder bookkeeping"
                    break
            if hit is not None:
                label = hit
                break
        totals[label] = totals.get(label, 0) + e["size"]
    return [
        f"{b / 2**30:8.2f} GiB  {lab}"
        for lab, b in sorted(totals.items(), key=lambda kv: -kv[1])
    ]


def _describe_batch(batch) -> str:  # type: ignore[no-untyped-def]
    """Patch size and, per modality, the tensor shape and real / encoded counts."""
    from olmoearth_pretrain.datatypes import MaskValue

    parts = []
    items = batch if isinstance(batch, tuple | list) else (batch,)
    for item in items:
        if isinstance(item, int):
            parts.append(f"patch_size={item}")
            continue
        fields = getattr(item, "_fields", None) or []
        for name in fields:
            value = getattr(item, name)
            if not torch.is_tensor(value) or name.endswith("_mask"):
                continue
            mask = getattr(item, f"{name}_mask", None)
            desc = f"{name}{list(value.shape)}"
            if torch.is_tensor(mask):
                desc += (
                    f" enc={int((mask == MaskValue.ONLINE_ENCODER.value).sum())}"
                    f" dec={int((mask == MaskValue.DECODER.value).sum())}"
                    f" missing={int((mask == MaskValue.MISSING.value).sum())}"
                    f" of {mask.numel()}"
                )
            parts.append(desc)
    return "; ".join(parts)


@dataclass
class PeakMemoryCallback(Callback):
    """Log what is live at each new peak step and at an OOM, with the batch's shape.

    Keeps a ring buffer of allocator events (long enough for a step). On a step whose
    peak beats every earlier one it replays the buffer to the peak and logs the live
    blocks by allocating line; an OutOfMemoryError in ``train_batch`` logs the live
    blocks at that moment (plus the batch) before re-raising.
    """

    def pre_train(self):  # noqa: D102
        torch.cuda.memory._record_memory_history(max_entries=1_500_000)
        self._spans = _module_spans()
        self._best = 0.0
        self._batch_desc = ""
        module = self.trainer.train_module
        original = module.train_batch

        def guarded(batch, *args, **kwargs):  # type: ignore[no-untyped-def]
            self._batch_desc = _describe_batch(batch)
            try:
                return original(batch, *args, **kwargs)
            except torch.OutOfMemoryError:
                snapshot = torch.cuda.memory._snapshot()
                live = [
                    {"size": blk["size"], "frames": blk.get("frames", [])}
                    for seg in snapshot["segments"]
                    for blk in seg["blocks"]
                    if blk["state"] == "active_allocated"
                ]
                lines = _group_by_site(live, 30)
                logger.info(
                    f"OOM at step {self.trainer.global_step}; batch: {self._batch_desc}\n"
                    f"live at OOM ({sum(b['size'] for b in live) / 2**30:.1f} GiB) by "
                    "module:\n"
                    + "\n".join(_group_by_module(live, self._spans))
                    + "\nby line:\n"
                    + "\n".join(lines)
                )
                raise

        module.train_batch = guarded  # type: ignore[method-assign]

    def pre_step(self, batch):  # noqa: D102
        torch.cuda.reset_peak_memory_stats()

    def post_step(self):  # noqa: D102
        peak = torch.cuda.max_memory_allocated() / 2**30
        logger.info(
            f"step {self.trainer.global_step} peak allocated {peak:.1f} GiB; "
            f"batch: {self._batch_desc}"
        )
        if peak <= self._best + 0.5 or self.trainer.global_step < 5:
            return
        self._best = peak
        snapshot = torch.cuda.memory._snapshot()
        traced_peak, lines = _live_by_site(snapshot)
        logger.info(
            f"NEW PEAK step {self.trainer.global_step}: {peak:.1f} GiB allocated "
            f"(traced {traced_peak:.1f} GiB); batch: {self._batch_desc}\nby module:\n"
            + "\n".join(_group_by_module(_peak_blocks(snapshot), self._spans))
            + "\nby line:\n"
            + "\n".join(lines)
        )


DIAG_STEPS = 30
# Parameter / activation groups for GradDiagCallback (name prefix -> group).
DIAG_GROUPS = (
    ("encoder.patch_embeddings", "patch_embed"),
    ("encoder.composite_encodings", "composite_enc"),
    ("encoder.narrow_stage", "narrow_stage"),
    ("encoder.slot_merge.gru", "merge.gru"),
    ("encoder.slot_merge.mlp", "merge.mlp"),
    ("encoder.slot_merge", "merge.other"),
    ("encoder.narrow_up", "narrow_up"),
    ("encoder.blocks.", "trunk.block"),
    ("encoder.perceiver", "perceiver"),
    ("encoder", "encoder.other"),
    ("decoder", "decoder"),
)


def _diag_group(name: str) -> str:
    for prefix, group in DIAG_GROUPS:
        if name.startswith(prefix):
            if group == "trunk.block":
                return f"trunk.{name.split('.')[2]}"
            if group == "narrow_stage" and ".blocks." in name:
                return f"narrow.{name.split('.blocks.')[1].split('.')[0]}"
            return group
    return name.split(".")[0]


class GradDiagCallback(Callback):
    """Per-module gradient norms, weight norms and activation RMS for DIAG_STEPS steps.

    Run from a trained checkpoint: after each backward (before the optimizer step) it
    logs each module group's gradient L2 norm, the attention q/k weight norms per
    block, and the RMS / max |x| of the narrow stage (and each of its blocks), slot
    merge, up-projection and first / last trunk block outputs; then a mean over the
    steps, and exits. For the attention of each narrow block and the first / last trunk
    block it also logs the max per-head q and k vector norms and their product times
    the softmax scale: an upper bound on any attention logit (logit growth shows here).
    """

    def pre_train(self) -> None:  # noqa: D102
        model = self.trainer.train_module.model
        self._acts: dict[str, list[tuple[float, float]]] = {}
        self._qk: dict[str, list[float]] = {}
        self._grads: dict[str, list[float]] = {}
        self._steps = 0
        names = dict(model.named_modules())
        blocks = sorted(
            int(n.split(".")[2])
            for n in names
            if n.startswith("encoder.blocks.") and n.count(".") == 2
        )
        watch = ["encoder.narrow_stage", "encoder.slot_merge", "encoder.narrow_up"]
        watch += [
            n
            for n in names
            if n.startswith("encoder.narrow_stage.blocks.") and n.count(".") == 3
        ]
        if blocks:
            watch += [f"encoder.blocks.{blocks[0]}", f"encoder.blocks.{blocks[-1]}"]
        attn_blocks = [w for w in watch if ".blocks." in w]

        def qk_hook(name, which):  # type: ignore[no-untyped-def]
            def fn(module, inputs, output):  # type: ignore[no-untyped-def]
                if torch.is_tensor(output):
                    n = output.detach().float().norm(dim=-1).max()
                    self._qk.setdefault(f"{name}.{which}", []).append(float(n))

            return fn

        for name in attn_blocks:
            for which in ("q", "k"):
                sub = f"{name}.attn.{which}_norm"
                if sub in names:
                    names[sub].register_forward_hook(qk_hook(name, which))
            if f"{name}.attn" in names:
                self._qk.setdefault(f"{name}.scale", []).append(
                    float(names[f"{name}.attn"].scale)
                )

        def hook(name):  # type: ignore[no-untyped-def]
            def fn(module, inputs, output):  # type: ignore[no-untyped-def]
                x = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(x) or not x.is_floating_point():
                    return
                x = x.detach().float()
                self._acts.setdefault(name, []).append(
                    (float(x.pow(2).mean().sqrt()), float(x.abs().max()))
                )

            return fn

        for name in watch:
            if name in names:
                names[name].register_forward_hook(hook(name))

    def pre_optim_step(self) -> None:  # noqa: D102
        model = self.trainer.train_module.model
        sq: dict[str, float] = {}
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            g = g.to_local() if hasattr(g, "to_local") else g
            group = _diag_group(name)
            sq[group] = sq.get(group, 0.0) + float(g.float().pow(2).sum())
        for group, v in sq.items():
            self._grads.setdefault(group, []).append(v**0.5)
        self._steps += 1
        logger.info(
            "DIAG step %d grads %s",
            self.trainer.global_step,
            {k: round(v**0.5, 3) for k, v in sorted(sq.items())},
        )
        if self._steps >= DIAG_STEPS:
            self._summary(model)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

    def _summary(self, model) -> None:  # type: ignore[no-untyped-def]
        import numpy as np

        weights: dict[str, float] = {}
        qk: dict[str, float] = {}
        for name, p in model.named_parameters():
            w = p.detach()
            w = w.to_local() if hasattr(w, "to_local") else w
            group = _diag_group(name)
            weights[group] = weights.get(group, 0.0) + float(w.float().pow(2).sum())
            if any(t in name for t in (".q.", ".k.", "q_proj", "k_proj", ".qkv.")):
                qk[name] = float(w.float().norm())
        summary = {
            "grad_norm_mean": {
                k: round(float(np.mean(v)), 3) for k, v in sorted(self._grads.items())
            },
            "grad_norm_max": {
                k: round(float(np.max(v)), 3) for k, v in sorted(self._grads.items())
            },
            "weight_norm": {k: round(v**0.5, 2) for k, v in sorted(weights.items())},
            "act_rms_mean_absmax": {
                k: [
                    round(float(np.mean([a for a, _ in v])), 3),
                    round(float(np.max([m for _, m in v])), 1),
                ]
                for k, v in self._acts.items()
            },
            "qk_weight_norm_top": dict(sorted(qk.items(), key=lambda kv: -kv[1])[:12]),
            "attn_max_qnorm_knorm_logitbound": {
                b: [
                    round(float(np.max(self._qk[f"{b}.q"])), 1),
                    round(float(np.max(self._qk[f"{b}.k"])), 1),
                    round(
                        float(np.max(self._qk[f"{b}.q"]))
                        * float(np.max(self._qk[f"{b}.k"]))
                        * self._qk[f"{b}.scale"][0],
                        1,
                    ),
                ]
                for b in sorted({k.rsplit(".", 1)[0] for k in self._qk})
                if f"{b}.q" in self._qk and f"{b}.k" in self._qk
            },
        }
        logger.info("DIAG SUMMARY %s", json.dumps(summary))


def build_trainer_config(common: CommonComponents):
    """The arm's trainer, stripped to a short profiled run."""
    trainer_config = _target(common).build_trainer_config(common)
    trainer_config.max_duration = Duration.steps(
        PROFILE_MAX_STEPS.get(_prefix(common), MAX_STEPS)
    )
    trainer_config.callbacks.pop("downstream_evaluator", None)
    trainer_config.callbacks["wandb"].enabled = False
    if _prefix(common) == "diag_allcap":
        # Runs from --trainer.load_path; the callback exits after DIAG_STEPS.
        trainer_config.max_duration = Duration.steps(10**7)
        checkpointer = trainer_config.callbacks["checkpointer"]
        checkpointer.save_interval = 10**9
        checkpointer.ephemeral_save_interval = None
        trainer_config.callbacks["grad_diag"] = GradDiagCallback()
        return trainer_config
    checkpointer = trainer_config.callbacks["checkpointer"]
    checkpointer.save_interval = 10**9
    checkpointer.ephemeral_save_interval = None
    profiler_cls = (
        RegionProfilerCallback
        if _prefix(common) in REGION_PREFIXES
        else ProfilerCallback
    )
    if _prefix(common) in PEAK_PREFIXES:
        trainer_config.max_duration = Duration.steps(
            PEAK_FORCED_MAX_STEPS
            if _prefix(common).startswith("prof_peakF")
            else PEAK_MAX_STEPS
        )
        trainer_config.callbacks["peak_memory"] = PeakMemoryCallback()
        return trainer_config
    if _prefix(common) in MEMORY_PREFIXES:
        trainer_config.max_duration = Duration.steps(MEMORY_START + MEMORY_STEPS + 2)
        trainer_config.callbacks["memory_snapshot"] = MemorySnapshotCallback()
        return trainer_config
    trainer_config.callbacks["profiler"] = profiler_cls(
        skip_first=PROFILE_SKIP,
        wait=1,
        warmup=PROFILE_WARMUP,
        active=PROFILE_ACTIVE,
        repeat=1,
        with_stack=False,
    )
    return trainer_config


def run() -> None:
    """Run the profiling job."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
