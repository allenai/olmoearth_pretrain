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
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import build_common_components as _base_common_components  # noqa: E402
from base import build_visualize_config  # noqa: E402
from olmo_core.train.callbacks import ProfilerCallback  # noqa: E402
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
}
# Prefixes whose profile also times each named module's FORWARD as a region (backward
# hooks break on modules returning dicts; backward is ~2x each region's forward).
REGION_PREFIXES = ("prof_regD", "prof_regv13")
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
        if torch.cuda.is_available():
            logger.info(
                "Peak GPU memory: "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
                f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved"
            )
        super()._on_trace_ready(prof)


def build_trainer_config(common: CommonComponents):
    """The arm's trainer, stripped to a short profiled run."""
    trainer_config = _target(common).build_trainer_config(common)
    trainer_config.max_duration = Duration.steps(
        PROFILE_MAX_STEPS.get(_prefix(common), MAX_STEPS)
    )
    trainer_config.callbacks.pop("downstream_evaluator", None)
    trainer_config.callbacks["wandb"].enabled = False
    checkpointer = trainer_config.callbacks["checkpointer"]
    checkpointer.save_interval = 10**9
    checkpointer.ephemeral_save_interval = None
    profiler_cls = (
        RegionProfilerCallback
        if _prefix(common) in REGION_PREFIXES
        else ProfilerCallback
    )
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
