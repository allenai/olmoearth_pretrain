"""Short profiled training runs to attribute step time and A/B speed flags.

Context: the v1.2 runs are purely GPU-compute-bound (data loading 0.05%,
callbacks ~1%, model 99.7% of the 0.66s step; H100s at ~335W/700W), so the
levers are kernel efficiency: flash-attn/varlen packing, torch.compile, fused
AdamW. This script runs a short (default 500-step) training job that is
identical to the production config except:

* ``max_duration`` is a few hundred steps and nothing is saved or loaded
  (no checkpointer callback, ``LoadStrategy.never``);
* in-loop evals are removed;
* olmo-core's ``ProfilerCallback`` records a torch.profiler trace on rank 0
  (written to ``<save_folder>/profiler``) for kernel-level attribution; and
* ``Encoder.remove_masked_tokens`` is instrumented to log the per-batch
  padding fraction (``benchmark/padding fraction``) -- because varlen packing
  removes pad tokens from ALL transformer compute, this number is directly
  the fraction of encoder time a flash/varlen run saves.

Compare arms via ``throughput/device/BPS`` in W&B; the run is steady-state so
~500 steps is representative. Launch the full ladder with
``launch_speed_benchmark.sh``; the flags under test are plain CLI overrides:

* flash:   --model.encoder_config.use_flash_attn=true
           --model.decoder_config.use_flash_attn=true
* compile: --train_module.compile_model=true
* fused:   --train_module.optim_config.fused=true

Model variant: set ``SPEED_BENCH_MODEL`` to ``base`` or ``regbtl``
(default ``auto``: regbtl when ``regbtl_v1_2_common`` is importable -- i.e. on
the register-bottleneck branch -- else base). The regbtl variant matches the
profiled production run (gdyn_d768_il_pdproj, nolsa); its register read
blocks intentionally stay on SDPA, so the flash override still composes.

``SPEED_BENCH_STEPS`` overrides the step count (the compile arm may need more
steps if recompilation on the variable patch-size/sequence-length batches is
frequent -- watch for step-time never stabilizing).
"""

import logging
import os
from dataclasses import dataclass

import torch
from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from base import build_model_config as _base_build_model_config
from base import build_trainer_config as _base_build_trainer_config
from olmo_core.train.callbacks import Callback, ProfilerCallback
from olmo_core.train.common import Duration, LoadStrategy, ReduceType

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.flexi_vit import Encoder
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

try:
    from regbtl_v1_2_common import build_regbtl_model_config

    _HAS_REGBTL = True
except ImportError:
    _HAS_REGBTL = False

logger = logging.getLogger(__name__)

BENCHMARK_STEPS = int(os.environ.get("SPEED_BENCH_STEPS", "500"))
MODEL_VARIANT = os.environ.get("SPEED_BENCH_MODEL", "auto")
WANDB_PROJECT = "2026_07_08_speed_benchmark"

# Trace steps ~56-61: past startup jitter, well before the end of the run.
PROFILER_SKIP_FIRST = 50
PROFILER_WARMUP = 5
PROFILER_ACTIVE = 5


# ---------------------------------------------------------------------------
# Padding instrumentation.
#
# Encoder.remove_masked_tokens right-pads every (micro)batch to the longest
# kept sequence, and the SDPA path computes full FLOPs over that padding. We
# wrap it to record fill = seq_lengths.sum() / (B * max_len). Stats stay on
# GPU and are synced once per step by the callback below, so the perturbation
# to the measured step time is negligible.
# ---------------------------------------------------------------------------

_PADDING_SAMPLES: list[tuple[torch.Tensor, torch.Tensor]] = []

_orig_remove_masked_tokens = Encoder.remove_masked_tokens


def _instrumented_remove_masked_tokens(x, mask):  # noqa: ANN001, ANN202
    out = _orig_remove_masked_tokens(x, mask)
    _, _, _, seq_lengths, max_length = out
    with torch.no_grad():
        fill = seq_lengths.sum().float() / (seq_lengths.numel() * max_length.float())
        _PADDING_SAMPLES.append((fill, max_length.float()))
    return out


Encoder.remove_masked_tokens = staticmethod(_instrumented_remove_masked_tokens)


@dataclass
class PaddingMonitorCallback(Callback):
    """Log the mean padding fraction and max sequence length each step."""

    def post_step(self) -> None:
        """Aggregate and record the padding stats captured during this step."""
        if not _PADDING_SAMPLES:
            return
        fills = torch.stack([fill for fill, _ in _PADDING_SAMPLES])
        max_lens = torch.stack([max_len for _, max_len in _PADDING_SAMPLES])
        _PADDING_SAMPLES.clear()
        self.trainer.record_metric(
            "benchmark/padding fraction", 1.0 - fills.mean(), ReduceType.mean
        )
        self.trainer.record_metric(
            "benchmark/max seq len", max_lens.mean(), ReduceType.mean
        )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model under test: regbtl when available/requested, else v1.2 base."""
    variant = MODEL_VARIANT
    if variant == "auto":
        variant = "regbtl" if _HAS_REGBTL else "base"
    if variant == "regbtl":
        if not _HAS_REGBTL:
            raise RuntimeError(
                "SPEED_BENCH_MODEL=regbtl but regbtl_v1_2_common is not importable; "
                "launch from the register-bottleneck branch or use SPEED_BENCH_MODEL=base"
            )
        logger.info(
            "Speed benchmark model variant: regbtl (gdyn_d768_il_pdproj, nolsa)"
        )
        return build_regbtl_model_config(common, latent_self_attn=False)
    logger.info("Speed benchmark model variant: v1.2 base")
    return _base_build_model_config(common)


def build_trainer_config(common: CommonComponents):
    """Production trainer config minus evals/checkpointing, plus profiling."""
    trainer_config = _base_build_trainer_config(common)
    trainer_config.max_duration = Duration.steps(BENCHMARK_STEPS)
    trainer_config.load_strategy = LoadStrategy.never
    # Benchmark runs save nothing and never resume; evals run elsewhere.
    trainer_config.callbacks.pop("checkpointer", None)
    trainer_config.callbacks.pop("downstream_evaluator", None)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    trainer_config.add_callback(
        "profiler",
        ProfilerCallback(
            skip_first=PROFILER_SKIP_FIRST,
            warmup=PROFILER_WARMUP,
            active=PROFILER_ACTIVE,
        ),
    )
    trainer_config.add_callback("padding_monitor", PaddingMonitorCallback())
    return trainer_config


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
