"""``pixreg_pixrecon`` at the v1.3 supervision weight (base 1.0 instead of 0.1).

Identical to ``pixreg_pixrecon.py`` (pixel-resolution d128 registers, ps 1..4, hw_p <=
24, per-cell map supervision, time-conditioned S2 L2A / S1 reconstruction) except the
LOSS BALANCE: the map-supervision base weight is the v1.3 release value of 1.0 rather
than the 0.1 the pixel-branch program inherited from its w0p1 lineage, and the two
reconstruction heads scale with it (0.05 -> 0.5 each) so the recon:map ratio is
unchanged. The MIM loss is the only term whose relative weight drops.

WHY: at ~120k steps the pixreg_pixrecon run's map-supervision losses sit 10-45% above
the query-token-compaction ablation (same register shape, w1.0) -- worldcover 0.50 vs
0.46, cdl 1.29 vs 1.08, srtm 0.045 vs 0.031 -- and its gradient norm is ~2.5x lower,
both consistent with the 10x lower weight rather than the pixel grid. The in-loop
evals are land-cover / crop probes, i.e. exactly what the map heads teach. This arm
also makes the pixel grid directly comparable to the w1 qtc run.

Everything else (sampler, microbatch, evals at 40k steps, W&B project) is imported
from ``pixreg_pixrecon``.
"""

import logging
import sys
from pathlib import Path

# Sibling arms import the pixreg_pixrecon builders from this directory and the
# release recipe from the directory above.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from pixreg_pixrecon import (  # noqa: E402
    PIXEL_RECON_WEIGHT,
    SUPERVISION_BASE_WEIGHT,
    apply_pixel_reconstruction,
    build_dataloader_config,
    build_pixreg_model_config,
    build_train_module_config,
)
from pixreg_pixrecon import (
    build_trainer_config as _pixreg_build_trainer_config,  # noqa: E402
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_3/experiments/pixreg_pixrecon_w1.py"

# v1.3's map-supervision base weight (pixreg_pixrecon trains at 0.1).
W1_SUPERVISION_BASE_WEIGHT = 1.0
# Reconstruction heads scale 10x with the map weight (0.05 -> 0.5 each) so the
# recon:map ratio is unchanged.
W1_PIXEL_RECON_WEIGHT = 0.5
# Guard the ratio against drift in either base constant.
assert (
    abs(
        W1_PIXEL_RECON_WEIGHT / PIXEL_RECON_WEIGHT
        - W1_SUPERVISION_BASE_WEIGHT / SUPERVISION_BASE_WEIGHT
    )
    < 1e-9
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 pixel registers + map supervision (w1.0) + S2 L2A / S1 reconstruction (0.5)."""
    config = build_pixreg_model_config(
        common, supervision_base_weight=W1_SUPERVISION_BASE_WEIGHT
    )
    return apply_pixel_reconstruction(config, weight=W1_PIXEL_RECON_WEIGHT)


def build_trainer_config(common: CommonComponents):
    """pixreg_pixrecon's trainer, with the eval job re-importing THIS module."""
    return _pixreg_build_trainer_config(common, module_path=MODULE_PATH)


def run() -> None:
    """Run the experiment."""
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
