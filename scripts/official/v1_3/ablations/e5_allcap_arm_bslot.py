"""E5 2x2 arm B_slot: arm A's 30-day slot grid and slot-start dates, captures redrawn every read.

Per slot a random S2 capture with cloud fraction <= ``ARM_B_CLOUD_TAU`` (least-cloudy fallback
when the slot has none) and a random S1 capture. See ``e5_allcap_arm_areal.py`` for the 2x2 and
``e5_allcap_common.py`` for the shared recipe. Run name ``agent-gandalf-e5-armBslot-30k-20261019``.
Note: tau is arm B's value (matched to arm A's mean cloud fraction for B's rule, not re-matched
for this per-slot rule), so B_slot's inputs are likely slightly cloudier than arm A's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e5_allcap_common import ARM_B_CLOUD_TAU, run_arm  # noqa: E402

from olmoearth_pretrain.data.allcap import ALLCAP_RANDOM_CLEAR_SLOT  # noqa: E402

if __name__ == "__main__":
    run_arm(ALLCAP_RANDOM_CLEAR_SLOT, ARM_B_CLOUD_TAU)
