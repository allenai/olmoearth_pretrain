"""E5 2x2 arm A_real: arm A's captures (least-cloudy S2 + midpoint S1 per 30-day slot), real dates.

With arm A (fixed captures, slot-start dates), arm B (fresh captures, real dates) and arm
B_slot (``e5_allcap_arm_bslot.py``: fresh captures, slot-start dates), this completes a 2x2 that
separates "captures redrawn every read" from "real dates" (Gandalf / Hollis 2026-10-19).
See ``e5_allcap_common.py``. Run name ``agent-gandalf-e5-armAreal-30k-20261019``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e5_allcap_common import run_arm  # noqa: E402

from olmoearth_pretrain.data.allcap import ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES  # noqa: E402

if __name__ == "__main__":
    run_arm(ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES)
