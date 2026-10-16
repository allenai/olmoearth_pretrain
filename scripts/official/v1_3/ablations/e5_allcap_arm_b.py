"""E5 arm B: allcap, S2-only, up to 12 random clear S2 captures (cloud <= tau), real dates.

See ``e5_allcap_common.py`` (set ``E5_MIM``). Run name ``agent-ilse-e5s2-armB-20261016``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e5_allcap_common import (  # noqa: E402
    ALLCAP_RANDOM_CLEAR,
    ARM_B_CLOUD_TAU,
    run_arm,
)

if __name__ == "__main__":
    run_arm(ALLCAP_RANDOM_CLEAR, ARM_B_CLOUD_TAU)
