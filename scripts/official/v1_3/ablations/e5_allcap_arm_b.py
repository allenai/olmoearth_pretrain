"""E5 arm B: allcap: up to 12 random clear S2 captures (cloud <= tau) with real dates.

See ``e5_allcap_common.py`` (fallback env vars ``E5_S2_ONLY`` / ``E5_MIM``). Run name ``agent-ilse-e5-armB-30k-20261016``.
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
