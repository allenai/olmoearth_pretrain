"""E5 arm A: allcap, S2-only, least-cloudy S2 capture per 30-day slot, stamped at slot start.

See ``e5_allcap_common.py`` (set ``E5_MIM``). Run name ``agent-ilse-e5s2-armA-20261016``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e5_allcap_common import ALLCAP_LEAST_CLOUDY_SLOT, run_arm  # noqa: E402

if __name__ == "__main__":
    run_arm(ALLCAP_LEAST_CLOUDY_SLOT)
