"""E5 arm A: allcap: least-cloudy S2 + midpoint S1 per 30-day slot, stamped at slot start.

See ``e5_allcap_common.py`` (fallback env vars ``E5_S2_ONLY`` / ``E5_MIM``). Run name ``agent-ilse-e5-armA-30k-20261016``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from e5_allcap_common import ALLCAP_LEAST_CLOUDY_SLOT, run_arm  # noqa: E402

if __name__ == "__main__":
    run_arm(ALLCAP_LEAST_CLOUDY_SLOT)
