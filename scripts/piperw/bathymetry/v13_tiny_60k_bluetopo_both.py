"""v1.3-tiny 60k: BlueTopo depth as a decode-only latent-MIM target and a supervision head.

Baseline: v13_tiny_60k.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_tiny_60k import BOTH, run  # noqa: E402

if __name__ == "__main__":
    run(bluetopo=BOTH)
