"""v1.3-tiny 60k: BlueTopo depth supervises the encoder; the sdb head is a detached probe.

Baseline: v13_tiny_60k_sdb_probe.py. See v13_tiny_60k.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_tiny_60k import run  # noqa: E402

if __name__ == "__main__":
    run(sdb_detach=True, bluetopo=True)
