"""v1.3-tiny 60k: both sdb and BlueTopo depth supervise the encoder.

Baseline: v13_tiny_60k_sdb_sup.py. See v13_tiny_60k.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_tiny_60k import run  # noqa: E402

if __name__ == "__main__":
    run(sdb_detach=False, bluetopo=True)
