"""Bathymetry direct-supervision pair, treatment arm: sdb head's loss reaches the encoder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v13_tiny_60k import run  # noqa: E402

if __name__ == "__main__":
    run(sdb_detach=False)
