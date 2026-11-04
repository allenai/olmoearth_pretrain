"""Window retrieval on v1.3, arm ``from_scratch``; see recipe.py for the recipe and launch commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recipe import run  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/window_retrieval/from_scratch.py"

if __name__ == "__main__":
    run("from_scratch", MODULE_PATH)
