"""Window retrieval on v1.3, arm ``end_to_end``; see recipe.py for the recipe and launch commands."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recipe import run  # noqa: E402

MODULE_PATH = "scripts/official/v1_3/window_retrieval/end_to_end.py"

if __name__ == "__main__":
    run("end_to_end", MODULE_PATH)
