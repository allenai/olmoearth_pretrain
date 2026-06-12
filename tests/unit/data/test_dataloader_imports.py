"""Import boundary tests for dataloader modules."""

import subprocess
import sys


def test_dataloader_import_does_not_import_train_masking() -> None:
    """The data package should not import training masking code at module import time."""
    script = """
import sys
import olmoearth_pretrain.data.dataloader
print("olmoearth_pretrain.train.masking" in sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
