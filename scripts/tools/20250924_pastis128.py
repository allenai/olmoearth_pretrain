"""Process PASTIS-R using the package processor and project defaults."""

from upath import UPath

from olmoearth_pretrain.evals.datasets.pastis_processor import (
    process_pastis as _process_pastis,
)
from olmoearth_pretrain.evals.datasets.pastis_processor import (
    process_pastis_orig_size as _process_pastis_orig_size,
)

DATA_DIR = UPath("/weka/dfive-default/helios/evaluation/PASTIS-R")
PASTIS_DIR = UPath("/weka/dfive-default/presto_eval_sets/pastis_r")
PASTIS_DIR_ORIG = UPath("/weka/dfive-default/presto_eval_sets/pastis_r_origsize")


def process_pastis(
    data_dir: str | UPath = DATA_DIR,
    output_dir: str | UPath = PASTIS_DIR,
) -> None:
    """Process PASTIS-R into 64x64 tiles."""
    _process_pastis(data_dir=str(data_dir), output_dir=str(output_dir))


def process_pastis_orig_size(
    data_dir: str | UPath = DATA_DIR,
    output_dir: str | UPath = PASTIS_DIR_ORIG,
) -> None:
    """Process PASTIS-R without splitting 128x128 tiles."""
    _process_pastis_orig_size(data_dir=str(data_dir), output_dir=str(output_dir))


if __name__ == "__main__":
    process_pastis_orig_size()
