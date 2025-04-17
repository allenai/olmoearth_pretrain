"""Concatenate the metadata files for one modality."""

import argparse
import csv

import tqdm
from upath import UPath

from helios.data.constants import Modality, ModalitySpec, TimeSpan

from .util import get_modality_dir, get_modality_temp_meta_dir


def make_meta_summary(
    helios_path: UPath, modality: ModalitySpec, time_span: TimeSpan
) -> None:
    """Create the concatenated metadata file for the specified modality.

    The data files and per-example temporary metadata files must be populated for the
    modality already. This function just concatenates those temporary metadata files
    together into one big CSV.

    Args:
        helios_path: Helios dataset path.
        modality: modality to write summary for.
        time_span: time span to write summary for.
    """
    # Concatenate the CSVs while keeping the header only from the first file that we
    # read.
    column_names: list[str] | None = None
    csv_rows = []
    modality_dir = get_modality_dir(helios_path, modality, time_span)
    meta_dir = get_modality_temp_meta_dir(helios_path, modality, time_span)
    meta_fnames = list(meta_dir.iterdir())
    for fname in tqdm.tqdm(meta_fnames):
        with fname.open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV at {fname} does not contain header")
            if column_names is None:
                column_names = list(reader.fieldnames)
            for csv_row in reader:
                csv_rows.append(csv_row)

    if column_names is None:
        raise ValueError(f"did not find any files in {meta_dir}")

    with (helios_path / f"{modality_dir.name}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create concatenated metadata file",
    )
    parser.add_argument(
        "--helios_path",
        type=str,
        help="Helios dataset path",
        required=True,
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality to summarize",
        required=True,
    )
    parser.add_argument(
        "--time_span",
        type=str,
        help="Time span to use (defaults to static)",
        default=TimeSpan.STATIC.value,
    )
    args = parser.parse_args()

    modality = Modality.get(args.modality)
    make_meta_summary(UPath(args.helios_path), modality, TimeSpan(args.time_span))
