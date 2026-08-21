"""Concatenate the metadata files for one modality."""

import argparse
import csv
import multiprocessing

import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec, TimeSpan

from .util import get_modality_dir, get_modality_temp_meta_dir


def _read_meta_file(fname: UPath) -> tuple[list[str], list[dict[str, str]]]:
    """Read the header and rows from one temporary metadata file."""
    with fname.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV at {fname} does not contain header")
        return list(reader.fieldnames), list(reader)


def make_meta_summary(
    olmoearth_path: UPath,
    modality: ModalitySpec,
    time_span: TimeSpan,
    workers: int = 32,
) -> None:
    """Create the concatenated metadata file for the specified modality.

    The data files and per-example temporary metadata files must be populated for the
    modality already. This function just concatenates those temporary metadata files
    together into one big CSV.

    Args:
        olmoearth_path: OlmoEarth Pretrain dataset path.
        modality: modality to write summary for.
        time_span: time span to write summary for.
        workers: number of worker processes to use.
    """
    # Concatenate the CSVs while keeping the header only from the first file that we
    # read.
    column_names: list[str] | None = None
    csv_rows = []
    modality_dir = get_modality_dir(olmoearth_path, modality, time_span)
    meta_dir = get_modality_temp_meta_dir(olmoearth_path, modality, time_span)
    meta_fnames = list(meta_dir.iterdir())
    with multiprocessing.Pool(workers) as pool:
        file_contents = pool.imap(_read_meta_file, meta_fnames)
        for fieldnames, rows in tqdm.tqdm(file_contents, total=len(meta_fnames)):
            if column_names is None:
                column_names = fieldnames
            csv_rows.extend(rows)

    if column_names is None:
        raise ValueError(f"did not find any files in {meta_dir}")

    with (olmoearth_path / f"{modality_dir.name}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create concatenated metadata file",
    )
    parser.add_argument(
        "--olmoearth_path",
        type=str,
        help="OlmoEarth Pretrain dataset path",
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
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers to use",
        default=32,
    )
    args = parser.parse_args()

    modality = Modality.get(args.modality)
    make_meta_summary(
        UPath(args.olmoearth_path),
        modality,
        TimeSpan(args.time_span),
        workers=args.workers,
    )
