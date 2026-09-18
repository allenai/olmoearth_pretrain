"""Concatenate the metadata files for one modality."""

import argparse
import csv
import multiprocessing

import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec, TimeSpan

from .util import get_modality_dir, get_modality_temp_meta_dir


def _read_meta_csv(fname: UPath) -> tuple[list[str], list[dict[str, str]]]:
    """Read one per-example metadata CSV, returning (fieldnames, rows)."""
    with fname.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV at {fname} does not contain header")
        return list(reader.fieldnames), list(reader)


def make_meta_summary(
    olmoearth_path: UPath,
    modality: ModalitySpec,
    time_span: TimeSpan,
    workers: int = 1,
) -> None:
    """Create the concatenated metadata file for the specified modality.

    The data files and per-example temporary metadata files must be populated for the
    modality already. This function just concatenates those temporary metadata files
    together into one big CSV.

    Args:
        olmoearth_path: OlmoEarth Pretrain dataset path.
        modality: modality to write summary for.
        time_span: time span to write summary for.
        workers: parallel readers (the per-example files are tiny, so on network
            filesystems this is dominated by per-file latency).
    """
    # Concatenate the CSVs while keeping the header only from the first file that we
    # read.
    column_names: list[str] | None = None
    csv_rows = []
    modality_dir = get_modality_dir(olmoearth_path, modality, time_span)
    meta_dir = get_modality_temp_meta_dir(olmoearth_path, modality, time_span)
    meta_fnames = list(meta_dir.iterdir())
    if workers > 1 and len(meta_fnames) > 1:
        with multiprocessing.Pool(workers) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(_read_meta_csv, meta_fnames, chunksize=64),
                    total=len(meta_fnames),
                )
            )
    else:
        results = [_read_meta_csv(fname) for fname in tqdm.tqdm(meta_fnames)]
    for fieldnames, rows in results:
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
    args = parser.parse_args()

    modality = Modality.get(args.modality)
    make_meta_summary(UPath(args.olmoearth_path), modality, TimeSpan(args.time_span))
