"""Concatenate the metadata files for one modality."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm
from upath import UPath

from olmoearth_pretrain.data.constants import Modality, ModalitySpec, TimeSpan

from .util import get_modality_dir, get_modality_temp_meta_dir

DEFAULT_READ_WORKERS = 128


def _read_csv(fname: UPath) -> tuple[list[str], list[dict[str, str]]]:
    """Read a single per-window metadata CSV and return (fieldnames, rows)."""
    with fname.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV at {fname} does not contain header")
        return list(reader.fieldnames), list(reader)


def make_meta_summary(
    olmoearth_path: UPath,
    modality: ModalitySpec,
    time_span: TimeSpan,
    read_workers: int = DEFAULT_READ_WORKERS,
) -> None:
    """Create the concatenated metadata file for the specified modality.

    The data files and per-example temporary metadata files must be populated for the
    modality already. This function just concatenates those temporary metadata files
    together into one big CSV.

    Args:
        olmoearth_path: OlmoEarth Pretrain dataset path.
        modality: modality to write summary for.
        time_span: time span to write summary for.
        read_workers: number of threads for concurrent CSV reads.
    """
    modality_dir = get_modality_dir(olmoearth_path, modality, time_span)
    meta_dir = get_modality_temp_meta_dir(olmoearth_path, modality, time_span)

    print(f"Listing files in {meta_dir} ...")
    meta_fnames = list(meta_dir.iterdir())
    print(f"Found {len(meta_fnames)} metadata files.")

    column_names: list[str] | None = None
    csv_rows: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=read_workers) as pool:
        futures = {pool.submit(_read_csv, fname): fname for fname in meta_fnames}
        for future in tqdm.tqdm(
            as_completed(futures), total=len(futures), desc="Reading CSVs"
        ):
            fieldnames, rows = future.result()
            if column_names is None:
                column_names = fieldnames
            csv_rows.extend(rows)

    if column_names is None:
        raise ValueError(f"did not find any files in {meta_dir}")

    print(f"Writing {len(csv_rows)} rows to summary CSV ...")
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
        "--read-workers",
        type=int,
        default=DEFAULT_READ_WORKERS,
        help=f"Number of threads for concurrent CSV reads (default: {DEFAULT_READ_WORKERS})",
    )
    args = parser.parse_args()

    modality = Modality.get(args.modality)
    make_meta_summary(
        UPath(args.olmoearth_path),
        modality,
        TimeSpan(args.time_span),
        read_workers=args.read_workers,
    )
