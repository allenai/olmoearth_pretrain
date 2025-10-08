"""Entrypoint for Sentinel-2 L1C jobs."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import materialize_dataset_windows
from rslearn.main import IngestHandler
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

GROUP = "res_10"


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(
        description="Ingest Sentinel-2 L1C images",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to the rslearn dataset for dataset creation",
        required=True,
    )
    parser.add_argument(
        "--windows",
        type=str,
        help="Comma-separated list of window names to ingest",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers",
        default=32,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    dataset = Dataset(ds_path)

    # Load the Window objects.
    windows: list[Window] = []
    for window_name in args.windows.split(","):
        window = Window.load(Window.get_window_root(ds_path, GROUP, window_name))
        windows.append(window)

    # Load the IngestHandler and corresponding jobs to run.
    ingest_handler = IngestHandler()
    ingest_handler.set_dataset(dataset)
    ingest_handler_jobs = ingest_handler.get_jobs(windows, args.workers)

    # Run the ingest jobs.
    # ingest_handler expects to be called with lists of jobs so we convert it to
    # single-element lists.
    ingest_handler_batches = [[job] for job in ingest_handler_jobs]
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(ingest_handler, ingest_handler_batches)
    for _ in tqdm.tqdm(outputs, total=len(ingest_handler_batches)):
        pass

    # Materialize data. It is by window so we can call materialize_dataset_windows
    # directly.
    materialize_dataset_windows_jobs = []
    for window in windows:
        materialize_dataset_windows_jobs.append(
            dict(
                dataset=dataset,
                windows=[window],
            )
        )
    outputs = star_imap_unordered(
        p, materialize_dataset_windows, materialize_dataset_windows_jobs
    )
    for _ in tqdm.tqdm(outputs, total=len(materialize_dataset_windows_jobs)):
        pass

    p.close()
