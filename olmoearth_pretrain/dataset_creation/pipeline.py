"""End-to-end pipeline: rslearn dataset → olmoearth tiffs → consolidated metadata → rasterized OSM → h5py.

Usage:
    python -m olmoearth_pretrain.dataset_creation.pipeline \
        --ds_path /path/to/rslearn_dataset \
        --olmoearth_path /path/to/olmoearth_output \
        --workers 8 --groups res_10.0

    # Skip steps that already completed:
    python -m olmoearth_pretrain.dataset_creation.pipeline \
        --ds_path ... --olmoearth_path ... \
        --skip convert

    # Only run h5 conversion (everything else done):
    python -m olmoearth_pretrain.dataset_creation.pipeline \
        --ds_path ... --olmoearth_path ... \
        --only h5
"""

import argparse
import multiprocessing
import sys

from upath import UPath

from olmoearth_pretrain.data.constants import Modality, TimeSpan

STEPS = ["convert", "metadata", "rasterize_osm", "h5"]

MODALITIES_FOR_H5 = [
    "sentinel2_l2a",
    "sentinel1",
    "worldcover",
    "openstreetmap_raster",
    "worldcereal",
    "srtm",
    "wri_canopy_height_map",
]

MODALITY_TIME_SPANS: list[tuple[Modality, TimeSpan]] = [
    (Modality.SENTINEL2_L2A, TimeSpan.YEAR),
    (Modality.SENTINEL1, TimeSpan.YEAR),
    (Modality.WORLDCOVER, TimeSpan.STATIC),
    (Modality.OPENSTREETMAP, TimeSpan.STATIC),
    (Modality.WORLDCEREAL, TimeSpan.STATIC),
    (Modality.SRTM, TimeSpan.STATIC),
    (Modality.WRI_CANOPY_HEIGHT_MAP, TimeSpan.STATIC),
]


def step_convert(ds_path: str, olmoearth_path: str, workers: int, groups: str) -> None:
    """Step 1: Convert rslearn windows to olmoearth tiffs + per-window metadata."""
    from rslearn.dataset import Dataset
    from rslearn.utils.mp import star_imap_unordered
    import tqdm

    from olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.convert_all import ALL_MODALITIES, CONVERTERS, TEMPORAL_MODALITIES, _convert_window

    group_list = [g.strip() for g in groups.split(",")]
    print(f"[convert] Converting modalities: {ALL_MODALITIES}")
    print(f"[convert] Groups: {group_list}")

    dataset = Dataset(UPath(ds_path))
    olmo_path = UPath(olmoearth_path)

    jobs = []
    for window in dataset.load_windows(workers=workers, show_progress=True, groups=group_list):
        jobs.append(dict(
            window=window,
            olmoearth_path=olmo_path,
            modalities=ALL_MODALITIES,
            use_temporal_stack=True,
        ))

    print(f"[convert] Processing {len(jobs)} windows with {workers} workers...")
    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, _convert_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
    print("[convert] Done.")


def step_metadata(olmoearth_path: str) -> None:
    """Step 2: Consolidate per-window temp metadata CSVs into one CSV per modality."""
    from olmoearth_pretrain.dataset_creation.make_meta_summary import make_meta_summary

    olmo_path = UPath(olmoearth_path)
    for modality, time_span in MODALITY_TIME_SPANS:
        print(f"[metadata] Consolidating {modality.name} ({time_span.value})...")
        try:
            make_meta_summary(olmo_path, modality, time_span)
        except (ValueError, FileNotFoundError) as e:
            print(f"[metadata]   skipped: {e}")
    print("[metadata] Done.")


def step_rasterize_osm(olmoearth_path: str, workers: int) -> None:
    """Step 3: Rasterize OSM GeoJSON vectors into tiff rasters."""
    import csv

    from rslearn.utils.geometry import Projection
    from rslearn.utils.mp import star_imap_unordered
    from rasterio.crs import CRS
    import tqdm

    from olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.rasterize_openstreetmap import (
        FACTOR,
        OUTPUT_RESOLUTION,
        OpenStreetMapRasterJob,
        _window_metadata_from_csv_row,
        rasterize_openstreetmap,
    )
    from olmoearth_pretrain.dataset_creation.util import get_modality_dir
    from olmoearth_pretrain.dataset.utils import get_modality_fname

    olmo_path = UPath(olmoearth_path)
    src_modality_dir = get_modality_dir(olmo_path, Modality.OPENSTREETMAP, TimeSpan.STATIC)
    src_metadata_fname = olmo_path / f"{src_modality_dir.name}.csv"

    if not src_metadata_fname.exists():
        print(f"[rasterize_osm] No {src_metadata_fname} found, skipping.")
        return

    with src_metadata_fname.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"got None for field names in {src_metadata_fname}")
        csv_rows = list(reader)

    print(f"[rasterize_osm] Rasterizing {len(csv_rows)} OSM samples...")
    rasterize_jobs = []
    for csv_row in csv_rows:
        window_metadata = _window_metadata_from_csv_row(csv_row)
        bounds = window_metadata.bounds
        if bounds is None:
            raise ValueError("window metadata must include bounds")
        rasterize_jobs.append(dict(
            job=OpenStreetMapRasterJob(
                in_fname=get_modality_fname(
                    olmo_path, Modality.OPENSTREETMAP, TimeSpan.STATIC,
                    window_metadata, 10, "geojson",
                ),
                out_fname=get_modality_fname(
                    olmo_path, Modality.OPENSTREETMAP_RASTER, TimeSpan.STATIC,
                    window_metadata, OUTPUT_RESOLUTION, "tif",
                ),
                projection=Projection(
                    CRS.from_string(window_metadata.crs),
                    OUTPUT_RESOLUTION, -OUTPUT_RESOLUTION,
                ),
                bounds=(
                    bounds[0] * FACTOR, bounds[1] * FACTOR,
                    bounds[2] * FACTOR, bounds[3] * FACTOR,
                ),
            )
        ))

    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, rasterize_openstreetmap, rasterize_jobs)
    for _ in tqdm.tqdm(outputs, total=len(rasterize_jobs)):
        pass
    p.close()

    dst_modality_dir = get_modality_dir(olmo_path, Modality.OPENSTREETMAP_RASTER, TimeSpan.STATIC)
    dst_metadata_fname = olmo_path / f"{dst_modality_dir.name}.csv"
    for csv_row in csv_rows:
        if csv_row["image_idx"] == "N/A":
            csv_row["image_idx"] = "0"
    with dst_metadata_fname.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("[rasterize_osm] Done.")


def step_h5(olmoearth_path: str) -> None:
    """Step 4: Convert olmoearth tiffs → h5py training dataset."""
    from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5pyConfig

    print(f"[h5] Converting to h5py...")
    config = ConvertToH5pyConfig(
        tile_path=olmoearth_path,
        supported_modality_names=MODALITIES_FOR_H5,
        multiprocessed_h5_creation=True,
        compression="zstd",
        compression_opts=3,
    )
    converter = config.build()
    converter.run()
    print("[h5] Done.")


def main() -> None:
    multiprocessing.set_start_method("forkserver", force=True)

    parser = argparse.ArgumentParser(description="Full rslearn → olmoearth → h5py pipeline")
    parser.add_argument("--ds_path", type=str, required=True, help="Source rslearn dataset path")
    parser.add_argument("--olmoearth_path", type=str, required=True, help="Destination OlmoEarth path")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--groups", type=str, default="res_10.0", help="rslearn window groups")
    parser.add_argument("--skip", type=str, default=None, help=f"Comma-separated steps to skip: {STEPS}")
    parser.add_argument("--only", type=str, default=None, help=f"Only run this step: {STEPS}")
    args = parser.parse_args()

    if args.only:
        steps_to_run = [args.only]
    else:
        skip = set(args.skip.split(",")) if args.skip else set()
        steps_to_run = [s for s in STEPS if s not in skip]

    print(f"Pipeline steps: {steps_to_run}")

    if "convert" in steps_to_run:
        step_convert(args.ds_path, args.olmoearth_path, args.workers, args.groups)

    if "metadata" in steps_to_run:
        step_metadata(args.olmoearth_path)

    if "rasterize_osm" in steps_to_run:
        step_rasterize_osm(args.olmoearth_path, args.workers)

    if "h5" in steps_to_run:
        step_h5(args.olmoearth_path)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
