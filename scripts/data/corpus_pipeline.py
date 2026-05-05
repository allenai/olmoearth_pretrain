"""Distributed corpus pipeline: rslearn ingest -> convert -> H5.

Three independently launchable steps, each distributable across Beaker shards.

Usage:
    # Step 1: Launch rslearn ingest across N shards
    python scripts/data/corpus_pipeline.py launch-rslearn \
        --corpus /weka/.../studio_corpus_lonlats.json \
        --rslearn-dir /weka/.../studio_corpus_full/rslearn \
        --rslearn-config /weka/.../rslearn_corpus_30/config.json \
        --num-shards 100 --workers 16 --cluster ai2/jupiter

    # Step 2: Launch convert across N shards
    python scripts/data/corpus_pipeline.py launch-convert \
        --corpus /weka/.../studio_corpus_lonlats.json \
        --rslearn-dir /weka/.../studio_corpus_full/rslearn \
        --olmoearth-dir /weka/.../studio_corpus_full/olmoearth \
        --num-shards 100 --workers 16 --cluster ai2/jupiter

    # Step 3a: Prepare H5 metadata (single machine)
    python scripts/data/corpus_pipeline.py prepare-h5 \
        --olmoearth-dir /weka/.../studio_corpus_full/olmoearth

    # Step 3b: Launch H5 writing across N machines
    python scripts/data/corpus_pipeline.py launch-h5 \
        --olmoearth-dir /weka/.../studio_corpus_full/olmoearth \
        --num-h5-shards 4 --cluster ai2/jupiter

    # Check progress
    python scripts/data/corpus_pipeline.py status \
        --corpus /weka/.../studio_corpus_lonlats.json \
        --rslearn-dir /weka/.../studio_corpus_full/rslearn \
        --olmoearth-dir /weka/.../studio_corpus_full/olmoearth
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import subprocess
import sys

from upath import UPath

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _progress_dir(base_dir: str, step: str) -> UPath:
    return UPath(base_dir) / "progress" / step


def _write_progress(
    base_dir: str, step: str, shard_id: int, status: str, detail: str = ""
) -> None:
    """Write a progress file for this shard. Called periodically by workers."""
    import time

    d = _progress_dir(base_dir, step)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "shard_id": shard_id,
        "status": status,
        "detail": detail,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    path = d / f"shard_{shard_id:05d}.json"
    with path.open("w") as f:
        json.dump(payload, f)


def _read_all_progress(base_dir: str, step: str) -> list[dict]:
    """Read all shard progress files for a step."""
    d = _progress_dir(base_dir, step)
    if not d.exists():
        return []
    results = []
    for p in sorted(d.glob("shard_*.json")):
        with p.open() as f:
            results.append(json.load(f))
    return results


# ---------------------------------------------------------------------------
# Worker: rslearn ingest
# ---------------------------------------------------------------------------


def cmd_rslearn_worker(args: argparse.Namespace) -> None:
    """Run rslearn prepare/ingest/materialize for one shard."""
    from olmoearth_pretrain.dataset_creation.create_windows.from_corpus import (
        attach_dataset_config,
        create_corpus_windows,
    )
    from olmoearth_pretrain.dataset_creation.distributed import get_shard, load_corpus

    entries = load_corpus(args.corpus)
    if args.max_samples:
        entries = entries[: args.max_samples]
    shard = get_shard(entries, args.shard_id, args.num_shards)
    rslearn_dir = args.rslearn_dir
    logger.info(
        f"rslearn-worker shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard.entries)} samples"
    )

    # Ensure config.json exists in rslearn dir
    if args.rslearn_config:
        attach_dataset_config(rslearn_dir, args.rslearn_config, force=True)

    _write_progress(
        rslearn_dir, "rslearn", args.shard_id, "running", "creating windows"
    )

    # Create windows for this shard (idempotent -- skips existing)
    create_corpus_windows(UPath(rslearn_dir), shard.entries, workers=args.workers)

    # Run rslearn CLI steps
    window_args = []
    for name in shard.window_names:
        window_args.extend(["--window", name])

    rslearn_steps = ["prepare", "ingest", "materialize"]
    for i, step in enumerate(rslearn_steps):
        _write_progress(
            rslearn_dir,
            "rslearn",
            args.shard_id,
            "running",
            f"{step} ({i + 1}/{len(rslearn_steps)})",
        )
        cmd = [
            sys.executable,
            "-m",
            "rslearn.main",
            "dataset",
            step,
            "--root",
            rslearn_dir,
            *window_args,
            "--workers",
            str(args.workers),
        ]
        for layer in getattr(args, "disabled_layers", []):
            cmd.extend(["--disabled-layers", layer])
        if step in ("ingest", "materialize"):
            cmd.extend(["--ignore-errors", "--retry-max-attempts", "3"])

        logger.info(f"Running: rslearn dataset {step}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            _write_progress(rslearn_dir, "rslearn", args.shard_id, "failed", step)
            logger.error(f"rslearn {step} failed with code {result.returncode}")
            sys.exit(result.returncode)

    _write_progress(rslearn_dir, "rslearn", args.shard_id, "done")
    logger.info(f"rslearn-worker shard {args.shard_id} complete")


# ---------------------------------------------------------------------------
# Worker: convert
# ---------------------------------------------------------------------------


def cmd_convert_worker(args: argparse.Namespace) -> None:
    """Run convert + metadata + rasterize_osm for one shard's windows."""
    import tqdm as tqdm_mod
    from rslearn.dataset import Dataset
    from rslearn.utils.mp import star_imap_unordered

    from olmoearth_pretrain.dataset_creation.distributed import get_shard, load_corpus
    from olmoearth_pretrain.dataset_creation.pipeline import (
        step_metadata,
        step_rasterize_osm,
    )
    from olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.convert_all import (
        ALL_MODALITIES,
        _convert_window,
    )

    entries = load_corpus(args.corpus)
    if args.max_samples:
        entries = entries[: args.max_samples]
    shard = get_shard(entries, args.shard_id, args.num_shards)
    shard_names = set(shard.window_names)
    logger.info(
        f"convert-worker shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard.entries)} samples"
    )

    _write_progress(
        args.olmoearth_dir, "convert", args.shard_id, "running", "loading windows"
    )

    # Convert only this shard's windows
    dataset = Dataset(UPath(args.rslearn_dir))
    olmo_path = UPath(args.olmoearth_dir)
    jobs = []
    for window in dataset.load_windows(
        workers=args.workers, show_progress=True, groups=["res_10.0"]
    ):
        if window.name not in shard_names:
            continue
        jobs.append(
            dict(
                window=window,
                olmoearth_path=olmo_path,
                modalities=ALL_MODALITIES,
                use_temporal_stack=True,
            )
        )

    _write_progress(
        args.olmoearth_dir,
        "convert",
        args.shard_id,
        "running",
        f"converting {len(jobs)} windows",
    )
    logger.info(f"[convert] Processing {len(jobs)} windows with {args.workers} workers")
    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, _convert_window, jobs)
    for _ in tqdm_mod.tqdm(outputs, total=len(jobs)):
        pass
    p.close()

    _write_progress(
        args.olmoearth_dir, "convert", args.shard_id, "running", "metadata + osm"
    )
    step_metadata(args.olmoearth_dir)
    step_rasterize_osm(args.olmoearth_dir, args.workers)

    _write_progress(args.olmoearth_dir, "convert", args.shard_id, "done")
    logger.info(f"convert-worker shard {args.shard_id} complete")


# ---------------------------------------------------------------------------
# Worker: prepare-h5 (single machine)
# ---------------------------------------------------------------------------


def cmd_prepare_h5(args: argparse.Namespace) -> None:
    """Scan olmoearth TIFFs, filter, assign global indices, write metadata.

    Produces sample_manifest.json + sample_metadata.csv + latlon_distribution.npy
    in the H5 output directory.
    """
    from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5pyConfig
    from olmoearth_pretrain.dataset_creation.pipeline import MODALITIES_FOR_H5

    logger.info(f"prepare-h5: scanning {args.olmoearth_dir}")
    config = ConvertToH5pyConfig(
        tile_path=args.olmoearth_dir,
        supported_modality_names=MODALITIES_FOR_H5,
        compression="zstd",
        compression_opts=3,
    )
    converter = config.build()

    # Phase 1: scan and filter
    samples = converter.get_and_filter_samples()
    logger.info(f"prepare-h5: {len(samples)} samples after filtering")

    # Phase 2: build (index, sample) tuples
    assert converter.num_subtiles is not None
    tuples: list[tuple[int, object]] = []
    for sample in samples:
        for j in range(converter.num_subtiles):
            tuples.append((j, sample))

    converter.set_h5py_dir(len(tuples))
    assert converter.h5py_dir is not None
    h5py_dir = converter.h5py_dir

    # Write metadata and latlon (these are fast)
    converter.save_compression_settings()
    converter.save_sample_metadata(tuples)
    converter.save_latlon_distribution(tuples)

    # Write manifest for h5-workers to consume
    manifest_path = h5py_dir / "sample_manifest.json"
    manifest = {
        "h5py_dir": str(h5py_dir),
        "olmoearth_dir": args.olmoearth_dir,
        "num_samples": len(tuples),
        "modalities": MODALITIES_FOR_H5,
        "compression": "zstd",
        "compression_opts": 3,
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"prepare-h5 complete: {len(tuples)} samples, h5py_dir={h5py_dir}")
    logger.info(f"Manifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# Worker: h5-worker (one of N machines)
# ---------------------------------------------------------------------------


def cmd_h5_worker(args: argparse.Namespace) -> None:
    """Write sample_{index}.h5 files for this shard's index range."""
    from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5pyConfig

    manifest_path = UPath(args.h5py_dir) / "sample_manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    h5py_dir = UPath(manifest["h5py_dir"])
    olmoearth_dir = manifest["olmoearth_dir"]
    num_samples = manifest["num_samples"]

    # Compute this shard's index range
    per_shard = num_samples // args.num_h5_shards
    remainder = num_samples % args.num_h5_shards
    start = args.h5_shard_id * per_shard + min(args.h5_shard_id, remainder)
    end = start + per_shard + (1 if args.h5_shard_id < remainder else 0)
    logger.info(
        f"h5-worker {args.h5_shard_id}/{args.num_h5_shards}: "
        f"samples [{start}, {end}) of {num_samples}"
    )

    config = ConvertToH5pyConfig(
        tile_path=olmoearth_dir,
        supported_modality_names=manifest["modalities"],
        multiprocessed_h5_creation=False,
        compression=manifest.get("compression"),
        compression_opts=manifest.get("compression_opts"),
    )
    converter = config.build()
    samples = converter.get_and_filter_samples()

    assert converter.num_subtiles is not None
    all_tuples: list[tuple[int, object]] = []
    for sample in samples:
        for j in range(converter.num_subtiles):
            all_tuples.append((j, sample))

    shard_tuples = all_tuples[start:end]
    converter.h5py_dir = h5py_dir

    logger.info(f"h5-worker: writing {len(shard_tuples)} H5 files")
    total = len(shard_tuples)
    for i, (global_idx, (sublock_idx, sample)) in enumerate(
        zip(range(start, end), shard_tuples)
    ):
        if i % max(1, total // 20) == 0:
            _write_progress(
                str(h5py_dir),
                "h5",
                args.h5_shard_id,
                "running",
                f"{i}/{total} ({100 * i // total}%)",
            )
        converter.process_sample_into_h5((global_idx, (sublock_idx, sample)))

    _write_progress(str(h5py_dir), "h5", args.h5_shard_id, "done")
    logger.info(f"h5-worker {args.h5_shard_id} complete")


# ---------------------------------------------------------------------------
# Launch commands
# ---------------------------------------------------------------------------


def cmd_launch_rslearn(args: argparse.Namespace) -> None:
    """Launch rslearn ingest across Beaker shards."""
    from olmoearth_pretrain.dataset_creation.distributed import (
        launch_beaker_jobs,
        load_corpus,
    )

    entries = load_corpus(args.corpus)
    if args.max_samples:
        entries = entries[: args.max_samples]
    logger.info(f"Corpus: {len(entries)} samples, {args.num_shards} shards")

    cmd_template = [
        "scripts/data/corpus_pipeline.py",
        "rslearn-worker",
        "--corpus",
        args.corpus,
        "--rslearn-dir",
        args.rslearn_dir,
        "--rslearn-config",
        args.rslearn_config,
        "--shard-id",
        "{shard_id}",
        "--num-shards",
        str(args.num_shards),
        "--workers",
        str(args.workers),
    ]
    for layer in args.disabled_layers:
        cmd_template.extend(["--disabled-layers", layer])
    if args.max_samples:
        cmd_template.extend(["--max-samples", str(args.max_samples)])
    launch_beaker_jobs(
        step_name="rslearn",
        worker_cmd_template=cmd_template,
        num_shards=args.num_shards,
        cluster=args.cluster,
    )


def cmd_launch_convert(args: argparse.Namespace) -> None:
    """Launch convert across Beaker shards."""
    from olmoearth_pretrain.dataset_creation.distributed import launch_beaker_jobs

    cmd_template = [
        "scripts/data/corpus_pipeline.py",
        "convert-worker",
        "--corpus",
        args.corpus,
        "--rslearn-dir",
        args.rslearn_dir,
        "--olmoearth-dir",
        args.olmoearth_dir,
        "--shard-id",
        "{shard_id}",
        "--num-shards",
        str(args.num_shards),
        "--workers",
        str(args.workers),
    ]
    for layer in args.disabled_layers:
        cmd_template.extend(["--disabled-layers", layer])
    if args.max_samples:
        cmd_template.extend(["--max-samples", str(args.max_samples)])
    launch_beaker_jobs(
        step_name="convert",
        worker_cmd_template=cmd_template,
        num_shards=args.num_shards,
        cluster=args.cluster,
    )


def cmd_launch_h5(args: argparse.Namespace) -> None:
    """Launch H5 writing across Beaker shards."""
    from olmoearth_pretrain.dataset_creation.distributed import launch_beaker_jobs

    cmd_template = [
        "scripts/data/corpus_pipeline.py",
        "h5-worker",
        "--h5py-dir",
        args.h5py_dir,
        "--h5-shard-id",
        "{shard_id}",
        "--num-h5-shards",
        str(args.num_h5_shards),
    ]
    launch_beaker_jobs(
        step_name="h5",
        worker_cmd_template=cmd_template,
        num_shards=args.num_h5_shards,
        cluster=args.cluster,
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _print_shard_progress(progress: list[dict], step_name: str) -> None:
    """Summarize shard progress files for a step."""
    if not progress:
        print(f"  {step_name}: no shard progress yet")
        return
    by_status: dict[str, list[dict]] = {}
    for p in progress:
        by_status.setdefault(p["status"], []).append(p)
    parts = []
    for s in ["done", "running", "failed"]:
        if s in by_status:
            count = len(by_status[s])
            parts.append(f"{count} {s}")
    print(f"  {step_name} shards: {', '.join(parts)} (of {len(progress)} reporting)")
    # Show running details
    for p in by_status.get("running", []):
        print(
            f"    shard {p['shard_id']}: {p.get('detail', '')} ({p.get('timestamp', '')})"
        )
    for p in by_status.get("failed", []):
        print(
            f"    shard {p['shard_id']}: FAILED {p.get('detail', '')} ({p.get('timestamp', '')})"
        )


def cmd_status(args: argparse.Namespace) -> None:
    """Report progress across all pipeline steps."""
    rslearn_dir = UPath(args.rslearn_dir) if args.rslearn_dir else None
    olmoearth_dir = UPath(args.olmoearth_dir) if args.olmoearth_dir else None

    if args.corpus:
        from olmoearth_pretrain.dataset_creation.distributed import load_corpus

        entries = load_corpus(args.corpus)
        print(f"Corpus: {len(entries)} samples")
    else:
        print("Corpus: not specified")

    # rslearn progress
    if rslearn_dir:
        windows_dir = rslearn_dir / "windows" / "res_10.0"
        if windows_dir.exists():
            window_dirs = [d for d in windows_dir.iterdir() if d.is_dir()]
            completed = 0
            for wd in window_dirs:
                layers_dir = wd / "layers"
                if not layers_dir.exists():
                    continue
                s2_dirs = [
                    d
                    for d in layers_dir.iterdir()
                    if d.name.startswith("sentinel2_l2a")
                ]
                if len(s2_dirs) >= 12:
                    completed += 1
            print(f"rslearn: {completed}/{len(window_dirs)} windows fully materialized")
        else:
            print("rslearn: no windows directory found")
        _print_shard_progress(
            _read_all_progress(args.rslearn_dir, "rslearn"), "rslearn"
        )

    # olmoearth progress
    if olmoearth_dir:
        s2_csv = olmoearth_dir / "10_sentinel2_l2a_monthly.csv"
        if s2_csv.exists():
            import csv

            with s2_csv.open() as f:
                n_rows = sum(1 for _ in csv.reader(f)) - 1
            print(f"convert: {n_rows} windows converted (from sentinel2 CSV)")
        else:
            print("convert: no sentinel2 CSV found yet")
        _print_shard_progress(
            _read_all_progress(args.olmoearth_dir, "convert"), "convert"
        )

        # H5 progress
        h5_dirs = list(olmoearth_dir.glob("h5py_data_*/*/*"))
        for hd in h5_dirs:
            h5_files = list(hd.glob("sample_*.h5"))
            manifest = hd / "sample_manifest.json"
            if manifest.exists():
                with manifest.open() as f:
                    m = json.load(f)
                total = m["num_samples"]
                print(f"h5: {len(h5_files)}/{total} H5 files written in {hd}")
                _print_shard_progress(_read_all_progress(str(hd), "h5"), "h5")
            else:
                print(f"h5: {len(h5_files)} H5 files in {hd} (no manifest)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the distributed corpus pipeline."""
    multiprocessing.set_start_method("forkserver", force=True)

    parser = argparse.ArgumentParser(
        description="Distributed corpus pipeline: rslearn -> convert -> H5"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- launch-rslearn --
    p = subparsers.add_parser("launch-rslearn", help="Launch rslearn ingest on Beaker")
    p.add_argument("--corpus", required=True, help="Corpus CSV or JSON path")
    p.add_argument("--rslearn-dir", required=True, help="Shared rslearn dataset dir")
    p.add_argument("--rslearn-config", required=True, help="rslearn config.json path")
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cluster", default="ai2/jupiter")
    p.add_argument("--disabled-layers", nargs="*", default=[], help="rslearn layers to skip")
    p.add_argument("--max-samples", type=int, default=None, help="Limit corpus to first N samples")
    p.set_defaults(func=cmd_launch_rslearn)

    # -- rslearn-worker --
    p = subparsers.add_parser("rslearn-worker", help="Run rslearn for one shard")
    p.add_argument("--corpus", required=True)
    p.add_argument("--rslearn-dir", required=True)
    p.add_argument("--rslearn-config", default=None)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--disabled-layers", nargs="*", default=[])
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_rslearn_worker)

    # -- launch-convert --
    p = subparsers.add_parser("launch-convert", help="Launch convert on Beaker")
    p.add_argument("--corpus", required=True)
    p.add_argument("--rslearn-dir", required=True)
    p.add_argument("--olmoearth-dir", required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--cluster", default="ai2/jupiter")
    p.add_argument("--disabled-layers", nargs="*", default=[])
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_launch_convert)

    # -- convert-worker --
    p = subparsers.add_parser("convert-worker", help="Run convert for one shard")
    p.add_argument("--corpus", required=True)
    p.add_argument("--rslearn-dir", required=True)
    p.add_argument("--olmoearth-dir", required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--disabled-layers", nargs="*", default=[])
    p.add_argument("--max-samples", type=int, default=None)
    p.set_defaults(func=cmd_convert_worker)

    # -- prepare-h5 --
    p = subparsers.add_parser("prepare-h5", help="Prepare H5 metadata (single machine)")
    p.add_argument("--olmoearth-dir", required=True)
    p.set_defaults(func=cmd_prepare_h5)

    # -- launch-h5 --
    p = subparsers.add_parser("launch-h5", help="Launch H5 writing on Beaker")
    p.add_argument("--h5py-dir", required=True, help="H5 output dir from prepare-h5")
    p.add_argument("--num-h5-shards", type=int, required=True)
    p.add_argument("--cluster", default="ai2/jupiter")
    p.set_defaults(func=cmd_launch_h5)

    # -- h5-worker --
    p = subparsers.add_parser("h5-worker", help="Write H5 files for one shard")
    p.add_argument("--h5py-dir", required=True, help="H5 output dir with manifest")
    p.add_argument("--h5-shard-id", type=int, required=True)
    p.add_argument("--num-h5-shards", type=int, required=True)
    p.set_defaults(func=cmd_h5_worker)

    # -- status --
    p = subparsers.add_parser("status", help="Report pipeline progress")
    p.add_argument("--corpus", default=None)
    p.add_argument("--rslearn-dir", default=None)
    p.add_argument("--olmoearth-dir", default=None)
    p.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
