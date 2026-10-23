"""Build the per-window h5 cache of eval datasets' every-capture twins (CPU job).

One file per twin window (``every_capture.CACHE_ROOT``) replaces a few hundred
GeoTIFF reads per window at eval time. Existing files are skipped unless they lack
a modality listed in ``--refresh_missing`` (e.g. ``landsat_l2`` once its fetch is
done). Finishes with a parity check: for a few windows, cached reads must equal
GeoTIFF reads at the eval crop.

Usage: ``python build_every_capture_cache.py pastis_year_aligned
africa_crop_mask_year_aligned --workers 64 [--limit N] [--refresh_missing landsat_l2]``.
"""

import argparse
import json
import time
from multiprocessing import get_context

import h5py
import numpy as np
from upath import UPath

from olmoearth_pretrain.evals.datasets import every_capture as ec
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)
from olmoearth_pretrain.evals.studio_ingest.registry import get_dataset_entry

_STATE: dict = {}


def _needs_build(path: UPath, refresh_missing: list[str]) -> bool:
    if not path.exists():
        return True
    if not refresh_missing:
        return False
    with path.open("rb") as f, h5py.File(f, "r") as h5:
        return any(m not in h5 for m in refresh_missing)


def _build(i: int) -> tuple[str, float, str | None]:
    window = _STATE["windows"][i]
    start = time.perf_counter()
    try:
        path = ec.cache_path(_STATE["source"], window.name)
        if _needs_build(path, _STATE["refresh_missing"]):
            ec.write_cache(window, _STATE["source"])
        return window.name, time.perf_counter() - start, None
    except Exception as e:  # noqa: BLE001 - reported, the job keeps going
        return window.name, time.perf_counter() - start, repr(e)


def _parity(source: ec.EveryCaptureSource, names: list[str]) -> None:
    """Cached reads equal GeoTIFF reads (a 16 x 16 crop and the full window).

    Run right after building, so a window's cached Landsat is current.
    """
    reader = ec.EveryCaptureReader(source)
    root = ec.CACHE_ROOT
    for name in names:
        for crop in ((slice(0, 16), slice(0, 16)), (None, None)):
            cached = ec.read_every_capture(reader, name, *crop)
            cached_l2 = ec.read_landsat_l2(reader, name, *crop)
            ec.CACHE_ROOT = "/nonexistent"  # force the GeoTIFF path
            direct = ec.read_every_capture(reader, name, *crop)
            direct_l2 = ec.read_landsat_l2(reader, name, *crop)
            ec.CACHE_ROOT = root
            if direct_l2 is not None:
                assert cached_l2 is not None, (name, "landsat_l2")
                cached["landsat_l2"], direct["landsat_l2"] = cached_l2, direct_l2
            for modality in direct:
                a, b = cached[modality], direct[modality]
                assert a[2] == b[2], (name, modality, "dates")
                np.testing.assert_array_equal(a[0], b[0])
                np.testing.assert_array_equal(a[1], b[1])
        print(json.dumps({"parity_ok": name}), flush=True)


def main() -> None:
    """Build the cache (see the module docstring)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh_missing", nargs="*", default=[])
    args = parser.parse_args()
    for dataset in args.datasets:
        entry = get_dataset_entry(dataset)
        source = ec.every_capture_source(entry.name, entry.source_path)
        windows = RslearnWindowProvider(
            UPath(source.root), groups=[source.group]
        ).load_windows()
        windows = sorted(windows, key=lambda w: w.name)[: args.limit]
        _STATE.update(
            windows=windows, source=source, refresh_missing=args.refresh_missing
        )
        start, done, failed = time.perf_counter(), 0, []
        with get_context("fork").Pool(args.workers) as pool:
            for name, seconds, error in pool.imap_unordered(
                _build, range(len(windows))
            ):
                done += 1
                if error is not None:
                    failed.append((name, error))
                if done % 100 == 0 or done == len(windows):
                    print(
                        json.dumps(
                            {
                                "dataset": dataset,
                                "done": done,
                                "of": len(windows),
                                "failed": len(failed),
                                "elapsed_s": round(time.perf_counter() - start),
                                "last_s": round(seconds, 1),
                            }
                        ),
                        flush=True,
                    )
        for name, error in failed[:20]:
            print(json.dumps({"failed": name, "error": error[:300]}), flush=True)
        _parity(source, [w.name for w in windows[:3]])


if __name__ == "__main__":
    main()
