"""Copy files from one location to another in parallel."""

import multiprocessing
import random
import shutil
import sys

import tqdm
from upath import UPath


def copy_file(job: tuple[UPath, UPath]) -> None:
    """Perform the specified copy job."""
    src_fname, dst_fname = job
    if dst_fname.exists():
        return
    with src_fname.open("rb") as src:
        with dst_fname.open("wb") as dst:
            shutil.copyfileobj(src, dst)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    src_dir = UPath(sys.argv[1])
    dst_dir = UPath(sys.argv[2])
    existing_fnames = set()
    for dst_fname in dst_dir.iterdir():
        existing_fnames.add(dst_fname.name)
    jobs = []
    for src_fname in src_dir.iterdir():
        if src_fname.is_dir():
            print(f"warning: skipping directory {src_fname}", flush=True)
            continue
        if src_fname.name in existing_fnames:
            continue
        dst_fname = dst_dir / src_fname.name
        jobs.append((src_fname, dst_fname))
    print(f"got {len(jobs)} files to copy", flush=True)
    random.shuffle(jobs)
    p = multiprocessing.Pool(128)
    outputs = p.imap_unordered(copy_file, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
