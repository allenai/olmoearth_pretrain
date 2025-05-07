"""Read data in random order to help move it from WEKA to SSD."""

import multiprocessing
import random
import sys

import tqdm
from upath import UPath

NUM_WORKERS = 128


def read_file(fname: UPath) -> int:
    """Perform the specified copy job."""
    with fname.open("rb") as f:
        data = f.read()
    return len(data)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    target_dir = UPath(sys.argv[1])
    fnames = [fname for fname in target_dir.iterdir() if fname.is_file()]
    print(f"got {len(fnames)} files to read", flush=True)
    random.shuffle(fnames)
    p = multiprocessing.Pool(NUM_WORKERS)
    outputs = p.imap_unordered(read_file, fnames)
    for _ in tqdm.tqdm(outputs, total=len(fnames)):
        pass
    p.close()
