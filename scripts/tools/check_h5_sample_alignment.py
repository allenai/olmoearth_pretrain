r"""Check whether two h5 builds address the SAME sample under the same index.

WHY THIS EXISTS. The OmniCloudMask cloud sidecars are keyed by the raw h5 sample id
(``dataset.__getitem__`` does ``index = sample_indices[args.idx]``, then reads
``sample_{index}.h5`` and ``sample_{index}.npz``). So a cache computed for one h5 build
is reusable by another build ONLY IF index i means the same sample in both. If it does
not, reuse is worse than no cloud masking at all: every sample gets a different
sample's cloud map, silently corrupting which decode targets are dropped, with nothing
in the logs to show for it.

The complete cache on weka was computed for ``cloud_mask_cache.DEFAULT_H5_DIR`` (the
old ``cdl_gse_..._wri_canopy_height_map`` map set). The new-maps builds
(``cdl_glo30_..._meta_canopy_height_...``, DN and Landsat-reflectance) have no cache of
their own. Both have 1,138,828 samples, which is suggestive but not proof.

WHAT IT COMPARES, cheapest and most decisive first:

* ``latlon`` -- exact equality. Two builds over the same sampling should agree bit for
  bit; this alone catches a reordering.
* ``timestamps`` -- exact equality. Catches a build that kept locations but re-drew the
  temporal window.
* ``sentinel2_l2a`` -- sha1 of the raw bytes. The STRONGEST available test: S2 is
  untouched by both the map swap and the Landsat DN->reflectance conversion, so if the
  index means the same sample the array must be byte-identical. A latlon match with an
  S2 mismatch means same location, different data -- reuse would still be wrong.

Landsat is deliberately NOT compared: it legitimately differs between a DN build and a
reflectance build, so a mismatch there proves nothing either way.

USAGE (needs the weka mount, so run it on a weka-mounted host or as a 0-GPU job):

    python scripts/tools/check_h5_sample_alignment.py \\
        --h5_dir_a /weka/.../osm_sampling/h5py_data_.../cdl_gse_.../1138828 \\
        --h5_dir_b /weka/.../osm_sampling_landsat_refl/h5py_data_.../cdl_glo30_.../1138828 \\
        --n 300

Exit code is 0 only when every probed index matches on all three, i.e. only when
reusing A's cloud cache for B is safe.
"""

import argparse
import hashlib
import logging
import sys

import h5py
import hdf5plugin  # noqa: F401  (registers the zstd filter these h5 files use)
import numpy as np
from upath import UPath

logger = logging.getLogger(__name__)

# Total samples in the OSM-sampling pretrain set; the index space both builds share.
DEFAULT_NUM_SAMPLES = 1138828
SAMPLE_FILE_PATTERN = "sample_{index}.h5"
# Compared for exact equality. Small, and enough on their own to catch a reordering.
EXACT_KEYS = ("latlon", "timestamps")
# Hashed rather than compared elementwise: full S2 stacks are large.
HASHED_KEY = "sentinel2_l2a"


def _read_probe(h5_dir: UPath, index: int) -> dict[str, object] | None:
    """Read the comparison keys for one sample, or None if the file is absent."""
    path = h5_dir / SAMPLE_FILE_PATTERN.format(index=index)
    if not path.exists():
        return None
    out: dict[str, object] = {}
    with path.open("rb") as f:
        with h5py.File(f, "r") as h5file:
            for key in EXACT_KEYS:
                out[key] = h5file[key][()] if key in h5file else None
            if HASHED_KEY in h5file:
                raw = np.ascontiguousarray(h5file[HASHED_KEY][()])
                out[HASHED_KEY] = (
                    raw.shape,
                    hashlib.sha1(raw.tobytes(), usedforsecurity=False).hexdigest(),
                )
            else:
                out[HASHED_KEY] = None
    return out


def _compare(a: dict[str, object], b: dict[str, object]) -> list[str]:
    """Return the names of the keys that disagree between two probes."""
    bad = []
    for key in EXACT_KEYS:
        av, bv = a[key], b[key]
        if av is None or bv is None:
            if av is not bv:
                bad.append(f"{key} (present in only one build)")
            continue
        if not np.array_equal(np.asarray(av), np.asarray(bv)):
            bad.append(key)
    if a[HASHED_KEY] != b[HASHED_KEY]:
        bad.append(HASHED_KEY)
    return bad


def check(
    h5_dir_a: str, h5_dir_b: str, n: int, num_samples: int, seed: int
) -> tuple[int, int, int]:
    """Probe n random indices in both builds. Returns (matched, mismatched, missing)."""
    dir_a, dir_b = UPath(h5_dir_a), UPath(h5_dir_b)
    rng = np.random.default_rng(seed)
    indices = rng.choice(num_samples, size=min(n, num_samples), replace=False)

    matched = mismatched = missing = 0
    for i, index in enumerate(sorted(int(x) for x in indices), start=1):
        probe_a = _read_probe(dir_a, index)
        probe_b = _read_probe(dir_b, index)
        if probe_a is None or probe_b is None:
            missing += 1
            which = "A" if probe_a is None else "B"
            logger.warning("index %d: absent from build %s", index, which)
            continue
        bad = _compare(probe_a, probe_b)
        if bad:
            mismatched += 1
            if mismatched <= 10:
                logger.error("index %d: DISAGREES on %s", index, ", ".join(bad))
        else:
            matched += 1
        if i % 50 == 0:
            logger.info(
                "%d/%d probed (%d matched, %d mismatched, %d missing)",
                i,
                len(indices),
                matched,
                mismatched,
                missing,
            )
    return matched, mismatched, missing


def main() -> int:
    """Parse args, run the check, and print a verdict."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5_dir_a", required=True, help="build whose cloud cache exists")
    ap.add_argument("--h5_dir_b", required=True, help="build that wants to reuse it")
    ap.add_argument("--n", type=int, default=300, help="indices to probe")
    ap.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    matched, mismatched, missing = check(
        args.h5_dir_a, args.h5_dir_b, args.n, args.num_samples, args.seed
    )
    total = matched + mismatched + missing
    print(
        f"\nprobed {total}: {matched} matched, {mismatched} mismatched, {missing} missing"
    )
    if mismatched or missing:
        print(
            "VERDICT: DO NOT reuse A's cloud cache for B. The index does not address "
            "the same sample in both builds, so the sidecars would be misapplied "
            "sample by sample, silently. Precompute the cache for B instead."
        )
        return 1
    print(
        "VERDICT: safe to reuse A's cloud cache for B on this evidence "
        f"({matched} indices agree on latlon, timestamps and the S2 bytes). "
        "Note the sidecars' landsat_cloud was still computed on A's Landsat "
        "radiometry; s2_cloud is unaffected."
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sys.exit(main())
