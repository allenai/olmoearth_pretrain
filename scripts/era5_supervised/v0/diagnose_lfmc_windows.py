"""Diagnose why lfmc_woody_eval resolves to 0 windows.

Run on the cluster (where Weka is mounted):

    cd /weka/dfive-default/hadriens/olmoearth_pretrain
    source ./.venv/bin/activate
    python3 scripts/era5_supervised/v0/diagnose_lfmc_windows.py

It loads the raw rslearn dataset (no index, no caching) and prints:
  1. The distribution of window *groups*.
  2. Cross-tabs of the ``split`` tag and the ``oep_eval`` tag.
  3. A step-by-step replay of the exact eval filter
     (groups=[spatial_split] -> tags={oep_eval:"", split:"train"})
     so we can see which step zeroes it out.
"""

from __future__ import annotations

from collections import Counter

from rslearn.dataset import Dataset
from upath import UPath

DATASET_PATH = (
    "/weka/dfive-default/rslearn-eai/datasets/lfmc/20251023/woody/scratch/dataset"
)
GROUP = "spatial_split"
SPLIT_TAG_KEY = "split"
EVAL_TAG_KEY = "oep_eval"


def _matches(options: dict, tags: dict) -> bool:
    """Replicate rslearn's tag matcher (empty value = key-existence check)."""
    for k, v in tags.items():
        if k not in options or (v and options[k] != v):
            return False
    return True


def main() -> None:
    """Load the lfmc dataset and print group/tag diagnostics."""
    ds = Dataset(UPath(DATASET_PATH))

    # 1. ALL windows, no filter.
    all_windows = ds.load_windows()
    print(f"Total windows in dataset (no filter): {len(all_windows)}")

    group_counts = Counter(w.group for w in all_windows)
    print("\nWindows per group:")
    for g, n in sorted(group_counts.items()):
        print(f"  {g!r}: {n}")

    # 2. Tag key presence + value distributions (across all windows).
    has_split = sum(1 for w in all_windows if SPLIT_TAG_KEY in w.options)
    has_eval = sum(1 for w in all_windows if EVAL_TAG_KEY in w.options)
    print(f"\nWindows with a {SPLIT_TAG_KEY!r} tag: {has_split}")
    print(f"Windows with an {EVAL_TAG_KEY!r} tag: {has_eval}")

    split_vals = Counter(w.options.get(SPLIT_TAG_KEY, "<missing>") for w in all_windows)
    print(f"\n{SPLIT_TAG_KEY!r} tag values:")
    for v, n in sorted(split_vals.items(), key=lambda kv: str(kv[0])):
        print(f"  {v!r}: {n}")

    eval_vals = Counter(
        repr(w.options.get(EVAL_TAG_KEY, "<missing>")) for w in all_windows
    )
    print(f"\n{EVAL_TAG_KEY!r} tag values:")
    for v, n in sorted(eval_vals.items()):
        print(f"  {v}: {n}")

    # 3. Cross-tab: oep_eval present x split value.
    print(f"\nCross-tab ({EVAL_TAG_KEY} present) x {SPLIT_TAG_KEY}:")
    cross: Counter = Counter()
    for w in all_windows:
        cross[
            (EVAL_TAG_KEY in w.options, w.options.get(SPLIT_TAG_KEY, "<missing>"))
        ] += 1
    for (has, split), n in sorted(cross.items(), key=lambda kv: str(kv[0])):
        print(f"  oep_eval={has}, split={split!r}: {n}")

    # 4. Replay the exact eval filter for the *train* split.
    print("\n--- Replaying eval filter: train split ---")
    grouped = ds.load_windows(groups=[GROUP])
    print(f"load_windows(groups=[{GROUP!r}]) -> {len(grouped)} windows")
    if not grouped:
        print(
            f"  !! Group {GROUP!r} matched 0 windows. Real groups are listed "
            "above — the registry/model.yaml 'groups' is likely wrong."
        )

    train_tags = {EVAL_TAG_KEY: "", SPLIT_TAG_KEY: "train"}
    after_tags = [w for w in grouped if _matches(w.options, train_tags)]
    print(f"after tags={train_tags} -> {len(after_tags)} windows")

    # Sample a few raw option dicts so we can eyeball exact keys/values.
    print("\nSample window.options (first 5 windows in dataset):")
    for w in all_windows[:5]:
        print(f"  group={w.group!r} name={w.name!r} options={w.options}")


if __name__ == "__main__":
    main()
