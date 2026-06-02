"""Smoke-test the 50Cities eval dataset wiring end-to-end.

Run on a machine with FIFTY_CITIES_DIR mounted (e.g. the weka cluster):

    python scripts/tools/20260602_test_fifty_cities.py

It checks, for every registered fifty_cities EVAL_TASK:
  * the dataset builds for train/valid/test via the real get_eval_dataset entry,
  * a sample has the expected single-timestep shapes and label range,
  * a batch collates through the eval collate fn used by the eval harness.
"""

import logging

import torch
from torch.utils.data import DataLoader

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets import get_eval_dataset, paths
from olmoearth_pretrain.evals.datasets.configs import dataset_to_config
from olmoearth_pretrain.evals.datasets.utils import eval_collate_fn
from olmoearth_pretrain.internal.all_evals import EVAL_TASKS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fifty_cities_test")


def main() -> None:
    """Build, sample, and collate every fifty_cities eval task."""
    print(f"FIFTY_CITIES_DIR = {paths.FIFTY_CITIES_DIR}")
    tasks = {k: v for k, v in EVAL_TASKS.items() if k.startswith("fifty_cities")}
    assert tasks, "No fifty_cities tasks registered in EVAL_TASKS"

    for name, task in tasks.items():
        cfg = dataset_to_config(task.dataset)
        print(
            f"\n=== {name} (dataset={task.dataset}, mods={task.input_modalities}) ==="
        )
        print(
            f"  config: num_classes={cfg.num_classes} hw={cfg.height_width} "
            f"timeseries={cfg.timeseries}"
        )

        for split in ["train", "valid", "test"]:
            ds = get_eval_dataset(
                task.dataset,
                split=split,
                norm_stats_from_pretrained=task.norm_stats_from_pretrained,
                input_modalities=task.input_modalities,
            )
            n = len(ds)
            if n == 0:
                print(f"  {split:5s}: EMPTY split (no tiles assigned)")
                continue
            sample, label = ds[0]

            # Per-modality shape: (H, W, T=1, C)
            shapes = {}
            for m in task.input_modalities:
                arr = getattr(sample, m)
                shapes[m] = tuple(arr.shape)
                assert arr.shape[0] == cfg.height_width, f"{name}/{split} bad H"
                assert arr.shape[1] == cfg.height_width, f"{name}/{split} bad W"
                assert arr.shape[2] == 1, f"{name}/{split} expected single timestep"
                assert torch.isfinite(arr).all(), f"{name}/{split} non-finite in {m}"

            assert label.shape == (cfg.height_width, cfg.height_width), (
                f"{name}/{split} bad label shape {tuple(label.shape)}"
            )
            valid = label[label >= 0]
            if valid.numel():
                lo, hi = int(valid.min()), int(valid.max())
                assert lo >= 0 and hi < cfg.num_classes, (
                    f"{name}/{split} label range [{lo},{hi}] outside "
                    f"[0,{cfg.num_classes - 1}]"
                )
            else:
                lo, hi = -1, -1  # this particular tile is all-ignore

            # Exercise the eval harness collate over a small batch.
            loader = DataLoader(ds, batch_size=min(4, n), collate_fn=eval_collate_fn)
            bsample, btarget = next(iter(loader))
            bshapes = {
                m: tuple(getattr(bsample, m).shape) for m in task.input_modalities
            }

            print(
                f"  {split:5s}: n={n:6d}  sample={shapes}  "
                f"label[min,max]=[{lo},{hi}]  batch_target={tuple(btarget.shape)} "
                f"batch_sample={bshapes}"
            )

    # Exercise the dataset-stats (non-pretrained) normalization path if the
    # stats file has been computed.
    if (paths.FIFTY_CITIES_DIR / "norm_stats.json").exists():
        ds = get_eval_dataset(
            "fifty_cities",
            split="train",
            norm_stats_from_pretrained=False,
            input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        )
        sample, _ = ds[0]
        s2 = sample.sentinel2_l2a
        print(
            f"\nnorm_stats_from_pretrained=False OK: S2 {tuple(s2.shape)} "
            f"range=[{float(s2.min()):.3f},{float(s2.max()):.3f}]"
        )
    else:
        print(
            "\n(norm_stats.json not found -- skipping non-pretrained norm check; "
            "run 20260602_fifty_cities_norm_stats.py to enable it.)"
        )

    print("\nAll fifty_cities tasks loaded, sampled, and collated OK.")


if __name__ == "__main__":
    main()
