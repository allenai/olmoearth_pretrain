"""Smoke check of the AllCaptures arms' eval tasks on real eval data (not a training run).

For every task of ``allcap_narrow.allcap_eval_tasks``: build the val dataset exactly as
the evaluator does, read ``N`` samples (timing each), collate them, and run the
random-init config-D encoder at the task's patch size on GPU. Prints one JSON line per
task: tokens/timesteps per modality, valid fraction, seconds per sample, forward
output shape, or the error. Usage: ``python smoke_allcap_evals.py [N]``.
"""

import json
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import allcap_narrow as A  # noqa: E402
import allcap_slot_n4d192 as D  # noqa: E402

from olmoearth_pretrain.datatypes import MaskValue  # noqa: E402
from olmoearth_pretrain.evals.datasets import get_eval_dataset  # noqa: E402
from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG  # noqa: E402
from olmoearth_pretrain.evals.datasets.utils import (  # noqa: E402
    eval_collate_fn_variable_time,
)
from olmoearth_pretrain.internal.experiment import SubCmd  # noqa: E402


def main() -> None:
    """Run the smoke check."""
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    common = A.build_common_components(
        D.MODULE_PATH, SubCmd.dry_run, "smoke", "ai2/jupiter", []
    )
    tasks = D.build_trainer_config(common).callbacks["downstream_evaluator"].tasks
    model = D.build_model_config(common).build()
    encoder = model.encoder.cuda().eval()
    for name, task in tasks.items():
        row: dict = {"task": name}
        try:
            kwargs: dict = {}
            if task.dataset not in DATASET_TO_CONFIG:
                for key in ("window_size",):
                    if getattr(task, key) is not None:
                        kwargs[key] = getattr(task, key)
                for key in (
                    "label_at_center_pixel",
                    "tile_samples",
                    "every_capture",
                    "twin_landsat_l2",
                ):
                    if getattr(task, key):
                        kwargs[key] = True
            t0 = time.perf_counter()
            ds = get_eval_dataset(
                eval_dataset=task.dataset,
                split="valid",
                norm_stats_from_pretrained=task.norm_stats_from_pretrained,
                input_modalities=task.input_modalities,
                norm_method=task.norm_method,
                **kwargs,
            )
            row["build_s"] = round(time.perf_counter() - t0, 1)
            row["len"] = len(ds)
            items = []
            t0 = time.perf_counter()
            for i in range(min(n, len(ds))):
                items.append(ds[i])
            row["s_per_sample"] = round(
                (time.perf_counter() - t0) / max(1, len(items)), 3
            )
            sample, labels = eval_collate_fn_variable_time(items)
            for m in sample.modalities:
                x = getattr(sample, m)
                mask = getattr(sample, f"{m}_mask")
                if x is None or mask is None:
                    continue
                valid = (mask != MaskValue.MISSING.value).float().mean().item()
                row[m] = {
                    "shape": list(x.shape),
                    "valid_frac": round(valid, 3),
                    "finite": bool(torch.isfinite(x).all()),
                }
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = encoder(
                    sample.to_device(torch.device("cuda")),
                    patch_size=task.patch_size,
                    input_res=10,
                )
            row["student"] = list(out["student_registers"].shape)
            row["labels"] = list(labels.shape)
            row["ok"] = True
        except Exception as e:  # report and continue with the next task
            row["error"] = repr(e)[:400]
            row["tb"] = traceback.format_exc()[-1200:]
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
