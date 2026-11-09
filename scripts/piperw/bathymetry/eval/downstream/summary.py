"""Paired effect of each depth-trained model vs its control, per task, for the results page -> summary.json.

Controls: v1.3 arms vs v13_sdbprobe, v1.2 sdb vs v1.2 base.
- BlueTopo: held-out-cell probe (bluetopo_eval/results_holdout.json), dR² with cell-bootstrap CI.
- Vessel: 5 probe seeds per model (vessel_seeds/results.json), difference of means, ±1.96 SE.
- Seagrass: macro F1, paired by fold and seed, bootstrap CI (v1.3: seagrass/results.csv; v1.2: round 1 fixdata).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/downstream"
)
ROUND1 = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-26-piperw-bathymetry-target/seagrass/results_fixdata.csv"
)
PAIRS = [
    ("v13_sdbsup", "v13_sdbprobe"),
    ("v13_btsup", "v13_sdbprobe"),
    ("v13_sdbbtsup", "v13_sdbprobe"),
    ("v12_sdb", "v12_base"),
]
VESSEL_NAME = {
    "v12_base": "base",
    "v12_sdb": "sdb",
    "v13_sdbprobe": "sdbprobe",
    "v13_sdbsup": "sdbsup",
    "v13_btsup": "btsup",
    "v13_sdbbtsup": "sdbbtsup",
}
F1 = "test_F1_f1/avg"


def bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> list[float]:
    """95% bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def seagrass_rows() -> dict:
    """Seagrass macro F1 keyed by (model, fold, method, seed)."""
    by = {}
    for path, prefix in ((HERE / "seagrass" / "results.csv", "v13_"), (ROUND1, "v12_")):
        for r in csv.DictReader(open(path)):
            if r[F1]:
                by[(prefix + r["arm"], r["fold"], r["method"], int(r["seed"]))] = float(
                    r[F1]
                )
    return by


def main() -> None:
    """Write summary.json with per-task effects and absolute scores."""
    bt = json.loads(
        (HERE.parent / "bluetopo_eval" / "results_holdout.json").read_text()
    )
    vessel = json.loads((HERE / "vessel_seeds" / "results.json").read_text())
    sg = seagrass_rows()

    tasks = [
        {"key": "bluetopo", "label": "Depth on held-out cells", "metric": "R²"},
        {"key": "vessel_length", "label": "Vessel length", "metric": "R²"},
        {"key": "vessel_type", "label": "Vessel type", "metric": "accuracy"},
        {
            "key": "seagrass_linprobe",
            "label": "Seagrass, linear probe",
            "metric": "macro F1",
        },
        {
            "key": "seagrass_finetune",
            "label": "Seagrass, fine-tune",
            "metric": "macro F1",
        },
    ]
    effects = defaultdict(dict)
    absolute = defaultdict(dict)
    for b, a in PAIRS:
        p = bt["paired"][f"{b} - {a} @200000"]
        effects["bluetopo"][b] = {
            "d": p["r2_diff"],
            "ci": p["r2_diff_ci95"],
            "n": f"{p['n_cells']} cells",
        }
        for key, metric in (("vessel_length", "r2"), ("vessel_type", "acc")):
            x, y = (
                np.array(vessel[VESSEL_NAME[a]][metric]),
                np.array(vessel[VESSEL_NAME[b]][metric]),
            )
            d = y.mean() - x.mean()
            h = 1.96 * np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
            effects[key][b] = {
                "d": float(d),
                "ci": [float(d - h), float(d + h)],
                "n": f"{len(x)} probe seeds",
            }
        for method in ("linprobe", "finetune"):
            d = np.array(
                [
                    sg[(b, f, m, s)] - v
                    for (arm, f, m, s), v in sg.items()
                    if arm == a and m == method and (b, f, m, s) in sg
                ]
            )
            effects[f"seagrass_{method}"][b] = {
                "d": float(d.mean()),
                "ci": bootstrap_ci(d),
                "n": f"{len(d)} fold×seed pairs",
            }

    for name in VESSEL_NAME:
        absolute["bluetopo"][name] = bt["probe"][f"{name}@200000"]["r2"]
        absolute["vessel_length"][name] = [
            float(np.mean(vessel[VESSEL_NAME[name]]["r2"])),
            float(np.std(vessel[VESSEL_NAME[name]]["r2"], ddof=1)),
        ]
        absolute["vessel_type"][name] = [
            float(np.mean(vessel[VESSEL_NAME[name]]["acc"])),
            float(np.std(vessel[VESSEL_NAME[name]]["acc"], ddof=1)),
        ]
        for method in ("linprobe", "finetune"):
            v = [x for (arm, _, m, _), x in sg.items() if arm == name and m == method]
            if v:
                absolute[f"seagrass_{method}"][name] = [
                    float(np.mean(v)),
                    float(np.std(v, ddof=1)),
                ]

    out = {
        "tasks": tasks,
        "pairs": [list(p) for p in PAIRS],
        "effects": effects,
        "absolute": absolute,
    }
    (HERE / "summary.json").write_text(json.dumps(out, indent=1))
    for t in tasks:
        print(
            t["label"],
            {
                b: f"{e['d']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
                for b, e in effects[t["key"]].items()
            },
        )


if __name__ == "__main__":
    main()
