"""Parse polygon-eval metrics from the v1.3 seagrass Beaker logs and compare arms (paired by fold/method/seed).

Adapted from 2026-10-26-piperw-bathymetry-target/seagrass/collect_results.py.

  python collect_results.py   # writes results.csv, prints per-cell means and paired deltas
"""

import csv
import json
import re
import subprocess  # nosec
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/downstream/seagrass"
)
ARMS = ["base", "btsup"]
PAIRS = [("base", "btsup")]
METRICS = ["test_F1_f1/avg", "test_F1_f1/cls_0", "test_F1_f1/cls_1"]
ROW = re.compile(r"│\s*(test_\S+)\s*│\s*([-0-9.eE]+)\s*│")
NAME = re.compile(r"bathy_v13(\w+?)_(mallorca|menorca)_(linprobe|finetune)_s(\d)")


def beaker(*args: str) -> str:
    """Run a beaker CLI command and return stdout."""
    return subprocess.run(  # nosec
        ["beaker", *args], capture_output=True, text=True, check=True
    ).stdout


def collect() -> list[dict]:
    """Metrics and exit code for every experiment in experiments.txt."""
    rows = []
    for url in (HERE / "experiments.txt").read_text().split():
        exp = url.rsplit("/", 1)[-1]
        info = json.loads(beaker("experiment", "get", exp, "--format", "json"))
        info = info[0] if isinstance(info, list) else info
        arm, fold, method, seed = NAME.search(info["name"]).groups()
        metrics = dict(ROW.findall(beaker("experiment", "logs", exp)))
        row = {
            "arm": arm,
            "fold": fold,
            "method": method,
            "seed": int(seed),
            "experiment": exp,
            "exit_code": info["jobs"][-1]["status"].get("exitCode"),
        }
        row.update({m: float(metrics[m]) if m in metrics else None for m in METRICS})
        rows.append(row)
    return rows


def bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    """Write results.csv and print per-arm means and paired deltas."""
    rows = collect()
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(
            sorted(rows, key=lambda r: (r["method"], r["fold"], r["arm"], r["seed"]))
        )

    for r in rows:
        if r[METRICS[0]] is None:
            print(
                f"[missing] {r['arm']} {r['fold']} {r['method']} s{r['seed']} exit={r['exit_code']}"
            )

    by = {
        (r["arm"], r["fold"], r["method"], r["seed"]): r
        for r in rows
        if r[METRICS[0]] is not None
    }
    for metric in METRICS:
        print(f"\n== {metric}  (mean ± sd over seeds)")
        cells = defaultdict(list)
        for (arm, fold, method, _), r in by.items():
            cells[(method, fold, arm)].append(r[metric])
        for method in ("linprobe", "finetune"):
            for fold in ("mallorca", "menorca"):
                line = [f"{method:9s} {fold:9s}"]
                for arm in ARMS:
                    v = np.array(cells.get((method, fold, arm), []))
                    line.append(
                        f"{arm} {v.mean():.3f}±{v.std(ddof=1) if len(v) > 1 else 0:.3f}"
                        if len(v)
                        else f"{arm} -"
                    )
                print("  ".join(line))

        for a, b in PAIRS:
            print(f"-- paired delta {b} - {a}, bootstrap 95% CI")
            for label, methods in (
                ("linprobe", ["linprobe"]),
                ("finetune", ["finetune"]),
                ("all", ["linprobe", "finetune"]),
            ):
                d = np.array(
                    [
                        by[(b, f, m, s)][metric] - by[(a, f, m, s)][metric]
                        for (x, f, m, s) in by
                        if x == a and m in methods and (b, f, m, s) in by
                    ]
                )
                if len(d):
                    lo, hi = bootstrap_ci(d)
                    print(
                        f"{label:9s} n={len(d):2d}  mean {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {b} better in {(d > 0).sum()}/{len(d)}"
                    )


if __name__ == "__main__":
    main()
