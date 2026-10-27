"""Collect the snow eval read-out into one JSON (for the report and the board).

Reads, under the dataset root: ``meta/label_stats.json``, ``meta/cells_split.parquet``,
``meta/candidates.parquet``, ``meta/baselines_windows_eval_<family>.json``,
``embeddings*/target_lab.json``, ``probe_results*/*.jsonl`` (the evaluator's
format, from ``emb_probe.py`` or in-loop runs) and ``meta/stations_summary.json``,
and writes ``<out>``:

* ``dataset``: cells / windows per region and split, label SDs;
* ``baselines``: val / test R2 per family, target and baseline;
* ``target_lab``: per (table, family, target, pooling) floor, 1601, gain, arm
  range, between-arm F and the 1306 trajectory (``usdm/target_lab.summarize``);
* ``probes``: per probe-results dir and task, F at the plateau LR (median val
  over the step-50k arms) and at each run's best LR, on val and test, as in the
  USDM session's ``flux_fstats.py``, plus floor and 1601 at the plateau LR;
* ``curves``: per task, the 1306 / 1305 / 1301 val R2 trajectories at the
  plateau LR;
* ``stations``: the GHCN-D check.

Usage::

    python scripts/era5_supervised/v0/snow/snow_results.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

V0_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V0_DIR / "usdm"))
from target_lab import summarize  # noqa: E402

SEED_RE = re.compile(r"_seed\d+$")
CURVE_RUNS = (
    "era5enc_1306_halo75_nogate",
    "era5enc_1305_naive70_nogate",
    "era5enc_1301_halo22",
)


def f_stat(groups: list[np.ndarray]) -> tuple[float, float, int]:
    """Between-arm over within-arm variance.

    Args:
        groups: Per arm, the seeds' values.

    Returns:
        ``(F, pooled within-arm SD, arms with >= 2 seeds)``.
    """
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 3:
        return float("nan"), float("nan"), len(groups)
    n = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    grand = np.average(means, weights=n)
    between = np.sum(n * (means - grand) ** 2) / (len(groups) - 1)
    within = np.sum([((g - g.mean()) ** 2).sum() for g in groups]) / (
        n.sum() - len(groups)
    )
    return float(between / within), float(np.sqrt(within)), len(groups)


def load_probes(results_dir: Path) -> pd.DataFrame:
    """Every (run, step, task, lr) val / test R2 of a probe-results dir.

    Args:
        results_dir: Directory of evaluator JSONL files.

    Returns:
        ``run, step, task, lr, val, test``.
    """
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            for lr, res in (rec.get("lrsweep") or {}).items():
                val, test = res.get("val") or {}, res.get("test") or {}
                if "r2" in val:
                    rows.append(dict(run=rec["run"], step=rec["step"], task=rec["task"],
                                     lr=float(lr), val=val["r2"], test=test.get("r2")))  # fmt: skip
    frame = pd.DataFrame(rows)
    return (
        frame.drop_duplicates(["run", "step", "task", "lr"], keep="last")
        if len(frame)
        else frame
    )


def probe_stats(frame: pd.DataFrame) -> dict:
    """F statistics, floor and 1601 per task (USDM ``flux_fstats.py`` convention).

    Args:
        frame: Output of :func:`load_probes`.

    Returns:
        ``{task: stats}``.
    """
    out = {}
    is_init = frame["run"].str.startswith("init_")
    final = frame[~is_init & frame["step"].isin([50000, 150000])
                  & ~frame["run"].str.startswith(CURVE_RUNS)]  # fmt: skip
    final = final.assign(arm=final["run"].map(lambda r: SEED_RE.sub("", r)))
    for task, part in final.groupby("task"):
        plateau = float(part.groupby("lr")["val"].median().idxmax())
        fixed = part[np.isclose(part["lr"], plateau)]
        best = part.loc[part.groupby("run")["val"].idxmax()]
        res: dict = {"plateau_lr": plateau}
        for mode, sel in (("fixed", fixed), ("best", best)):
            for split in ("val", "test"):
                groups = [g[split].dropna().to_numpy() for _, g in sel.groupby("arm")]
                value, sd, k = f_stat(groups)
                dof = sum(len(g) for g in groups if len(g) > 1) - k
                res[f"{mode}_{split}"] = dict(
                    F=value, pooled_sd=sd, arms=k,
                    p=float(stats.f.sf(value, k - 1, dof)) if k > 2 else None,
                )  # fmt: skip
        init = frame[
            is_init & (frame["task"] == task) & np.isclose(frame["lr"], plateau)
        ]
        ref = fixed[fixed["arm"] == "era5enc_1601_doyc"]
        res.update(
            floor_val=float(init["val"].mean()), floor_test=float(init["test"].mean()),
            floor_sd=float(init["val"].std(ddof=1)) if len(init) > 1 else None,
            ref_val=float(ref["val"].mean()), ref_test=float(ref["test"].mean()),
            ref_sd=float(ref["val"].std(ddof=1)) if len(ref) > 1 else None,
            arm_means={a: float(g["val"].mean()) for a, g in fixed.groupby("arm")},
        )  # fmt: skip
        curves = {}
        for run in CURVE_RUNS:
            traj = frame[(frame["run"] == run) & (frame["task"] == task)
                         & np.isclose(frame["lr"], plateau)].sort_values("step")  # fmt: skip
            if len(traj):
                curves[run] = dict(
                    steps=traj["step"].tolist(), val=traj["val"].round(4).tolist()
                )
        res["curves"] = curves
        out[task] = res
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/weka/dfive-default/rslearn-eai/datasets/snow")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    meta = root / "meta"
    out: dict = {}
    if (meta / "candidates.parquet").exists():
        cand = pd.read_parquet(meta / "candidates.parquet",
                               columns=["family", "cell_id", "wy", "split", "region", "sdd_anom"])  # fmt: skip
        post = cand[cand["family"] == "post"]
        out["dataset"] = dict(
            cells=int(cand["cell_id"].nunique()),
            by_region=post.groupby("region").agg(cells=("cell_id", "nunique"),
                                                 windows=("wy", "size")).to_dict("index"),
            by_split=post.groupby("split").size().to_dict(),
            sdd_anom_sd=float(post["sdd_anom"].std()),
        )  # fmt: skip
    if (meta / "label_stats.json").exists():
        out["label_stats"] = json.loads((meta / "label_stats.json").read_text())
    out["baselines"] = {p.stem.replace("baselines_windows_", ""): json.loads(p.read_text())
                        for p in sorted(meta.glob("baselines_windows_*.json"))}  # fmt: skip
    out["target_lab"] = {}
    for path in sorted(root.glob("embeddings*/target_lab.json")):
        results = json.loads(path.read_text())
        for metric in ("val", "test"):
            frames = []
            for family, by_target in results.items():
                frame = summarize(by_target, metric)
                frame.insert(0, "family", family)
                frames.append(frame)
            if frames:
                out["target_lab"].setdefault(path.parent.name, {})[metric] = (
                    pd.concat(frames).round(4).to_dict("records"))  # fmt: skip
    out["probes"] = {}
    for results_dir in sorted(root.glob("probe_results*")):
        frame = load_probes(results_dir)
        if len(frame):
            out["probes"][results_dir.name] = probe_stats(frame)
    if (meta / "stations_summary.json").exists():
        out["stations"] = json.loads((meta / "stations_summary.json").read_text())
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {args.out}: {', '.join(out)}")


if __name__ == "__main__":
    main()
