"""Summarize the FLUXNET probe results written by ``probe_queue.py`` / in-loop evals.

Reads every ``<results_dir>/*.jsonl`` (one line per (run, step, task) eval, see
``Era5DownstreamEvaluatorCallback.results_jsonl``) and prints, per task:

* ``lr``: median val R2 per probe LR over the step-50k arm checkpoints (the
  plateau that picks the registry ``probe_lr``);
* ``floor``: random-init (``*_init*``) and step-0 values;
* ``curves``: for runs with many checkpoints, peak value and step, final value,
  convergence step (first step within 10 % of the floor-to-final gain), late
  SD (steps >= 40k), step-to-step jitter (SD of successive differences) and the
  Spearman correlation of value with step (monotonicity);
* ``arms``: seed mean +- SD per arm at its last checkpoint, the between-arm F
  statistic (how well the task separates arms relative to seed noise) and
  Welch t / p of each arm against a reference arm;
* ``agreement``: Spearman correlation across arms between task means.

Metric: val R2 at a fixed probe LR (``--lr``, else the registry LR), or the best
R2 over the LR grid with ``--best-lr``.

Usage::

    python scripts/era5_supervised/v0/fluxnet/analyze_probes.py --results-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED_RE = re.compile(r"_seed\d+$")


def load(results_dir: Path) -> pd.DataFrame:
    """Flatten every JSONL record into one row per (run, step, task, lr).

    Args:
        results_dir: Directory of ``*.jsonl`` files.

    Returns:
        Columns ``run, step, task, lr, registry, val_r2, val_rmse, test_r2,
        test_rmse, effective_rank``.
    """
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            base = dict(run=rec["run"], step=rec["step"], task=rec["task"],
                        effective_rank=rec.get("geometry", {}).get(
                            f"eval_other/{rec['task']}/effective_rank"))  # fmt: skip
            val, test = rec.get("val") or {}, rec.get("test") or {}
            rows.append(dict(base, lr=float(rec["probe_lr"]), registry=True,
                             val_r2=val.get("r2"), val_rmse=val.get("rmse"),
                             test_r2=test.get("r2"), test_rmse=test.get("rmse")))  # fmt: skip
            for lr, res in (rec.get("lrsweep") or {}).items():
                v, t = res.get("val") or {}, res.get("test") or {}
                rows.append(dict(base, lr=float(lr), registry=False,
                                 val_r2=v.get("r2"), val_rmse=v.get("rmse"),
                                 test_r2=t.get("r2"), test_rmse=t.get("rmse")))  # fmt: skip
    frame = pd.DataFrame(rows)
    # A requeued sweep can log a checkpoint twice: keep the last.
    return frame.drop_duplicates(["run", "step", "task", "lr", "registry"], keep="last")


def arm_of(run: str) -> str:
    """Arm name of a run: drop the probe suffix and the seed suffix.

    Args:
        run: W&B run name, e.g. ``era5enc_1601_doyc_seed1_flux``.

    Returns:
        ``era5enc_1601_doyc``.
    """
    name = re.sub(r"_(flux|reprobe)$", "", run)
    return SEED_RE.sub("", name)


def metric_frame(frame: pd.DataFrame, lr: float | None, best_lr: bool) -> pd.DataFrame:
    """One value per (run, step, task).

    Args:
        frame: Output of :func:`load`.
        lr: Fixed probe LR to read (None: the registry LR).
        best_lr: Take the best val R2 over the LR grid instead.

    Returns:
        Columns ``run, step, task, value, test`` (value = val R2).
    """
    if best_lr:
        sweep = frame[~frame["registry"]].dropna(subset=["val_r2"])
        idx = sweep.groupby(["run", "step", "task"])["val_r2"].idxmax()
        picked = sweep.loc[idx]
    elif lr is not None:
        picked = frame[(~frame["registry"]) & np.isclose(frame["lr"], lr)]
    else:
        picked = frame[frame["registry"]]
    return picked.rename(columns={"val_r2": "value", "test_r2": "test"})[
        ["run", "step", "task", "value", "test", "lr"]
    ]


def lr_plateau(frame: pd.DataFrame) -> pd.DataFrame:
    """Median val R2 per (task, LR) over the last checkpoint of every trained run.

    Args:
        frame: Output of :func:`load`.

    Returns:
        Tasks x LRs table.
    """
    sweep = frame[(~frame["registry"]) & (frame["step"] > 0)]
    last = sweep.groupby("run")["step"].transform("max") == sweep["step"]
    return sweep[last].pivot_table(
        index="task", columns="lr", values="val_r2", aggfunc="median"
    )


def curve_stats(values: pd.DataFrame) -> pd.DataFrame:
    """Shape statistics of each (run, task) curve with >= 5 checkpoints.

    Args:
        values: Output of :func:`metric_frame`.

    Returns:
        One row per (run, task).
    """
    rows = []
    for (run, task), curve in values.groupby(["run", "task"]):
        curve = curve.sort_values("step")
        if len(curve) < 5:
            continue
        steps, v = curve["step"].to_numpy(), curve["value"].to_numpy()
        floor, final = v[0], v[-1]
        gain = final - floor
        within = np.abs(v - final) <= 0.1 * abs(gain) if gain else np.ones_like(v, bool)
        conv = next(
            s
            for s, ok in zip(steps, within)
            if ok and within[list(steps).index(s) :].all()
        )
        late = v[steps >= 40000]
        rows.append(dict(
            run=run, task=task, step0=floor, final=final, peak=v.max(),
            peak_step=int(steps[v.argmax()]), gain=gain, conv_step=int(conv),
            late_sd=float(late.std()) if len(late) > 1 else np.nan,
            jitter=float(np.diff(v).std()), spearman=float(stats.spearmanr(steps, v)[0]),
        ))  # fmt: skip
    return pd.DataFrame(rows)


def arm_stats(values: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Seed mean / SD per arm at each run's last checkpoint, with Welch tests.

    Args:
        values: Output of :func:`metric_frame`.
        reference: Arm the others are tested against.

    Returns:
        One row per (task, arm).
    """
    last = values[values.groupby("run")["step"].transform("max") == values["step"]]
    last = last[last["step"] > 0].assign(arm=lambda f: f["run"].map(arm_of))
    rows = []
    for task, part in last.groupby("task"):
        ref = part.loc[part["arm"] == reference, "value"].to_numpy()
        for arm, group in part.groupby("arm"):
            v = group["value"].to_numpy()
            t, p = (stats.ttest_ind(v, ref, equal_var=False)
                    if len(v) > 1 and len(ref) > 1 and arm != reference else (np.nan, np.nan))  # fmt: skip
            rows.append(dict(task=task, arm=arm, n=len(v), mean=v.mean(),
                             sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                             test_mean=group["test"].mean(), t=t, p=p))  # fmt: skip
    return pd.DataFrame(rows)


def between_arm_f(arms: pd.DataFrame) -> pd.Series:
    """One-way ANOVA-style F per task: between-arm variance over seed variance.

    Args:
        arms: Output of :func:`arm_stats` (arms with >= 2 seeds are used).

    Returns:
        F per task.
    """
    out = {}
    for task, part in arms[arms["n"] > 1].groupby("task"):
        n = part["n"].to_numpy()
        grand = np.average(part["mean"], weights=n)
        between = np.sum(n * (part["mean"] - grand) ** 2) / (len(part) - 1)
        within = np.sum((n - 1) * part["sd"] ** 2) / (n.sum() - len(part))
        out[task] = between / within
    return pd.Series(out, name="F")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results-dir",
        default="/weka/dfive-default/rslearn-eai/datasets/fluxnet/probe_results",
    )
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--best-lr", action="store_true")
    ap.add_argument("--reference", default="era5enc_1601_doyc")
    ap.add_argument("--runs", default=None, help="regex filter on run names")
    ap.add_argument("--out", default=None, help="also write the tables as JSON here")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    frame = load(Path(args.results_dir))
    if args.runs:
        frame = frame[frame["run"].str.contains(args.runs)]
    values = metric_frame(frame, args.lr, args.best_lr)
    print(
        "== LR plateau (median val R2 at each run's last checkpoint)\n",
        lr_plateau(frame).round(3),
    )
    floors = values[values["step"] == 0].pivot_table(
        index="run", columns="task", values="value"
    )
    print("\n== floors (step 0 / random init)\n", floors.round(3).T)
    curves = curve_stats(values)
    if not curves.empty:
        print(
            "\n== curves\n",
            curves.round(4).sort_values(["task", "run"]).to_string(index=False),
        )
    arms = arm_stats(values, args.reference)
    if not arms.empty:
        print("\n== arms (last checkpoint)\n", arms.round(4).to_string(index=False))
        print(
            "\n== between-arm F (seed-noise normalized)\n", between_arm_f(arms).round(2)
        )
        means = arms.pivot_table(index="arm", columns="task", values="mean")
        print(
            "\n== Spearman agreement across arms\n",
            means.corr(method="spearman").round(2),
        )
    if args.out:
        Path(args.out).write_text(json.dumps({
            "plateau": lr_plateau(frame).to_dict(), "curves": curves.to_dict("records"),
            "arms": arms.to_dict("records"),
        }, default=float, indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
