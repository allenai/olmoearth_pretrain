"""Arm-level rank correlations of the snow evals with the other ERA5 evals.

Do the snow evals rank encoders like the other ERA5 evals?
For the step-50k arms of the v1.4-v1.6 probe queue (seed means), collects:

* the in-loop basket (LFMC, DFMC, wheat, maize, burn risk, landslide) from the
  USDM session's ``old_evals.json`` (per arm id, three seeds; negative RMSE /
  F1 as logged in-loop);
* FLUXNET ``fluxnet_rgpp_eval`` and USDM ``usdm_dpast26_eval`` (val R2 at the
  plateau LR of their probe-results JSONL, the USDM session's convention);
* the snow evals: val R2 at the plateau LR of ``emb_probe`` JSONL
  (``probe_results`` / ``probe_results_last4``) or, with ``--snow-source lab``,
  the ridge proxy of ``embeddings/target_lab.json``;

and prints the Spearman correlation matrix across arms (higher = better for
every metric) and each snow eval's mean correlation with the basket. An eval
that ranks arms like the others adds little; one that ranks them differently is
an independent axis (or noise: read next to its between-arm F).

Usage::

    python scripts/era5_supervised/v0/snow/cross_eval.py --out cross_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED_RE = re.compile(r"_seed\d+$")
ARM_ID_RE = re.compile(r"era5enc_(\d{4})")
OLD_EVALS = Path("/weka/dfive-default/hadriens/scratch/usdm/old_evals.json")
FLUX_DIR = Path("/weka/dfive-default/rslearn-eai/datasets/fluxnet/probe_results")
USDM_DIR = Path("/weka/dfive-default/rslearn-eai/datasets/usdm/probe_results_last4")
SNOW_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/snow")
CURVE_RUNS = ("1301", "1305", "1306")


def arm_id(run: str) -> str | None:
    """Four-digit arm id of a run name (``era5enc_1601_doyc_seed1`` -> ``1601``).

    Args:
        run: Run name.

    Returns:
        The id, or None for random inits and unnamed runs.
    """
    match = ARM_ID_RE.match(run)
    return match.group(1) if match else None


def plateau_means(results_dir: Path, task: str) -> dict[str, float]:
    """Per-arm seed-mean val R2 of one task at its plateau LR (step-50k arms).

    Args:
        results_dir: Directory of evaluator JSONL files.
        task: Task name.

    Returns:
        ``{arm id: mean val R2}``.
    """
    rows = []
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("task") != task or rec.get("step") not in (50000, 150000):
                continue
            arm = arm_id(SEED_RE.sub("", re.sub(r"_(flux|reprobe)$", "", rec["run"])))
            if arm is None or arm in CURVE_RUNS:
                continue
            for lr, res in (rec.get("lrsweep") or {}).items():
                val = (res.get("val") or {}).get("r2")
                if val is not None:
                    rows.append(dict(run=rec["run"], arm=arm, lr=float(lr), val=val))
    frame = pd.DataFrame(rows).drop_duplicates(["run", "lr"], keep="last")
    if frame.empty:
        return {}
    plateau = frame.groupby("lr")["val"].median().idxmax()
    fixed = frame[np.isclose(frame["lr"], plateau)]
    return fixed.groupby("arm")["val"].mean().to_dict()


def lab_means(path: Path, family: str, target: str, pooling: str) -> dict[str, float]:
    """Per-arm seed-mean val R2 of the target lab's ridge proxy.

    Args:
        path: ``target_lab.json``.
        family: Window family.
        target: Target.
        pooling: Pooling.

    Returns:
        ``{arm id: mean val R2}``.
    """
    sets = json.loads(path.read_text())[family][target][pooling]
    values: dict[str, list[float]] = {}
    for name, scores in sets.items():
        run, _, step = name.partition("@")
        arm = arm_id(SEED_RE.sub("", run))
        if arm is None or arm in CURVE_RUNS or step not in ("50000", "150000"):
            continue
        values.setdefault(arm, []).append(scores["val"])
    return {a: float(np.mean(v)) for a, v in values.items()}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--snow-source", choices=["probe", "lab"], default="probe")
    args = ap.parse_args()
    table: dict[str, dict[str, float]] = {}
    old = json.loads(OLD_EVALS.read_text())
    for task, arms in old.items():
        table[task] = {
            a: float(np.mean(v["step50k"])) for a, v in arms.items() if v.get("step50k")
        }
    table["fluxnet_rgpp_eval"] = plateau_means(FLUX_DIR, "fluxnet_rgpp_eval")
    table["usdm_dpast26_eval"] = plateau_means(USDM_DIR, "usdm_dpast26_eval")
    snow = {
        "snow_sdd_anom_l0_eval": ("l0", "sdd_anom", "last4"),
        "snow_sdd_anom_p30_eval": ("p30", "sdd_anom", "last4"),
        "snow_sdd_anom_p60_eval": ("p60", "sdd_anom", "last4"),
        "snow_scd_post_eval": ("post", "scd", "last4"),
        "snow_sdd_dowy_post_eval": ("post", "sdd_dowy", "mean"),
    }
    for task, (family, target, pooling) in snow.items():
        if args.snow_source == "lab":
            lab = SNOW_ROOT / "embeddings" / "target_lab.json"
            try:
                table[f"{task}[{pooling}]"] = lab_means(lab, family, target, pooling)
            except KeyError:
                continue
        else:
            sub = "probe_results" if pooling == "mean" else f"probe_results_{pooling}"
            means = plateau_means(SNOW_ROOT / sub, task)
            if means:
                table[f"{task}[{pooling}]"] = means
    frame = pd.DataFrame(table)
    frame = frame.dropna(axis=1, how="all")
    arms = frame.dropna().index
    corr = frame.loc[arms].rank().corr(method="pearson")
    pd.set_option("display.width", 250)
    print(f"{len(arms)} arms with every eval: {sorted(arms)}")
    print(corr.round(2).to_string())
    basket = [c for c in corr.columns if not c.startswith("snow_")]
    for col in corr.columns:
        if col.startswith("snow_"):
            print(
                f"{col:38s} mean rank corr with basket {corr.loc[col, basket].mean():+.2f}"
            )
    Path(args.out).write_text(json.dumps(dict(
        arms=sorted(arms), values=frame.loc[arms].to_dict(), spearman=corr.to_dict(),
    ), indent=1))  # fmt: skip


if __name__ == "__main__":
    main()
