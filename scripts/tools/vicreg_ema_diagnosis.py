"""Deep diagnosis of EMA+VICReg runs: why is loss going down but evals not improving?

Pulls train metrics and eval metrics separately (they're logged at different steps),
then analyzes trends for overfitting, collapse, and EMA staleness.
"""

import numpy as np
import pandas as pd
import wandb

WANDB_ENTITY = "eai-ai2"
PROJECT = "2026_02_08_masked_neg"

ALL_RUNS = {
    # EMA runs
    "yaf8h3wd": ("vicreg_v3_patchvarcov_sat", True),
    "4issro26": ("vicreg_half", True),
    "yzunsg1d": ("vicreg_views_v2", True),
    "iywvcczi": ("vicreg_both_no_inv_v3", True),
    "x7d1dsw5": ("vicreg_v3_no_inv", True),
    # No-EMA runs
    "av32er6u": ("vicreg_no_ema", False),
    "8b8ygm8n": ("sigreg_no_ema", False),
    "c78de6c7": ("patch_varcov_no_ema", False),
    "ah0nbkl7": ("vicreg_both_no_ema", False),
    # Baseline
    "udyy8b18": ("BASELINE", True),
}

TRAIN_KEYS = ["train/InfoNCE", "train/VICRegViews", "train/PatchVarCov",
              "train/ModalityPatchDiscMasked", "train/ema_decay", "train/total_loss"]
EVAL_KEYS = ["eval/m-eurosat", "eval/m_so2sat", "eval/mados", "eval/pastis"]
EMBED_KEYS = [
    "eval_embed_diagnostics/embed_diag_eurosat/effective_rank",
    "eval_embed_diagnostics/embed_diag_eurosat/uniformity",
    "eval_embed_diagnostics/embed_diag_eurosat/cosine_sim_mean",
]


def pull_history_split(api, run_id, label, samples=200):
    """Pull train and eval history separately then merge on _step."""
    run = api.run(f"{WANDB_ENTITY}/{PROJECT}/{run_id}")
    total = run.summary.get("_step", 0)
    print(f"  {label} ({run_id}): {total} steps, state={run.state}")

    train_df = run.history(keys=["_step"] + TRAIN_KEYS, samples=samples, pandas=True)
    eval_df = run.history(keys=["_step"] + EVAL_KEYS + EMBED_KEYS, samples=samples, pandas=True)

    # Drop all-NaN columns
    train_df = train_df.dropna(axis=1, how="all")
    eval_df = eval_df.dropna(axis=1, how="all")

    print(f"    train: {len(train_df)} rows, cols={list(train_df.columns)}")
    print(f"    eval:  {len(eval_df)} rows, cols={list(eval_df.columns)}")

    return train_df, eval_df


def quartile_trend(series):
    """Compare first quartile mean vs last quartile mean."""
    s = series.dropna()
    if len(s) < 8:
        return None, None, "INSUFFICIENT"
    n = len(s) // 4
    early = s.iloc[:n].mean()
    late = s.iloc[-n:].mean()
    if late < early * 0.95:
        status = "DROPPING"
    elif late > early * 1.05:
        status = "RISING"
    else:
        status = "FLAT"
    return early, late, status


def diagnose_run(label, train_df, eval_df, is_ema):
    """Diagnose one run for overfitting / collapse / staleness."""
    tag = "[EMA]" if is_ema else "[no-EMA]"
    print(f"\n{'='*70}")
    print(f"  {tag} {label}")

    if "_step" in train_df.columns and len(train_df) > 0:
        print(f"  Train steps: {train_df['_step'].min():.0f} -> {train_df['_step'].max():.0f}")
    if "_step" in eval_df.columns and len(eval_df) > 0:
        print(f"  Eval steps:  {eval_df['_step'].min():.0f} -> {eval_df['_step'].max():.0f}")

    # Loss trends
    print(f"\n  --- Loss Trends ---")
    for k in TRAIN_KEYS:
        if k not in train_df.columns:
            continue
        early, late, status = quartile_trend(train_df[k])
        if early is None:
            continue
        print(f"    {k:<45s} {early:.6f} -> {late:.6f}  [{status}]")

    # Eval trends
    print(f"\n  --- Eval Trends ---")
    eval_improving = 0
    eval_regressing = 0
    for k in EVAL_KEYS:
        if k not in eval_df.columns:
            continue
        s = eval_df[k].dropna()
        if len(s) < 4:
            continue
        early, late, status = quartile_trend(eval_df[k])
        if early is None:
            continue
        peak = s.max()
        peak_step = eval_df.loc[s.idxmax(), "_step"] if s.idxmax() in eval_df.index else "?"
        # Check if peak is way before the end (sign of overfitting)
        last_step = eval_df["_step"].max() if "_step" in eval_df.columns else 0
        peak_pct = (peak_step / last_step * 100) if isinstance(peak_step, (int, float)) and last_step > 0 else "?"
        print(f"    {k:<45s} {early:.4f} -> {late:.4f}  peak={peak:.4f} @ step {peak_step} ({peak_pct}%)  [{status}]")

        if status == "IMPROVING":
            eval_improving += 1
        elif status in ("DROPPING", "RISING"):
            eval_regressing += 1

    # Embed trends
    print(f"\n  --- Embedding Diagnostics ---")
    for k in EMBED_KEYS:
        if k not in eval_df.columns:
            continue
        early, late, status = quartile_trend(eval_df[k])
        if early is None:
            continue
        print(f"    {k.split('/')[-1]:<35s} {early:.4f} -> {late:.4f}  [{status}]")

    # Verdict
    print(f"\n  --- VERDICT ---")
    if "_step" in train_df.columns and "train/InfoNCE" in train_df.columns:
        _, _, loss_status = quartile_trend(train_df["train/InfoNCE"])
        if loss_status == "DROPPING" and eval_regressing > eval_improving:
            print(f"    ⚠️  LOSS DROPPING + EVALS REGRESSING = LIKELY OVERFITTING or REP COLLAPSE")
        elif loss_status == "DROPPING" and eval_improving == 0 and eval_regressing == 0:
            print(f"    ⚠️  LOSS DROPPING + EVALS FLAT = LOSS NOT TRANSFERRING TO DOWNSTREAM")
        elif loss_status == "FLAT":
            print(f"    ℹ️  LOSS FLAT = TRAINING HAS SATURATED")
        else:
            print(f"    ✓  Loss={loss_status}, Evals: {eval_improving} improving, {eval_regressing} regressing")
    else:
        print(f"    (no InfoNCE data for verdict)")


def cross_run_table(all_data):
    """Print final snapshot comparison."""
    print(f"\n{'='*70}")
    print("CROSS-RUN FINAL SNAPSHOT")
    print(f"{'='*70}")
    rows = []
    for run_id, (label, is_ema) in ALL_RUNS.items():
        if run_id not in all_data:
            continue
        train_df, eval_df = all_data[run_id]
        row = {"run": label, "ema": is_ema}
        # Last train values
        for k in ["train/InfoNCE", "train/VICRegViews"]:
            if k in train_df.columns:
                s = train_df[k].dropna()
                row[k.split("/")[-1] + "_final"] = s.iloc[-1] if len(s) > 0 else None
        # Last eval values
        for k in EVAL_KEYS:
            if k in eval_df.columns:
                s = eval_df[k].dropna()
                row[k.split("/")[-1]] = s.iloc[-1] if len(s) > 0 else None
                row[k.split("/")[-1] + "_max"] = s.max() if len(s) > 0 else None
        for k in EMBED_KEYS:
            if k in eval_df.columns:
                s = eval_df[k].dropna()
                row[k.split("/")[-1]] = s.iloc[-1] if len(s) > 0 else None
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.4f"))
    df.to_csv("vicreg_ema_diagnosis_snapshot.csv", index=False)
    print("\nSaved vicreg_ema_diagnosis_snapshot.csv")


def main():
    api = wandb.Api(timeout=120)
    all_data = {}

    for run_id, (label, is_ema) in ALL_RUNS.items():
        try:
            train_df, eval_df = pull_history_split(api, run_id, label)
            all_data[run_id] = (train_df, eval_df)
        except Exception as e:
            print(f"  FAILED {label}: {e}")

    print("\n\n" + "#" * 70)
    print("# PER-RUN DIAGNOSIS")
    print("#" * 70)

    for run_id, (label, is_ema) in ALL_RUNS.items():
        if run_id not in all_data:
            continue
        train_df, eval_df = all_data[run_id]
        diagnose_run(label, train_df, eval_df, is_ema)

    cross_run_table(all_data)


if __name__ == "__main__":
    main()
