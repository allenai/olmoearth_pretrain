"""Analyze VICReg experiment runs: pull eval metrics, embed diagnostics, and training losses.

Usage:
    python -m scripts.tools.vicreg_analysis
"""

import json

import pandas as pd
import wandb

WANDB_ENTITY = "eai-ai2"
PROJECT = "2026_02_08_masked_neg"

VICREG_RUN_IDS = {
    # Currently running / finished
    "av32er6u": "vicreg_no_ema_v1",
    "8b8ygm8n": "sigreg_no_ema_v1",
    "c78de6c7": "patch_varcov_no_ema_v1",
    "iywvcczi": "vicreg_both_no_inv_v3",
    "yaf8h3wd": "vicreg_v3_patch_varcov_sat",
    "zkga5xl1": "vicreg_v3_no_inv_sat_only",
    # Crashed (partial data)
    "ah0nbkl7": "vicreg_both_no_ema_v1",
    "x7d1dsw5": "vicreg_v3_no_inv",
    "3ki792zz": "vicreg-multi-level-d6-w1",
    "zpppv2j2": "vicreg-pooled-target-align-mse-w1",
    "0fafpvxl": "vicreg_5x_v1",
    "6zbqo9mh": "vicreg_no_var_v1",
    "cu0sm24c": "vicreg-sigreg-replace-vicreg-w05",
    "efnustdj": "vicreg_2x_v1",
    "yzunsg1d": "vicreg_views_v2",
    "4issro26": "vicreg_half_v1",
    "3wpp91xz": "vicreg_views_v1",
}

BASELINE_ID = "udyy8b18"

EVAL_KEYS = [
    "eval/m-eurosat", "eval/m_so2sat", "eval/mados", "eval/pastis",
    "eval/geo_ecosystem_annual_test", "eval/canada_wildfire_sat_eval_split", "eval/yemen_crop",
]

EMBED_DIAG_KEYS = [
    "eval_embed_diagnostics/embed_diag_eurosat/effective_rank",
    "eval_embed_diagnostics/embed_diag_eurosat/uniformity",
    "eval_embed_diagnostics/embed_diag_eurosat/cosine_sim_mean",
    "eval_embed_diagnostics/embed_diag_so2sat/effective_rank",
    "eval_embed_diagnostics/embed_diag_so2sat/uniformity",
    "eval_embed_diagnostics/embed_diag_so2sat/cosine_sim_mean",
]

TRAIN_KEYS = [
    "train/ModalityPatchDiscMasked", "train/InfoNCE", "train/VICRegViews",
    "train/ModalityPatchDiscMasked+VICRegViews", "train/ema_decay", "train/PatchVarCov",
]


def get_summary_table(api: wandb.Api) -> pd.DataFrame:
    """Pull summary metrics for all VICReg runs + baseline."""
    all_ids = {BASELINE_ID: "BASELINE (EMA, no VICReg)", **VICREG_RUN_IDS}
    rows = []
    for run_id, label in all_ids.items():
        print(f"Fetching summary: {label} ({run_id})")
        run = api.run(f"{WANDB_ENTITY}/{PROJECT}/{run_id}")
        sm = run.summary
        row = {
            "run_id": run_id,
            "display_name": label,
            "state": run.state,
            "step": sm.get("_step", 0),
        }
        for k in EVAL_KEYS + EMBED_DIAG_KEYS + TRAIN_KEYS:
            row[k] = sm.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def get_history(api: wandb.Api, run_id: str, keys: list[str], samples: int = 100) -> pd.DataFrame:
    """Pull sampled time-series history for a run."""
    run = api.run(f"{WANDB_ENTITY}/{PROJECT}/{run_id}")
    print(f"Fetching history: {run.display_name} ({run_id}), {len(keys)} keys, {samples} samples")
    history = run.history(keys=["_step"] + keys, samples=samples, pandas=True)
    return history


def main():
    api = wandb.Api(timeout=120)

    # 1) Summary table
    print("=" * 80)
    print("STEP 1: Summary metrics for all runs")
    print("=" * 80)
    summary_df = get_summary_table(api)
    summary_df.to_csv("vicreg_analysis_summary.csv", index=False)
    print(f"\nSaved vicreg_analysis_summary.csv ({len(summary_df)} runs)")

    # Print eval table
    print("\n--- Downstream Eval Metrics ---")
    eval_cols = ["display_name", "state", "step"] + EVAL_KEYS
    print(summary_df[eval_cols].to_string(index=False, float_format="%.4f"))

    print("\n--- Embedding Diagnostics ---")
    diag_cols = ["display_name", "state", "step"] + EMBED_DIAG_KEYS
    print(summary_df[diag_cols].to_string(index=False, float_format="%.4f"))

    print("\n--- Training Losses ---")
    train_cols = ["display_name", "state", "step"] + TRAIN_KEYS
    print(summary_df[train_cols].to_string(index=False, float_format="%.4f"))

    # 2) Training curves for key runs
    print("\n" + "=" * 80)
    print("STEP 2: Training curves for key runs")
    print("=" * 80)
    curve_keys = EVAL_KEYS + [
        "eval_embed_diagnostics/embed_diag_eurosat/effective_rank",
        "eval_embed_diagnostics/embed_diag_eurosat/uniformity",
        "train/InfoNCE",
    ]
    key_runs = {
        BASELINE_ID: "BASELINE",
        "av32er6u": "vicreg_no_ema",
        "8b8ygm8n": "sigreg_no_ema",
        "c78de6c7": "patch_varcov_no_ema",
        "iywvcczi": "vicreg_both_no_inv_v3",
        "yaf8h3wd": "vicreg_v3_patch_varcov_sat_EMA",
        "4issro26": "vicreg_half_EMA",
    }
    all_curves = []
    for run_id, label in key_runs.items():
        try:
            df = get_history(api, run_id, curve_keys, samples=80)
            df["run"] = label
            all_curves.append(df)
        except Exception as e:
            print(f"  WARNING: failed for {label}: {e}")

    if all_curves:
        curves_df = pd.concat(all_curves, ignore_index=True)
        curves_df.to_csv("vicreg_analysis_curves.csv", index=False)
        print(f"\nSaved vicreg_analysis_curves.csv ({len(curves_df)} rows)")

    # 3) Print at-step comparison (align at ~300k steps for comparable snapshot)
    print("\n" + "=" * 80)
    print("STEP 3: Comparable snapshot at ~300k steps")
    print("=" * 80)
    target_step = 300000
    for run_id, label in key_runs.items():
        try:
            df = get_history(api, run_id, EVAL_KEYS + EMBED_DIAG_KEYS[:2], samples=200)
            closest = df.iloc[(df["_step"] - target_step).abs().argsort()[:1]]
            step = int(closest["_step"].iloc[0])
            eurosat = closest["eval/m-eurosat"].iloc[0] if "eval/m-eurosat" in closest.columns else None
            mados = closest["eval/mados"].iloc[0] if "eval/mados" in closest.columns else None
            er = closest.get("eval_embed_diagnostics/embed_diag_eurosat/effective_rank", pd.Series([None])).iloc[0]
            unif = closest.get("eval_embed_diagnostics/embed_diag_eurosat/uniformity", pd.Series([None])).iloc[0]
            print(f"  {label:<35s} step={step:>7d}  eurosat={eurosat}  mados={mados}  eff_rank={er}  uniformity={unif}")
        except Exception as e:
            print(f"  {label}: failed ({e})")

    print("\nDone!")


if __name__ == "__main__":
    main()
