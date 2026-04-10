r"""Download W&B run history to CSV (uses wandb.Api — run locally with WANDB_API_KEY set).

Example — training loss near the embed-diag spike (170k–200k):

  uv run python scripts/tools/fetch_wandb_run_history.py \\
    --path eai-ai2/2026_02_08_masked_neg/1aw0m3lf \\
    --keys 'train/InfoNCE,train/ModalityPatchDiscMasked,optim/LR (group 0),train/ema_decay' \\
    --min-step 170000 --max-step 200000 \\
    --output /tmp/train_1aw0m3lf.csv

Example — embedding diagnostics run:

  uv run python scripts/tools/fetch_wandb_run_history.py \\
    --path eai-ai2/2026_embedding_diagnostics/s1tihp1u \\
    --keys checkpoint_step,eval_embed_diagnostics/m_eurosat/effective_rank \\
    --output /tmp/embed_diag.csv

Merge on training step (after exporting both CSVs):

  import pandas as pd
  t = pd.read_csv("/tmp/train_1aw0m3lf.csv")
  e = pd.read_csv("/tmp/embed_diag.csv")
  # training logs use _step == optimizer step; embed sweep uses checkpoint_step
  m = t.merge(e, left_on="_step", right_on="checkpoint_step", how="inner")
  m.to_csv("/tmp/merged.csv", index=False)
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import wandb


def main() -> None:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        required=True,
        help="entity/project/run_id e.g. eai-ai2/2026_02_08_masked_neg/1aw0m3lf",
    )
    p.add_argument(
        "--keys",
        default=None,
        help="Comma-separated metric keys (default: all columns from history)",
    )
    p.add_argument("--min-step", type=int, default=None)
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output CSV path",
    )
    args = p.parse_args()

    api = wandb.Api(timeout=120)
    run = api.run(args.path)
    key_list = (
        [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else None
    )

    # scan_history is efficient for large runs; returns iterator of dicts
    rows: list[dict] = []
    for row in run.scan_history(
        keys=key_list, min_step=args.min_step, max_step=args.max_step
    ):
        rows.append(row)

    if not rows:
        print("No rows returned (check keys / step range / run id).", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    # Prefer trainer step column for merging across runs
    if "_step" in df.columns:
        df = df.sort_values("_step")
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.output}")


if __name__ == "__main__":
    main()
