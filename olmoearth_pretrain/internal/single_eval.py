"""Run a single evaluation task for an OlmoEarth checkpoint.

Thin wrapper around full_eval_sweep that accepts --task instead of --task-skip-names.

e.g. python -m olmoearth_pretrain.internal.single_eval \
        --cluster=ai2/jupiter \
        --defaults_only \
        --checkpoint_path=/weka/.../step667200 \
        --module_path=scripts/official/base.py \
        --task=m_eurosat
"""

import argparse
import sys
from logging import getLogger

from olmoearth_pretrain.internal.all_evals import EVAL_TASKS

logger = getLogger(__name__)


def main() -> None:
    """Run a single eval task."""
    # Pull --task before handing the rest to full_eval_sweep's parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--task",
        type=str,
        required=True,
        help="Name of the single eval task to run (must be a key in EVAL_TASKS).",
    )
    known, remaining = pre.parse_known_args()

    task = known.task
    if task not in EVAL_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Must be one of: {sorted(EVAL_TASKS.keys())}"
        )

    # Skip every task except the target.
    skip = ",".join(t for t in EVAL_TASKS if t != task)

    # Inject --task-skip-names and --defaults_only into argv for full_eval_sweep.
    sys.argv = [sys.argv[0]] + remaining + [f"--task-skip-names={skip}", "--defaults_only"]

    from olmoearth_pretrain.internal import full_eval_sweep
    full_eval_sweep.main()


if __name__ == "__main__":
    main()
