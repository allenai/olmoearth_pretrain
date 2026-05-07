r"""Fan out dump_embeddings.py across all models in paper Table 2.

Iterates every (foundation_model, model_size) pair plus the four OlmoEarth
checkpoints, building one ``dump_embeddings.py`` invocation per group. Use
``--print_only`` to inspect the plan before launching.

Per-model JSON groups are read from
``data/max_eval_settings/max_eval_settings_per_group_merged.json`` (external
FMs) and ``data/max_eval_settings/{nano,tiny,base,large}_settings.json``
(OlmoEarth).

Example::

    python3 scripts/joe/dump_paper_embeddings.py \\
        --save_embeddings_dir=/weka/dfive-default/olmoearth_pretrain/paper_embeddings \\
        --cluster=ai2/saturn-cirrascale \\
        --print_only
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess  # nosec
import threading
from logging import getLogger

logger = getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Use the *enriched* JSONs that carry an explicit ``norm_mode`` per task
# (added by ``scripts/joe/enrich_eval_settings_json.py``). Without them, the
# partition step can pick the wrong sweep arm for models like Galileo where
# ``norm_stats_from_pretrained`` is hardcoded False regardless of mode.
MERGED_JSON = "data/max_eval_settings/max_eval_settings_per_group_merged.enriched.json"

# Paper Table 2 uses 64x64 pastis only; drop the 128x128 variants by default.
DEFAULT_EXCLUDED_TASKS = (
    "pastis128_sentinel1",
    "pastis128_sentinel2",
    "pastis128_sentinel1_sentinel2",
)

# (model name in BaselineModelName, JSON-group suffix, optional --size value).
EXTERNAL_FM_GROUPS: list[tuple[str, str | None]] = [
    ("anysat", None),
    ("clay", "large"),
    ("copernicusfm", None),
    ("croma", "base"),
    ("croma", "large"),
    ("dino_v3", "dinov3_vitb16"),
    ("dino_v3", "dinov3_vitl16"),
    ("dino_v3", "dinov3_vith16plus"),
    ("dino_v3", "dinov3_vit7b16"),
    ("dino_v3", "dinov3_vitl16_sat"),
    ("dino_v3", "dinov3_vit7b16_sat"),
    ("galileo", "nano"),
    ("galileo", "tiny"),
    ("galileo", "base"),
    ("panopticon", None),
    ("presto", None),
    ("prithvi_v2", "Prithvi-EO-2.0-300M"),
    ("prithvi_v2", "Prithvi-EO-2.0-600M"),
    ("satlas", "base"),
    ("terramind", "base"),
    ("terramind", "large"),
    ("tessera", None),
]

# OlmoEarth: (size, settings JSON, group key inside that JSON, module path,
# checkpoint path on Weka).
OLMOEARTH_RUNS: list[tuple[str, str, str, str, str]] = [
    (
        "nano",
        "data/max_eval_settings/nano_settings.enriched.json",
        "nano_lr0.001_wd0.002",
        "scripts/official/nano.py",
        "/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000",
    ),
    (
        "tiny",
        "data/max_eval_settings/tiny_settings.enriched.json",
        "tiny_lr0.0002_wd0.02",
        "scripts/official/tiny.py",
        "/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000",
    ),
    (
        "base",
        "data/max_eval_settings/base_settings.enriched.json",
        "phase2.0_base_lr0.0001_wd0.02",
        "scripts/official/base.py",
        "/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200",
    ),
    (
        "large",
        "data/max_eval_settings/large_settings.enriched.json",
        "phase2.0_large_lr0.0001_wd0.002",
        "scripts/official/large.py",
        "/weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/step560000",
    ),
]


def _group_key(model: str, size: str | None) -> str:
    """Build the JSON top-level key for a (model, size) pair."""
    return model if size is None else f"{model}_{size}"


def _check_groups_exist() -> None:
    """Sanity check: every (model, size) we plan to launch is in the merged JSON."""
    merged_path = os.path.join(REPO_ROOT, MERGED_JSON)
    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            f"{merged_path} missing; we are likely on the wrong branch. "
            f"Use `git worktree add ... origin/gabi/pre-removal`."
        )
    with open(merged_path) as f:
        merged = json.load(f)
    missing = [
        _group_key(m, s)
        for m, s in EXTERNAL_FM_GROUPS
        if _group_key(m, s) not in merged
    ]
    if missing:
        raise KeyError(f"Groups missing from {MERGED_JSON}: {missing}")


def _build_external_cmd(
    model: str,
    size: str | None,
    args: argparse.Namespace,
) -> str:
    """Build the dump_embeddings.py command for an external FM group."""
    parts = [
        "python3 olmoearth_pretrain/internal/dump_embeddings.py",
        f"--cluster={args.cluster}",
        f"--settings_json={MERGED_JSON}",
        f"--settings_group={_group_key(model, size)}",
        f"--save_embeddings_dir={args.save_embeddings_dir}",
        f"--embedding_dump_dtype={args.embedding_dump_dtype}",
        f"--model={model}",
    ]
    if size is not None:
        parts.append(f"--size={size}")
    if not args.include_pastis128:
        parts.append("--exclude_tasks=" + ",".join(DEFAULT_EXCLUDED_TASKS))
    parts.extend(_launch_args(args))
    if args.dry_run:
        parts.append("--dry_run")
    if args.print_only:
        parts.append("--print_only")
    if args.project_name:
        parts.append(f"--project_name={args.project_name}")
    return " ".join(parts)


def _launch_args(args: argparse.Namespace) -> list[str]:
    """Build the --priority / --launch_clusters / --num_gpus pass-through args."""
    parts = [f"--priority={args.priority}", f"--num_gpus={args.num_gpus}"]
    if args.launch_clusters:
        parts.append(f"--launch_clusters={args.launch_clusters}")
    return parts


def _build_olmoearth_cmd(
    size: str,
    settings_json: str,
    group_key: str,
    module_path: str,
    checkpoint_path: str,
    args: argparse.Namespace,
) -> str:
    """Build the dump_embeddings.py command for one OlmoEarth checkpoint."""
    parts = [
        "python3 olmoearth_pretrain/internal/dump_embeddings.py",
        f"--cluster={args.cluster}",
        f"--settings_json={settings_json}",
        f"--settings_group={group_key}",
        f"--save_embeddings_dir={args.save_embeddings_dir}",
        f"--save_subdir=olmoearth_{size}",
        f"--embedding_dump_dtype={args.embedding_dump_dtype}",
        f"--module_path={module_path}",
        f"--checkpoint_path={checkpoint_path}",
        f"--run_name=dump_olmoearth_{size}",
    ]
    if not args.include_pastis128:
        parts.append("--exclude_tasks=" + ",".join(DEFAULT_EXCLUDED_TASKS))
    parts.extend(_launch_args(args))
    if args.dry_run:
        parts.append("--dry_run")
    if args.print_only:
        parts.append("--print_only")
    if args.project_name:
        parts.append(f"--project_name={args.project_name}")
    return " ".join(parts)


def main() -> None:
    """Generate (and optionally run) one dump command per group."""
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True)
    p.add_argument("--save_embeddings_dir", required=True)
    p.add_argument(
        "--embedding_dump_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    p.add_argument("--project_name", default=None)
    p.add_argument(
        "--external_only",
        action="store_true",
        help="Skip the four OlmoEarth checkpoints; useful when only baselines are needed.",
    )
    p.add_argument(
        "--olmoearth_only",
        action="store_true",
        help="Skip external FMs; only re-extract OlmoEarth embeddings.",
    )
    p.add_argument(
        "--filter",
        default=None,
        help="Substring filter on group key (e.g. 'galileo' or 'olmoearth_base').",
    )
    p.add_argument(
        "--include_pastis128",
        action="store_true",
        help="By default we drop the pastis128 variants (paper Table 2 reports 64x64). "
        "Pass this flag to keep them in.",
    )
    p.add_argument(
        "--priority",
        default="normal",
        choices=["low", "normal", "high", "urgent"],
        help="Beaker job priority for every launched run.",
    )
    p.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="--launch.num_gpus passed to every Beaker run.",
    )
    p.add_argument(
        "--launch_clusters",
        default=None,
        help="Comma-separated Beaker clusters (e.g. ai2/jupiter,ai2/saturn,ai2/neptune). "
        "Becomes --launch.clusters=[...] on every launched run.",
    )
    p.add_argument(
        "--parallel",
        type=int,
        default=8,
        help="Number of Beaker submissions to fire in parallel. The submit "
        "step is I/O-bound (image push), so 8 is comfortable; bump higher if "
        "Beaker isn't rate-limiting.",
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--print_only", action="store_true")
    args = p.parse_args()

    _check_groups_exist()

    cmds: list[str] = []
    if not args.olmoearth_only:
        for model, size in EXTERNAL_FM_GROUPS:
            key = _group_key(model, size)
            if args.filter and args.filter not in key:
                continue
            cmds.append(_build_external_cmd(model, size, args))
    if not args.external_only:
        for size, settings_json, gkey, mod, ckpt in OLMOEARTH_RUNS:
            tag = f"olmoearth_{size}"
            if args.filter and args.filter not in tag and args.filter not in gkey:
                continue
            cmds.append(
                _build_olmoearth_cmd(size, settings_json, gkey, mod, ckpt, args)
            )

    print(f"# Will run {len(cmds)} dump commands:")
    for cmd in cmds:
        print(cmd)
        print()
    if args.print_only:
        return
    failures: list[tuple[str, int]] = []
    print_lock = threading.Lock()

    def _run(cmd: str) -> tuple[str, int]:
        result = subprocess.run(cmd, shell=True, check=False)  # nosec
        with print_lock:
            tag = "ok" if result.returncode == 0 else f"FAIL rc={result.returncode}"
            print(f"  [{tag}] {cmd[:160]}...")
        return cmd, result.returncode

    # Beaker submissions are I/O-bound (image push); thread pool is fine.
    with cf.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for cmd, rc in ex.map(_run, cmds):
            if rc != 0:
                failures.append((cmd, rc))
    if failures:
        print(f"\n{len(failures)} of {len(cmds)} launches failed:")
        for cmd, rc in failures:
            print(f"  rc={rc}: {cmd[:200]}...")


if __name__ == "__main__":
    main()
