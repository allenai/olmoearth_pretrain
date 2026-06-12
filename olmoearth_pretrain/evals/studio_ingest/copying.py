"""Dataset copy helpers for Studio ingestion."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess  # nosec B404
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm
from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.paths import WEKA_EVAL_DATASETS_BASE_PATH
from olmoearth_pretrain.evals.studio_ingest.tags import tags_match_options

logger = logging.getLogger(__name__)

# Base path for eval datasets on Weka
EVAL_DATASETS_BASE_PATH = WEKA_EVAL_DATASETS_BASE_PATH


def _check_weka_exists() -> bool:
    """Check if Weka filesystem path exists."""
    return Path("/weka").exists()


def _try_copy_config_json(source_path: str, dest_path: str) -> None:
    """Copy config.json from source to destination if it exists."""
    src = UPath(source_path) / "config.json"
    if not src.exists():
        logger.info("  config.json not found in source, skipping")
        return
    dst = UPath(dest_path) / "config.json"
    with src.open("rb") as f:
        data = f.read()
    with dst.open("wb") as f:
        f.write(data)
    logger.info("  Copied config.json")


def _model_config_sources(model_config_path: str) -> tuple[UPath, UPath]:
    """Return the config directory and model YAML source for a config path."""
    path = UPath(model_config_path)
    if path.suffix in {".yaml", ".yml"}:
        return path.parent, path
    return path, path / "model.yaml"


def _ensure_config_json(dataset_path: str, model_config_path: str) -> None:
    """Ensure config.json exists in the dataset folder."""
    config_json = Path(dataset_path) / "config.json"

    if config_json.exists() and not config_json.is_symlink():
        return

    model_config_dir, _ = _model_config_sources(model_config_path)
    dataset_json = model_config_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(
            f"config.json missing from {dataset_path} and no dataset.json "
            f"found in {model_config_dir}"
        )

    if config_json.is_symlink():
        logger.info(f"  Removing broken config.json symlink in {dataset_path}")
        config_json.unlink()

    logger.info(f"  config.json missing, copying dataset.json from {model_config_dir}")
    with dataset_json.open("rb") as f:
        data = f.read()
    with open(config_json, "wb") as f:
        f.write(data)
    logger.info("  Wrote config.json to dataset folder")


def _copy_model_yaml(dataset_path: str, model_config_path: str) -> None:
    """Copy model.yaml into the dataset folder for canonical access at eval time."""
    dest = Path(dataset_path) / "model.yaml"
    if dest.exists():
        logger.info("  model.yaml already exists in dataset folder, skipping copy")
        return

    _, src = _model_config_sources(model_config_path)
    if not src.exists():
        raise FileNotFoundError(f"model.yaml not found at {model_config_path}")

    logger.info(f"  Copying model.yaml from {src} to dataset folder")
    with src.open("rb") as f:
        data = f.read()
    with open(dest, "wb") as f:
        f.write(data)
    logger.info("  Wrote model.yaml to dataset folder")


def prepare_copied_dataset_config(dataset_path: str, model_config_path: str) -> None:
    """Ensure copied datasets contain canonical config.json and model.yaml files."""
    _ensure_config_json(dataset_path, model_config_path)
    _copy_model_yaml(dataset_path, model_config_path)


def _copy_from_gcs(
    source_path: str,
    dest_path: str,
    source_groups: list[str] | None = None,
    source_tags: dict[str, str] | None = None,
) -> str:
    """Copy dataset from GCS using gsutil with parallel transfers."""
    if source_tags:
        raise NotImplementedError(
            "Tag-filtered copy is not supported for GCS sources. "
            "Download the dataset locally first, then ingest from a local path."
        )
    logger.info("  Copy method: gsutil (parallel GCS transfer)")

    Path(dest_path).mkdir(parents=True, exist_ok=True)

    _try_copy_config_json(source_path, dest_path)

    if source_groups:
        logger.info(f"  Copying only groups: {source_groups}")
        for group in source_groups:
            group_src = f"{source_path}/windows/{group}"
            group_dst_parent = f"{dest_path}/windows"
            Path(group_dst_parent).mkdir(parents=True, exist_ok=True)
            logger.info(f"  Running: gsutil -m cp -r {group_src} {group_dst_parent}")
            subprocess.run(  # nosec B603 B607
                ["gsutil", "-m", "cp", "-r", group_src, group_dst_parent], check=True
            )
    else:
        windows_src = f"{source_path}/windows"
        logger.info(f"  Running: gsutil -m cp -r {windows_src} {dest_path}")
        subprocess.run(["gsutil", "-m", "cp", "-r", windows_src, dest_path], check=True)  # nosec B603 B607

    logger.info("  gsutil copy complete")
    return dest_path


def _copy_from_gcs_tar(
    source_path: str,
    dest_path: str,
) -> str:
    """Download a .tar.gz archive from GCS and extract it to dest_path."""
    logger.info("  Copy method: gsutil download + tar extract")

    Path(dest_path).mkdir(parents=True, exist_ok=True)

    archive_name = Path(source_path).name
    local_archive = Path(dest_path) / archive_name

    logger.info(f"  Downloading {source_path} -> {local_archive}")
    subprocess.run(["gsutil", "cp", source_path, str(local_archive)], check=True)  # nosec B603 B607

    logger.info(f"  Extracting {local_archive} -> {dest_path}")
    subprocess.run(["tar", "xzf", str(local_archive), "-C", dest_path], check=True)  # nosec B603 B607

    logger.info(f"  Removing archive {local_archive}")
    local_archive.unlink()

    entries = [p for p in Path(dest_path).iterdir() if p.name != archive_name]
    if len(entries) == 1 and entries[0].is_dir():
        dataset_path = str(entries[0])
        logger.info(f"  Extracted dataset directory: {dataset_path}")
        return dataset_path

    logger.info("  tar extract complete")
    return dest_path


def _tar_copy_cmd(src: str, dst: str, use_pv: bool) -> str:
    """Build a streaming tar copy command, optionally with pv progress."""
    quoted_src = shlex.quote(src)
    quoted_dst = shlex.quote(dst)
    if use_pv:
        return f"tar cf - -C {quoted_src} . | pv | tar xf - -C {quoted_dst}"
    return f"tar cf - -C {quoted_src} . | tar xf - -C {quoted_dst}"


def _window_matches_tags(
    window_metadata_path: Path,
    source_tags: dict[str, str],
) -> bool:
    """Check whether a window's metadata.json matches all required tags."""
    try:
        with open(window_metadata_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    options = meta.get("options", {})
    return isinstance(options, dict) and tags_match_options(options, source_tags)


def _collect_matching_windows(
    source_path: str,
    source_groups: list[str] | None,
    source_tags: dict[str, str],
) -> list[tuple[str, str]]:
    """Scan source windows and return (group, window_name) pairs matching tags."""
    windows_dir = Path(source_path) / "windows"
    if not windows_dir.exists():
        return []

    groups = source_groups or [d.name for d in windows_dir.iterdir() if d.is_dir()]
    logger.info("  Scanning groups: %s", groups)

    all_window_dirs: list[tuple[str, Path]] = []
    for group in groups:
        group_dir = windows_dir / group
        if not group_dir.is_dir():
            continue
        for window_dir in group_dir.iterdir():
            if window_dir.is_dir():
                all_window_dirs.append((group, window_dir))

    matched: list[tuple[str, str]] = []
    pbar = tqdm(all_window_dirs, desc="Scanning windows for tags", unit="win")
    for group, window_dir in pbar:
        meta_path = window_dir / "metadata.json"
        if meta_path.exists() and _window_matches_tags(meta_path, source_tags):
            matched.append((group, window_dir.name))
        pbar.set_postfix(matched=len(matched))
    pbar.close()

    logger.info(
        "  Tag scan complete: %d/%d windows matched tags %s",
        len(matched),
        len(all_window_dirs),
        source_tags,
    )
    return matched


def _copy_filtered_windows(
    source_path: str,
    dest_path: str,
    matched_windows: list[tuple[str, str]],
) -> None:
    """Copy only the matched windows from source to destination."""
    num_workers = int(os.environ.get("OLMOEARTH_INGEST_WORKERS", "8"))
    total = len(matched_windows)
    logger.info("  Copying %d matched windows (workers=%d)...", total, num_workers)

    def _copy_one(group: str, wname: str) -> str:
        src = Path(source_path) / "windows" / group / wname
        dst = Path(dest_path) / "windows" / group / wname
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(dst))
        return wname

    pbar = tqdm(total=total, desc="Copying windows", unit="win")
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [
            pool.submit(_copy_one, group, wname) for group, wname in matched_windows
        ]
        for future in as_completed(futures):
            future.result()
            pbar.update(1)
    pbar.close()

    logger.info("  Finished copying %d windows", total)


def _copy_local(
    source_path: str,
    dest_path: str,
    source_groups: list[str] | None = None,
    source_tags: dict[str, str] | None = None,
) -> str:
    """Copy dataset locally using streaming tar pipe."""
    if dest_path.startswith("/weka") and not _check_weka_exists():
        raise RuntimeError(
            "Weka filesystem path /weka does not exist. "
            "Cannot copy dataset. Ensure Weka is available before running ingestion."
        )

    Path(dest_path).mkdir(parents=True, exist_ok=True)

    _try_copy_config_json(source_path, dest_path)

    if source_tags:
        logger.info("  Copy method: tag-filtered per-window copy")
        matched = _collect_matching_windows(source_path, source_groups, source_tags)
        if not matched:
            raise ValueError(
                f"No windows in {source_path} matched tags {source_tags}. "
                "Check that the tag key/values are correct."
            )
        _copy_filtered_windows(source_path, dest_path, matched)
    elif source_groups:
        has_pv = shutil.which("pv") is not None
        logger.info(
            "  Copy method: streaming tar pipe%s",
            " (with pv progress)" if has_pv else "",
        )
        logger.info(f"  Copying only groups: {source_groups}")
        for group in source_groups:
            group_src = f"{source_path}/windows/{group}"
            group_dst = f"{dest_path}/windows/{group}"
            Path(group_dst).mkdir(parents=True, exist_ok=True)

            cmd = _tar_copy_cmd(group_src, group_dst, has_pv)
            logger.info(f"  Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)  # nosec B602
            logger.info(f"  Copied group '{group}'")
    else:
        has_pv = shutil.which("pv") is not None
        logger.info(
            "  Copy method: streaming tar pipe%s",
            " (with pv progress)" if has_pv else "",
        )
        cmd = _tar_copy_cmd(source_path, dest_path, has_pv)
        logger.info(f"  Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)  # nosec B602
        logger.info("  Copy complete")

    return dest_path


def _copy_generic(
    source_path: str,
    dest_path: str,
    source_groups: list[str] | None = None,
    source_tags: dict[str, str] | None = None,
) -> str:
    """Fallback copy using UPath for unknown storage backends."""
    logger.info("  Copy method: UPath (generic, sequential)")

    source = UPath(source_path)
    dest = UPath(dest_path)

    dest.mkdir(parents=True, exist_ok=True)

    _try_copy_config_json(source_path, dest_path)

    if source_tags:
        logger.info("  Using tag-filtered copy (generic)")
        matched = _collect_matching_windows(source_path, source_groups, source_tags)
        if not matched:
            raise ValueError(f"No windows in {source_path} matched tags {source_tags}.")
        for group, wname in matched:
            _copy_directory_recursive(
                source / "windows" / group / wname,
                dest / "windows" / group / wname,
            )
        logger.info("  Copied %d matched windows", len(matched))
        return dest_path

    windows_src = source / "windows"
    windows_dst = dest / "windows"
    if windows_src.exists():
        if source_groups:
            logger.info(f"    Copying only groups: {source_groups}")
            for group in source_groups:
                group_src = windows_src / group
                group_dst = windows_dst / group
                if group_src.exists():
                    _copy_directory_recursive(group_src, group_dst)
                    logger.info(f"    Copied group '{group}'")
        else:
            _copy_directory_recursive(windows_src, windows_dst)
            logger.info("    Copied windows directory")

    return dest_path


def _copy_directory_recursive(src: UPath, dst: UPath) -> None:
    """Recursively copy a directory using UPath."""
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.is_dir():
            _copy_directory_recursive(item, dst / item.name)
        else:
            with item.open("rb") as f:
                data = f.read()
            with (dst / item.name).open("wb") as f:
                f.write(data)


def _resolve_dataset_root(path: str) -> str:
    """Find the actual rslearn dataset root within a directory."""
    p = Path(path)
    if (p / "windows").exists():
        return path
    subdirs = [
        d for d in p.iterdir() if d.is_dir() and d.name != ".rslearn_dataset_index"
    ]
    if len(subdirs) == 1 and (subdirs[0] / "windows").exists():
        resolved = str(subdirs[0])
        logger.info(f"  Resolved dataset root to nested directory: {resolved}")
        return resolved
    return path


def copy_dataset(
    source_path: str,
    name: str,
    source_groups: list[str] | None = None,
    source_tags: dict[str, str] | None = None,
    untar_source: bool = False,
) -> str:
    """Copy an rslearn dataset to the configured eval dataset location."""
    dest_path = f"{EVAL_DATASETS_BASE_PATH}/{name}"

    logger.info("=== Dataset Copy ===")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Destination: {dest_path}")
    if source_tags:
        logger.info(f"  Filtering to tags: {source_tags}")
    if source_groups:
        logger.info(f"  Filtering to groups: {source_groups}")
    if not source_groups and not source_tags:
        logger.info("  Copying all groups (no tag/group filter)")

    if Path(dest_path).exists():
        if source_groups or source_tags:
            raise ValueError(
                f"Destination {dest_path} already exists, but this ingest requested "
                "source group/tag filters. Refusing to reuse an existing copy because "
                "it may have been created with different filters."
            )
        logger.warning("  Destination already exists, skipping copy...")
        return _resolve_dataset_root(dest_path)

    if untar_source and source_path.startswith("gs://"):
        actual_path = _copy_from_gcs_tar(source_path, dest_path)
    elif source_path.startswith("gs://"):
        actual_path = _copy_from_gcs(source_path, dest_path, source_groups, source_tags)
    elif source_path.startswith("/weka") or source_path.startswith("/"):
        actual_path = _copy_local(source_path, dest_path, source_groups, source_tags)
    else:
        actual_path = _copy_generic(source_path, dest_path, source_groups, source_tags)

    logger.info(f"  Dataset copy complete: {actual_path}")
    return actual_path
