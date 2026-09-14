"""Process SwissCrop25 (https://huggingface.co/datasets/EOA-team/SwissCrop25) into OlmoEarth eval tensors.

SwissCrop25 ships 128x128 px Sentinel-2 L2A cubes (10 m, UTM 32N) with a ~2-3 day revisit,
a CloudSEN12+ mask, and per-parcel crop labels rasterised as per-LNF-code coverage fractions.
OlmoEarth consumes at most 12 timesteps, so each cube-year is reduced to 12 monthly
cloud-aware mosaics in the pretraining style (observations in a month ranked by clear
fraction, pixels filled from the clearest observation down). Labels follow the SwissCrop25
loader: coverage fractions are summed per target class after mapping LNF codes to the
non-excluded ``Crop_Label`` taxonomy (70 classes: 65 crops + 5 non-crop land cover), then
argmax; pixels with no coverage are background (0).

Input layout (Hugging Face repo, downloaded shards only):
    sentinel2/{year}/{year}_{shard}.tar   tar of <tile>.zarr.zip (bands s2_B01..s2_B12 (no B10),
                                          s2_mask, time = days since 1 Jan)
    labels/{year}.tar                     tar of <tile>.zarr.zip (lnf_code: (n_codes, 128, 128), band: codes)
    metadata/crop_classes.csv             LNF_code -> Crop_Label (+ Exclude flag)

Output layout:
    {out}/classes.json                    {"classes": [name_1..name_70], "lnf_to_class": {code: idx}}
    {out}/{year}/{tile}.pt                {"s2": uint16 [12, 12, 128, 128] (T, C in OlmoEarth band order),
                                           "months": int64 [12] yyyymm, "n_obs": int64 [12],
                                           "target": int16 [128, 128] (0 = background, 1..70 classes)}
"""

import argparse
import glob
import io
import json
import logging
import os
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import fsspec
import numpy as np
import pandas as pd
import torch
import zarr

logger = logging.getLogger(__name__)

# OlmoEarth Sentinel-2 band order (Modality.SENTINEL2_L2A.band_order). All are present in SwissCrop25.
S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]
CLEAR = 0  # s2_mask: 0 clear, 1 cloud, 2 shadow, 3 snow, 4 null
NODATA = 65535  # "missing data fill value" in the cube attrs
TILE_RE = re.compile(r"(S2_\d+_\d+)_\d{8}_\d{8}\.zarr")


def tile_key(name: str) -> str:
    """Location key shared across years, e.g. S2_262620_5110700."""
    m = TILE_RE.search(name)
    if m is None:
        raise ValueError(f"unexpected tile name {name}")
    return m.group(1)


def open_zarr_zip(data: bytes) -> zarr.Group:
    """Open a zipped zarr v2 group from bytes (SwissCrop25 tar members)."""
    mapper = fsspec.filesystem("zip", fo=io.BytesIO(data)).get_mapper("")
    try:
        return zarr.open_consolidated(mapper, mode="r", zarr_format=2)
    except Exception:
        return zarr.open_group(mapper, mode="r", zarr_format=2)


def build_class_mapping(csv_path: str) -> tuple[list[str], dict[int, int]]:
    """LNF code -> class index (1-based, alphabetical Crop_Label of non-excluded rows; 0 = background)."""
    df = pd.read_csv(csv_path)
    df = df[df["Exclude"] != True]  # noqa: E712
    df = df.dropna(subset=["Crop_Label"])
    classes = sorted(df["Crop_Label"].unique())
    idx = {name: i + 1 for i, name in enumerate(classes)}
    lnf_to_class = {int(r.LNF_code): idx[r.Crop_Label] for r in df.itertuples()}
    return classes, lnf_to_class


def label_from_group(
    g: zarr.Group, lnf_to_class: dict[int, int], num_classes: int
) -> np.ndarray:
    """Aggregate per-LNF coverage fractions by class and argmax (SwissCrop25 loader convention)."""
    fr = np.asarray(g["lnf_code"][:], dtype=np.float32)
    codes = np.asarray(g["band"][:]).astype(int)
    if fr.ndim == 2:  # legacy single-band format
        return np.vectorize(lambda c: lnf_to_class.get(int(c), 0))(
            fr.astype(int)
        ).astype(np.int16)
    agg = np.zeros((num_classes + 1,) + fr.shape[1:], dtype=np.float32)
    for band_idx, code in enumerate(codes):
        agg[lnf_to_class.get(int(code), 0)] += fr[band_idx]
    # Background (code 0 / excluded codes) competes in the argmax, as in the SwissCrop25 loader.
    return np.argmax(agg, axis=0).astype(np.int16)


def monthly_mosaics(
    g: zarr.Group, year: int, offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """12 cloud-aware monthly mosaics: (12, C, H, W) uint16, yyyymm (12,), n_obs (12,)."""
    days = np.asarray(g["time"][:]).astype(int)
    units = str(g["time"].attrs.get("units", f"days since {year}-01-01"))
    m_units = re.search(r"since (\d{4}-\d{2}-\d{2})", units)
    if m_units is None:
        raise ValueError(f"unparseable time units: {units!r}")
    base = np.datetime64(m_units.group(1))
    dates = base + days.astype("timedelta64[D]")
    months = dates.astype("datetime64[M]").astype(int) % 12  # 0..11
    bands = np.stack(
        [np.asarray(g[f"s2_{b}"][:]) for b in S2_BANDS], axis=1
    )  # (T, C, H, W) uint16
    mask = np.asarray(
        g["s2_mask"][:]
    ).copy()  # (T, H, W): 0 clear, 1 cloud, 2 shadow, 3 snow, 4 null
    mask[(bands == NODATA).any(axis=1)] = 4  # fill value in any band -> not clear
    clear_frac = (mask == CLEAR).reshape(len(days), -1).mean(1)
    if offset:
        bands = np.clip(bands.astype(np.int32) - offset, 0, None).astype(np.uint16)
    bands[bands == NODATA] = 0
    T, C, H, W = bands.shape
    out = np.zeros((12, C, H, W), dtype=np.uint16)
    n_obs = np.zeros(12, dtype=np.int64)
    for m in range(12):
        idx = np.where(months == m)[0]
        n_obs[m] = len(idx)
        if len(idx) == 0:
            continue
        idx = idx[np.argsort(-clear_frac[idx], kind="stable")]
        best = idx[0]
        comp = bands[best].copy()
        filled = mask[best] == CLEAR
        for i in idx[1:]:
            if filled.all():
                break
            sel = (~filled) & (mask[i] == CLEAR)
            comp[:, sel] = bands[i][:, sel]
            filled |= sel
        out[m] = comp
    yyyymm = np.array([year * 100 + m + 1 for m in range(12)], dtype=np.int64)
    return out, yyyymm, n_obs


def load_label_members(label_tar: str) -> dict[str, tuple[int, int]]:
    """Tile key -> (offset, size) of the tile's .zarr.zip inside the label tar."""
    members = {}
    with tarfile.open(label_tar) as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".zarr.zip"):
                members[tile_key(os.path.basename(m.name))] = (m.offset_data, m.size)
    return members


def process_shard(
    shard: str,
    year: int,
    label_tar: str,
    label_members: dict,
    lnf_to_class: dict,
    num_classes: int,
    out_dir: str,
    offset: int,
    min_months: int,
) -> dict:
    """Process every tile in one S2 shard; returns counts."""
    os.makedirs(out_dir, exist_ok=True)
    stats = {"tiles": 0, "written": 0, "no_label": 0, "few_months": 0, "errors": 0}
    with tarfile.open(shard) as tf, open(label_tar, "rb") as lf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".zarr.zip")):
                continue
            stats["tiles"] += 1
            key = tile_key(os.path.basename(m.name))
            out_path = os.path.join(out_dir, f"{key}.pt")
            if os.path.exists(out_path):
                stats["written"] += 1
                continue
            if key not in label_members:
                stats["no_label"] += 1
                continue
            try:
                fobj = tf.extractfile(m)
                if fobj is None:
                    raise ValueError(f"not a regular tar member: {m.name}")
                g = open_zarr_zip(fobj.read())
                s2, yyyymm, n_obs = monthly_mosaics(g, year, offset)
                if (n_obs > 0).sum() < min_months:
                    stats["few_months"] += 1
                    continue
                off, size = label_members[key]
                lf.seek(off)
                lg = open_zarr_zip(lf.read(size))
                target = label_from_group(lg, lnf_to_class, num_classes)
                torch.save(
                    {
                        "s2": torch.from_numpy(s2),
                        "months": torch.from_numpy(yyyymm),
                        "n_obs": torch.from_numpy(n_obs),
                        "target": torch.from_numpy(target),
                    },
                    out_path,
                )
                stats["written"] += 1
            except Exception as e:  # keep going; report at the end
                stats["errors"] += 1
                logger.warning(f"{shard} {key}: {type(e).__name__}: {e}")
    return stats


def main() -> None:
    """CLI: process downloaded SwissCrop25 shards into per-cube-year tensors."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir", default="/weka/dfive-default/presto_eval_sets/swisscrop25/raw"
    )
    ap.add_argument("--out-dir", default="/data/swisscrop25/processed")
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021, 2022, 2023, 2024, 2025],
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--min-months",
        type=int,
        default=12,
        help="drop cube-years with fewer observed months",
    )
    ap.add_argument(
        "--offset-from-year",
        type=int,
        default=None,
        help="subtract the 1000 BOA_ADD_OFFSET for years >= this (only if the cubes still carry it)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    classes, lnf_to_class = build_class_mapping(
        os.path.join(args.data_dir, "metadata", "crop_classes.csv")
    )
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(
        {"classes": classes, "lnf_to_class": lnf_to_class, "s2_bands": S2_BANDS},
        open(os.path.join(args.out_dir, "classes.json"), "w"),
        indent=1,
    )
    logger.info(f"{len(classes)} classes; {len(lnf_to_class)} LNF codes mapped")

    jobs = []
    with ProcessPoolExecutor(args.workers) as ex:
        for year in args.years:
            shards = sorted(
                glob.glob(
                    os.path.join(args.data_dir, "sentinel2", str(year), f"{year}_*.tar")
                )
            )
            label_tar = os.path.join(args.data_dir, "labels", f"{year}.tar")
            if not shards or not os.path.exists(label_tar):
                logger.warning(
                    f"{year}: {len(shards)} shards, labels present={os.path.exists(label_tar)}; skipping"
                )
                continue
            members = load_label_members(label_tar)
            offset = (
                1000
                if (args.offset_from_year is not None and year >= args.offset_from_year)
                else 0
            )
            logger.info(
                f"{year}: {len(shards)} shards, {len(members)} labelled tiles, offset={offset}"
            )
            for s in shards:
                jobs.append(
                    ex.submit(
                        process_shard,
                        s,
                        year,
                        label_tar,
                        members,
                        lnf_to_class,
                        len(classes),
                        os.path.join(args.out_dir, str(year)),
                        offset,
                        args.min_months,
                    )
                )
        total: dict[str, int] = {}
        for f in as_completed(jobs):
            st = f.result()
            for k, v in st.items():
                total[k] = total.get(k, 0) + v
            logger.info(f"shard done: {st}")
    logger.info(f"TOTAL {total}")


if __name__ == "__main__":
    main()
