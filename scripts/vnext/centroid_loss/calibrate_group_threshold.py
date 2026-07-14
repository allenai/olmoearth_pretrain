"""Calibrate group_threshold for the centroid-contrastive loss.

Builds a random-init v1.2 baseline encoder (distributionally equivalent to
the frozen-projection target encoder), runs real H5 samples through the
exit-at-0 target path, and reports target-grouping statistics per modality
at a grid of thresholds: groups per sample, largest-group fraction,
singleton fraction, and the fraction of degenerate (<2 groups) samples.

Run: PYTHONPATH=. python scripts/vnext/centroid_loss/calibrate_group_threshold.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDatasetConfig
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
)
MODALITIES = [
    "sentinel2_l2a",
    "sentinel1",
    "landsat",
    "worldcover",
    "srtm",
    "openstreetmap_raster",
    "wri_canopy_height_map",
    "cdl",
    "worldcereal",
]
THRESHOLDS = [0.5, 0.7, 0.9]
NUM_SAMPLES = 64


def build_target_encoder():
    """Random-init v1.2 base encoder (frozen-projection distribution)."""
    tok = TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(
                band_groups=[
                    [
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
                ]
            ),
            "landsat": ModalityTokenization(
                band_groups=[
                    ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
                ]
            ),
        }
    )
    return EncoderConfig(
        embedding_size=768,
        num_heads=12,
        depth=12,
        mlp_ratio=4.0,
        supported_modality_names=MODALITIES,
        max_patch_size=8,
        min_patch_size=1,
        drop_path=0.0,
        max_sequence_length=12,
        tokenization_config=tok,
        patch_embed_hidden_sizes=[64],
        position_encoding="rope_3d_mixed",
        rope_mixed_base=10000.0,
        rope_temporal_coordinate_scale=1.0 / 30.0,
    ).build()


def groups_stats(
    targets: torch.Tensor, threshold: float, center: bool = True
) -> tuple[int, int, float]:
    """(n_tokens, n_groups, largest_group_frac) for one sample's targets."""
    t = targets.float()
    if center:
        t = t - t.mean(dim=0, keepdim=True)
    t = F.normalize(t, dim=-1)
    n = t.shape[0]
    adj = (t @ t.T) >= threshold
    labels = torch.arange(n, device=t.device)
    for _ in range(64):
        new = torch.minimum(
            labels, torch.where(adj, labels.unsqueeze(0), n).min(dim=-1).values
        )
        if torch.equal(new, labels):
            break
        labels = new
    uniq, counts = labels.unique(return_counts=True)
    return n, len(uniq), counts.max().item() / n


def main() -> None:
    """Run the calibration and print per-modality stats per threshold."""
    torch.manual_seed(0)
    encoder = build_target_encoder().cuda().eval()
    ds = OlmoEarthDatasetConfig(h5py_dir=H5_DIR, training_modalities=MODALITIES).build()
    exit_cfg = {m: 0 for m in MODALITIES}

    per_mod: dict[str, dict[float, list]] = {
        m: {th: [] for th in THRESHOLDS} for m in MODALITIES
    }
    torch.manual_seed(1234)
    indices = torch.randint(0, 1_000_000, (NUM_SAMPLES,)).tolist()
    with torch.inference_mode():
        for i in indices:
            try:
                _, sample = ds[GetItemArgs(idx=i, patch_size=4, sampled_hw_p=8)]
            except Exception:  # noqa: BLE001  # nosec B112 - sparse corpus entries
                continue
            fields: dict = {}
            for name, val in sample.as_dict().items():
                if val is None:
                    continue
                v = torch.as_tensor(val).unsqueeze(0)
                if name == "timestamps":
                    fields[name] = v.long()
                else:
                    fields[name] = v.float()
                    fields[f"{name}_mask"] = torch.full(
                        v.shape, MaskValue.ONLINE_ENCODER.value, dtype=torch.long
                    )
            batch = MaskedOlmoEarthSample(**fields).to_device(torch.device("cuda"))
            out = encoder.forward(batch, patch_size=4, token_exit_cfg=exit_cfg)
            tam = out["tokens_and_masks"]
            for m in MODALITIES:
                tok = getattr(tam, m, None)
                if tok is None:
                    continue
                flat = tok.reshape(-1, tok.shape[-1])
                if flat.shape[0] < 4:
                    continue
                for th in THRESHOLDS:
                    per_mod[m][th].append(groups_stats(flat, th))

    print(
        f"\n{'modality':>24} {'th':>5} {'tok':>6} {'groups':>7} "
        f"{'compress':>9} {'maxgrp%':>8} {'G<2%':>6}"
    )
    for m in MODALITIES:
        for th in THRESHOLDS:
            rows = per_mod[m][th]
            if not rows:
                continue
            n = sum(r[0] for r in rows) / len(rows)
            g = sum(r[1] for r in rows) / len(rows)
            big = sum(r[2] for r in rows) / len(rows)
            degen = sum(1 for r in rows if r[1] < 2) / len(rows)
            print(
                f"{m:>24} {th:>5.2f} {n:>6.0f} {g:>7.1f} "
                f"{n / g:>8.1f}x {100 * big:>7.1f} {100 * degen:>5.1f}"
            )


if __name__ == "__main__":
    main()
