"""test_e5_masking_targets: which modalities the E5 recipes actually decode.

Pushes fixture allcap batches through the E5 arm configs (``e5_allcap_common``: training
modalities, dataset, and the dataloader's configured masking) and asserts which
modalities appear as DECODER targets, so a change to a recipe's inputs that silently
removes a sensor's latent target trips here.

* default (S1 + S2, v1.3 masking): S2 is a decode target in some but not all samples;
* ``E5_S2_ONLY=1`` + v1.3 masking: S2 is NEVER decoded (KNOWN FAILURE MODE: with one
  single-bandset sensor, ``random_time_with_decode`` makes it encode-only and only the
  maps are targets; see e5_allcap_ab/README.md, 2026-10-16);
* ``E5_S2_ONLY=1`` + ``E5_MIM=within_s2``: S2 is decoded in every sample.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from upath import UPath

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.internal.experiment import SubCmd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/official/v1_3/ablations"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5_allcap_common as e5  # noqa: E402
from test_allcap import _write_allcap_h5  # noqa: E402

N_FILES = 4
N_SAMPLES = 48
EXTRA_MAPS = {  # the four maps the allcap fixture lacks, [H, W, 1, C]
    "openstreetmap_raster": 30,
    "wri_canopy_height_map": 1,
    "cdl": 1,
    "worldcereal": 8,
}


@pytest.fixture(scope="module")
def e5_allcap_dir(tmp_path_factory: pytest.TempPathFactory) -> UPath:
    """Allcap-layout H5s carrying every E5 training modality."""
    d = tmp_path_factory.mktemp("e5") / "allcap" / str(N_FILES)
    d.mkdir(parents=True)
    rng = np.random.default_rng(11)
    for i in range(N_FILES):
        path = d / f"sample_{i}.h5"
        _write_allcap_h5(path, rng)
        with h5py.File(path, "a") as f:
            for name, c in EXTRA_MAPS.items():
                f.create_dataset(
                    name, data=rng.random((16, 16, 1, c)).astype(np.float32)
                )
    mods = [
        "sentinel2_l2a",
        "sentinel1",
        "landsat_l2",
        "worldcover",
        "srtm",
        *EXTRA_MAPS,
    ]
    pd.DataFrame(
        [{"sample_index": i, **{m: 1 for m in mods}} for i in range(N_FILES)]
    ).to_csv(d / "sample_metadata.csv", index=False)
    np.save(d / "latlon_distribution.npy", np.zeros((N_FILES, 2)))
    return UPath(d)


def _s2_decoded_per_sample(
    h5_dir: UPath, monkeypatch: pytest.MonkeyPatch, env: dict[str, str], selection: str
) -> tuple[torch.Tensor, list[str], object]:
    for k in ("E5_S2_ONLY", "E5_MIM"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    common = e5.build_common_components("t", SubCmd.dry_run, "t", "local", [])
    strategy = e5.build_dataloader_config(common).masking_config.build()
    ds = OlmoEarthDataset(
        h5py_dir=h5_dir,
        training_modalities=common.training_modalities,
        dtype=np.float32,
        normalize=False,
        allcap_selection=selection,
        allcap_cloud_tau=0.5,
    )
    ds.prepare()
    np.random.seed(0)
    torch.manual_seed(0)
    items = [
        ds[
            GetItemArgs(
                idx=i % N_FILES, patch_size=1, sampled_hw_p=4, token_budget=None
            )
        ]
        for i in range(N_SAMPLES)
    ]
    _, masked = collate_single_masked_batched(items, None, strategy)
    s2 = masked.sentinel2_l2a_mask.reshape(N_SAMPLES, -1)
    for m in ("worldcover", "srtm", *EXTRA_MAPS):
        mk = getattr(masked, f"{m}_mask")
        assert ((mk == MaskValue.DECODER.value) | (mk == MaskValue.MISSING.value)).all()
    return (s2 == MaskValue.DECODER.value).any(1), common.training_modalities, masked


SELECTIONS = [e5.ALLCAP_LEAST_CLOUDY_SLOT, e5.ALLCAP_RANDOM_CLEAR]


@pytest.mark.parametrize("selection", SELECTIONS)
def test_e5_masking_targets_default_s1_s2(
    e5_allcap_dir: UPath, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    """Default E5 (S1 + S2, v1.3 masking): S2 is a latent target in some samples."""
    dec, mods, masked = _s2_decoded_per_sample(
        e5_allcap_dir, monkeypatch, {}, selection
    )
    assert "sentinel1" in mods and "landsat" not in mods
    frac = dec.float().mean().item()
    assert 0.2 < frac < 0.8, frac  # cross-sensor split: ~half the samples
    s1 = masked.sentinel1_mask.reshape(N_SAMPLES, -1)
    s1_dec = (s1 == MaskValue.DECODER.value).any(1)
    # A sensor is never both input and target in one sample; S1 is the target otherwise.
    assert not (dec & s1_dec).any()
    assert (dec | s1_dec).float().mean().item() > 0.9


@pytest.mark.parametrize("selection", SELECTIONS)
def test_e5_masking_targets_s2_only_v13_known_failure(
    e5_allcap_dir: UPath, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    """KNOWN FAILURE MODE: S2-only + v1.3 masking never decodes S2 (maps-only targets)."""
    dec, mods, _ = _s2_decoded_per_sample(
        e5_allcap_dir, monkeypatch, {"E5_S2_ONLY": "1"}, selection
    )
    assert "sentinel1" not in mods
    assert not dec.any()


@pytest.mark.parametrize("selection", SELECTIONS)
def test_e5_masking_targets_s2_only_within_s2(
    e5_allcap_dir: UPath, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    """S2-only + within_s2 (the canary fallback): S2 is decoded in every sample."""
    dec, _, _ = _s2_decoded_per_sample(
        e5_allcap_dir,
        monkeypatch,
        {"E5_S2_ONLY": "1", "E5_MIM": "within_s2"},
        selection,
    )
    assert dec.all()


def test_e5_within_s2_requires_s2_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """within_s2 with S1 + S2 is rejected (it would only alter the S1-less samples)."""
    monkeypatch.delenv("E5_S2_ONLY", raising=False)
    monkeypatch.setenv("E5_MIM", "within_s2")
    with pytest.raises(ValueError):
        e5.build_common_components("t", SubCmd.dry_run, "t", "local", [])
