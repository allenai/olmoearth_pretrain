"""Unit tests for the Tessera v2 export pipeline (pure logic)."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from olmoearth_pretrain.evals.datasets.tessera_v2_export import (
    DATASETS,
    FETCH_LAYERS,
    FETCH_LAYERS_CONFIG,
    TESSERA_S2_INDICES,
    check_names_unique,
    fetch_time_range,
    resolve_spec,
    s1_raw_to_tessera_units,
    scl_to_valid_mask,
    write_fetch_config,
)
from olmoearth_pretrain.evals.models.tessera.tessera_v2_infer import (
    encode_tile,
    get_bin_size,
)
from olmoearth_pretrain.evals.models.tessera.tessera_v2_model import (
    S2_BAND_ORDER,
    PixelStudent,
    count_params,
)


def test_scl_to_valid_mask() -> None:
    """SCL nodata/saturated/dark/shadow/cloud are invalid; cirrus is valid."""
    scl = np.arange(12)
    mask = scl_to_valid_mask(scl)
    assert mask.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]


def test_s1_raw_to_tessera_units() -> None:
    """Replicates their (20*log10(raw)+50)*200 int16 encoding with 0 sentinel."""
    raw = np.array([0.06, 1.0, 0.0, -1.0, np.nan, np.inf, 1e6], dtype=np.float32)
    out = s1_raw_to_tessera_units(raw)
    assert out.dtype == np.float32
    # 20*log10(0.06) = -24.437 -> (50 - 24.437) * 200 = 5112 (int16-rounded).
    assert out[0] == 5112.0
    assert out[1] == 10000.0
    # Non-positive / non-finite raw values become the 0 missing sentinel.
    assert out[2] == out[3] == out[4] == 0.0
    # Clipped to int16 range.
    assert out[6] == 32767.0


def test_tessera_s2_band_order() -> None:
    """The reorder indices map config band order onto the model's band order."""
    concat_order = [
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
    ]
    assert [concat_order[i] for i in TESSERA_S2_INDICES] == S2_BAND_ORDER


def test_fetch_layers_match_the_pastis_config() -> None:
    """The shared fetch config matches pastis's inline copy, bar max_matches.

    pastis_rslearn declares these layers inline (they predate the shared
    file). If the two drift, a v2 export on another dataset stops mirroring
    what pastis was built from.

    max_matches is the one deliberate difference: pastis was fetched at the
    original 150 for S2, which truncated 4.94% of ethiopia_crops windows (and
    from the END of the year, since the layer sorts datetime ascending), so
    the shared config raises it. Everything else must stay identical.
    """
    shared = json.loads(FETCH_LAYERS_CONFIG.read_text())["layers"]
    pastis = json.loads(
        (Path(FETCH_LAYERS_CONFIG).parent / "config_pastis_rslearn.json").read_text()
    )["layers"]
    assert sorted(shared) == sorted(FETCH_LAYERS)
    for name in FETCH_LAYERS:
        a, b = copy.deepcopy(shared[name]), copy.deepcopy(pastis[name])
        for layer in (a, b):
            layer["data_source"]["query_config"].pop("max_matches")
        assert a == b, name
    assert (
        shared["sentinel2_l2a_all"]["data_source"]["query_config"]["max_matches"]
        > (pastis["sentinel2_l2a_all"]["data_source"]["query_config"]["max_matches"])
    )


def _window(
    name: str = "w", group: str = "g", time_range: tuple | None = None
) -> SimpleNamespace:
    """Minimal stand-in for an rslearn Window."""
    return SimpleNamespace(name=name, group=group, time_range=time_range)


def _year_range(start: int, end: int) -> tuple[datetime, datetime]:
    """Calendar range (Jan 1 start, Jan 1 end)."""
    return (datetime(start, 1, 1, tzinfo=UTC), datetime(end, 1, 1, tzinfo=UTC))


def test_fetch_time_range_reads_the_windows_own_year() -> None:
    """A re-anchored window supplies its own product year."""
    assert fetch_time_range(_window(time_range=_year_range(2020, 2021)), None) == (
        _year_range(2020, 2021)
    )


def test_fetch_time_range_rejects_non_calendar_windows() -> None:
    """A Sept-Sept window is not a product year: report, do not guess."""
    sept = (datetime(2018, 9, 1, tzinfo=UTC), datetime(2019, 8, 27, tzinfo=UTC))
    reason = fetch_time_range(_window(time_range=sept), None)
    assert isinstance(reason, str) and "not a calendar year" in reason
    assert isinstance(fetch_time_range(_window(), None), str)
    # ... unless the year is pinned, which is what pastis does.
    assert fetch_time_range(_window(time_range=sept), 2019) == _year_range(2019, 2020)


def test_fetch_time_range_rejects_multi_year_windows() -> None:
    """(Jan 1 2019, Jan 1 2021) is calendar-aligned but is not one year."""
    reason = fetch_time_range(_window(time_range=_year_range(2019, 2021)), None)
    assert isinstance(reason, str) and "!= 1 year" in reason


def test_check_names_unique_flags_cross_group_collisions() -> None:
    """Infer matches fetch windows by name, so names must not repeat."""
    check_names_unique([_window("a", "g1"), _window("b", "g2")])
    with pytest.raises(SystemExit, match="occur in more than one group"):
        check_names_unique([_window("a", "g1"), _window("a", "g2")])


def test_resolve_spec_presets_and_overrides() -> None:
    """Presets pin pastis to 2019 and leave year-aligned datasets per-window."""
    assert resolve_spec("pastis_rslearn").year == 2019
    assert resolve_spec("africa_crop_mask_year_aligned").year is None
    assert (
        resolve_spec("ethiopia_crops_year_aligned").fetch_group
        == "ethiopia_crops_tessera_v2"
    )
    # Overrides must not leak back into the preset table.
    assert resolve_spec("pastis_rslearn", year=2020).year == 2020
    assert DATASETS["pastis_rslearn"].year == 2019
    with pytest.raises(SystemExit, match="Unknown --dataset"):
        resolve_spec("nope")
    with pytest.raises(SystemExit, match="pass --dataset or --fetch_group"):
        resolve_spec(None)


def test_write_fetch_config_keeps_storage_and_leaves_the_dataset_alone(
    tmp_path: Path,
) -> None:
    """Only the layers are replaced, and config.json is never touched."""
    config = tmp_path / "config.json"
    storage = {"class_path": "rslearn.dataset.storage.sqlite.SQLiteWindowStorage"}
    original = {"layers": {"gse": {"type": "raster"}}, "storage": storage}
    config.write_text(json.dumps(original))

    out = write_fetch_config(str(tmp_path))
    written = json.loads(Path(str(out)).read_text())
    # rslearn instantiates the window storage from whatever config it is
    # handed, so the fetch config has to carry the dataset's own.
    assert written["storage"] == storage
    assert set(written["layers"]) == set(FETCH_LAYERS)
    assert json.loads(config.read_text()) == original


def test_student_param_counts_match_model_cards() -> None:
    """Vendored architecture reproduces the published student sizes exactly."""
    sizes = {
        (36, 2, 384): 1_066_402,  # nano
        (64, 4, 1024): 7_112_322,  # small
        (110, 4, 1792): 21_031_506,  # medium
        (160, 4, 2560): 43_831_170,  # large
    }
    for (latent, layers, ffn), expected in sizes.items():
        model = PixelStudent(latent_dim=latent, num_layers=layers, dim_feedforward=ffn)
        assert count_params(model) == expected


def test_bin_sizes() -> None:
    """Valid-observation counts bucketize to the {8,...,256} ladder."""
    assert get_bin_size(0) == 0
    assert get_bin_size(1) == 8
    assert get_bin_size(43) == 48
    assert get_bin_size(135) == 136
    assert get_bin_size(500) == 256


def test_encode_tile_shapes_and_mask_handling() -> None:
    """encode_tile runs the full bucketize/pad/forward path on a tiny tile."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = PixelStudent(latent_dim=8, num_layers=1, nhead=2, dim_feedforward=16)
    model.eval()

    t_s2, t_s1, h, w = 5, 4, 3, 3
    s2 = rng.uniform(0, 4000, (t_s2, h, w, 10)).astype(np.float32)
    masks = np.ones((t_s2, h, w), dtype=np.uint8)
    masks[:, 0, 0] = 0  # one fully-cloudy pixel: s2_bin drops to 0
    s1 = rng.uniform(2000, 8000, (t_s1, h, w, 2)).astype(np.float32)

    out = encode_tile(
        model,
        s2_bands=s2,
        s2_doys=np.array([10, 60, 150, 220, 300]),
        s2_masks=masks,
        s1_asc_bands=s1,
        s1_asc_doys=np.array([15, 105, 195, 285]),
        s1_desc_bands=None,
        s1_desc_doys=None,
        batch_pixels=4,
        device=torch.device("cpu"),
    )
    assert out.shape == (h, w, 128)
    assert np.isfinite(out).all()
    # The final non-affine LayerNorm makes every pixel mean-0/std-1.
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-4)
