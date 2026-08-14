"""Tests for the per-pixel cloud-aware monthly compositor."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from rasterio.crs import CRS
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.evals.datasets import pixel_mosaic_export as pme
from olmoearth_pretrain.evals.datasets.tessera_v2_export import INVALID_SCL_CLASSES


class TestSeverityRanking:
    """The per-class ordering, which is new machinery in this repo."""

    def test_clear_tier_matches_tessera_minus_cirrus(self) -> None:
        """Rank 0 must mean what "clear" means to Tessera, so the arms agree."""
        clear = {c for c in range(12) if pme.SCL_SEVERITY[c] == pme.CLEAR_SEVERITY}
        assert clear == set(range(12)) - set(INVALID_SCL_CLASSES) - {10}

    def test_every_documented_class_is_ranked(self) -> None:
        """Every SCL class the product defines must have a rank."""
        ranked = {c for tier in pme.SCL_SEVERITY_TIERS for c in tier}
        assert ranked == set(range(12))

    def test_ordering_is_cloud_free_before_cloud(self) -> None:
        """Surface classes must outrank haze, and haze must outrank cloud."""
        assert pme.SCL_SEVERITY[4] < pme.SCL_SEVERITY[10] < pme.SCL_SEVERITY[3]
        assert pme.SCL_SEVERITY[8] < pme.SCL_SEVERITY[9] < pme.SCL_SEVERITY[0]

    def test_unknown_class_sorts_last(self) -> None:
        """A future class 12 must not silently rank as clear."""
        assert pme.SCL_SEVERITY[12] == pme.WORST_SEVERITY
        assert pme.SCL_SEVERITY[255] == pme.WORST_SEVERITY
        assert pme.WORST_SEVERITY > pme.SCL_SEVERITY[0]


class TestPeriodIndex:
    """Bucketing must reproduce the mo layers' 30d-at-0d..330d offsets."""

    def test_period_boundaries(self) -> None:
        """Period edges must land where the 30d/0d..330d offsets put them."""
        assert pme.period_index(np.array([1]))[0] == 0
        assert pme.period_index(np.array([30]))[0] == 0
        assert pme.period_index(np.array([31]))[0] == 1
        assert pme.period_index(np.array([360]))[0] == 11

    def test_tail_of_year_is_dropped(self) -> None:
        """Days 361+ fall in no mo layer, so they must fall in none here."""
        assert list(pme.period_index(np.array([361, 365, 366]))) == [-1, -1, -1]

    def test_every_period_reachable(self) -> None:
        """All twelve periods must be reachable, each covering 30 days."""
        doys = np.arange(1, 361)
        assert set(pme.period_index(doys).tolist()) == set(range(12))
        counts = np.bincount(pme.period_index(doys))
        assert (counts == pme.PERIOD_DAYS).all()

    def test_midpoint_inside_its_own_period(self) -> None:
        """The tie-break midpoint must fall inside the period it breaks ties for."""
        for period in range(pme.MONTHS):
            mid = pme.period_midpoint_doy(period)
            assert pme.period_index(np.array([int(mid)]))[0] == period


class TestSelectBest:
    """Per-pixel selection: tier first, then nearest the period midpoint."""

    def test_prefers_clear_over_cloud(self) -> None:
        """A clear observation must win regardless of its date."""
        severity = np.array([[[5]], [[0]]], dtype=np.uint8)  # (2,1,1)
        chosen = pme.select_best(severity, np.array([2, 20]), period=0)
        assert chosen[0, 0] == 1

    def test_falls_back_to_least_contaminated(self) -> None:
        """No clear observation: take the least bad, never a hole."""
        severity = np.array([[[7]], [[4]], [[6]]], dtype=np.uint8)
        chosen = pme.select_best(severity, np.array([2, 10, 20]), period=0)
        assert chosen[0, 0] == 1

    def test_tie_breaks_toward_period_midpoint(self) -> None:
        """Among equally clear observations, take the one nearest the midpoint."""
        severity = np.zeros((3, 1, 1), dtype=np.uint8)
        # period 0 midpoint is day 15.5; day 14 is nearest.
        chosen = pme.select_best(severity, np.array([2, 14, 29]), period=0)
        assert chosen[0, 0] == 1

    def test_full_tie_is_deterministic_earliest(self) -> None:
        """A complete tie must resolve the same way on every run."""
        severity = np.zeros((2, 1, 1), dtype=np.uint8)
        # Equidistant from the 15.5 midpoint -> argmin takes the first.
        chosen = pme.select_best(severity, np.array([15, 16]), period=0)
        assert chosen[0, 0] == 0

    def test_selection_is_per_pixel(self) -> None:
        """Different pixels may come from different acquisitions."""
        severity = np.array(
            [[[0, 5], [5, 5]], [[5, 0], [0, 0]]], dtype=np.uint8
        )  # (2,2,2)
        chosen = pme.select_best(severity, np.array([10, 20]), period=0)
        np.testing.assert_array_equal(chosen, np.array([[0, 1], [1, 1]]))


def _fake_reader(
    imagery: np.ndarray, scl: np.ndarray, doys: np.ndarray
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Stub _read_scenes, dispatching on which band sets were asked for."""

    def read(
        window: Any,
        layer_name: str,
        band_sets: Any,
        resampling: Any = None,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        if list(band_sets) == [pme.SCL_BAND_SET]:
            return scl, doys
        return imagery, doys

    return read


class TestCompositeWindow:
    """End-to-end composite for one window, with I/O severed."""

    @pytest.fixture
    def scenes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Three acquisitions over a 2x2 window; periods 0, 0, 1.

        Scene 0 is clear only at (1,1); scene 1 is clear only at (0,0) and
        cloud-medium elsewhere (so it still beats scene 0's cloud-high); scene 2
        is wholly clear and alone in period 1. Band values encode the scene, so
        the assertions read which acquisition each pixel came from.
        """
        n_bands = len(pme.composite_bands())
        imagery = np.stack(
            [
                np.full((2, 2, n_bands), (i + 1) * 100.0, dtype=np.float32)
                for i in range(3)
            ]
        )
        scl = np.empty((3, 2, 2, 1), dtype=np.float32)
        scl[0, ..., 0] = np.array([[9, 9], [9, 4]])
        scl[1, ..., 0] = np.array([[4, 8], [8, 8]])
        scl[2, ..., 0] = 4
        return imagery, scl, np.array([5, 10, 40])

    def test_picks_per_pixel_across_acquisitions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scenes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """The whole point: one period, different source scenes per pixel."""
        imagery, scl, doys = scenes
        monkeypatch.setattr(pme, "_read_scenes", _fake_reader(imagery, scl, doys))
        out_imagery, out_scl, centre_doy, hist = pme.composite_window(SimpleNamespace())

        first = out_imagery[pme.S2_LAYERS[0]]
        # (1,1) is the only pixel where scene 0 was the least contaminated.
        np.testing.assert_allclose(first[..., 0], np.array([[200, 200], [200, 100]]))
        np.testing.assert_array_equal(
            out_scl[pme.SCL_LAYERS[0]], np.array([[4, 8], [8, 4]])
        )
        # Two pixels landed on clear, two on cloud-medium.
        assert hist[pme.CLEAR_SEVERITY] == 2 + 4  # period 0 has 2, period 1 all 4

    def test_single_acquisition_passes_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scenes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """A period with one acquisition must be that acquisition, untouched."""
        imagery, scl, doys = scenes
        monkeypatch.setattr(pme, "_read_scenes", _fake_reader(imagery, scl, doys))
        out_imagery, out_scl, _, _ = pme.composite_window(SimpleNamespace())

        second = out_imagery[pme.S2_LAYERS[1]]
        assert (second[..., :10] == 300).all()
        assert (out_scl[pme.SCL_LAYERS[1]] == 4).all()

    def test_dtypes_and_shapes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scenes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Imagery is float32 (it carries negative fill) and SCL uint8."""
        imagery, scl, doys = scenes
        monkeypatch.setattr(pme, "_read_scenes", _fake_reader(imagery, scl, doys))
        out_imagery, out_scl, _, _ = pme.composite_window(SimpleNamespace())

        assert out_imagery[pme.S2_LAYERS[0]].dtype == np.float32
        assert out_imagery[pme.S2_LAYERS[0]].shape == (
            2,
            2,
            len(pme.composite_bands()),
        )
        assert out_scl[pme.SCL_LAYERS[0]].dtype == np.uint8
        assert out_scl[pme.SCL_LAYERS[0]].shape == (2, 2)

    def test_empty_periods_are_absent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scenes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Rslearn never materializes an empty month; nor do we."""
        imagery, scl, doys = scenes
        monkeypatch.setattr(pme, "_read_scenes", _fake_reader(imagery, scl, doys))
        out_imagery, out_scl, _, _ = pme.composite_window(SimpleNamespace())

        assert set(out_imagery) == {pme.S2_LAYERS[0], pme.S2_LAYERS[1]}
        assert set(out_scl) == {pme.SCL_LAYERS[0], pme.SCL_LAYERS[1]}

    def test_zero_scenes_is_a_coverage_gap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No S2 at all is an archive gap, not a failure."""
        empty = np.zeros((0, 2, 2, len(pme.composite_bands())), dtype=np.float32)
        monkeypatch.setattr(
            pme,
            "_read_scenes",
            _fake_reader(empty, np.zeros((0, 2, 2, 1), np.float32), np.zeros(0, int)),
        )
        with pytest.raises(pme.NoS2ScenesError):
            pme.composite_window(SimpleNamespace(group="g", name="w"))

    def test_scene_ordering_mismatch_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        scenes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Imagery and SCL must describe the same scenes, in the same order."""
        imagery, scl, doys = scenes

        def read(
            window: Any,
            layer_name: str,
            band_sets: Any,
            resampling: Any = None,
            **kwargs: Any,
        ) -> tuple[np.ndarray, np.ndarray]:
            if list(band_sets) == [pme.SCL_BAND_SET]:
                return scl, doys[::-1]
            return imagery, doys

        monkeypatch.setattr(pme, "_read_scenes", read)
        with pytest.raises(ValueError, match="ordering diverged"):
            pme.composite_window(SimpleNamespace())


class TestBandOrder:
    """The composite must land in the band order the eval config expects."""

    def test_matches_year_aligned_model_yaml(self) -> None:
        """The composite band order must equal the eval config bands list."""
        path = "data/rslearn_dataset_configs/ethiopia_crops_year_aligned/model.yaml"
        with open(path) as f:
            config = yaml.safe_load(f)
        bands = config["data"]["init_args"]["inputs"]["sentinel2_l2a"]["bands"]
        # The 12-band composite must match the eval config exactly. The 10-band
        # form (used on fetch groups that predate B01/B09) is its prefix, so the
        # band subset is the only difference.
        assert pme.composite_bands(include_60m=True) == bands
        assert bands[: len(pme.composite_bands())] == (
            pme.composite_bands(include_60m=False)
        )


class TestWriteComposite:
    """Writing to a real rslearn dataset on disk."""

    @staticmethod
    def _window(tmp_path: Path) -> Window:
        """A single 4x4 window in a throwaway rslearn dataset."""
        ds_path = UPath(tmp_path) / "dataset"
        ds_path.mkdir(parents=True)
        with (ds_path / "config.json").open("w") as f:
            json.dump({"layers": {}}, f)
        window = Window(
            storage=Dataset(ds_path).storage,
            group="default",
            name="w1",
            projection=Projection(CRS.from_epsg(32610), 10, -10),
            bounds=(0, 0, 4, 4),
            time_range=(
                datetime(2020, 1, 1, tzinfo=UTC),
                datetime(2021, 1, 1, tzinfo=UTC),
            ),
        )
        window.save()
        return window

    def _payload(self, periods: range) -> tuple[dict, dict, dict, list[str]]:
        """Imagery/SCL/centre-DOY for the given periods, plus the band list."""
        bands = pme.composite_bands()
        imagery = {
            pme.S2_LAYERS[p]: np.full((4, 4, len(bands)), 100 + p, dtype=np.float32)
            for p in periods
        }
        scl = {pme.SCL_LAYERS[p]: np.full((4, 4), 4, dtype=np.uint8) for p in periods}
        # A plausible selected acquisition: mid-period.
        centre_doy = {pme.S2_LAYERS[p]: pme.PERIOD_DAYS * p + 15 for p in periods}
        return imagery, scl, centre_doy, bands

    def test_every_period_lands_under_its_own_bands(self, tmp_path: Path) -> None:
        """Regression: a shadowed loop variable put periods 1-11 under ["SCL"]."""
        window = self._window(tmp_path)
        imagery, scl, centre_doy, bands = self._payload(range(pme.MONTHS))
        pme.write_composite(window, imagery, scl, centre_doy, bands)

        completed = {name for name, _ in window.list_completed_layers()}
        assert completed == set(pme.S2_LAYERS) | set(pme.SCL_LAYERS)
        for period in range(pme.MONTHS):
            s2_dir = window.get_raster_dir(pme.S2_LAYERS[period], bands)
            assert s2_dir.exists(), pme.S2_LAYERS[period]
            # The imagery must NOT have been filed under the SCL band dir.
            wrong = window.get_raster_dir(pme.S2_LAYERS[period], pme.SCL_BAND_SET)
            assert not wrong.exists(), pme.S2_LAYERS[period]

    def test_stamps_the_selected_acquisition_not_the_period(
        self, tmp_path: Path
    ) -> None:
        """The parent stamps the chosen scene's instant; we must match it.

        Verified against the parent 2026-08-14: its mo layers carry
        (acquisition, acquisition) zero-length ranges. Stamping the period start
        instead would shift this arm ~15 days early and give every window an
        identical time signal.
        """
        window = self._window(tmp_path)
        stamp = pme.acquisition_time_range(window, doy=17)
        assert stamp[0] == stamp[1], "must be a zero-length range like the parent"
        assert (stamp[0] - window.time_range[0]).days == 16  # doy 17 == Jan 17

        imagery, scl, centre_doy, bands = self._payload(range(2))
        pme.write_composite(window, imagery, scl, centre_doy, bands)
        r = GeotiffRasterFormat().decode_raster(
            window.get_raster_dir(pme.S2_LAYERS[1], bands),
            window.projection,
            window.bounds,
        )
        expected = pme.acquisition_time_range(window, centre_doy[pme.S2_LAYERS[1]])
        assert r.timestamps == [expected]

    def test_missing_periods_are_not_marked(self, tmp_path: Path) -> None:
        """A window with only some periods must not claim the rest."""
        window = self._window(tmp_path)
        imagery, scl, centre_doy, bands = self._payload(range(3))
        pme.write_composite(window, imagery, scl, centre_doy, bands)

        completed = {name for name, _ in window.list_completed_layers()}
        assert completed == set(pme.S2_LAYERS[:3]) | set(pme.SCL_LAYERS[:3])

    def test_no_layer_marked_before_all_rasters_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The resume guard reads period 0, so marking must come last.

        Otherwise an interrupted window is skipped forever, missing periods
        1-11.
        """
        window = self._window(tmp_path)
        imagery, scl, centre_doy, bands = self._payload(range(pme.MONTHS))
        calls: list[str] = []
        real = pme.GeotiffRasterFormat.encode_raster

        def encode(self_: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append("encode")
            if len(calls) == 5:
                raise RuntimeError("interrupted")
            return real(self_, *args, **kwargs)

        monkeypatch.setattr(pme.GeotiffRasterFormat, "encode_raster", encode)
        with pytest.raises(RuntimeError, match="interrupted"):
            pme.write_composite(window, imagery, scl, centre_doy, bands)

        assert list(window.list_completed_layers()) == []


class TestPatchConfig:
    """The clone inherits the parent's band sets; they must be rewritten."""

    @staticmethod
    def _dataset(tmp_path: Path) -> UPath:
        """A dataset whose config carries the PARENT's three-band-set layers."""
        ds_path = UPath(tmp_path) / "dataset"
        ds_path.mkdir(parents=True)
        parent_s2 = {
            "type": "raster",
            "band_sets": [
                {"bands": ["B02", "B03", "B04", "B08"], "dtype": "uint16"},
                {
                    "bands": ["B05", "B06", "B07", "B8A", "B11", "B12"],
                    "dtype": "uint16",
                    "zoom_offset": -1,
                },
                {"bands": ["B01", "B09"], "dtype": "uint16", "zoom_offset": -2},
            ],
            "data_source": {"name": "parent"},
        }
        parent_scl = {
            "type": "raster",
            "band_sets": [
                {"bands": ["SCL"], "dtype": "uint8", "zoom_offset": -1},
            ],
        }
        layers: dict[str, Any] = {name: dict(parent_s2) for name in pme.S2_LAYERS}
        layers.update({name: dict(parent_scl) for name in pme.SCL_LAYERS})
        layers["sentinel1_mo01"] = {"type": "raster", "band_sets": []}
        with (ds_path / "config.json").open("w") as f:
            json.dump({"layers": layers}, f)
        return ds_path

    def test_rewrites_to_one_band_set_at_window_resolution(
        self, tmp_path: Path
    ) -> None:
        """Without this rslearn looks for band-set dirs the composite never wrote."""
        ds_path = self._dataset(tmp_path)
        changed = pme.patch_config(str(ds_path))

        assert set(changed) == set(pme.S2_LAYERS) | set(pme.SCL_LAYERS)
        config = json.loads((ds_path / "config.json").read_text())
        s2 = config["layers"][pme.S2_LAYERS[0]]["band_sets"]
        assert s2 == [{"bands": pme.composite_bands(), "dtype": "float32"}]
        # zoom_offset must be gone: everything is on the window grid now.
        assert all("zoom_offset" not in bs for bs in s2)
        scl = config["layers"][pme.SCL_LAYERS[0]]["band_sets"]
        assert scl == [{"bands": ["SCL"], "dtype": "uint8"}]
        # Untouched layers stay untouched, and provenance survives.
        assert config["layers"]["sentinel1_mo01"]["band_sets"] == []
        assert config["layers"][pme.S2_LAYERS[0]]["data_source"] == {"name": "parent"}

    def test_backs_up_once_and_is_idempotent(self, tmp_path: Path) -> None:
        """Re-running must not clobber the backup or report spurious changes."""
        ds_path = self._dataset(tmp_path)
        original = (ds_path / "config.json").read_text()
        pme.patch_config(str(ds_path))
        backup = (ds_path / pme.CONFIG_BACKUP_NAME).read_text()
        assert backup == original

        assert pme.patch_config(str(ds_path)) == []
        assert (ds_path / pme.CONFIG_BACKUP_NAME).read_text() == original

    def test_declares_all_twelve_bands(self, tmp_path: Path) -> None:
        """The written composite is always twelve channels, so declare twelve."""
        ds_path = self._dataset(tmp_path)
        pme.patch_config(str(ds_path))
        config = json.loads((ds_path / "config.json").read_text())
        bands = config["layers"][pme.S2_LAYERS[0]]["band_sets"][0]["bands"]
        assert bands == pme.composite_bands()
        assert bands[-2:] == ["B01", "B09"]
        assert config["layers"][pme.S2_LAYERS[0]]["band_sets"][0]["dtype"] == "float32"

    def test_refuses_a_dataset_without_monthly_layers(self, tmp_path: Path) -> None:
        """Guard against being pointed at the wrong dataset."""
        ds_path = UPath(tmp_path) / "other"
        ds_path.mkdir(parents=True)
        with (ds_path / "config.json").open("w") as f:
            json.dump({"layers": {"sentinel1_mo01": {}}}, f)
        with pytest.raises(SystemExit, match="declares no"):
            pme.patch_config(str(ds_path))


class TestTwelveChannelOutput:
    """The model indexes S2 channels 10/11, so the composite must supply them."""

    def test_unread_bands_get_their_normalized_zero_value(self) -> None:
        """Band dropout zeroes AFTER normalization, so raw 0 would be wrong."""
        picked = np.full((3, 3, 10), 1234.6, dtype=np.float32)
        out = pme._to_twelve_bands(picked)

        assert out.shape == (3, 3, 12)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[..., :10], 1234.6, rtol=1e-6)
        # Negative, which is why the composite is float32 and not the parent's uint16.
        assert out[..., 10].min() < 0 and out[..., 11].min() < 0
        np.testing.assert_allclose(out[0, 0, 10], pme.normalized_zero_value("B01"))
        np.testing.assert_allclose(out[0, 0, 11], pme.normalized_zero_value("B09"))

    def test_fill_survives_the_evals_own_normalizer(self) -> None:
        """The load-bearing assertion: the filler must normalize to exactly 0.

        Runs the real Normalizer the eval uses (Strategy.COMPUTED over the
        pretraining config), not a reimplementation of its formula.
        """
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy

        out = pme._to_twelve_bands(np.zeros((1, 1, 10), dtype=np.float32))
        normalized = Normalizer(Strategy.COMPUTED).normalize(
            Modality.SENTINEL2_L2A, out
        )
        np.testing.assert_allclose(normalized[0, 0, 10], 0.0, atol=1e-6)
        np.testing.assert_allclose(normalized[0, 0, 11], 0.0, atol=1e-6)

    def test_twelve_read_bands_pass_through(self) -> None:
        """With B01/B09 materialized nothing is filled."""
        picked = np.arange(12, dtype=np.float32).reshape(1, 1, 12)
        out = pme._to_twelve_bands(picked)

        np.testing.assert_allclose(out[0, 0], np.arange(12, dtype=np.float32))

    def test_channel_positions_match_the_model_band_order(self) -> None:
        """B01/B09 must land at 10/11, where compute_indices looks for them."""
        bands = pme.composite_bands()
        assert bands.index("B01") == 10
        assert bands.index("B09") == 11
        # And that order is the ModalitySpec's, which the tokenizer indexes against.
        from olmoearth_pretrain.data.constants import Modality

        assert bands == list(Modality.SENTINEL2_L2A.band_order)
