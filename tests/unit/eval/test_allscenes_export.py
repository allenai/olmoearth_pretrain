"""Tests for the all-scenes S2 exporter's scene thinning."""

from datetime import UTC, datetime, timedelta

from olmoearth_pretrain.evals.datasets import allscenes_export as ase


def scenes(*days: float) -> list[tuple[int, datetime]]:
    """Build ``(group_idx, instant)`` pairs from days-since-Jan-1-2020."""
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return [(i, start + timedelta(days=day)) for i, day in enumerate(days)]


def days_of(kept: list[tuple[int, datetime]]) -> list[float]:
    """Days since Jan 1 2020 of a selection, for readable assertions."""
    start = datetime(2020, 1, 1, tzinfo=UTC)
    return [(instant - start).total_seconds() / 86400 for _idx, instant in kept]


class TestThinToCap:
    """Selection is the only judgement call in the export; pin its behaviour."""

    def test_short_years_are_untouched(self) -> None:
        """A window with fewer looks than the cap must keep all of them."""
        year = scenes(*range(0, 300, 30))
        assert ase.thin_to_cap(year, cap=36) == year

    def test_exactly_at_the_cap_is_untouched(self) -> None:
        """The cap is a maximum, not a target to thin down to."""
        year = scenes(*range(36))
        assert ase.thin_to_cap(year, cap=36) == year

    def test_no_cap_keeps_everything(self) -> None:
        """Cap <= 0 is the documented "keep every scene" escape hatch."""
        year = scenes(*range(100))
        assert ase.thin_to_cap(year, cap=0) == year
        assert ase.thin_to_cap(year, cap=-1) == year

    def test_thins_to_exactly_the_cap(self) -> None:
        """A full year must come back at the budget it was given."""
        year = scenes(*[5 * i for i in range(73)])
        assert len(ase.thin_to_cap(year, cap=36)) == 36

    def test_output_stays_chronological(self) -> None:
        """Item groups are renumbered by output position, so order is the axis."""
        kept = ase.thin_to_cap(scenes(*[5 * i for i in range(73)]), cap=36)
        assert days_of(kept) == sorted(days_of(kept))

    def test_selection_is_a_subset(self) -> None:
        """Thinning selects real acquisitions; it must never synthesize one."""
        year = scenes(*[5 * i for i in range(73)])
        kept = ase.thin_to_cap(year, cap=36)
        assert set(kept).issubset(set(year))
        assert len(set(kept)) == len(kept)

    def test_spreads_over_the_year_not_the_list(self) -> None:
        """The reason selection is time-based rather than index-based.

        MGRS overlap zones return two scenes for the same date. Here the first
        half of the year is duplicated and the second is not, so taking every
        k-th list entry would spend most of the budget before July. Selection
        must instead come out roughly balanced across the two halves.
        """
        duplicated = [day for day in range(0, 180, 5) for _ in (0, 1)]
        singles = list(range(180, 360, 5))
        kept_days = days_of(ase.thin_to_cap(scenes(*duplicated, *singles), cap=36))

        first_half = sum(1 for day in kept_days if day < 180)
        assert 14 <= first_half <= 22, kept_days

    def test_prefers_distinct_dates_over_duplicates(self) -> None:
        """With a same-day pair available, the budget should not spend both slots.

        Four acquisitions, two of them on the same day, thinned to three: the
        two distinct later dates must both survive.
        """
        kept_days = days_of(ase.thin_to_cap(scenes(10, 10, 100, 200), cap=3))
        assert 100 in kept_days
        assert 200 in kept_days

    def test_is_deterministic(self) -> None:
        """Two runs over the same dataset must produce the same eval inputs."""
        year = scenes(*[4.9 * i for i in range(74)])
        assert ase.thin_to_cap(year, cap=36) == ase.thin_to_cap(year, cap=36)

    def test_single_instant_year_falls_back_to_truncation(self) -> None:
        """A zero-length span has no spacing to respect; it must not divide by it."""
        year = scenes(*[7.0] * 5)
        assert len(ase.thin_to_cap(year, cap=3)) == 3


class TestSceneLayerConfig:
    """The config the export writes must describe what the fetch group holds."""

    def test_declares_only_materialized_band_sets(self) -> None:
        """B01/B09 never landed on disk, so declaring them would break reads."""
        declared = {band for bs in ase.SCENE_BAND_SETS for band in bs["bands"]}
        assert "B01" not in declared
        assert "B09" not in declared

    def test_model_bands_are_the_optical_sets_in_order(self) -> None:
        """model.yaml's `bands:` must match the band-set concatenation order."""
        assert ase.SCENE_BANDS == [
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

    def test_scl_is_stored_but_not_a_model_band(self) -> None:
        """SCL rides along for a future masked sibling, unread by these tasks."""
        declared = {band for bs in ase.SCENE_BAND_SETS for band in bs["bands"]}
        assert "SCL" in declared
        assert "SCL" not in ase.SCENE_BANDS

    def test_layer_has_no_data_source(self) -> None:
        """A source is one stray materialize from refetching a year per window."""
        assert "data_source" not in ase.SCENE_LAYER_CONFIG
        assert set(ase.SCENE_LAYER_CONFIG) == {"type", "band_sets"}

    def test_monthly_s2_layers_are_the_ones_dropped(self) -> None:
        """The rsync leaves the monthly rasters behind; the config must agree.

        S1 and Landsat monthlies stay -- they are unchanged across both arms and
        the model.yaml still declares them.
        """
        assert "sentinel2_l2a_mo01".startswith(ase.DROP_LAYER_PREFIXES)
        assert "sentinel2_scl_mo01".startswith(ase.DROP_LAYER_PREFIXES)
        assert not "sentinel1_mo01".startswith(ase.DROP_LAYER_PREFIXES)
        assert not "landsat_mo01".startswith(ase.DROP_LAYER_PREFIXES)
        assert not ase.S2_LAYER.startswith(ase.DROP_LAYER_PREFIXES)
