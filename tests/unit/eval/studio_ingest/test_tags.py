"""Tests for Studio ingest tag matching helpers."""

from olmoearth_pretrain.evals.studio_ingest.tags import tags_match_options


def test_tags_match_options_supports_exact_and_key_exists() -> None:
    """Tag matching should support exact values and key-exists checks."""
    options = {"split": "train", "region": "west"}

    assert tags_match_options(options, {"split": "train"})
    assert tags_match_options(options, {"region": ""})
    assert not tags_match_options(options, {"split": "val"})
    assert not tags_match_options(options, {"missing": ""})


def test_tags_match_options_handles_empty_requirements_and_missing_options() -> None:
    """Empty requirements should match, but missing options should not match tags."""
    assert tags_match_options(None, {})
    assert not tags_match_options(None, {"split": "train"})
