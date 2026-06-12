"""Tag matching helpers for Studio ingest."""

from __future__ import annotations

from collections.abc import Mapping


def tags_match_options(
    options: Mapping[str, object] | None,
    required_tags: Mapping[str, str],
) -> bool:
    """Return whether window options match all required tags.

    An empty required tag value means the key only needs to exist.
    """
    if not required_tags:
        return True
    if not options:
        return False

    for key, value in required_tags.items():
        if key not in options:
            return False
        if value and options[key] != value:
            return False
    return True
