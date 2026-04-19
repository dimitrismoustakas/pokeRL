"""Helpers for team preview order parsing and validation."""

import numpy as np


def parse_team_preview_order(team_order: str, team_size: int, pick_size: int) -> np.ndarray:
    """Parse a Showdown ``/team`` command into zero-based team slot indices."""
    prefix = "/team "
    if not isinstance(team_order, str) or not team_order.startswith(prefix):
        raise ValueError(f"Expected a Showdown team preview command starting with {prefix!r}, got: {team_order!r}")

    raw_order = team_order[len(prefix):].strip()
    if len(raw_order) != pick_size:
        raise ValueError(
            f"Expected {pick_size} team preview picks in {team_order!r}, got {len(raw_order)} characters."
        )
    if not raw_order.isdigit():
        raise ValueError(f"Team preview command must contain only slot digits, got: {team_order!r}")

    picks = np.array([int(char) - 1 for char in raw_order], dtype=np.int64)
    if (picks < 0).any() or (picks >= team_size).any():
        raise ValueError(
            f"Team preview picks {picks.tolist()} are out of range for a team of size {team_size}."
        )
    if len(set(picks.tolist())) != pick_size:
        raise ValueError(f"Team preview command must not repeat slots, got: {team_order!r}")
    return picks
