"""Outcome metrics shared by evaluation entry points."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any


_WILSON_95_Z = 1.959963984540054


def _wilson_interval(successes: float, games: int) -> list[float]:
    """Return a two-sided 95% Wilson score interval."""
    if games == 0:
        return [0.0, 1.0]

    proportion = successes / games
    z_squared = _WILSON_95_Z**2
    denominator = 1.0 + z_squared / games
    center = (proportion + z_squared / (2.0 * games)) / denominator
    margin = (
        _WILSON_95_Z
        * math.sqrt(
            proportion * (1.0 - proportion) / games + z_squared / (4.0 * games**2)
        )
        / denominator
    )
    lower = 0.0 if successes == 0.0 else max(0.0, center - margin)
    upper = 1.0 if successes == games else min(1.0, center + margin)
    return [lower, upper]


def _summary_from_counts(wins: int, losses: int, draws: int) -> dict[str, Any]:
    games = wins + losses + draws
    win_rate = wins / games if games else 0.0
    score_successes = wins + 0.5 * draws
    score_rate = score_successes / games if games else 0.0
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "score_rate": score_rate,
        "win_rate_ci95": _wilson_interval(float(wins), games),
        # Wilson intervals are binomial intervals. For the ternary battle
        # outcome, this uses the common fractional-success approximation:
        # every draw contributes half a success and half a failure.
        "score_rate_ci95": _wilson_interval(score_successes, games),
    }


def outcome_summary(rewards: Iterable[float]) -> dict[str, Any]:
    """Count wins, draws, and losses from the candidate’s perspective."""
    wins = losses = draws = 0
    for reward_value in rewards:
        reward = float(reward_value)
        if not math.isfinite(reward):
            raise ValueError("rewards must contain only finite values")
        if reward > 0.0:
            wins += 1
        elif reward < 0.0:
            losses += 1
        else:
            draws += 1

    return _summary_from_counts(wins, losses, draws)
