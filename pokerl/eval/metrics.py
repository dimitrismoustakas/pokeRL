"""Outcome metrics shared by evaluation entry points."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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


def outcome_summary(
    rewards: Iterable[float] | None = None,
    *,
    wins: int | None = None,
    losses: int | None = None,
    draws: int | None = None,
) -> dict[str, Any]:
    """Summarize rewards or explicit W/D/L counts with 95% Wilson intervals.

    Positive rewards are wins, negative rewards are losses, and zero rewards
    are draws. ``score_rate`` gives each draw half credit. Its interval applies
    the Wilson formula to the corresponding fractional success count, which is
    an approximation because battle scores are not binary observations.

    Pass either an iterable of rewards or all three explicit ``wins``,
    ``losses``, and ``draws`` keyword counts. Empty reward input and zero counts
    produce zero rates and uninformative ``[0.0, 1.0]`` intervals. Non-finite
    rewards and invalid counts are rejected rather than silently miscounted.
    """
    supplied_counts = (wins, losses, draws)
    if rewards is not None and any(value is not None for value in supplied_counts):
        raise ValueError("pass either rewards or explicit W/D/L counts, not both")
    if rewards is None:
        if any(value is None for value in supplied_counts):
            raise ValueError("pass rewards or provide all of wins, losses, and draws")
        explicit_counts = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }
        valid_wins, valid_losses, valid_draws = _outcome_counts(
            explicit_counts, "outcome"
        )
        return _summary_from_counts(valid_wins, valid_losses, valid_draws)

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


def _outcome_counts(summary: Mapping[str, object], label: str) -> tuple[int, int, int]:
    counts: list[int] = []
    for key in ("wins", "losses", "draws"):
        value = summary[key]
        if isinstance(value, bool):
            raise ValueError(f"{label} {key} must be a non-negative integer")
        count = int(value)  # type: ignore[arg-type]
        if count < 0 or count != value:
            raise ValueError(f"{label} {key} must be a non-negative integer")
        counts.append(count)
    return counts[0], counts[1], counts[2]


def mirrored_candidate_summary(
    forward: Mapping[str, object],
    reverse: Mapping[str, object],
) -> dict[str, Any]:
    """Combine mirrored matchups from the original candidate's perspective.

    ``forward`` is already expressed from the candidate's perspective.
    ``reverse`` is expressed from its opposing player's perspective, so its
    wins and losses are inverted before the aggregate metrics are calculated.
    """
    forward_wins, forward_losses, forward_draws = _outcome_counts(forward, "forward")
    reverse_wins, reverse_losses, reverse_draws = _outcome_counts(reverse, "reverse")
    return _summary_from_counts(
        forward_wins + reverse_losses,
        forward_losses + reverse_wins,
        forward_draws + reverse_draws,
    )
