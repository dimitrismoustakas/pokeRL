import math

import numpy as np
import pytest

from pokerl.eval.metrics import mirrored_candidate_summary, outcome_summary


def test_outcome_summary_counts_by_sign_and_scores_draws_as_half() -> None:
    summary = outcome_summary(np.array([2.5, 0.0, -0.25, 7.0, -3.0, -0.0]))

    assert summary["games"] == 6
    assert summary["wins"] == 2
    assert summary["losses"] == 2
    assert summary["draws"] == 2
    assert summary["win_rate"] == pytest.approx(1.0 / 3.0)
    assert summary["score_rate"] == pytest.approx(0.5)
    assert summary["win_rate_ci95"] == pytest.approx(
        [0.09677141110578052, 0.700006684861608]
    )
    assert summary["score_rate_ci95"] == pytest.approx(
        [0.1876163064826506, 0.8123836935173494]
    )


def test_outcome_summary_empty_input_has_uninformative_intervals() -> None:
    assert outcome_summary(iter(())) == {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "win_rate": 0.0,
        "score_rate": 0.0,
        "win_rate_ci95": [0.0, 1.0],
        "score_rate_ci95": [0.0, 1.0],
    }


def test_outcome_summary_accepts_explicit_wdl_counts() -> None:
    from_rewards = outcome_summary([1.0, 1.0, -1.0, 0.0])

    assert outcome_summary(wins=2, losses=1, draws=1) == from_rewards


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wins": 1, "losses": 0},
        {"wins": 1, "losses": -1, "draws": 0},
        {"wins": 1.5, "losses": 0, "draws": 0},
        {"wins": True, "losses": 0, "draws": 0},
    ],
)
def test_outcome_summary_rejects_incomplete_or_invalid_wdl_counts(kwargs) -> None:
    with pytest.raises(ValueError):
        outcome_summary(**kwargs)


def test_outcome_summary_rejects_mixed_rewards_and_wdl_counts() -> None:
    with pytest.raises(ValueError, match="either rewards or explicit"):
        outcome_summary([1.0], wins=1, losses=0, draws=0)


@pytest.mark.parametrize("rewards", [[math.nan], [math.inf], [-math.inf]])
def test_outcome_summary_rejects_non_finite_rewards(rewards) -> None:
    with pytest.raises(ValueError, match="finite"):
        outcome_summary(rewards)


def test_wilson_interval_stays_inside_probability_bounds_at_extremes() -> None:
    all_wins = outcome_summary([1.0] * 10)
    all_losses = outcome_summary([-1.0] * 10)

    assert all_wins["win_rate_ci95"][1] == 1.0
    assert all_wins["score_rate_ci95"][1] == 1.0
    assert all_losses["win_rate_ci95"][0] == 0.0
    assert all_losses["score_rate_ci95"][0] == 0.0


def test_mirrored_candidate_summary_inverts_reverse_wins_and_losses() -> None:
    forward = outcome_summary([1.0, 1.0, -1.0, 0.0])
    reverse = outcome_summary([1.0, -1.0, -1.0, 0.0])

    combined = mirrored_candidate_summary(forward, reverse)

    assert combined["games"] == 8
    assert combined["wins"] == 4
    assert combined["losses"] == 2
    assert combined["draws"] == 2
    assert combined["win_rate"] == 0.5
    assert combined["score_rate"] == 0.625
    assert combined == outcome_summary([1.0] * 4 + [-1.0] * 2 + [0.0] * 2)


@pytest.mark.parametrize(
    "bad_summary",
    [
        {"wins": -1, "losses": 0, "draws": 0},
        {"wins": 1.5, "losses": 0, "draws": 0},
        {"wins": True, "losses": 0, "draws": 0},
    ],
)
def test_mirrored_candidate_summary_rejects_invalid_counts(bad_summary) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        mirrored_candidate_summary(
            bad_summary,
            {"wins": 0, "losses": 0, "draws": 0},
        )
