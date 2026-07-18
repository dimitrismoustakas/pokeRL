import math

import numpy as np
import pytest

from scripts.evaluate_preview_policy import (
    enumerate_preview_actions,
    summarize_preview_distributions,
)


def test_enumerate_preview_actions_covers_every_bring_and_ordered_lead() -> None:
    actions = enumerate_preview_actions(team_size=6, pick_size=4)

    assert actions.shape == (180, 4)
    assert actions.dtype == np.int64
    assert len({tuple(row) for row in actions}) == 180
    assert all(len(set(row)) == 4 for row in actions)
    assert all(np.all((row >= 0) & (row < 6)) for row in actions)
    assert all(tuple(row[2:]) == tuple(sorted(row[2:])) for row in actions)

    bring_counts: dict[tuple[int, ...], int] = {}
    for row in actions:
        bring = tuple(sorted(int(slot) for slot in row))
        bring_counts[bring] = bring_counts.get(bring, 0) + 1
    assert len(bring_counts) == 15
    assert set(bring_counts.values()) == {12}


def test_distribution_summary_reports_exact_opponent_information() -> None:
    actions = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [0, 2, 1, 3],
        ],
        dtype=np.int64,
    )
    probabilities = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    summary = summarize_preview_distributions(probabilities, actions)

    assert summary["opponent_count"] == 2
    assert summary["legal_action_count"] == 3
    assert summary["distinct_most_probable_full_choices"] == 2
    assert summary["distinct_most_probable_bring_choices"] == 1
    assert summary["distinct_most_probable_lead_choices"] == 2
    assert summary["exact_policy_entropy_mean_nats"] == 0.0
    assert summary["max_probability_mean"] == 1.0
    assert summary["mutual_information_nats"] == pytest.approx(math.log(2.0))
    assert summary["mutual_information_bits"] == pytest.approx(1.0)
    assert summary["mean_total_variation_to_mean_policy"] == pytest.approx(0.5)


def test_distribution_summary_identical_uniform_policies_have_no_information() -> None:
    actions = enumerate_preview_actions(team_size=4, pick_size=2)
    probabilities = np.full((3, len(actions)), 1.0 / len(actions))

    summary = summarize_preview_distributions(probabilities, actions)

    assert summary["distinct_most_probable_full_choices"] == 1
    assert summary["exact_policy_entropy_mean_nats"] == pytest.approx(
        math.log(len(actions))
    )
    assert summary["max_probability_mean"] == pytest.approx(1.0 / len(actions))
    assert summary["mutual_information_nats"] == 0.0
    assert summary["mean_total_variation_to_mean_policy"] == pytest.approx(0.0)


def test_distribution_summary_rejects_unnormalized_rows() -> None:
    actions = enumerate_preview_actions(team_size=4, pick_size=2)
    probabilities = np.full((1, len(actions)), 0.5)

    with pytest.raises(ValueError, match="sum to one"):
        summarize_preview_distributions(probabilities, actions)
