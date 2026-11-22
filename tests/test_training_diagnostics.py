import pytest
import torch

from scripts.train import summarize_first_tera_turns, summarize_observation_history


def test_summarize_observation_history_reports_causal_input_coverage() -> None:
    obs = {
        "recent_event_meta": torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 1.0, 0.25]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        ),
        "recent_event_mask": torch.tensor(
            [
                [[False, False, True], [False, True, True]],
                [[True, True, True], [False, False, False]],
            ]
        ),
        "prev_own_valid": torch.tensor(
            [[[True, True], [True, False]], [[False, False], [True, True]]]
        ),
    }

    metrics = summarize_observation_history(obs)

    assert metrics["history_event_segment_coverage"] == pytest.approx(0.75)
    assert metrics["history_partial_segment_fraction"] == pytest.approx(1 / 3)
    assert metrics["history_overflow_segment_fraction"] == pytest.approx(1 / 3)
    assert metrics["history_events_per_valid_segment"] == pytest.approx(2.0)
    assert metrics["history_previous_intent_slot_coverage"] == pytest.approx(0.625)


def test_summarize_first_tera_turns_keeps_nonuse_in_denominator() -> None:
    summary = summarize_first_tera_turns([1, 3, 3], completed_battles=5)

    assert summary["tera_used_fraction"] == pytest.approx(0.6)
    assert summary["turn_one_fraction"] == pytest.approx(0.2)
    assert summary["avg_first_tera_turn"] == pytest.approx(7 / 3)
    assert summary["first_tera_turn_distribution"] == [
        {"turn": 1, "count": 1},
        {"turn": 3, "count": 2},
    ]
