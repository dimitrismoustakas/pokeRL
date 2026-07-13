from collections import Counter
from types import SimpleNamespace

import pytest

from pokerl.env.actions import SWITCH_OFFSET, TARGET_NONE
from pokerl.eval.fixed_bot import (
    _classify_slot_action,
    _record_action_opportunities,
    _summarize_action_stats,
)


def _move(move_id: str, *, base_power: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        id=move_id,
        base_power=base_power,
        target=SimpleNamespace(name="NORMAL"),
    )


def _battle(
    moves: list[SimpleNamespace],
    *,
    switches: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        available_moves=[moves, []],
        available_switches=[switches or [], []],
        opponent_active_pokemon=[],
    )


def test_action_opportunities_count_each_family_once_per_slot_decision() -> None:
    battle = _battle(
        [
            _move("protect"),
            _move("detect"),
            _move("trickroom"),
            _move("followme"),
            _move("fakeout"),
            _move("tailwind"),
        ],
        switches=[object(), object()],
    )

    assert _record_action_opportunities(battle, slot=0) == Counter(
        {
            "protect_like_opportunities": 1,
            "trick_room_opportunities": 1,
            "redirection_opportunities": 1,
            "fake_out_opportunities": 1,
            "tailwind_opportunities": 1,
            "switch_opportunities": 1,
        }
    )
    assert _record_action_opportunities(battle, slot=2) == Counter()


@pytest.mark.parametrize(
    ("move_id", "diagnostic"),
    [
        ("detect", "protect_like"),
        ("trickroom", "trick_room"),
        ("ragepowder", "redirection"),
        ("fakeout", "fake_out"),
        ("tailwind", "tailwind"),
    ],
)
def test_classification_matches_chosen_move_to_available_family(
    move_id: str,
    diagnostic: str,
) -> None:
    battle = _battle([_move(move_id)])

    stats = _classify_slot_action(
        battle,
        slot=0,
        action_idx=TARGET_NONE,
        tera_choice=0,
    )
    summary = _summarize_action_stats(stats)

    assert stats[f"{diagnostic}_opportunities"] == 1
    assert stats[f"{diagnostic}_actions"] == 1
    assert summary["rates"][f"{diagnostic}_choice_rate_when_available"] == 1.0

    if move_id == "detect":
        assert stats["protect_actions"] == 0
        assert summary["rates"]["protect_rate"] == 0.0


def test_switch_classification_records_choice_and_opportunity() -> None:
    battle = _battle([_move("tailwind")], switches=[object()])

    stats = _classify_slot_action(
        battle,
        slot=0,
        action_idx=SWITCH_OFFSET,
        tera_choice=0,
    )
    summary = _summarize_action_stats(stats)

    assert stats["switch_opportunities"] == 1
    assert stats["switch_actions"] == 1
    assert summary["rates"]["switch_choice_rate_when_available"] == 1.0
    assert stats["tailwind_opportunities"] == 1
    assert stats["tailwind_actions"] == 0


def test_action_summary_normalizes_choices_by_opportunity_count() -> None:
    summary = _summarize_action_stats(
        Counter(
            {
                "slot_decisions": 12,
                "protect_like_actions": 1,
                "protect_like_opportunities": 4,
                "trick_room_actions": 2,
                "trick_room_opportunities": 2,
                "redirection_actions": 1,
                "redirection_opportunities": 5,
                "fake_out_opportunities": 3,
                "tailwind_actions": 1,
                "switch_actions": 2,
                "switch_opportunities": 8,
            }
        )
    )

    rates = summary["rates"]
    assert rates["protect_like_choice_rate_when_available"] == pytest.approx(0.25)
    assert rates["trick_room_choice_rate_when_available"] == 1.0
    assert rates["redirection_choice_rate_when_available"] == pytest.approx(0.2)
    assert rates["fake_out_choice_rate_when_available"] == 0.0
    assert rates["tailwind_choice_rate_when_available"] == 0.0
    assert rates["switch_choice_rate_when_available"] == pytest.approx(0.25)
