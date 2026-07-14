from collections import Counter
from types import SimpleNamespace

import pytest
from poke_env.battle import Field

from pokerl.env.actions import SWITCH_OFFSET, TARGET_NONE
from pokerl.eval.fixed_bot import (
    _classify_battle_mechanics,
    _classify_protect_outcomes,
    _classify_slot_action,
    _record_action_opportunities,
    _summarize_battle_behavior,
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
    first_turn: bool = True,
    protect_counter: int = 0,
    trick_room_active: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        available_moves=[moves, []],
        available_switches=[switches or [], []],
        opponent_active_pokemon=[],
        active_pokemon=[
            SimpleNamespace(
                first_turn=first_turn,
                protect_counter=protect_counter,
            ),
            None,
        ],
        fields={Field.TRICK_ROOM: 1} if trick_room_active else {},
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
            "protect_like_fresh_opportunities": 1,
            "protect_like_prior_0_opportunities": 1,
            "trick_room_opportunities": 1,
            "trick_room_setup_opportunities": 1,
            "redirection_opportunities": 1,
            "fake_out_opportunities": 1,
            "fake_out_first_turn_opportunities": 1,
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


def test_contextual_move_opportunities_distinguish_stateful_mechanics() -> None:
    battle = _battle(
        [_move("protect"), _move("trickroom"), _move("fakeout")],
        first_turn=False,
        protect_counter=1,
        trick_room_active=True,
    )

    stats = _record_action_opportunities(battle, slot=0)

    assert stats["protect_like_repeat_opportunities"] == 1
    assert stats["protect_like_fresh_opportunities"] == 0
    assert stats["trick_room_cancel_opportunities"] == 1
    assert stats["trick_room_setup_opportunities"] == 0
    assert stats["fake_out_late_opportunities"] == 1
    assert stats["fake_out_first_turn_opportunities"] == 0


@pytest.mark.parametrize(
    ("move_id", "battle_kwargs", "expected_group"),
    [
        ("protect", {"protect_counter": 1}, "protect_like_repeat"),
        ("trickroom", {"trick_room_active": True}, "trick_room_cancel"),
        ("fakeout", {"first_turn": False}, "fake_out_late"),
    ],
)
def test_contextual_move_actions_match_pre_action_state(
    move_id: str,
    battle_kwargs: dict[str, object],
    expected_group: str,
) -> None:
    stats = _classify_slot_action(
        _battle([_move(move_id)], **battle_kwargs),
        slot=0,
        action_idx=TARGET_NONE,
        tera_choice=0,
    )

    assert stats[f"{expected_group}_opportunities"] == 1
    assert stats[f"{expected_group}_actions"] == 1


def test_battle_mechanics_report_battle_normalized_choice_rates() -> None:
    aggregate = Counter()
    aggregate.update(
        _classify_battle_mechanics(
            Counter(
                trick_room_setup_opportunities=3,
                trick_room_setup_actions=1,
                fake_out_late_opportunities=2,
            )
        )
    )
    aggregate.update(
        _classify_battle_mechanics(
            Counter(
                trick_room_setup_opportunities=1,
                protect_like_repeat_opportunities=2,
                protect_like_repeat_actions=1,
            )
        )
    )

    summary = _summarize_action_stats(aggregate)

    assert aggregate["trick_room_setup_opportunity_battles"] == 2
    assert aggregate["trick_room_setup_action_battles"] == 1
    assert summary["rates"][
        "trick_room_setup_battle_choice_rate_when_available"
    ] == pytest.approx(0.5)
    assert summary["rates"][
        "fake_out_late_battle_choice_rate_when_available"
    ] == 0.0
    assert summary["rates"][
        "protect_like_repeat_battle_choice_rate_when_available"
    ] == 1.0


def test_protect_outcomes_use_same_pokemon_success_streak() -> None:
    fresh_success = SimpleNamespace(protect_counter=1)
    repeat_success = SimpleNamespace(protect_counter=2)
    repeat_failure = SimpleNamespace(protect_counter=0)

    stats = _classify_protect_outcomes(
        [
            (fresh_success, 0),
            (repeat_success, 1),
            (repeat_failure, 1),
        ]
    )
    summary = _summarize_action_stats(stats)

    assert stats["protect_like_attempts"] == 3
    assert stats["protect_like_successes"] == 2
    assert summary["rates"]["protect_like_fresh_success_rate"] == 1.0
    assert summary["rates"]["protect_like_repeat_success_rate"] == 0.5
    assert summary["protect_chain"][1]["success_rate_when_chosen"] == 0.5


def test_battle_behavior_reports_tera_use_timing_and_slot() -> None:
    summary = _summarize_battle_behavior(
        [
            {
                "tera_used": True,
                "first_tera_turn": 1,
                "first_tera_slot": 0,
                "won": True,
            },
            {
                "tera_used": True,
                "first_tera_turn": 3,
                "first_tera_slot": 1,
                "won": False,
            },
            {"tera_used": False, "won": False},
        ]
    )

    assert summary["tera_game_rate"] == pytest.approx(2 / 3)
    assert summary["turn_one_tera_rate_all_games"] == pytest.approx(1 / 3)
    assert summary["turn_one_tera_rate_when_used"] == pytest.approx(1 / 2)
    assert summary["first_tera_slot_distribution"] == [
        {"slot": 0, "count": 1, "rate": 0.5},
        {"slot": 1, "count": 1, "rate": 0.5},
    ]
