from __future__ import annotations

import logging

import numpy as np
import pytest
from poke_env.battle import DoubleBattle, Move
from poke_env.player import SingleBattleOrder

import pokerl.env.actions as actions_module
from pokerl.env.actions import _hydrate_move_for_request, build_legal_action_state


class _UnreadableRequestBattle:
    @property
    def last_request(self):
        raise AssertionError("hydrated request moves must not rescan request JSON")


class _RequestBattle:
    def __init__(self, move_id: str, target: str) -> None:
        self.last_request = {
            "active": [
                {
                    "moves": [
                        {
                            "id": move_id,
                            "target": target,
                            "disabled": False,
                        }
                    ]
                }
            ]
        }


def test_fully_hydrated_request_move_is_reused_without_request_json_scan() -> None:
    move = Move("fakeout", gen=9)
    move.request_index = 3
    move.request_target = "normal"

    hydrated = _hydrate_move_for_request(move, _UnreadableRequestBattle(), slot=0)

    assert hydrated is move
    assert SingleBattleOrder(hydrated, move_target=1).message == "/choose move 3 1"


@pytest.mark.parametrize(
    ("move_id", "request_target"),
    [
        ("icespinner", "normal"),
        ("dazzlinggleam", "allAdjacentFoes"),
        ("outrage", "randomNormal"),
        ("protect", "self"),
        ("acupressure", "adjacentAllyOrSelf"),
    ],
)
def test_request_metadata_fast_path_matches_fallback_order_semantics(
    move_id: str,
    request_target: str,
) -> None:
    current_move = Move(move_id, gen=9)
    current_move.request_index = 1
    current_move.request_target = request_target

    incomplete_move = Move(move_id, gen=9)
    incomplete_move.request_index = 1
    battle = _RequestBattle(move_id, request_target)

    fast = _hydrate_move_for_request(current_move, battle, slot=0)
    fallback = _hydrate_move_for_request(incomplete_move, battle, slot=0)

    assert fast is current_move
    assert fallback is not incomplete_move
    assert fast.id == fallback.id
    assert fast.gen == fallback.gen
    assert fast.request_index == fallback.request_index
    assert fast.request_target == fallback.request_target
    assert fast.deduced_target == fallback.deduced_target
    assert fast.target == fallback.target
    assert fast.non_ghost_target == fallback.non_ghost_target
    assert (
        SingleBattleOrder(fast, move_target=1).message
        == SingleBattleOrder(fallback, move_target=1).message
    )


def test_hydrated_targetless_request_move_is_reused() -> None:
    move = Move("outrage", gen=9)
    move.request_index = 1
    battle = _RequestBattle("outrage", "randomNormal")
    battle.last_request["active"][0]["moves"][0].pop("target")

    hydrated = _hydrate_move_for_request(move, battle, slot=0)

    assert hydrated is move
    assert hydrated.request_target is None
    assert SingleBattleOrder(hydrated).message == "/choose move 1"


def _request_pokemon(
    ident: str,
    species: str,
    *,
    active: bool,
    ability: str,
    moves: list[str],
) -> dict[str, object]:
    return {
        "ident": ident,
        "details": f"{species}, L50",
        "condition": "150/150",
        "active": active,
        "stats": {"atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        "moves": moves,
        "baseAbility": ability,
        "ability": ability,
        "item": "",
        "pokeball": "pokeball",
    }


def _parsed_double_battle() -> DoubleBattle:
    battle = DoubleBattle("metadata-integration", "player", logging.getLogger(), gen=9)
    first_moves = ["fakeout", "protect", "snarl", "partingshot"]
    second_moves = ["grassyglide", "protect", "woodhammer", "uturn"]
    request = {
        "active": [
            {
                "canTerastallize": "Fire",
                "moves": [
                    {"id": "fakeout", "target": "normal", "disabled": False},
                    {"id": "protect", "target": "self", "disabled": False},
                    {
                        "id": "snarl",
                        "target": "allAdjacentFoes",
                        "disabled": False,
                    },
                    {"id": "partingshot", "target": "normal", "disabled": False},
                ],
            },
            {
                "moves": [
                    {
                        "id": "grassyglide",
                        "target": "normal",
                        "disabled": False,
                    },
                    {"id": "protect", "target": "self", "disabled": False},
                    {"id": "woodhammer", "target": "normal", "disabled": False},
                    {"id": "uturn", "target": "normal", "disabled": False},
                ]
            },
        ],
        "side": {
            "id": "p1",
            "name": "player",
            "pokemon": [
                _request_pokemon(
                    "p1: Incineroar",
                    "Incineroar",
                    active=True,
                    ability="intimidate",
                    moves=first_moves,
                ),
                _request_pokemon(
                    "p1: Rillaboom",
                    "Rillaboom",
                    active=True,
                    ability="grassysurge",
                    moves=second_moves,
                ),
                _request_pokemon(
                    "p1: Flutter Mane",
                    "Flutter Mane",
                    active=False,
                    ability="protosynthesis",
                    moves=["moonblast", "shadowball", "protect", "icywind"],
                ),
                _request_pokemon(
                    "p1: Urshifu",
                    "Urshifu-Rapid-Strike",
                    active=False,
                    ability="unseenfist",
                    moves=["surgingstrikes", "closecombat", "aquajet", "protect"],
                ),
            ],
        },
    }
    battle.parse_request(request)
    return battle


def test_real_double_battle_request_fast_path_matches_forced_clone_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    battle = _parsed_double_battle()
    assert all(
        move.request_index is not None and move.request_target is not None
        for slot_moves in battle.available_moves
        for move in slot_moves
    )
    fast_state = build_legal_action_state(battle)

    def forced_clone(move: Move, current_battle: DoubleBattle, slot: int) -> Move:
        clone = Move(move.id, gen=move.gen)
        request = actions_module._request_move_entry(move, current_battle, slot)
        clone.request_index = move.request_index
        if request is not None and request.get("target") is not None:
            clone.request_target = request["target"]
        return clone

    monkeypatch.setattr(actions_module, "_hydrate_move_for_request", forced_clone)
    fallback_state = build_legal_action_state(battle)

    np.testing.assert_array_equal(fast_state.joint_mask, fallback_state.joint_mask)
    np.testing.assert_array_equal(fast_state.can_tera, fallback_state.can_tera)
    for slot, (fast_orders, fallback_orders) in enumerate(
        zip(
            fast_state.slot_action_orders,
            fallback_state.slot_action_orders,
            strict=True,
        )
    ):
        assert fast_orders.keys() == fallback_orders.keys()
        assert [
            actions_module.action_to_order(
                action,
                battle,
                slot,
                action_state=fast_state,
            ).message
            for action in fast_orders
        ] == [
            actions_module.action_to_order(
                action,
                battle,
                slot,
                action_state=fallback_state,
            ).message
            for action in fallback_orders
        ]
