from __future__ import annotations

import copy
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.player import SingleBattleOrder

import pokerl.env.actions as actions_module
from pokerl.config import ModelConfig
from pokerl.data import GameData
from pokerl.env.actions import (
    NUM_TARGETS,
    SWITCH_OFFSET,
    _hydrate_move_for_request,
    action_to_order,
    build_legal_action_state,
)
from pokerl.env.observation import OWN_STATS_OFFSET, ObservationBuilder
from pokerl.eval.fixed_bot import max_power_slot_action
from pokerl.model.agent import PokeAgent


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


def _with_disabled_fake_out() -> DoubleBattle:
    battle = _parsed_double_battle()
    request = copy.deepcopy(battle.last_request)
    request["active"][0]["moves"][0]["disabled"] = True
    battle.parse_request(request)
    return battle


def test_brought_reserves_match_the_observation_and_switch_actions() -> None:
    battle = _parsed_double_battle()
    game_data = GameData()
    builder = ObservationBuilder(game_data)
    six = copy.deepcopy(battle.last_request)
    six["side"]["pokemon"].extend(
        [
            _request_pokemon(
                "p1: Gengar",
                "Gengar",
                active=False,
                ability="cursedbody",
                moves=["hex"],
            ),
            _request_pokemon(
                "p1: Raichu",
                "Raichu",
                active=False,
                ability="static",
                moves=["thunder"],
            ),
        ]
    )
    battle.parse_request(six)
    preview = builder.build(battle)
    assert not preview["own_mask"].any()

    for reserve_slots in ((2, 3), (5, 4)):
        four = copy.deepcopy(six)
        four["side"]["pokemon"] = [
            six["side"]["pokemon"][i] for i in (0, 1, *reserve_slots)
        ]
        battle.parse_request(four)
        obs = builder.build(battle)
        state = build_legal_action_state(battle)
        assert len(battle.team) == 6
        assert obs["own_mask"].tolist() == [False] * 4 + [True] * 2
        for row, preview_row in enumerate(reserve_slots, start=2):
            assert obs["own_species"][row] == preview["own_species"][preview_row]
            switch = state.slot_action_orders[0][SWITCH_OFFSET + row]
            assert game_data.species_to_idx[switch.species] == obs["own_species"][row]

    four["side"]["pokemon"][2] = dict(four["side"]["pokemon"][2], condition="0 fnt")
    battle.parse_request(four)
    obs = builder.build(battle)
    assert not obs["own_mask"][2]
    assert obs["own_continuous"][2, 0] == 0
    assert (
        SWITCH_OFFSET + 2 not in build_legal_action_state(battle).slot_action_orders[0]
    )


def test_own_stats_follow_the_request_and_opponent_stats_stay_private() -> None:
    battle = _parsed_double_battle()
    builder = ObservationBuilder(GameData())
    request = copy.deepcopy(battle.last_request)
    before = builder.build(battle)
    slot = request["side"]["pokemon"][0]
    slot["condition"] = "130/191"
    slot["stats"] = {"atk": 140, "def": 93, "spa": 88, "spd": 115, "spe": 178}
    battle.parse_request(request)
    battle.switch("p2a: Incineroar", "Incineroar, L50", "100/100")
    battle.opponent_active_pokemon[0].stats = slot["stats"]
    obs = builder.build(battle)
    np.testing.assert_array_equal(
        obs["own_continuous"][0, OWN_STATS_OFFSET:] * 256, [191, 140, 93, 88, 115, 178]
    )
    assert before["own_continuous"][0, -1] != obs["own_continuous"][0, -1]
    assert not obs["opp_mask"][0]
    assert not obs["opp_continuous"][:, OWN_STATS_OFFSET:].any()


@pytest.mark.parametrize(
    ("species", "element"),
    [
        ("ogerpon", "grass"),
        ("ogerponwellspring", "water"),
        ("ogerponhearthflame", "fire"),
        ("ogerponcornerstonetera", "rock"),
        ("smeargle", "grass"),
    ],
)
def test_ivy_cudgel_uses_the_active_form_in_its_request_slot(species, element) -> None:
    gd = GameData()
    move = Move("ivycudgel", gen=9)
    move.request_index = 3
    battle = SimpleNamespace(
        active_pokemon=[Pokemon(gen=9, species=species), None],
        available_moves=[[move], []],
    )
    ids, metadata = ObservationBuilder(gd)._encode_own_request_moves(battle)
    assert ids[0].tolist() == [0, 0, gd.move_to_idx["ivycudgel"], 0]
    assert metadata[0, :, 0].tolist() == [0, 0, gd.type_to_idx[element], 0]


def test_tera_action_features_use_original_stab_and_the_changed_attack_type() -> None:
    gd = GameData()
    head = PokeAgent(ModelConfig(), gd).action_head
    builder = ObservationBuilder(gd)
    battle = _parsed_double_battle()
    req = copy.deepcopy(battle.last_request)
    req["side"]["pokemon"][0] = _request_pokemon(
        "p1: Dragonite",
        "Dragonite",
        active=True,
        ability="multiscale",
        moves=["dragonclaw", "extremespeed", "terablast"],
    )
    req["side"]["pokemon"][0]["teraType"] = "Normal"
    req["active"][0] = {
        "canTerastallize": "Normal",
        "moves": [
            {"id": m, "target": "normal", "disabled": False}
            for m in req["side"]["pokemon"][0]["moves"]
        ],
    }
    battle.parse_request(req)
    battle.switch("p2a: Arcanine", "Arcanine, L50", "100/100")

    def features(can_tera):
        obs = {k: torch.from_numpy(v[None]) for k, v in builder.build(battle).items()}
        return head._candidate_action_features(
            battle_obs=obs,
            actor_slot=0,
            can_tera_slot=torch.tensor([can_tera]),
            batch_size=1,
            device=torch.device("cpu"),
        )[0]

    pre = features(True)
    battle.active_pokemon[0].terastallize("Normal")
    post = features(False)
    # Columns 14 and 15 encode current STAB minus one and incremental STAB.
    assert pre[[0, NUM_TARGETS], 14].tolist() == [0.5, 0.0]
    assert pre[[0, NUM_TARGETS], 15].tolist() == [0.0, 0.5]
    assert post[[0, NUM_TARGETS], 14].tolist() == [0.5, 0.5]

    req["side"]["pokemon"][0]["teraType"] = "Water"
    req["active"][0]["canTerastallize"] = "Water"
    battle = _parsed_double_battle()
    battle.parse_request(req)
    battle.switch("p2a: Arcanine", "Arcanine, L50", "100/100")
    blast = NUM_TARGETS * 2
    before_water = features(True)
    assert before_water[blast, 19].item() == 1 / 4
    # Power gain includes the change from neutral Normal to super-effective Water.
    assert before_water[blast, 23].item() == pytest.approx(
        (80 / 250) * (2 * 1.5 - 1) / 4
    )
    battle.active_pokemon[0].terastallize("Water")
    water = features(False)
    assert water[blast, 14].item() == 0.5
    assert water[blast, 19].item() == 2 / 4  # Water against Fire.
    assert post[blast, 19].item() == 1 / 4
    assert water[blast, 13].item() == 1  # Equal attacking stats use Special.
    battle.active_pokemon[0].boost("atk", 1)
    assert features(False)[blast, 12].item() == 1

    battle.active_pokemon[0].terastallize("Stellar")
    obs = builder.build(battle)
    t1, t2 = head._defensive_type_pair(
        torch.from_numpy(obs["own_types"]), torch.from_numpy(obs["own_continuous"])
    )
    assert [t1[0].item(), t2[0].item()] == [
        gd.type_to_idx["dragon"],
        gd.type_to_idx["flying"],
    ]


def test_terminal_observation_has_no_choices_from_the_last_request() -> None:
    battle = _parsed_double_battle()
    battle.won_by("player")
    assert battle.available_moves[0]
    obs = ObservationBuilder(GameData()).build(battle)
    assert not obs["own_request_moves"].any()
    assert not obs["own_request_move_metadata"].any()


def _move_order_messages_by_slot(
    battle: DoubleBattle,
) -> dict[int, set[str]]:
    state = build_legal_action_state(battle)
    messages: dict[int, set[str]] = {}
    for action_index in state.slot_action_orders[0]:
        if action_index >= SWITCH_OFFSET:
            continue
        move_slot = action_index // NUM_TARGETS
        messages.setdefault(move_slot, set()).add(
            action_to_order(
                action_index,
                battle,
                slot=0,
                action_state=state,
            ).message
        )
    return messages


def test_disabled_move_keeps_showdown_request_slots_and_order_messages() -> None:
    battle = _with_disabled_fake_out()
    game_data = GameData(gen=9)

    assert [(move.id, move.request_index) for move in battle.available_moves[0]] == [
        ("protect", 2),
        ("snarl", 3),
        ("partingshot", 4),
    ]
    messages = _move_order_messages_by_slot(battle)

    assert set(messages) == {1, 2, 3}
    assert messages[1] == {"/choose move 2"}
    assert messages[2] == {"/choose move 3"}
    assert all(message.startswith("/choose move 4") for message in messages[3])
    slot_mask = build_legal_action_state(battle).joint_mask[0]
    assert (
        max_power_slot_action(battle, 0, slot_mask, np.random.default_rng(0))
        // NUM_TARGETS
        == 2  # Snarl is request move 3 after Fake Out is disabled.
    )
    observation = ObservationBuilder(game_data).build(battle)
    assert observation["own_request_moves"][0].tolist() == [
        0,
        game_data.move_to_idx["protect"],
        game_data.move_to_idx["snarl"],
        game_data.move_to_idx["partingshot"],
    ]


def test_single_move_request_uses_request_slot_zero_and_live_move_identity() -> None:
    battle = _parsed_double_battle()
    request = copy.deepcopy(battle.last_request)
    request["active"][0]["moves"] = [
        {"id": "snarl", "target": "allAdjacentFoes", "disabled": False}
    ]
    battle.parse_request(request)

    assert [(move.id, move.request_index) for move in battle.available_moves[0]] == [
        ("snarl", 1)
    ]
    assert _move_order_messages_by_slot(battle) == {0: {"/choose move 1"}}

    game_data = GameData(gen=9)
    observation = ObservationBuilder(game_data).build(battle)
    assert observation["own_request_moves"][0].tolist() == [
        game_data.move_to_idx["snarl"],
        0,
        0,
        0,
    ]


def test_forced_struggle_request_uses_struggle_identity() -> None:
    battle = _parsed_double_battle()
    request = copy.deepcopy(battle.last_request)
    request["active"][0]["moves"] = [
        {"id": "struggle", "target": "randomNormal", "disabled": False}
    ]
    battle.parse_request(request)
    game_data = GameData(gen=9)

    observation = ObservationBuilder(game_data).build(battle)

    assert observation["own_request_moves"][0].tolist() == [
        game_data.move_to_idx["struggle"],
        0,
        0,
        0,
    ]
    assert _move_order_messages_by_slot(battle) == {0: {"/choose move 1"}}


@pytest.mark.parametrize("request_index", [None, True, 5])
def test_invalid_available_move_request_index_fails_loudly(request_index) -> None:
    battle = _parsed_double_battle()
    battle.available_moves[0][0].request_index = request_index

    with pytest.raises(RuntimeError, match="request index"):
        build_legal_action_state(battle)


def test_duplicate_available_move_request_index_fails_loudly() -> None:
    battle = _parsed_double_battle()
    battle.available_moves[0][1].request_index = 1

    with pytest.raises(RuntimeError, match="share Showdown request index 1"):
        build_legal_action_state(battle)


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
