from types import SimpleNamespace

import numpy as np
import torch
from poke_env.battle import (
    Move,
    Pokemon,
    PublicBattleEvent,
    PublicBattleEventKind,
    PublicBattleEventSegment,
    PublicMoveTargetOutcome,
)
from pokerl.config import ModelConfig
from pokerl.data import MOVE_SCOPE_TARGETED, GameData
from pokerl.env.actions import (
    INTENT_MOVE,
    INTENT_SWITCH,
    LegalActionState,
    SemanticActionIntent,
    SemanticSlotIntent,
    TERA_SLOT_1,
    semantic_action_intent,
)
from pokerl.env.observation import (
    ACTIVE_MECHANICS_ENTERED,
    ACTIVE_MECHANICS_FIRST_TURN,
    ACTIVE_MECHANICS_ACTIVE_TURNS,
    ACTIVE_MECHANICS_PROTECT_CHAIN,
    ACTIVE_MECHANICS_RESOLUTION_OFFSET,
    ACTIVE_MECHANICS_TERA_EVENT,
    ACTIVE_RESOLUTION_MOVE,
    ACTIVE_RESOLUTION_NONE,
    ACTIVE_RESOLUTION_SWITCH,
    CONTINUOUS_DIM,
    ACTIVE_TURNS_OFFSET,
    FIRST_TURN_OFFSET,
    MAX_RECENT_EVENTS,
    NUM_PUBLIC_EVENT_FLAGS,
    POSITION_IMPLICIT,
    POSITION_OPP_1,
    POSITION_OPP_2,
    POSITION_OWN_2,
    PROTECT_COUNTER_OFFSET,
    TARGET_OUTCOME_HIT_COUNT,
    TARGET_OUTCOME_MISSED,
    TARGET_OUTCOME_PRESENT,
    TARGET_OUTCOME_PROTECTED,
    ObservationBuilder,
    build_team_preview_observation,
)
from pokerl.model.agent import PokeAgent
from pokerl.model.encoders import RecentEventEncoder


PACKED_TEAM = "]".join(
    [
        "Pikachu||lightball|static|thunderbolt,protect|Timid|||||50|",
        "Raichu||lifeorb|static|thunderbolt,protect|Timid|||||50|",
        "Bulbasaur||eviolite|overgrow|gigadrain,protect|Bold|||||50|",
        "Charmander||charcoal|blaze|flamethrower,protect|Timid|||||50|",
        "Squirtle||leftovers|torrent|surf,protect|Bold|||||50|",
        "Eevee||choicescarf|runaway|quickattack,protect|Jolly|||||50|",
    ]
)


def test_observation_encodes_public_order_and_semantic_own_intent() -> None:
    game_data = GameData(gen=9)
    builder = ObservationBuilder(game_data, team_size=6)
    segment = PublicBattleEventSegment(
        events=(
            PublicBattleEvent(
                PublicBattleEventKind.MOVE,
                actor="p2a: Opponent",
                target="p1b: Ally",
                move="Moonblast",
            ),
            PublicBattleEvent(
                PublicBattleEventKind.SWITCH,
                actor="p1b: Ally",
                details="Incineroar, L50, M",
            ),
            PublicBattleEvent(
                PublicBattleEventKind.TERA,
                actor="p2b: Partner",
                details="Grass",
            ),
        ),
        turn=2,
        partial=True,
        overflow_count=2,
    )
    battle = SimpleNamespace(player_role="p1", recent_public_events=segment)
    own_intent = SemanticActionIntent(
        slots=(
            SemanticSlotIntent(
                kind=INTENT_MOVE,
                move_id="fakeout",
                target=0,
                terastallize=True,
            ),
            SemanticSlotIntent(
                kind=INTENT_SWITCH,
                switch_species="rillaboom",
            ),
        )
    )
    obs: dict[str, np.ndarray] = {}

    builder._encode_transition_history(battle, own_intent, obs)

    assert obs["recent_event_mask"].tolist()[:4] == [False, False, False, True]
    assert obs["recent_event_actor"].tolist()[:3] == [POSITION_OPP_1, 2, 4]
    assert obs["recent_event_target"][0] == POSITION_OWN_2
    assert obs["recent_event_move"][0] == game_data.move_to_idx["moonblast"]
    assert obs["recent_event_species"][1] == game_data.species_to_idx["incineroar"]
    np.testing.assert_allclose(obs["recent_event_meta"], [1.0, 1.0, 2 / 12])
    assert obs["prev_own_target"][0] == POSITION_OPP_1
    assert obs["prev_own_tera"].tolist() == [True, False]
    assert obs["prev_own_switch_species"][1] == game_data.species_to_idx["rillaboom"]


def test_active_mechanics_bind_public_events_to_current_slots() -> None:
    game_data = GameData(gen=9)
    builder = ObservationBuilder(game_data, team_size=6)
    segment = PublicBattleEventSegment(
        events=(
            PublicBattleEvent(
                PublicBattleEventKind.MOVE,
                actor="p2a: Opponent",
                move="Moonblast",
            ),
            PublicBattleEvent(
                PublicBattleEventKind.SWITCH,
                actor="p1b: Ally",
                details="Incineroar, L50, M",
            ),
            PublicBattleEvent(
                PublicBattleEventKind.TERA,
                actor="p2b: Partner",
                details="Grass",
            ),
        ),
        turn=2,
        partial=False,
    )
    obs: dict[str, np.ndarray] = {}
    builder._encode_transition_history(
        SimpleNamespace(player_role="p1", recent_public_events=segment),
        None,
        obs,
    )
    obs["own_active_indices"] = np.asarray([0, 1], dtype=np.int64)
    obs["opp_active_indices"] = np.asarray([0, 1], dtype=np.int64)
    obs["own_active_present"] = np.asarray([True, True])
    obs["opp_active_present"] = np.asarray([True, True])
    obs["own_continuous"] = np.zeros((6, CONTINUOUS_DIM), dtype=np.float32)
    obs["opp_continuous"] = np.zeros((6, CONTINUOUS_DIM), dtype=np.float32)
    obs["own_continuous"][0, FIRST_TURN_OFFSET] = 1.0
    obs["own_continuous"][0, PROTECT_COUNTER_OFFSET] = 0.5
    obs["own_continuous"][0, ACTIVE_TURNS_OFFSET] = 0.5

    builder._encode_active_mechanics(obs)

    mechanics = obs["active_mechanics"]
    assert mechanics[0, ACTIVE_MECHANICS_FIRST_TURN] == 1.0
    assert mechanics[0, ACTIVE_MECHANICS_PROTECT_CHAIN] == 0.5
    assert mechanics[0, ACTIVE_MECHANICS_ACTIVE_TURNS] == 0.5
    assert mechanics[1, ACTIVE_MECHANICS_ENTERED] == 1.0
    assert (
        mechanics[
            1,
            ACTIVE_MECHANICS_RESOLUTION_OFFSET + ACTIVE_RESOLUTION_SWITCH,
        ]
        == 1.0
    )
    assert (
        mechanics[
            2,
            ACTIVE_MECHANICS_RESOLUTION_OFFSET + ACTIVE_RESOLUTION_MOVE,
        ]
        == 1.0
    )
    assert obs["active_mechanics_move"][2] == game_data.move_to_idx["moonblast"]
    assert mechanics[3, ACTIVE_MECHANICS_TERA_EVENT] == 1.0


def test_active_mechanics_do_not_claim_a_resolution_after_event_overflow() -> None:
    game_data = GameData(gen=9)
    builder = ObservationBuilder(game_data, team_size=6)
    segment = PublicBattleEventSegment(
        events=(
            PublicBattleEvent(
                PublicBattleEventKind.MOVE,
                actor="p2a: Opponent",
                move="Moonblast",
            ),
        ),
        turn=3,
        partial=False,
        overflow_count=1,
    )
    obs: dict[str, np.ndarray] = {}
    builder._encode_transition_history(
        SimpleNamespace(player_role="p1", recent_public_events=segment),
        None,
        obs,
    )
    obs["own_active_indices"] = np.asarray([0, 1], dtype=np.int64)
    obs["opp_active_indices"] = np.asarray([0, 1], dtype=np.int64)
    obs["own_active_present"] = np.asarray([True, True])
    obs["opp_active_present"] = np.asarray([True, True])
    obs["own_continuous"] = np.zeros((6, CONTINUOUS_DIM), dtype=np.float32)
    obs["opp_continuous"] = np.zeros((6, CONTINUOUS_DIM), dtype=np.float32)

    builder._encode_active_mechanics(obs)

    mechanics = obs["active_mechanics"]
    assert (
        mechanics[
            2,
            ACTIVE_MECHANICS_RESOLUTION_OFFSET + ACTIVE_RESOLUTION_NONE,
        ]
        == 1.0
    )
    assert obs["active_mechanics_move"][2] == 0


def test_active_mechanics_do_not_bind_tera_event_to_switch_replacement() -> None:
    game_data = GameData(gen=9)
    builder = ObservationBuilder(game_data, team_size=6)
    segment = PublicBattleEventSegment(
        events=(
            PublicBattleEvent(
                PublicBattleEventKind.TERA,
                actor="p2a: Outgoing",
                details="Grass",
            ),
            PublicBattleEvent(
                PublicBattleEventKind.SWITCH,
                actor="p2a: Replacement",
                details="Incineroar, L50, M",
            ),
        ),
        turn=3,
        partial=False,
    )
    obs: dict[str, np.ndarray] = {}
    builder._encode_transition_history(
        SimpleNamespace(player_role="p1", recent_public_events=segment),
        None,
        obs,
    )
    obs["own_active_indices"] = np.asarray([0, 1], dtype=np.int64)
    obs["opp_active_indices"] = np.asarray([0, 1], dtype=np.int64)
    obs["own_active_present"] = np.asarray([True, True])
    obs["opp_active_present"] = np.asarray([True, True])
    obs["own_continuous"] = np.zeros((6, CONTINUOUS_DIM), dtype=np.float32)
    obs["opp_continuous"] = np.zeros((6, CONTINUOUS_DIM), dtype=np.float32)

    builder._encode_active_mechanics(obs)

    mechanics = obs["active_mechanics"]
    assert mechanics[2, ACTIVE_MECHANICS_TERA_EVENT] == 0.0
    assert mechanics[2, ACTIVE_MECHANICS_ENTERED] == 1.0
    assert (
        mechanics[
            2,
            ACTIVE_MECHANICS_RESOLUTION_OFFSET + ACTIVE_RESOLUTION_SWITCH,
        ]
        == 1.0
    )


def test_team_preview_history_is_zero_and_fixed_shape() -> None:
    obs = build_team_preview_observation(
        GameData(gen=9),
        team_size=6,
        own_packed_team=PACKED_TEAM,
        opponent_packed_team=PACKED_TEAM,
    )

    assert obs["recent_event_kind"].shape == (MAX_RECENT_EVENTS,)
    assert obs["recent_event_mask"].all()
    assert not obs["prev_own_valid"].any()
    assert not obs["recent_event_meta"].any()


def test_observation_encodes_called_move_and_per_target_resolution() -> None:
    game_data = GameData(gen=9)
    builder = ObservationBuilder(game_data, team_size=6)
    segment = PublicBattleEventSegment(
        events=(
            PublicBattleEvent(
                PublicBattleEventKind.MOVE,
                actor="p2a: Opponent",
                move="Rock Slide",
                spread=True,
                source_effect="move: Sleep Talk",
                target_outcomes=(
                    PublicMoveTargetOutcome(
                        target="p1a: Lead",
                        missed=True,
                    ),
                    PublicMoveTargetOutcome(
                        target="p1b: Partner",
                        hit_count=2,
                        activations=("move: Protect",),
                    ),
                ),
            ),
        ),
        turn=2,
        partial=False,
    )
    battle = SimpleNamespace(player_role="p1", recent_public_events=segment)
    obs: dict[str, np.ndarray] = {}

    builder._encode_transition_history(battle, None, obs)

    assert obs["recent_event_source_move"][0] == game_data.move_to_idx["sleeptalk"]
    assert obs["recent_event_move_scope"][0] == MOVE_SCOPE_TARGETED
    lead = obs["recent_event_target_outcomes"][0, 0]
    partner = obs["recent_event_target_outcomes"][0, 1]
    assert lead[TARGET_OUTCOME_PRESENT] == 1.0
    assert lead[TARGET_OUTCOME_MISSED] == 1.0
    assert partner[TARGET_OUTCOME_PROTECTED] == 1.0
    assert partner[TARGET_OUTCOME_HIT_COUNT] == 0.2
    assert (
        obs["recent_event_target_activation_move"][0, 1]
        == (game_data.move_to_idx["protect"])
    )


def test_semantic_intent_resolves_request_local_move_and_switch_slots() -> None:
    move = Move("fakeout", gen=9)
    switch = Pokemon(gen=9, species="rillaboom")
    state = LegalActionState(
        slot_action_orders=(
            {1: move},
            {20: switch},
        ),
        slot_masks=(np.zeros(27, dtype=bool), np.zeros(27, dtype=bool)),
        joint_mask=np.zeros((28, 27), dtype=bool),
        can_tera=np.array([True, False]),
    )

    intent = semantic_action_intent(np.array([1, 20, TERA_SLOT_1]), state)

    assert intent.slots[0].kind == INTENT_MOVE
    assert intent.slots[0].move_id == "fakeout"
    assert intent.slots[0].terastallize
    assert intent.slots[1].kind == INTENT_SWITCH
    assert intent.slots[1].switch_species == "rillaboom"


def test_recent_event_encoder_is_zero_without_history_and_receives_gradient() -> None:
    encoder = RecentEventEncoder(
        num_moves=16,
        num_species=16,
        num_types=8,
        num_event_kinds=7,
        num_positions=6,
        num_intent_kinds=4,
    )
    obs = {
        "field": torch.zeros(2, 36),
        "recent_event_kind": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_actor": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_target": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_move": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_species": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_tera_type": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_flags": torch.zeros(2, 12, NUM_PUBLIC_EVENT_FLAGS),
        "recent_event_mask": torch.ones(2, 12, dtype=torch.bool),
        "recent_event_meta": torch.zeros(2, 3),
        "prev_own_kind": torch.zeros(2, 2, dtype=torch.long),
        "prev_own_move": torch.zeros(2, 2, dtype=torch.long),
        "prev_own_target": torch.full((2, 2), POSITION_IMPLICIT, dtype=torch.long),
        "prev_own_switch_species": torch.zeros(2, 2, dtype=torch.long),
        "prev_own_tera": torch.zeros(2, 2, dtype=torch.bool),
        "prev_own_valid": torch.zeros(2, 2, dtype=torch.bool),
    }
    assert torch.equal(encoder(obs), torch.zeros(2, encoder.out_dim))
    assert torch.equal(
        encoder({"field": obs["field"]}),
        encoder(obs),
    )

    obs["recent_event_kind"][0, 0] = 1
    obs["recent_event_actor"][0, 0] = POSITION_OPP_2
    obs["recent_event_move"][0, 0] = 3
    obs["recent_event_mask"][0, 0] = False
    obs["recent_event_meta"][0, 0] = 1.0
    output = encoder(obs)
    output[0].sum().backward()

    assert output[0].abs().sum() > 0
    assert encoder.event_projection[0].weight.grad is not None


def test_team_preview_bypasses_zero_recent_event_history() -> None:
    game_data = GameData(gen=9)
    preview_obs = build_team_preview_observation(
        game_data,
        team_size=6,
        own_packed_team=PACKED_TEAM,
        opponent_packed_team=PACKED_TEAM,
    )
    obs_batch = {
        key: torch.from_numpy(np.stack([value, value]))
        for key, value in preview_obs.items()
    }
    agent = PokeAgent(ModelConfig(), game_data).eval()

    def fail_if_called(_obs):
        raise AssertionError("team preview evaluated zero recent-event history")

    agent.spatial.recent_event_encoder.forward = fail_if_called
    with torch.inference_mode():
        result = agent.team_preview(
            obs_batch,
            include_value=False,
            include_policy_stats=False,
            include_opponent_preview=False,
        )

    assert result["picks"].shape == (2, 4)
