from unittest.mock import MagicMock

import numpy as np
import pytest
from poke_env.battle import Battle, PublicBattleEventKind

from pokerl.data import MOVE_SCOPE_SCOPED, MOVE_SCOPE_TARGETED, GameData
from pokerl.env.actions import (
    INTENT_MOVE,
    INTENT_PASS,
    TARGET_OPPONENT_1,
    SemanticActionIntent,
    SemanticSlotIntent,
)
from pokerl.env.observation import (
    EVENT_FLAG_ACTION_FAILED,
    EVENT_FLAG_NO_TARGET,
    EVENT_FLAG_SPREAD,
    EVENT_KIND_TO_INDEX,
    NUM_PUBLIC_EVENT_FLAGS,
    NUM_PUBLIC_TARGET_POSITIONS,
    NUM_TARGET_OUTCOME_FEATURES,
    ObservationBuilder,
    POSITION_IMPLICIT,
    POSITION_OPP_1,
    POSITION_OWN_1,
    TARGET_OUTCOME_DECLARED,
    TARGET_OUTCOME_FAILED,
    TARGET_OUTCOME_HIT_COUNT,
    TARGET_OUTCOME_IMMUNE,
    TARGET_OUTCOME_MISSED,
    TARGET_OUTCOME_PRESENT,
)
from pokerl.public_consequence import (
    MOVE_BRANCH_ACTION_FAILED,
    MOVE_BRANCH_NO_TARGET,
    MOVE_BRANCH_SCOPED,
    MOVE_BRANCH_TARGET_PHASE,
    NONMOVE_REASON_CANT,
    NONMOVE_REASON_FAINTED,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
    TARGET_INTERACTION_FAILED,
    TARGET_INTERACTION_IMMUNE,
    TARGET_INTERACTION_MISS,
    encode_public_consequence,
    encode_public_field,
)


MOVE_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.MOVE]
CANT_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.CANT]
FAINT_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.FAINT]


def _observation() -> dict[str, np.ndarray]:
    outcomes = np.zeros(
        (4, NUM_PUBLIC_TARGET_POSITIONS, NUM_TARGET_OUTCOME_FEATURES),
        dtype=np.float32,
    )
    outcomes[0, 2, TARGET_OUTCOME_PRESENT] = 1.0
    outcomes[0, 2, TARGET_OUTCOME_DECLARED] = 1.0
    return {
        "field": np.linspace(0.0, 1.0, PUBLIC_CONSEQUENCE_FIELD_DIM).astype(np.float32),
        "recent_event_meta": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "recent_event_kind": np.asarray([MOVE_EVENT, 0, 0, 0], dtype=np.int64),
        "recent_event_actor": np.asarray([POSITION_OWN_1, 0, 0, 0]),
        "recent_event_move": np.asarray([11, 0, 0, 0], dtype=np.int64),
        "recent_event_source_move": np.zeros(4, dtype=np.int64),
        "recent_event_move_scope": np.asarray(
            [MOVE_SCOPE_TARGETED, 0, 0, 0], dtype=np.int64
        ),
        "recent_event_flags": np.zeros((4, NUM_PUBLIC_EVENT_FLAGS), dtype=np.float32),
        "recent_event_target_outcomes": outcomes,
        "recent_event_target_activation_move": np.zeros(
            (4, NUM_PUBLIC_TARGET_POSITIONS), dtype=np.int64
        ),
        "recent_event_mask": np.asarray([False, True, True, True]),
        "prev_own_valid": np.asarray([True, False]),
        "prev_own_kind": np.asarray([INTENT_MOVE, INTENT_PASS], dtype=np.int64),
        "prev_own_move": np.asarray([11, 0], dtype=np.int64),
        "prev_own_target": np.asarray([POSITION_OPP_1, POSITION_IMPLICIT]),
    }


def test_target_phase_labels_declared_route_and_conditional_interactions() -> None:
    obs = _observation()

    labels = encode_public_consequence(obs)

    assert labels.move_emitted_targets.tolist() == [1.0, 0.0]
    assert labels.move_emitted_masks.tolist() == [True, False]
    assert labels.move_branch_labels[0] == MOVE_BRANCH_TARGET_PHASE
    assert labels.move_branch_masks.tolist() == [True, False]
    assert labels.route_targets[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert labels.route_masks[0].all()
    assert labels.interaction_masks[0].tolist() == [False, False, True, False]
    assert labels.interaction_targets[0, 2] == 0.0
    assert not labels.interaction_reason_masks[0].any()
    assert labels.cant_masks[0].tolist() == [True, False]
    np.testing.assert_array_equal(labels.field_targets[0], obs["field"])
    np.testing.assert_array_equal(encode_public_field(obs), obs["field"])


def test_spread_targets_keep_separate_miss_and_immunity_markers() -> None:
    obs = _observation()
    obs["recent_event_flags"][0, EVENT_FLAG_SPREAD] = 1.0
    obs["prev_own_target"][0] = POSITION_IMPLICIT
    outcomes = obs["recent_event_target_outcomes"][0]
    outcomes[2, TARGET_OUTCOME_MISSED] = 1.0
    outcomes[3, TARGET_OUTCOME_PRESENT] = 1.0
    outcomes[3, TARGET_OUTCOME_DECLARED] = 1.0
    outcomes[3, TARGET_OUTCOME_IMMUNE] = 1.0

    labels = encode_public_consequence(obs)

    assert labels.route_targets[0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert labels.interaction_targets[0].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert labels.interaction_reason_labels[0, 2] == TARGET_INTERACTION_MISS
    assert labels.interaction_reason_labels[0, 3] == TARGET_INTERACTION_IMMUNE
    assert labels.cant_masks[0].tolist() == [True, True]


@pytest.mark.parametrize(
    ("flag", "scope", "expected_branch"),
    (
        (EVENT_FLAG_ACTION_FAILED, MOVE_SCOPE_TARGETED, MOVE_BRANCH_ACTION_FAILED),
        (EVENT_FLAG_NO_TARGET, MOVE_SCOPE_TARGETED, MOVE_BRANCH_NO_TARGET),
        (None, MOVE_SCOPE_SCOPED, MOVE_BRANCH_SCOPED),
    ),
)
def test_move_branch_is_explicit_and_does_not_invent_target_routes(
    flag: int | None,
    scope: int,
    expected_branch: int,
) -> None:
    obs = _observation()
    obs["recent_event_move_scope"][0] = scope
    obs["recent_event_target_outcomes"][:] = 0.0
    if flag is not None:
        obs["recent_event_flags"][0, flag] = 1.0

    labels = encode_public_consequence(obs)

    assert labels.move_branch_labels[0] == expected_branch
    assert not labels.route_masks[0].any()


@pytest.mark.parametrize(
    ("event_kind", "expected_reason"),
    ((CANT_EVENT, NONMOVE_REASON_CANT), (FAINT_EVENT, NONMOVE_REASON_FAINTED)),
)
def test_nonemitted_move_has_conditional_public_reason(
    event_kind: int,
    expected_reason: int,
) -> None:
    obs = _observation()
    obs["recent_event_kind"][0] = event_kind
    obs["recent_event_move"][0] = 0 if event_kind == CANT_EVENT else 11
    obs["recent_event_move_scope"][0] = 0

    labels = encode_public_consequence(obs)

    assert labels.move_emitted_targets[0] == 0.0
    assert labels.move_emitted_masks[0]
    assert labels.nonmove_reason_labels[0] == expected_reason
    assert labels.nonmove_reason_masks[0]
    assert not labels.move_branch_masks[0]


def test_called_move_aligns_to_the_selected_source_move() -> None:
    obs = _observation()
    obs["recent_event_move"][0] = 999
    obs["recent_event_source_move"][0] = 11

    labels = encode_public_consequence(obs)

    assert labels.move_emitted_masks[0]
    assert labels.move_branch_labels[0] == MOVE_BRANCH_TARGET_PHASE


def test_public_hit_count_is_conditional_on_unmarked_declared_target() -> None:
    obs = _observation()
    obs["recent_event_target_outcomes"][0, 2, TARGET_OUTCOME_HIT_COUNT] = 0.3

    labels = encode_public_consequence(obs)

    assert labels.hit_count_masks[0, 2]
    assert labels.hit_count_labels[0, 2] == 2


def test_minor_target_presence_is_not_mislabeled_as_move_route() -> None:
    obs = _observation()
    outcomes = obs["recent_event_target_outcomes"][0]
    outcomes[0, TARGET_OUTCOME_PRESENT] = 1.0
    outcomes[0, TARGET_OUTCOME_FAILED] = 1.0

    labels = encode_public_consequence(obs)

    assert labels.route_targets[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert not labels.interaction_masks[0, 0]


def test_parting_shot_subeffect_failure_remains_target_interaction() -> None:
    game_data = GameData(gen=9)
    builder = ObservationBuilder(game_data)
    battle = Battle("parting-shot-clear-amulet", "agent", MagicMock(), gen=9)
    battle.player_role = "p1"
    battle.turn = 3
    battle._track_public_battle_event(
        ["", "move", "p1a: Pivot", "Parting Shot", "p2a: Target"]
    )
    battle._track_public_battle_event(
        [
            "",
            "-fail",
            "p2a: Target",
            "unboost",
            "[from] item: Clear Amulet",
            "[of] p1a: Pivot",
        ]
    )
    battle._track_public_battle_event(
        ["", "switch", "p1a: Replacement", "Incineroar, L50", "100/100"]
    )
    battle.won_by("agent")
    intent = SemanticActionIntent(
        slots=(
            SemanticSlotIntent(
                kind=INTENT_MOVE,
                move_id="partingshot",
                target=TARGET_OPPONENT_1,
            ),
            SemanticSlotIntent(kind=INTENT_PASS),
        )
    )
    observation: dict[str, np.ndarray] = {}

    builder._encode_transition_history(battle, intent, observation)
    labels = encode_public_consequence(observation)

    assert observation["recent_event_flags"][0, EVENT_FLAG_ACTION_FAILED] == 0.0
    assert labels.move_branch_labels[0] == MOVE_BRANCH_TARGET_PHASE
    assert labels.route_targets[0, 2] == 1.0
    assert labels.interaction_targets[0, 2] == 1.0
    assert labels.interaction_reason_labels[0, 2] == TARGET_INTERACTION_FAILED


@pytest.mark.parametrize(
    "meta",
    ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.1]),
)
def test_incomplete_or_overflowed_segments_are_not_supervised(
    meta: list[float],
) -> None:
    obs = _observation()
    obs["recent_event_meta"] = np.asarray(meta, dtype=np.float32)

    labels = encode_public_consequence(obs)

    assert not labels.move_emitted_masks.any()
    assert not labels.route_masks.any()
    assert not labels.field_masks.any()
