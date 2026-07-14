from unittest.mock import MagicMock

import numpy as np
import pytest
from poke_env.battle import Battle, PublicBattleEventKind

from pokerl.data import GameData
from pokerl.env.actions import (
    INTENT_MOVE,
    INTENT_PASS,
    INTENT_SWITCH,
    TARGET_OPPONENT_1,
    SemanticActionIntent,
    SemanticSlotIntent,
)
from pokerl.env.observation import (
    EVENT_KIND_TO_INDEX,
    FIELD_TURN_OFFSET,
    ObservationBuilder,
    POSITION_IMPLICIT,
    POSITION_OPP_2,
    POSITION_OWN_1,
    POSITION_OWN_2,
)
from pokerl.public_consequence import (
    PUBLIC_CONSEQUENCE_CANT,
    PUBLIC_CONSEQUENCE_EXECUTED,
    PUBLIC_CONSEQUENCE_FAILED,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
    PUBLIC_CONSEQUENCE_MISSED,
    encode_public_consequence,
    encode_public_field,
)


MOVE_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.MOVE]
CANT_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.CANT]


def _observation() -> dict[str, np.ndarray]:
    return {
        "field": np.arange(FIELD_TURN_OFFSET + 1, dtype=np.float32),
        "recent_event_meta": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "recent_event_kind": np.asarray(
            [MOVE_EVENT, MOVE_EVENT, CANT_EVENT, 0],
            dtype=np.int64,
        ),
        "recent_event_actor": np.asarray(
            [POSITION_OWN_1, POSITION_OWN_2, POSITION_OPP_2, 0],
            dtype=np.int64,
        ),
        "recent_event_move": np.asarray([11, 22, 33, 0], dtype=np.int64),
        "recent_event_flags": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "recent_event_mask": np.asarray([False, False, False, True]),
        "prev_own_valid": np.asarray([True, True]),
        "prev_own_kind": np.asarray([INTENT_MOVE, INTENT_MOVE], dtype=np.int64),
        "prev_own_move": np.asarray([11, 22], dtype=np.int64),
        "prev_own_target": np.asarray(
            [POSITION_OPP_2, POSITION_IMPLICIT],
            dtype=np.int64,
        ),
    }


def test_public_consequence_matches_exact_moves_and_public_joint_outcomes() -> None:
    obs = _observation()

    labels = encode_public_consequence(obs)

    assert labels.resolution_labels.tolist() == [
        PUBLIC_CONSEQUENCE_EXECUTED,
        PUBLIC_CONSEQUENCE_MISSED,
    ]
    assert labels.resolution_masks.tolist() == [True, True]
    assert labels.cant_targets.tolist() == [[0.0, 1.0], [0.0, 1.0]]
    assert labels.cant_masks.tolist() == [[False, True], [True, True]]
    assert labels.field_masks.tolist() == [True, True]
    expected_field = obs["field"][:PUBLIC_CONSEQUENCE_FIELD_DIM]
    np.testing.assert_array_equal(labels.field_targets[0], expected_field)
    np.testing.assert_array_equal(labels.field_targets[1], expected_field)
    np.testing.assert_array_equal(encode_public_field(obs), expected_field)


def test_public_consequence_distinguishes_failed_move_from_miss() -> None:
    obs = _observation()
    obs["recent_event_flags"][0] = [1.0, 0.0, 0.0]
    obs["prev_own_valid"][1] = False

    labels = encode_public_consequence(obs)

    assert labels.resolution_labels[0] == PUBLIC_CONSEQUENCE_FAILED
    assert labels.resolution_masks.tolist() == [True, False]


def test_public_consequence_does_not_attribute_cant_to_implicit_non_spread_move() -> (
    None
):
    obs = _observation()
    obs["prev_own_target"][0] = POSITION_IMPLICIT
    obs["recent_event_flags"][0, 2] = False

    labels = encode_public_consequence(obs)

    assert labels.resolution_masks[0]
    assert not labels.cant_masks[0].any()


def test_public_consequence_attributes_cant_to_both_for_implicit_spread_move() -> None:
    obs = _observation()
    obs["prev_own_target"][0] = POSITION_IMPLICIT
    obs["recent_event_flags"][0, 2] = True

    labels = encode_public_consequence(obs)

    assert labels.resolution_masks[0]
    assert labels.cant_masks[0].tolist() == [True, True]


def test_public_consequence_accepts_lone_cant_without_move_id() -> None:
    obs = _observation()
    obs["recent_event_kind"] = np.asarray([CANT_EVENT, 0, 0, 0])
    obs["recent_event_actor"] = np.asarray([POSITION_OWN_1, 0, 0, 0])
    obs["recent_event_move"] = np.zeros(4, dtype=np.int64)
    obs["recent_event_mask"] = np.asarray([False, True, True, True])
    obs["prev_own_valid"][1] = False

    labels = encode_public_consequence(obs)

    assert labels.resolution_labels[0] == PUBLIC_CONSEQUENCE_CANT
    assert labels.resolution_masks.tolist() == [True, False]


def test_public_consequence_masks_lone_cant_with_mismatched_move_id() -> None:
    obs = _observation()
    obs["recent_event_kind"] = np.asarray([CANT_EVENT, 0, 0, 0])
    obs["recent_event_actor"] = np.asarray([POSITION_OWN_1, 0, 0, 0])
    obs["recent_event_move"] = np.asarray([999, 0, 0, 0], dtype=np.int64)
    obs["recent_event_mask"] = np.asarray([False, True, True, True])
    obs["prev_own_valid"][1] = False

    labels = encode_public_consequence(obs)

    assert labels.resolution_labels.tolist() == [-1, -1]
    assert not labels.resolution_masks.any()


@pytest.mark.parametrize(
    "meta",
    (
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0 / 12.0],
    ),
)
def test_public_consequence_masks_incomplete_segments(meta: list[float]) -> None:
    obs = _observation()
    obs["recent_event_meta"] = np.asarray(meta, dtype=np.float32)

    labels = encode_public_consequence(obs)

    assert not labels.resolution_masks.any()
    assert not labels.cant_masks.any()
    assert not labels.field_masks.any()


def test_terminal_battle_segment_labels_without_masking_live_partial_turn() -> None:
    game_data = GameData(gen=9)
    builder = ObservationBuilder(game_data)
    battle = Battle("terminal-consequence", "agent", MagicMock(), gen=9)
    battle.player_role = "p1"
    battle.turn = 5
    battle._track_public_battle_event(
        ["", "move", "p1a: Lead", "Fake Out", "p2a: Target"]
    )
    intent = SemanticActionIntent(
        slots=(
            SemanticSlotIntent(
                kind=INTENT_MOVE,
                move_id="fakeout",
                target=TARGET_OPPONENT_1,
            ),
            SemanticSlotIntent(kind=INTENT_PASS),
        )
    )
    observation = _observation()

    builder._encode_transition_history(battle, intent, observation)
    live_labels = encode_public_consequence(observation)
    assert observation["recent_event_meta"].tolist() == [1.0, 1.0, 0.0]
    assert not live_labels.resolution_masks.any()

    battle.won_by("agent")
    builder._encode_transition_history(battle, intent, observation)
    terminal_labels = encode_public_consequence(observation)

    assert observation["recent_event_meta"].tolist() == [1.0, 0.0, 0.0]
    assert terminal_labels.resolution_labels.tolist() == [
        PUBLIC_CONSEQUENCE_EXECUTED,
        -1,
    ]
    assert terminal_labels.resolution_masks.tolist() == [True, False]
    assert terminal_labels.field_masks.tolist() == [True, False]


@pytest.mark.parametrize("mode", ("transformed", "ambiguous", "no_event"))
def test_public_consequence_masks_non_exact_move_alignment(mode: str) -> None:
    obs = _observation()
    obs["prev_own_valid"][1] = False
    if mode == "transformed":
        obs["recent_event_move"][0] = 999
    elif mode == "ambiguous":
        obs["recent_event_kind"][1] = MOVE_EVENT
        obs["recent_event_actor"][1] = POSITION_OWN_1
        obs["recent_event_move"][1] = 11
    else:
        obs["recent_event_mask"][0] = True

    labels = encode_public_consequence(obs)

    assert not labels.resolution_masks[0]
    assert not labels.cant_masks[0].any()
    assert not labels.field_masks[0]


def test_public_consequence_masks_invalid_intent_and_missing_schema() -> None:
    obs = _observation()
    obs["prev_own_kind"][:] = INTENT_SWITCH

    labels = encode_public_consequence(obs)
    missing = encode_public_consequence({"field": obs["field"]})

    assert not labels.resolution_masks.any()
    assert not missing.resolution_masks.any()
    assert missing.resolution_labels.tolist() == [-1, -1]
    assert encode_public_field({}) is None


def test_public_consequence_does_not_supervise_cant_for_self_target() -> None:
    obs = _observation()
    obs["prev_own_target"][0] = POSITION_OWN_1
    obs["prev_own_valid"][1] = False

    labels = encode_public_consequence(obs)

    assert labels.resolution_masks[0]
    assert not labels.cant_masks[0].any()
