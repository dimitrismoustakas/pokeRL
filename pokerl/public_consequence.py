"""Public, action-aligned consequence labels for auxiliary training.

The labels are a conditional likelihood of exact public protocol facts.  They
do not assign value to those facts: PPO's terminal reward remains the only
source of strategic preference.  The scope is the immediate resolved segment;
delayed effects are retained in recurrent public history but are not attributed
to an earlier action when the protocol does not provide exact causal alignment.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from poke_env.battle import PublicBattleEventKind

from .data import MOVE_SCOPE_SCOPED, MOVE_SCOPE_TARGETED, UNKNOWN_IDX
from .env.actions import INTENT_MOVE
from .env.observation import (
    EVENT_FLAG_ACTION_FAILED,
    EVENT_FLAG_NO_TARGET,
    EVENT_FLAG_SPREAD,
    EVENT_KIND_TO_INDEX,
    FIELD_TRICK_ROOM_OFFSET,
    NUM_PUBLIC_EVENT_FLAGS,
    NUM_PUBLIC_TARGET_POSITIONS,
    NUM_TARGET_OUTCOME_FEATURES,
    POSITION_IMPLICIT,
    POSITION_OPP_1,
    POSITION_OPP_2,
    POSITION_OWN_1,
    POSITION_OWN_2,
    TARGET_OUTCOME_BLOCKED,
    TARGET_OUTCOME_DECLARED,
    TARGET_OUTCOME_FAILED,
    TARGET_OUTCOME_HIT_COUNT,
    TARGET_OUTCOME_IMMUNE,
    TARGET_OUTCOME_MISSED,
    TARGET_OUTCOME_PROTECTED,
)


NONMOVE_REASON_CANT = 0
NONMOVE_REASON_FAINTED = 1
NUM_NONMOVE_REASONS = 2

MOVE_BRANCH_ACTION_FAILED = 0
MOVE_BRANCH_NO_TARGET = 1
MOVE_BRANCH_SCOPED = 2
MOVE_BRANCH_TARGET_PHASE = 3
NUM_MOVE_BRANCHES = 4

TARGET_INTERACTION_MISS = 0
TARGET_INTERACTION_IMMUNE = 1
TARGET_INTERACTION_PROTECTED = 2
TARGET_INTERACTION_BLOCKED = 3
TARGET_INTERACTION_FAILED = 4
NUM_TARGET_INTERACTION_REASONS = 5

MAX_PUBLIC_HIT_COUNT = 10
PUBLIC_CONSEQUENCE_FIELD_DIM = FIELD_TRICK_ROOM_OFFSET + 1
PUBLIC_CONSEQUENCE_OBSERVATION_KEYS = (
    "field",
    "recent_event_meta",
    "recent_event_kind",
    "recent_event_actor",
    "recent_event_move",
    "recent_event_source_move",
    "recent_event_move_scope",
    "recent_event_flags",
    "recent_event_target_outcomes",
    "recent_event_target_activation_move",
    "recent_event_mask",
    "prev_own_valid",
    "prev_own_kind",
    "prev_own_move",
    "prev_own_target",
)

_MOVE_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.MOVE]
_CANT_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.CANT]
_FAINT_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.FAINT]
_OWN_POSITIONS = (POSITION_OWN_1, POSITION_OWN_2)
_OPPONENT_POSITIONS = (POSITION_OPP_1, POSITION_OPP_2)


@dataclass(frozen=True, slots=True)
class PublicConsequenceLabels:
    """Trainer-only labels aligned to the two selected own move intents."""

    move_emitted_targets: np.ndarray  # (2,) float32
    move_emitted_masks: np.ndarray  # (2,) bool
    nonmove_reason_labels: np.ndarray  # (2,) int64
    nonmove_reason_masks: np.ndarray  # (2,) bool
    move_branch_labels: np.ndarray  # (2,) int64
    move_branch_masks: np.ndarray  # (2,) bool
    route_targets: np.ndarray  # (2, 4) float32
    route_masks: np.ndarray  # (2, 4) bool
    interaction_targets: np.ndarray  # (2, 4) float32
    interaction_masks: np.ndarray  # (2, 4) bool
    interaction_reason_labels: np.ndarray  # (2, 4) int64
    interaction_reason_masks: np.ndarray  # (2, 4) bool
    hit_count_labels: np.ndarray  # (2, 4) int64, zero-based 1..10
    hit_count_masks: np.ndarray  # (2, 4) bool
    cant_targets: np.ndarray  # (2, 2) float32
    cant_masks: np.ndarray  # (2, 2) bool
    field_targets: np.ndarray  # (2, 33) float32
    field_masks: np.ndarray  # (2,) bool


def encode_public_field(
    observation: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    """Return the public field prefix predicted by the consequence head."""

    field_value = observation.get("field")
    if field_value is None:
        return None
    field = np.asarray(field_value)
    if field.ndim != 1 or field.shape[0] < PUBLIC_CONSEQUENCE_FIELD_DIM:
        return None
    return np.asarray(field[:PUBLIC_CONSEQUENCE_FIELD_DIM], dtype=np.float32).copy()


def encode_public_consequence(
    observation: Mapping[str, np.ndarray],
) -> PublicConsequenceLabels:
    """Encode exact public outcomes for the previously selected own moves."""

    labels = _empty_labels()
    arrays = _validated_arrays(observation)
    if arrays is None or not _is_complete_segment(arrays["recent_event_meta"]):
        return labels

    event_mask = arrays["recent_event_mask"]
    event_kind = arrays["recent_event_kind"]
    event_actor = arrays["recent_event_actor"]
    event_move = arrays["recent_event_move"]
    event_source_move = arrays["recent_event_source_move"]
    event_move_scope = arrays["recent_event_move_scope"]
    event_flags = arrays["recent_event_flags"]
    event_target_outcomes = arrays["recent_event_target_outcomes"]
    public_field = encode_public_field(observation)
    present = ~event_mask

    opponent_cant = np.asarray(
        [
            np.any(
                present
                & (event_kind == _CANT_EVENT)
                & (event_actor == opponent_position)
            )
            for opponent_position in _OPPONENT_POSITIONS
        ],
        dtype=np.float32,
    )

    for slot, own_position in enumerate(_OWN_POSITIONS):
        if not arrays["prev_own_valid"][slot]:
            continue
        if arrays["prev_own_kind"][slot] != INTENT_MOVE:
            continue
        own_move = int(arrays["prev_own_move"][slot])
        if own_move == UNKNOWN_IDX:
            continue

        own_move_events = np.flatnonzero(
            present & (event_kind == _MOVE_EVENT) & (event_actor == own_position)
        )
        exact_move_events = own_move_events[
            (event_move[own_move_events] == own_move)
            | (event_source_move[own_move_events] == own_move)
        ]
        own_cant_events = np.flatnonzero(
            present & (event_kind == _CANT_EVENT) & (event_actor == own_position)
        )
        compatible_cant_events = own_cant_events[
            (event_move[own_cant_events] == UNKNOWN_IDX)
            | (event_move[own_cant_events] == own_move)
        ]
        own_faint_events = np.flatnonzero(
            present & (event_kind == _FAINT_EVENT) & (event_actor == own_position)
        )

        event_index: int | None = None
        nonmove_reason: int | None = None
        if (
            own_move_events.size == 1
            and exact_move_events.size == 1
            and own_cant_events.size == 0
        ):
            event_index = int(exact_move_events[0])
            labels.move_emitted_targets[slot] = 1.0
            labels.move_emitted_masks[slot] = True
        elif (
            own_move_events.size == 0
            and own_cant_events.size == 1
            and compatible_cant_events.size == 1
        ):
            event_index = int(compatible_cant_events[0])
            nonmove_reason = NONMOVE_REASON_CANT
        elif (
            own_move_events.size == 0
            and own_cant_events.size == 0
            and own_faint_events.size == 1
        ):
            event_index = int(own_faint_events[0])
            nonmove_reason = NONMOVE_REASON_FAINTED
        else:
            continue

        if nonmove_reason is not None:
            labels.move_emitted_targets[slot] = 0.0
            labels.move_emitted_masks[slot] = True
            labels.nonmove_reason_labels[slot] = nonmove_reason
            labels.nonmove_reason_masks[slot] = True
        else:
            assert event_index is not None
            branch = _move_branch(
                event_scope=int(event_move_scope[event_index]),
                event_flags=event_flags[event_index],
            )
            if branch is not None:
                labels.move_branch_labels[slot] = branch
                labels.move_branch_masks[slot] = True
                if branch == MOVE_BRANCH_TARGET_PHASE:
                    _encode_target_phase(
                        labels,
                        slot,
                        event_target_outcomes[event_index],
                    )

        assert event_index is not None
        labels.cant_targets[slot] = opponent_cant
        labels.cant_masks[slot] = _opponent_cant_mask(
            int(arrays["prev_own_target"][slot]),
            spread=(
                event_kind[event_index] == _MOVE_EVENT
                and bool(event_flags[event_index, EVENT_FLAG_SPREAD])
            ),
        )
        if public_field is not None:
            labels.field_targets[slot] = public_field
            labels.field_masks[slot] = True

    return labels


def _move_branch(*, event_scope: int, event_flags: np.ndarray) -> int | None:
    if bool(event_flags[EVENT_FLAG_NO_TARGET]):
        return MOVE_BRANCH_NO_TARGET
    if bool(event_flags[EVENT_FLAG_ACTION_FAILED]):
        return MOVE_BRANCH_ACTION_FAILED
    if event_scope == MOVE_SCOPE_SCOPED:
        return MOVE_BRANCH_SCOPED
    if event_scope == MOVE_SCOPE_TARGETED:
        return MOVE_BRANCH_TARGET_PHASE
    return None


def _encode_target_phase(
    labels: PublicConsequenceLabels,
    slot: int,
    outcomes: np.ndarray,
) -> None:
    labels.route_masks[slot] = True
    # Route is the target set declared on the move protocol line.  Minor
    # effects can mention additional Pokemon (for example a self-switch after
    # Parting Shot); those remain useful history but are not move targets.
    routed = outcomes[:, TARGET_OUTCOME_DECLARED] > 0.5
    labels.route_targets[slot] = routed.astype(np.float32)
    for position in np.flatnonzero(routed):
        outcome = outcomes[position]
        interaction_facts = (
            outcome[TARGET_OUTCOME_MISSED] > 0.5,
            outcome[TARGET_OUTCOME_IMMUNE] > 0.5,
            outcome[TARGET_OUTCOME_PROTECTED] > 0.5,
            outcome[TARGET_OUTCOME_BLOCKED] > 0.5,
            outcome[TARGET_OUTCOME_FAILED] > 0.5,
        )
        has_interaction_marker = any(interaction_facts)
        labels.interaction_targets[slot, position] = float(has_interaction_marker)
        labels.interaction_masks[slot, position] = True
        if has_interaction_marker:
            # These are public interaction markers, not a judgment that the
            # whole move failed.  A target `-fail`/`-block` may only suppress a
            # subeffect while damage, switching, or another scope still works.
            # Protect-like activations can accompany a generic block marker;
            # normalize that public alias to the more informative reason.
            if interaction_facts[TARGET_INTERACTION_PROTECTED]:
                reason = TARGET_INTERACTION_PROTECTED
                compatible = not (
                    interaction_facts[TARGET_INTERACTION_MISS]
                    or interaction_facts[TARGET_INTERACTION_IMMUNE]
                    or interaction_facts[TARGET_INTERACTION_FAILED]
                )
            else:
                active_reasons = [
                    reason_index
                    for reason_index, active in enumerate(interaction_facts)
                    if active
                ]
                compatible = len(active_reasons) == 1
                reason = active_reasons[0] if active_reasons else -1
            if compatible and reason >= 0:
                labels.interaction_reason_labels[slot, position] = reason
                labels.interaction_reason_masks[slot, position] = True
        else:
            normalized_hit_count = float(outcome[TARGET_OUTCOME_HIT_COUNT])
            if normalized_hit_count > 0.0:
                hit_count = int(round(normalized_hit_count * MAX_PUBLIC_HIT_COUNT))
                if 1 <= hit_count <= MAX_PUBLIC_HIT_COUNT:
                    labels.hit_count_labels[slot, position] = hit_count - 1
                    labels.hit_count_masks[slot, position] = True


def _empty_labels() -> PublicConsequenceLabels:
    return PublicConsequenceLabels(
        move_emitted_targets=np.zeros(2, dtype=np.float32),
        move_emitted_masks=np.zeros(2, dtype=bool),
        nonmove_reason_labels=np.full(2, -1, dtype=np.int64),
        nonmove_reason_masks=np.zeros(2, dtype=bool),
        move_branch_labels=np.full(2, -1, dtype=np.int64),
        move_branch_masks=np.zeros(2, dtype=bool),
        route_targets=np.zeros((2, NUM_PUBLIC_TARGET_POSITIONS), dtype=np.float32),
        route_masks=np.zeros((2, NUM_PUBLIC_TARGET_POSITIONS), dtype=bool),
        interaction_targets=np.zeros(
            (2, NUM_PUBLIC_TARGET_POSITIONS), dtype=np.float32
        ),
        interaction_masks=np.zeros((2, NUM_PUBLIC_TARGET_POSITIONS), dtype=bool),
        interaction_reason_labels=np.full(
            (2, NUM_PUBLIC_TARGET_POSITIONS), -1, dtype=np.int64
        ),
        interaction_reason_masks=np.zeros((2, NUM_PUBLIC_TARGET_POSITIONS), dtype=bool),
        hit_count_labels=np.full((2, NUM_PUBLIC_TARGET_POSITIONS), -1, dtype=np.int64),
        hit_count_masks=np.zeros((2, NUM_PUBLIC_TARGET_POSITIONS), dtype=bool),
        cant_targets=np.zeros((2, 2), dtype=np.float32),
        cant_masks=np.zeros((2, 2), dtype=bool),
        field_targets=np.zeros((2, PUBLIC_CONSEQUENCE_FIELD_DIM), dtype=np.float32),
        field_masks=np.zeros(2, dtype=bool),
    )


def _validated_arrays(
    observation: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    expected_shapes = {
        "recent_event_meta": (3,),
        "prev_own_valid": (2,),
        "prev_own_kind": (2,),
        "prev_own_move": (2,),
        "prev_own_target": (2,),
    }
    arrays: dict[str, np.ndarray] = {}
    for key, shape in expected_shapes.items():
        value = observation.get(key)
        if value is None:
            return None
        array = np.asarray(value)
        if array.shape != shape:
            return None
        arrays[key] = array

    event_keys = (
        "recent_event_kind",
        "recent_event_actor",
        "recent_event_move",
        "recent_event_source_move",
        "recent_event_move_scope",
        "recent_event_mask",
    )
    event_length: int | None = None
    for key in event_keys:
        value = observation.get(key)
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim != 1:
            return None
        if event_length is None:
            event_length = array.shape[0]
        elif array.shape[0] != event_length:
            return None
        arrays[key] = array

    flags = np.asarray(observation.get("recent_event_flags"))
    if flags.shape != (event_length, NUM_PUBLIC_EVENT_FLAGS):
        return None
    arrays["recent_event_flags"] = flags
    outcomes = np.asarray(observation.get("recent_event_target_outcomes"))
    if outcomes.shape != (
        event_length,
        NUM_PUBLIC_TARGET_POSITIONS,
        NUM_TARGET_OUTCOME_FEATURES,
    ):
        return None
    arrays["recent_event_target_outcomes"] = outcomes
    return arrays


def _is_complete_segment(meta: np.ndarray) -> bool:
    return bool(meta[0] > 0.5 and meta[1] == 0.0 and meta[2] == 0.0)


def _opponent_cant_mask(intent_target: int, *, spread: bool) -> np.ndarray:
    if intent_target == POSITION_OPP_1:
        return np.asarray([True, False], dtype=bool)
    if intent_target == POSITION_OPP_2:
        return np.asarray([False, True], dtype=bool)
    if intent_target == POSITION_IMPLICIT and spread:
        return np.asarray([True, True], dtype=bool)
    return np.zeros(2, dtype=bool)
