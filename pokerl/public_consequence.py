"""Public, action-aligned consequence labels for auxiliary training.

The encoder in this module consumes one *returned* (therefore unbatched)
observation.  It never inspects simulator-private state.  Labels are emitted
only when the recent-event segment is complete and one public event can be
matched unambiguously to a previously submitted own move.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from poke_env.battle import PublicBattleEventKind

from .data import UNKNOWN_IDX
from .env.actions import INTENT_MOVE
from .env.observation import (
    EVENT_KIND_TO_INDEX,
    FIELD_TRICK_ROOM_OFFSET,
    POSITION_IMPLICIT,
    POSITION_OPP_1,
    POSITION_OPP_2,
    POSITION_OWN_1,
    POSITION_OWN_2,
)


PUBLIC_CONSEQUENCE_EXECUTED = 0
PUBLIC_CONSEQUENCE_FAILED = 1
PUBLIC_CONSEQUENCE_MISSED = 2
PUBLIC_CONSEQUENCE_CANT = 3
NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES = 4

PUBLIC_CONSEQUENCE_FIELD_DIM = FIELD_TRICK_ROOM_OFFSET + 1
PUBLIC_CONSEQUENCE_OBSERVATION_KEYS = (
    "field",
    "recent_event_meta",
    "recent_event_kind",
    "recent_event_actor",
    "recent_event_move",
    "recent_event_flags",
    "recent_event_mask",
    "prev_own_valid",
    "prev_own_kind",
    "prev_own_move",
    "prev_own_target",
)

_MOVE_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.MOVE]
_CANT_EVENT = EVENT_KIND_TO_INDEX[PublicBattleEventKind.CANT]
_OWN_POSITIONS = (POSITION_OWN_1, POSITION_OWN_2)
_OPPONENT_POSITIONS = (POSITION_OPP_1, POSITION_OPP_2)


@dataclass(frozen=True, slots=True)
class PublicConsequenceLabels:
    """Trainer-only labels aligned to the two previously selected own moves."""

    resolution_labels: np.ndarray  # (2,) int64
    resolution_masks: np.ndarray  # (2,) bool
    cant_targets: np.ndarray  # (2, 2) float32
    cant_masks: np.ndarray  # (2, 2) bool
    field_targets: np.ndarray  # (2, 33) float32
    field_masks: np.ndarray  # (2,) bool


def encode_public_field(
    observation: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    """Return the public field prefix used by the consequence head.

    The prefix includes weather, terrain, both sides' public conditions, and
    Trick Room.  It deliberately stops before either side's Tera-use flag and
    the absolute turn counter.
    """

    field_value = observation.get("field")
    if field_value is None:
        return None
    field = np.asarray(field_value)
    if field.ndim != 1 or field.shape[0] < PUBLIC_CONSEQUENCE_FIELD_DIM:
        return None
    return np.asarray(
        field[:PUBLIC_CONSEQUENCE_FIELD_DIM],
        dtype=np.float32,
    ).copy()


def encode_public_consequence(
    observation: Mapping[str, np.ndarray],
) -> PublicConsequenceLabels:
    """Encode exact next-public consequences from one returned observation.

    A previous move is supervised only when actor and move id identify exactly
    one own ``MOVE``, or when the actor has one lone ``CANT`` and no ``MOVE``.
    This masks transformed moves, missing public events, and duplicate or
    otherwise ambiguous matches rather than inventing a target.
    """

    labels = _empty_labels()
    arrays = _validated_arrays(observation)
    if arrays is None or not _is_complete_segment(arrays["recent_event_meta"]):
        return labels

    event_mask = arrays["recent_event_mask"]
    event_kind = arrays["recent_event_kind"]
    event_actor = arrays["recent_event_actor"]
    event_move = arrays["recent_event_move"]
    event_flags = arrays["recent_event_flags"]
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
        exact_move_events = own_move_events[event_move[own_move_events] == own_move]
        own_cant_events = np.flatnonzero(
            present & (event_kind == _CANT_EVENT) & (event_actor == own_position)
        )
        compatible_cant_events = own_cant_events[
            (event_move[own_cant_events] == UNKNOWN_IDX)
            | (event_move[own_cant_events] == own_move)
        ]

        # CANT protocol messages often omit the intended move.  A lone CANT
        # for the exact actor is still unambiguous, but it is never used as a
        # fallback when a (possibly transformed) MOVE event by that actor also
        # exists.
        if (
            own_move_events.size == 1
            and exact_move_events.size == 1
            and own_cant_events.size == 0
        ):
            event_index = int(exact_move_events[0])
        elif (
            own_move_events.size == 0
            and own_cant_events.size == 1
            and compatible_cant_events.size == 1
        ):
            event_index = int(compatible_cant_events[0])
        else:
            continue
        if event_kind[event_index] == _CANT_EVENT:
            resolution = PUBLIC_CONSEQUENCE_CANT
        elif event_flags[event_index, 1]:
            # poke-env also marks a miss as failed; the more specific class
            # therefore takes precedence.
            resolution = PUBLIC_CONSEQUENCE_MISSED
        elif event_flags[event_index, 0]:
            resolution = PUBLIC_CONSEQUENCE_FAILED
        else:
            resolution = PUBLIC_CONSEQUENCE_EXECUTED

        labels.resolution_labels[slot] = resolution
        labels.resolution_masks[slot] = True

        cant_mask = _opponent_cant_mask(
            int(arrays["prev_own_target"][slot]),
            spread=bool(event_flags[event_index, 2]),
        )
        labels.cant_targets[slot] = opponent_cant
        labels.cant_masks[slot] = cant_mask

        if public_field is not None:
            labels.field_targets[slot] = public_field
            labels.field_masks[slot] = True

    return labels


def _empty_labels() -> PublicConsequenceLabels:
    return PublicConsequenceLabels(
        resolution_labels=np.full(2, -1, dtype=np.int64),
        resolution_masks=np.zeros(2, dtype=bool),
        cant_targets=np.zeros((2, 2), dtype=np.float32),
        cant_masks=np.zeros((2, 2), dtype=bool),
        field_targets=np.zeros(
            (2, PUBLIC_CONSEQUENCE_FIELD_DIM),
            dtype=np.float32,
        ),
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

    flags_value = observation.get("recent_event_flags")
    if flags_value is None:
        return None
    flags = np.asarray(flags_value)
    if flags.ndim != 2 or flags.shape != (event_length, 3):
        return None
    arrays["recent_event_flags"] = flags
    return arrays


def _is_complete_segment(meta: np.ndarray) -> bool:
    return bool(meta[0] > 0.5 and meta[1] == 0.0 and meta[2] == 0.0)


def _opponent_cant_mask(intent_target: int, *, spread: bool) -> np.ndarray:
    if intent_target == POSITION_OPP_1:
        return np.asarray([True, False], dtype=bool)
    if intent_target == POSITION_OPP_2:
        return np.asarray([False, True], dtype=bool)
    # TARGET_NONE / POSITION_IMPLICIT is shared by spread attacks and by
    # self/field moves such as Protect and Trick Room.  Only the matched
    # public MOVE event can distinguish those cases without consulting
    # simulator-private state or move-dex heuristics.
    if intent_target == POSITION_IMPLICIT and spread:
        return np.asarray([True, True], dtype=bool)
    return np.zeros(2, dtype=bool)
