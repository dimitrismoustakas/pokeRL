"""Public, action-aligned consequence labels for auxiliary training.

The labels are a conditional likelihood of exact public protocol facts.  They
do not assign value to those facts: PPO's terminal reward remains the only
source of strategic preference.  The scope is the immediate resolved segment;
delayed effects are retained in recurrent public history but are not attributed
to an earlier action when the protocol does not provide exact causal alignment.
"""

from collections.abc import Mapping

import numpy as np

from .data import MOVE_SCOPE_SCOPED, MOVE_SCOPE_TARGETED, UNKNOWN_IDX
from .env.actions import INTENT_MOVE
from .env.observation import (
    EVENT_FLAG_ACTION_FAILED,
    EVENT_FLAG_NO_TARGET,
    EVENT_FLAG_SPREAD,
    EVENT_MOVE,
    EVENT_CANT,
    EVENT_FAINT,
    FIELD_TRICK_ROOM_OFFSET,
    NUM_PUBLIC_TARGET_POSITIONS,
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
    "recent_event_mask",
    "prev_own_valid",
    "prev_own_kind",
    "prev_own_move",
    "prev_own_target",
)

# Positions match the interaction reason labels used by the prediction head.
_INTERACTION_MARKERS = (
    TARGET_OUTCOME_MISSED,
    TARGET_OUTCOME_IMMUNE,
    TARGET_OUTCOME_PROTECTED,
    TARGET_OUTCOME_BLOCKED,
    TARGET_OUTCOME_FAILED,
)
_OWN_POSITIONS = (POSITION_OWN_1, POSITION_OWN_2)
_OPPONENT_POSITIONS = (POSITION_OPP_1, POSITION_OPP_2)


def empty_consequence_batch(size: int) -> dict[str, np.ndarray]:
    slots = (size, 2)
    targets = (*slots, NUM_PUBLIC_TARGET_POSITIONS)
    opponents = (*slots, 2)
    return {
        "move_emitted_targets": np.empty(slots, np.float32),
        "move_emitted_masks": np.empty(slots, bool),
        "nonmove_reason_labels": np.empty(slots, np.int64),
        "nonmove_reason_masks": np.empty(slots, bool),
        "move_branch_labels": np.empty(slots, np.int64),
        "move_branch_masks": np.empty(slots, bool),
        "route_targets": np.empty(targets, np.float32),
        "route_masks": np.empty(targets, bool),
        "interaction_targets": np.empty(targets, np.float32),
        "interaction_masks": np.empty(targets, bool),
        "interaction_reason_labels": np.empty(targets, np.int64),
        "interaction_reason_masks": np.empty(targets, bool),
        "hit_count_labels": np.empty(targets, np.int64),
        "hit_count_masks": np.empty(targets, bool),
        "cant_targets": np.empty(opponents, np.float32),
        "cant_masks": np.empty(opponents, bool),
        "field_targets": np.empty((*slots, PUBLIC_CONSEQUENCE_FIELD_DIM), np.float32),
        "field_masks": np.empty(slots, bool),
    }


def encode_consequence_batch(
    obs: Mapping[str, np.ndarray], out: dict[str, np.ndarray]
) -> None:
    """Write slot-aligned consequences for a batch of post-action player observations."""
    for name, values in out.items():
        values.fill(-1 if name.endswith("_labels") else 0)
    if "recent_event_meta" not in obs:
        return
    kind = obs["recent_event_kind"]
    if kind.shape[1] == 0:
        return
    meta = obs["recent_event_meta"]
    complete = (meta[:, 0] > 0.5) & (meta[:, 1] == 0) & (meta[:, 2] == 0)
    chosen = obs["prev_own_move"]
    eligible = (
        complete[:, None]
        & obs["prev_own_valid"]
        & (obs["prev_own_kind"] == INTENT_MOVE)
        & (chosen != UNKNOWN_IDX)
    )
    actor = obs["recent_event_actor"]
    present = ~obs["recent_event_mask"]
    # Each slot can be supervised only when one public event resolves its intent.
    own = (actor[:, None, :] == np.asarray(_OWN_POSITIONS)[None, :, None]) & present[
        :, None, :
    ]
    moves = own & (kind == EVENT_MOVE)[:, None, :]
    cants = own & (kind == EVENT_CANT)[:, None, :]
    faints = own & (kind == EVENT_FAINT)[:, None, :]
    move_count = moves.sum(axis=-1)
    cant_count = cants.sum(axis=-1)
    event_move = obs["recent_event_move"][:, None, :]
    matches = (event_move == chosen[:, :, None]) | (
        obs["recent_event_source_move"][:, None, :] == chosen[:, :, None]
    )
    emitted = (
        eligible
        & (move_count == 1)
        & (cant_count == 0)
        & (moves & matches).any(axis=-1)
    )
    compatible = (event_move == UNKNOWN_IDX) | (event_move == chosen[:, :, None])
    stopped = (
        eligible
        & (move_count == 0)
        & (cant_count == 1)
        & (cants & compatible).any(axis=-1)
    )
    fainted = (
        eligible & (move_count == 0) & (cant_count == 0) & (faints.sum(axis=-1) == 1)
    )
    resolved = emitted | stopped | fainted
    out["move_emitted_targets"][:] = emitted
    out["move_emitted_masks"][:] = resolved
    out["nonmove_reason_labels"][stopped] = NONMOVE_REASON_CANT
    out["nonmove_reason_labels"][fainted] = NONMOVE_REASON_FAINTED
    out["nonmove_reason_masks"][:] = stopped | fainted

    lanes, slots = np.nonzero(emitted)
    events = moves[lanes, slots].argmax(axis=1)
    flags = obs["recent_event_flags"][lanes, events]
    scope = obs["recent_event_move_scope"][lanes, events]
    branch = np.select(
        (
            flags[:, EVENT_FLAG_NO_TARGET] != 0,
            flags[:, EVENT_FLAG_ACTION_FAILED] != 0,
            scope == MOVE_SCOPE_SCOPED,
            scope == MOVE_SCOPE_TARGETED,
        ),
        (
            MOVE_BRANCH_NO_TARGET,
            MOVE_BRANCH_ACTION_FAILED,
            MOVE_BRANCH_SCOPED,
            MOVE_BRANCH_TARGET_PHASE,
        ),
        default=-1,
    )
    out["move_branch_labels"][lanes, slots] = branch
    out["move_branch_masks"][lanes, slots] = branch >= 0
    targeted = branch == MOVE_BRANCH_TARGET_PHASE
    _encode_target_phase(
        out,
        lanes[targeted],
        slots[targeted],
        obs["recent_event_target_outcomes"][lanes[targeted], events[targeted]],
    )

    spread = np.zeros_like(resolved)
    spread[lanes, slots] = flags[:, EVENT_FLAG_SPREAD] != 0
    target = obs["prev_own_target"]
    for index, position in enumerate(_OPPONENT_POSITIONS):
        cant = (present & (actor == position) & (kind == EVENT_CANT)).any(axis=-1)
        out["cant_targets"][:, :, index] = resolved & cant[:, None]
        out["cant_masks"][:, :, index] = resolved & (
            (target == position) | ((target == POSITION_IMPLICIT) & spread)
        )
    field = obs.get("field")
    if (
        field is not None
        and field.ndim == 2
        and field.shape[1] >= PUBLIC_CONSEQUENCE_FIELD_DIM
    ):
        np.copyto(
            out["field_targets"],
            field[:, None, :PUBLIC_CONSEQUENCE_FIELD_DIM],
            where=resolved[:, :, None],
        )
        out["field_masks"][:] = resolved


def _encode_target_phase(out, lanes, slots, outcomes):
    # Only targets declared on the move line are part of its route.
    routed = outcomes[:, :, TARGET_OUTCOME_DECLARED] > 0.5
    out["route_targets"][lanes, slots] = routed
    out["route_masks"][lanes, slots] = True
    markers = outcomes[..., _INTERACTION_MARKERS] > 0.5
    counts = markers.sum(axis=-1)
    marked = routed & (counts > 0)
    out["interaction_targets"][lanes, slots] = marked
    out["interaction_masks"][lanes, slots] = routed
    # Protect and generic block may refer to the same interaction.
    alias = (
        markers[:, :, TARGET_INTERACTION_PROTECTED]
        & markers[:, :, TARGET_INTERACTION_BLOCKED]
    )
    identified = routed & ((counts == 1) | ((counts == 2) & alias))
    out["interaction_reason_masks"][lanes, slots] = identified
    out["interaction_reason_labels"][lanes, slots] = np.where(
        identified, markers.argmax(axis=-1), -1
    )
    hits = np.rint(
        outcomes[:, :, TARGET_OUTCOME_HIT_COUNT].astype(np.float64)
        * MAX_PUBLIC_HIT_COUNT
    )
    counted = routed & ~marked & (hits >= 1) & (hits <= MAX_PUBLIC_HIT_COUNT)
    out["hit_count_labels"][lanes, slots] = np.where(counted, hits - 1, -1).astype(
        np.int64
    )
    out["hit_count_masks"][lanes, slots] = counted
