"""Action space for doubles battles: indices, masking, and conversion to battle orders."""

import torch
import numpy as np
from poke_env.environment import DoubleBattle, Pokemon, Move
from poke_env.player import BattleOrder


# --- Action layout per active slot ---
#
#   0..11:  move_i * 3 + target_j        (4 moves × 3 targets, no tera)
#  12..23:  move_i * 3 + target_j + 12   (4 moves × 3 targets, with tera)
#  24..27:  switch to bench_k             (up to 4 bench pokemon)
#  28:      pass                          (slot has no action this turn)

NUM_MOVES = 4
NUM_TARGETS = 3   # opposing_1, opposing_2, ally
TERA_OFFSET = NUM_MOVES * NUM_TARGETS         # 12
SWITCH_OFFSET = 2 * NUM_MOVES * NUM_TARGETS   # 24
MAX_SWITCHES = 4
PASS_ACTION = SWITCH_OFFSET + MAX_SWITCHES    # 28
ACTIONS_PER_SLOT = PASS_ACTION + 1            # 29

# Target index -> Showdown move_target value, per acting slot.
# In Showdown doubles: -1 = opp left, -2 = opp right, +N = ally position.
_TARGET_MAP = {
    0: {0: -1, 1: -2, 2: 2},   # slot 0 (left): ally is position 2
    1: {0: -1, 1: -2, 2: 1},   # slot 1 (right): ally is position 1
}

# Move target types that require no target choice (single valid action per move)
_NO_CHOICE_TARGETS = {
    "self", "allAdjacentFoes", "allAdjacent", "all", "adjacentAllyOrSelf",
    "foeSide", "allyTeam", "allyOrSelf", "randomNormal",
}


def decode_action(action_idx: int) -> tuple[str, int, int, bool]:
    """Decode a flat action index into (kind, move_or_switch_idx, target, tera).

    Returns:
        kind: "move", "switch", or "pass"
        idx: move index (0-3) or switch index (0-3)
        target: target index (0-2) for moves, -1 otherwise
        tera: whether to terastallize
    """
    if action_idx == PASS_ACTION:
        return "pass", -1, -1, False

    if action_idx >= SWITCH_OFFSET:
        return "switch", action_idx - SWITCH_OFFSET, -1, False

    tera = action_idx >= TERA_OFFSET
    if tera:
        action_idx -= TERA_OFFSET
    move_idx = action_idx // NUM_TARGETS
    target_idx = action_idx % NUM_TARGETS
    return "move", move_idx, target_idx, tera


def action_to_order(
    action_idx: int, battle: DoubleBattle, slot: int
) -> BattleOrder | None:
    """Convert a flat action index to a poke-env BattleOrder."""
    kind, idx, target, tera = decode_action(action_idx)

    if kind == "pass":
        return None

    if kind == "switch":
        pokemon = battle.available_switches[slot][idx]
        return BattleOrder(pokemon)

    # Move order
    move = battle.available_moves[slot][idx]
    showdown_target = _resolve_target(move, slot, target, battle)
    return BattleOrder(
        move,
        move_target=showdown_target,
        terastallize=tera,
    )


def build_action_mask(battle: DoubleBattle, slot: int) -> np.ndarray:
    """Build a boolean mask of valid actions for one active slot.

    Returns shape (ACTIONS_PER_SLOT,), True = valid.
    """
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)

    pokemon = battle.active_pokemon[slot]
    if pokemon is None or pokemon.fainted:
        mask[PASS_ACTION] = True
        return mask

    # Force switch: only switches are valid (and possibly pass if no switches)
    if battle.force_switch[slot]:
        switches = battle.available_switches[slot]
        for k, _ in enumerate(switches):
            mask[SWITCH_OFFSET + k] = True
        if not switches:
            mask[PASS_ACTION] = True
        return mask

    # Moves
    moves = battle.available_moves[slot]
    can_tera = battle.can_tera[slot] if hasattr(battle, "can_tera") else False
    opp_pokemon = battle.opponent_active_pokemon

    for i, move in enumerate(moves):
        valid_targets = _valid_targets_for_move(move, slot, battle)
        for j in valid_targets:
            mask[i * NUM_TARGETS + j] = True
            if can_tera:
                mask[TERA_OFFSET + i * NUM_TARGETS + j] = True

    # Switches
    switches = battle.available_switches[slot]
    for k, _ in enumerate(switches):
        mask[SWITCH_OFFSET + k] = True

    # If nothing is valid (shouldn't happen), allow pass as fallback
    if not mask.any():
        mask[PASS_ACTION] = True

    return mask


def _valid_targets_for_move(
    move: Move, slot: int, battle: DoubleBattle
) -> list[int]:
    """Which of our 3 target indices are valid for this move from this slot."""
    target_type = move.target if hasattr(move, "target") else "normal"

    if target_type in _NO_CHOICE_TARGETS:
        return [0]  # no choice -- use first slot as placeholder

    if target_type == "adjacentAlly":
        # Can only target ally
        ally_slot = 1 - slot
        ally = battle.active_pokemon[ally_slot]
        if ally and not ally.fainted:
            return [2]
        return []

    # "normal" and other single-target types: can target opp_1, opp_2, ally
    valid = []
    for j, opp in enumerate(battle.opponent_active_pokemon):
        if opp and not opp.fainted:
            valid.append(j)
    # Ally
    ally_slot = 1 - slot
    ally = battle.active_pokemon[ally_slot]
    if ally and not ally.fainted:
        valid.append(2)
    return valid


def _resolve_target(
    move: Move, slot: int, target_idx: int, battle: DoubleBattle
) -> int:
    """Convert our abstract target index to a Showdown move_target integer."""
    target_type = move.target if hasattr(move, "target") else "normal"
    if target_type in _NO_CHOICE_TARGETS:
        return 0  # Showdown ignores the target for these
    return _TARGET_MAP[slot][target_idx]
