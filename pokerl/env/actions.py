"""Action space for doubles battles: indices, masking, and conversion to battle orders."""

import numpy as np
from poke_env.battle import DoubleBattle, Move
from poke_env.player import PassBattleOrder, SingleBattleOrder


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
) -> SingleBattleOrder:
    """Convert a flat action index to a poke-env single-slot order."""
    if any(battle.force_switch) and not battle.force_switch[slot]:
        return PassBattleOrder()

    kind, idx, target, tera = decode_action(action_idx)

    if kind == "pass":
        return PassBattleOrder()

    if kind == "switch":
        pokemon = battle.available_switches[slot][idx]
        return SingleBattleOrder(pokemon)

    # Move order
    move = battle.available_moves[slot][idx]
    showdown_target = _resolve_target(move, slot, target, battle)
    return SingleBattleOrder(
        move,
        move_target=showdown_target,
        terastallize=tera,
    )


def build_action_mask(battle: DoubleBattle, slot: int) -> np.ndarray:
    """Build a boolean mask of valid actions for one active slot.

    Returns shape (ACTIONS_PER_SLOT,), True = valid.
    """
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)

    if any(battle.force_switch) and not battle.force_switch[slot]:
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

    pokemon = battle.active_pokemon[slot]
    if pokemon is None or pokemon.fainted:
        mask[PASS_ACTION] = True
        return mask

    # Moves
    moves = battle.available_moves[slot]
    can_tera = bool(battle.can_tera[slot])

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

    if not mask.any():
        raise RuntimeError(
            f"build_action_mask produced an empty mask for slot {slot} — "
            f"pokemon={getattr(pokemon, 'species', None)!r}, "
            f"moves={[m.id for m in moves]}, switches={len(switches)}, "
            f"force_switch={battle.force_switch}. "
            "This indicates a bug in action mask construction."
        )

    return mask


def _valid_targets_for_move(
    move: Move, slot: int, battle: DoubleBattle
) -> list[int]:
    """Which of our 3 target indices are valid for this move from this slot."""
    pokemon = battle.active_pokemon[slot]
    if pokemon is None:
        return [0]

    showdown_targets = battle.get_possible_showdown_targets(move, pokemon)
    valid_targets = []
    for showdown_target in showdown_targets:
        target_idx = _showdown_target_to_index(showdown_target, slot, battle)
        if target_idx is not None and target_idx not in valid_targets:
            valid_targets.append(target_idx)

    return valid_targets or [0]


def _resolve_target(
    move: Move, slot: int, target_idx: int, battle: DoubleBattle
) -> int:
    """Convert our abstract target index to a Showdown move_target integer.

    The sampled action should always be valid (the mask prevents illegal ones),
    so if the target can't be resolved that indicates a mask/action bug.
    """
    pokemon = battle.active_pokemon[slot]
    if pokemon is None:
        return battle.EMPTY_TARGET_POSITION

    possible_targets = battle.get_possible_showdown_targets(move, pokemon)
    if target_idx == 0 and battle.EMPTY_TARGET_POSITION in possible_targets:
        return battle.EMPTY_TARGET_POSITION

    showdown_target = _target_index_to_showdown(target_idx, slot, battle)
    if showdown_target in possible_targets:
        return showdown_target

    raise RuntimeError(
        f"_resolve_target: computed showdown_target={showdown_target} is not in "
        f"possible_targets={possible_targets} for move={move.id!r}, slot={slot}, "
        f"target_idx={target_idx}. This means the action mask allowed an invalid target — "
        "check build_action_mask and _valid_targets_for_move."
    )


def _showdown_target_to_index(
    showdown_target: int, slot: int, battle: DoubleBattle
) -> int | None:
    if showdown_target == battle.EMPTY_TARGET_POSITION:
        return 0

    ally_position = (
        battle.POKEMON_2_POSITION if slot == 0 else battle.POKEMON_1_POSITION
    )
    mapping = {
        battle.OPPONENT_1_POSITION: 0,
        battle.OPPONENT_2_POSITION: 1,
        ally_position: 2,
    }
    return mapping.get(showdown_target)


def _target_index_to_showdown(
    target_idx: int, slot: int, battle: DoubleBattle
) -> int:
    ally_position = battle.POKEMON_2_POSITION if slot == 0 else battle.POKEMON_1_POSITION
    mapping = {
        0: battle.OPPONENT_1_POSITION,
        1: battle.OPPONENT_2_POSITION,
        2: ally_position,
    }
    return mapping.get(target_idx, battle.EMPTY_TARGET_POSITION)
