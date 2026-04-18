"""Action space helpers for doubles battles.

The model acts in a fixed discrete space. Single-slot move targets come from
``poke-env``'s request-aware Showdown target resolver, and joint legality is
still delegated to ``DoubleBattleOrder.join_orders``.
"""

from dataclasses import dataclass

from poke_env.battle import DoubleBattle, Move
from poke_env.player import DoubleBattleOrder, PassBattleOrder, SingleBattleOrder

import numpy as np


# --- Action layout per active slot ---
#
#   0..19:  move_i * 5 + target_j        (4 moves x 5 targets, no tera)
#  20..39:  move_i * 5 + target_j + 20   (4 moves x 5 targets, with tera)
#  40..45:  switch to own_team_slot_k    (stable team slot, 0..5)
#  46:      pass                         (slot has no action this turn)
#
# Target order:
#   0: opposing_1
#   1: opposing_2
#   2: ally
#   3: self
#   4: no explicit target

NUM_MOVES = 4
NUM_TARGETS = 5
MAX_TEAM_SLOTS = 6
TARGET_OPPONENT_1 = 0
TARGET_OPPONENT_2 = 1
TARGET_ALLY = 2
TARGET_SELF = 3
TARGET_NONE = 4
TERA_OFFSET = NUM_MOVES * NUM_TARGETS          # 20
SWITCH_OFFSET = 2 * NUM_MOVES * NUM_TARGETS    # 40
PASS_ACTION = SWITCH_OFFSET + MAX_TEAM_SLOTS   # 46
ACTIONS_PER_SLOT = PASS_ACTION + 1             # 47
FIRST_SLOT_MASK_ROW = 0


@dataclass(frozen=True)
class LegalActionState:
    """Per-request legality tables derived from the live Showdown request."""

    slot_action_orders: tuple[dict[int, SingleBattleOrder], dict[int, SingleBattleOrder]]
    joint_mask: np.ndarray


def decode_action(action_idx: int) -> tuple[str, int, int, bool]:
    """Decode a flat action index into (kind, move_or_switch_idx, target, tera)."""
    if action_idx == PASS_ACTION:
        return "pass", -1, TARGET_NONE, False

    if action_idx >= SWITCH_OFFSET:
        return "switch", action_idx - SWITCH_OFFSET, TARGET_NONE, False

    tera = action_idx >= TERA_OFFSET
    if tera:
        action_idx -= TERA_OFFSET
    move_idx = action_idx // NUM_TARGETS
    target_idx = action_idx % NUM_TARGETS
    return "move", move_idx, target_idx, tera


def build_legal_action_state(battle: DoubleBattle) -> LegalActionState:
    """Build and index all legal orders for the current live request once."""
    slot_action_orders = _build_slot_action_orders(battle)
    joint_mask = _build_joint_mask(slot_action_orders, battle)
    return LegalActionState(slot_action_orders=slot_action_orders, joint_mask=joint_mask)


def build_action_mask(
    battle: DoubleBattle,
    slot: int,
    action_state: LegalActionState | None = None,
) -> np.ndarray:
    """Build a slot-local mask from the current Showdown request."""
    if action_state is None:
        action_state = build_legal_action_state(battle)
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)
    for action_idx in action_state.slot_action_orders[slot]:
        mask[action_idx] = True
    return mask


def build_joint_action_mask(
    battle: DoubleBattle,
    action_state: LegalActionState | None = None,
) -> np.ndarray:
    """Build a joint legality table from joined doubles orders.

    Returns shape ``(ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT)``:
    - row 0 contains legal first-slot actions
    - row 1 + a1 contains legal second-slot actions conditioned on first-slot
      action ``a1``
    """
    if action_state is None:
        action_state = build_legal_action_state(battle)
    return action_state.joint_mask


def _build_joint_mask(
    slot_action_orders: tuple[dict[int, SingleBattleOrder], dict[int, SingleBattleOrder]],
    battle: DoubleBattle,
) -> np.ndarray:
    joint_mask = np.zeros((ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)

    joined_orders = DoubleBattleOrder.join_orders(
        list(slot_action_orders[0].values()),
        list(slot_action_orders[1].values()),
    )
    if not joined_orders:
        if set(slot_action_orders[0]) == {PASS_ACTION} and set(slot_action_orders[1]) == {PASS_ACTION}:
            joint_mask[FIRST_SLOT_MASK_ROW, PASS_ACTION] = True
            joint_mask[PASS_ACTION + 1, PASS_ACTION] = True
            return joint_mask
        raise RuntimeError(
            "poke-env produced no legal joined doubles orders for the current request. "
            "This indicates the request state and action mapping are inconsistent."
        )

    for order in joined_orders:
        a1 = _order_to_action_idx(order.first_order, battle, slot=0)
        a2 = _order_to_action_idx(order.second_order, battle, slot=1)
        joint_mask[FIRST_SLOT_MASK_ROW, a1] = True
        joint_mask[a1 + 1, a2] = True

    return joint_mask


def action_to_order(
    action_idx: int,
    battle: DoubleBattle,
    slot: int,
    action_state: LegalActionState | None = None,
) -> SingleBattleOrder:
    """Convert a flat action index to the exact current legal order."""
    if action_state is None:
        action_state = build_legal_action_state(battle)
    try:
        return action_state.slot_action_orders[slot][action_idx]
    except KeyError as exc:
        raise RuntimeError(
            f"Action {action_idx} is not legal for slot {slot} in the current request. "
            "The policy should only sample actions allowed by build_joint_action_mask()."
        ) from exc


def _build_slot_action_orders(
    battle: DoubleBattle,
) -> tuple[dict[int, SingleBattleOrder], dict[int, SingleBattleOrder]]:
    raw_orders = (
        _build_slot_orders(battle, slot=0),
        _build_slot_orders(battle, slot=1),
    )
    return (
        _index_orders(raw_orders[0], battle, slot=0),
        _index_orders(raw_orders[1], battle, slot=1),
    )


def _build_slot_orders(battle: DoubleBattle, slot: int) -> list[SingleBattleOrder]:
    trapped = getattr(battle, "trapped", [False, False])
    force_switch = battle.force_switch
    orders: list[SingleBattleOrder] = []

    if any(force_switch) and not force_switch[slot]:
        return [PassBattleOrder()]

    if not trapped[slot]:
        orders.extend(SingleBattleOrder(pokemon) for pokemon in battle.available_switches[slot])

    if all(force_switch) and len(battle.available_switches[0]) == 1:
        orders.append(PassBattleOrder())
        return orders

    active_pokemon = battle.active_pokemon[slot]
    if active_pokemon is not None and not force_switch[slot]:
        for move in battle.available_moves[slot]:
            hydrated_move, target_locs = _request_aware_move_targets(move, active_pokemon, battle, slot)
            for target_loc in target_locs:
                orders.append(SingleBattleOrder(hydrated_move, move_target=target_loc))
                if battle.can_tera[slot]:
                    orders.append(
                        SingleBattleOrder(hydrated_move, move_target=target_loc, terastallize=True)
                    )

    if not orders:
        orders.append(PassBattleOrder())
    return orders


def _index_orders(
    orders: list[SingleBattleOrder],
    battle: DoubleBattle,
    slot: int,
) -> dict[int, SingleBattleOrder]:
    indexed: dict[int, SingleBattleOrder] = {}
    for order in orders:
        action_idx = _order_to_action_idx(order, battle, slot)
        existing = indexed.get(action_idx)
        if existing is not None and existing.message != order.message:
            raise RuntimeError(
                f"Two different legal orders mapped to the same action id {action_idx} "
                f"for slot {slot}: {existing.message!r} vs {order.message!r}. "
                "Expand the action encoding so these orders are distinguishable."
            )
        indexed.setdefault(action_idx, order)

    if not indexed:
        indexed[PASS_ACTION] = PassBattleOrder()
    return indexed


def _order_to_action_idx(order: SingleBattleOrder, battle: DoubleBattle, slot: int) -> int:
    if isinstance(order, PassBattleOrder) or order.message == "/choose pass":
        return PASS_ACTION

    if isinstance(order.order, str):
        raise RuntimeError(
            f"Unsupported string-backed battle order {order.message!r}. "
            "Only move, switch, and pass orders should appear during training."
        )

    if isinstance(order.order, Move):
        if order.mega or order.z_move or order.dynamax:
            raise RuntimeError(
                f"Unsupported special move modifier in legal order {order.message!r}. "
                "This action space currently supports plain and terastallized moves only."
            )

        move_idx = _find_available_move_index(battle.available_moves[slot], order.order)
        target_idx = _showdown_target_to_index(order.move_target, slot, battle)
        action_idx = move_idx * NUM_TARGETS + target_idx
        if order.terastallize:
            action_idx += TERA_OFFSET
        return action_idx

    ordered_team = _ordered_own_team(battle)
    team_slot = _find_team_slot(ordered_team, order.order)
    if team_slot is None:
        raise RuntimeError(
            f"Legal switch order {order.message!r} referenced a Pokemon that is not in the "
            "ordered team. This indicates battle.team and valid_orders are out of sync."
        )
    return SWITCH_OFFSET + team_slot


def _find_available_move_index(available_moves: list[Move], selected_move: Move) -> int:
    for move_idx, move in enumerate(available_moves):
        if move is selected_move:
            return move_idx

    selected_id = Move.retrieve_id(selected_move.id)
    matching_indices = [
        move_idx
        for move_idx, move in enumerate(available_moves)
        if Move.retrieve_id(move.id) == selected_id
    ]
    if len(matching_indices) == 1:
        return matching_indices[0]

    raise RuntimeError(
        f"Legal move order referenced {selected_move.id!r}, but it was not uniquely present in "
        "battle.available_moves for this slot. This indicates valid_orders and available_moves "
        "are out of sync."
    )


def _request_move_entry(move: Move, battle: DoubleBattle, slot: int) -> dict | None:
    request = getattr(battle, "last_request", None)
    if not request:
        return None

    active_requests = request.get("active")
    if not active_requests or slot >= len(active_requests):
        return None

    move_requests = active_requests[slot].get("moves")
    if not move_requests:
        return None

    move_id = Move.retrieve_id(move.id)
    for move_request in move_requests:
        if move_request.get("disabled", False):
            continue
        if move_request.get("id") == move_id:
            return move_request
    return None


def _hydrate_move_for_request(move: Move, battle: DoubleBattle, slot: int) -> Move:
    """Copy a move and attach the current request target, if any.

    ``poke-env`` uses ``Move.request_target`` to disambiguate target classes such
    as ``randomNormal``. Cloning avoids leaking per-turn request state across
    reused move objects.
    """
    hydrated = Move(move.id, gen=move.gen)
    move_request = _request_move_entry(move, battle, slot)
    target = None if move_request is None else move_request.get("target")
    if target is not None:
        hydrated.request_target = target
    return hydrated


def _request_aware_move_targets(
    move: Move,
    active_pokemon,
    battle: DoubleBattle,
    slot: int,
) -> tuple[Move, list[int]]:
    hydrated_move = _hydrate_move_for_request(move, battle, slot)
    move_request = _request_move_entry(move, battle, slot)
    if (
        move_request is not None
        and move_request.get("target") is None
        and _is_random_normal_move(move)
    ):
        # Showdown sometimes omits a request target for initial Outrage/Thrash-style
        # selections even though a foe target is still required. Preserve poke-env's
        # occupancy filtering by reusing its adjacent-foe resolver.
        hydrated_move.request_target = "adjacentFoe"
    return hydrated_move, _legal_target_locs(hydrated_move, active_pokemon, battle)


def _ordered_own_team(battle: DoubleBattle) -> list:
    team = list(battle.team.values())
    active_set = {pokemon for pokemon in battle.active_pokemon if pokemon is not None}
    active = [pokemon for pokemon in team if pokemon in active_set]
    bench = [pokemon for pokemon in team if pokemon not in active_set]
    ordered = active + bench
    while len(ordered) < MAX_TEAM_SLOTS:
        ordered.append(None)
    return ordered[:MAX_TEAM_SLOTS]


def _find_team_slot(team_list: list, pokemon) -> int | None:
    for index, team_member in enumerate(team_list):
        if team_member is pokemon:
            return index
    return None


def _showdown_target_to_index(
    showdown_target: int,
    slot: int,
    battle: DoubleBattle,
) -> int:
    ally_position = battle.POKEMON_2_POSITION if slot == 0 else battle.POKEMON_1_POSITION
    self_position = battle.POKEMON_1_POSITION if slot == 0 else battle.POKEMON_2_POSITION

    mapping = {
        battle.OPPONENT_1_POSITION: TARGET_OPPONENT_1,
        battle.OPPONENT_2_POSITION: TARGET_OPPONENT_2,
        ally_position: TARGET_ALLY,
        self_position: TARGET_SELF,
        battle.EMPTY_TARGET_POSITION: TARGET_NONE,
    }
    try:
        return mapping[showdown_target]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported showdown target {showdown_target} for slot {slot}. "
            "Expand the action encoding to represent this target."
        ) from exc


def _legal_target_locs(move: Move, active_pokemon, battle: DoubleBattle) -> list[int]:
    target_locs = battle.get_possible_showdown_targets(move, active_pokemon)
    if not target_locs:
        raise RuntimeError(
            f"poke-env produced no legal Showdown targets for move {move.id!r}. "
            "This indicates the request state and target mapping are inconsistent."
        )
    return list(target_locs)


def _is_random_normal_move(move: Move) -> bool:
    target = move.target
    return target is not None and target.name == "RANDOM_NORMAL"
