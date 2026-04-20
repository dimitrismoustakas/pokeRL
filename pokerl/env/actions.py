"""Action space helpers for doubles battles.

The model acts in a fixed discrete space. Single-slot move targets come from
``poke-env``'s request-aware Showdown target resolver. Joint legality is
computed directly from SingleBattleOrder attributes without allocating
DoubleBattleOrder objects.
"""

from dataclasses import dataclass

from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.player import PassBattleOrder, SingleBattleOrder

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
    slot_masks: tuple[np.ndarray, np.ndarray]
    joint_mask: np.ndarray


@dataclass(frozen=True)
class _SlotActionState:
    action_orders: dict[int, SingleBattleOrder]
    mask: np.ndarray
    joined_orders: list[SingleBattleOrder]
    action_by_order_id: dict[int, int]


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
    ordered_team = _ordered_own_team(battle)
    slot_states = (
        _build_slot_action_state(battle, slot=0, ordered_team=ordered_team),
        _build_slot_action_state(battle, slot=1, ordered_team=ordered_team),
    )
    joint_mask = _build_joint_mask(slot_states)
    return LegalActionState(
        slot_action_orders=(slot_states[0].action_orders, slot_states[1].action_orders),
        slot_masks=(slot_states[0].mask, slot_states[1].mask),
        joint_mask=joint_mask,
    )


def build_action_mask(
    battle: DoubleBattle,
    slot: int,
    action_state: LegalActionState | None = None,
) -> np.ndarray:
    """Build a slot-local mask from the current Showdown request."""
    if action_state is None:
        action_state = build_legal_action_state(battle)
    return action_state.slot_masks[slot]


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
    slot_states: tuple[_SlotActionState, _SlotActionState],
) -> np.ndarray:
    joint_mask = np.zeros((ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)
    orders0 = slot_states[0].action_orders
    orders1 = slot_states[1].action_orders
    any_compatible = False

    for a1, o1 in orders0.items():
        for a2, o2 in orders1.items():
            if _orders_compatible(o1, o2):
                joint_mask[FIRST_SLOT_MASK_ROW, a1] = True
                joint_mask[a1 + 1, a2] = True
                any_compatible = True

    if not any_compatible:
        if set(orders0) == {PASS_ACTION} and set(orders1) == {PASS_ACTION}:
            # Both slots are forced to pass — allow the double-pass combination.
            joint_mask[FIRST_SLOT_MASK_ROW, PASS_ACTION] = True
            joint_mask[PASS_ACTION + 1, PASS_ACTION] = True
            return joint_mask
        raise RuntimeError(
            "No legal paired doubles orders for the current request. "
            "This indicates the request state and action mapping are inconsistent."
        )

    return joint_mask


def _orders_compatible(o1: SingleBattleOrder, o2: SingleBattleOrder) -> bool:
    """Return True if o1 and o2 can be combined into a legal DoubleBattleOrder.

    Replicates all filters from DoubleBattleOrder.join_orders without allocating
    DoubleBattleOrder objects or formatting message strings.
    """
    if o1.mega and o2.mega:
        return False
    if o1.z_move and o2.z_move:
        return False
    if o1.dynamax and o2.dynamax:
        return False
    if o1.terastallize and o2.terastallize:
        return False
    # Switch conflict: same Pokemon object means the same switch target.
    # Using identity (is) instead of str() because available_switches always
    # yields the canonical Pokemon object from battle.team — no string formatting needed.
    if isinstance(o1.order, Pokemon) and isinstance(o2.order, Pokemon) and o1.order is o2.order:
        return False
    if isinstance(o1, PassBattleOrder) and isinstance(o2, PassBattleOrder):
        return False
    return True


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


def _build_slot_action_state(
    battle: DoubleBattle,
    slot: int,
    ordered_team: list,
) -> _SlotActionState:
    trapped = getattr(battle, "trapped", [False, False])
    force_switch = battle.force_switch
    action_orders: dict[int, SingleBattleOrder] = {}
    joined_orders: list[SingleBattleOrder] = []
    action_by_order_id: dict[int, int] = {}
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)

    if any(force_switch) and not force_switch[slot]:
        return _build_pass_slot_state(mask)

    if not trapped[slot]:
        # O(1) lookup per switch instead of O(team_size) linear search.
        team_slot_by_id = {id(p): i for i, p in enumerate(ordered_team) if p is not None}
        for pokemon in battle.available_switches[slot]:
            team_slot = team_slot_by_id.get(id(pokemon))
            if team_slot is None:
                raise RuntimeError(
                    f"Available switch target {pokemon.species!r} was not found in the ordered team. "
                    "This indicates battle.team and available_switches are out of sync."
                )
            _add_order(
                action_orders=action_orders,
                action_by_order_id=action_by_order_id,
                joined_orders=joined_orders,
                mask=mask,
                action_idx=SWITCH_OFFSET + team_slot,
                order=SingleBattleOrder(pokemon),
                slot=slot,
            )

    if all(force_switch) and len(battle.available_switches[0]) == 1:
        _add_order(
            action_orders=action_orders,
            action_by_order_id=action_by_order_id,
            joined_orders=joined_orders,
            mask=mask,
            action_idx=PASS_ACTION,
            order=PassBattleOrder(),
            slot=slot,
        )
        return _finalize_slot_state(action_orders, mask, joined_orders, action_by_order_id)

    active_pokemon = battle.active_pokemon[slot]
    if active_pokemon is not None and not force_switch[slot]:
        # Build target mapping once per slot instead of once per (move, target) call.
        target_mapping = _build_target_mapping(slot, battle)
        for move_idx, move in enumerate(battle.available_moves[slot]):
            hydrated_move, target_locs = _request_aware_move_targets(move, active_pokemon, battle, slot)
            for target_loc in target_locs:
                target_idx = target_mapping.get(target_loc)
                if target_idx is None:
                    raise RuntimeError(
                        f"Unsupported showdown target {target_loc} for slot {slot}. "
                        "Expand the action encoding to represent this target."
                    )
                action_idx = move_idx * NUM_TARGETS + target_idx
                _add_order(
                    action_orders=action_orders,
                    action_by_order_id=action_by_order_id,
                    joined_orders=joined_orders,
                    mask=mask,
                    action_idx=action_idx,
                    order=SingleBattleOrder(hydrated_move, move_target=target_loc),
                    slot=slot,
                )
                if battle.can_tera[slot]:
                    _add_order(
                        action_orders=action_orders,
                        action_by_order_id=action_by_order_id,
                        joined_orders=joined_orders,
                        mask=mask,
                        action_idx=action_idx + TERA_OFFSET,
                        order=SingleBattleOrder(hydrated_move, move_target=target_loc, terastallize=True),
                        slot=slot,
                    )

    if not action_orders:
        return _build_pass_slot_state(mask)
    return _finalize_slot_state(action_orders, mask, joined_orders, action_by_order_id)


def _build_pass_slot_state(mask: np.ndarray) -> _SlotActionState:
    pass_order = PassBattleOrder()
    action_orders = {PASS_ACTION: pass_order}
    mask[PASS_ACTION] = True
    return _SlotActionState(
        action_orders=action_orders,
        mask=mask,
        joined_orders=[pass_order],
        action_by_order_id={id(pass_order): PASS_ACTION},
    )


def _finalize_slot_state(
    action_orders: dict[int, SingleBattleOrder],
    mask: np.ndarray,
    joined_orders: list[SingleBattleOrder],
    action_by_order_id: dict[int, int],
) -> _SlotActionState:
    return _SlotActionState(
        action_orders=action_orders,
        mask=mask,
        joined_orders=joined_orders,
        action_by_order_id=action_by_order_id,
    )


def _add_order(
    *,
    action_orders: dict[int, SingleBattleOrder],
    action_by_order_id: dict[int, int],
    joined_orders: list[SingleBattleOrder],
    mask: np.ndarray,
    action_idx: int,
    order: SingleBattleOrder,
    slot: int,
) -> None:
    existing = action_orders.get(action_idx)
    if existing is not None:
        if existing.message != order.message:
            raise RuntimeError(
                f"Two different legal orders mapped to the same action id {action_idx} "
                f"for slot {slot}: {existing.message!r} vs {order.message!r}. "
                "Expand the action encoding so these orders are distinguishable."
            )
        return

    action_orders[action_idx] = order
    mask[action_idx] = True
    joined_orders.append(order)
    action_by_order_id[id(order)] = action_idx


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


def _request_aware_move_targets(
    move: Move,
    active_pokemon,
    battle: DoubleBattle,
    slot: int,
) -> tuple[Move, list[int]]:
    # Fetch the request entry once; pass it to the hydrator to avoid a second lookup.
    move_request = _request_move_entry(move, battle, slot)
    hydrated_move = Move(move.id, gen=move.gen)
    target = None if move_request is None else move_request.get("target")
    if target is not None:
        hydrated_move.request_target = target
    elif move_request is not None and _is_random_normal_move(move):
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


def _build_target_mapping(slot: int, battle: DoubleBattle) -> dict[int, int]:
    """Build the showdown-position → action-index map once per slot per step."""
    ally_position = battle.POKEMON_2_POSITION if slot == 0 else battle.POKEMON_1_POSITION
    self_position = battle.POKEMON_1_POSITION if slot == 0 else battle.POKEMON_2_POSITION
    return {
        battle.OPPONENT_1_POSITION: TARGET_OPPONENT_1,
        battle.OPPONENT_2_POSITION: TARGET_OPPONENT_2,
        ally_position: TARGET_ALLY,
        self_position: TARGET_SELF,
        battle.EMPTY_TARGET_POSITION: TARGET_NONE,
    }


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
