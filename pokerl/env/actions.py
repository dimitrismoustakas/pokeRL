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
#   0..19:  move_i * 5 + target_j        (4 moves x 5 targets)
#  20..25:  switch to own_team_slot_k    (stable team slot, 0..5)
#  26:      pass                         (slot has no action this turn)
#
# Terastallization is a separate joint decision made after both slot actions:
#   0: no tera
#   1: tera slot 0
#   2: tera slot 1
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
SWITCH_OFFSET = NUM_MOVES * NUM_TARGETS        # 20
PASS_ACTION = SWITCH_OFFSET + MAX_TEAM_SLOTS   # 26
ACTIONS_PER_SLOT = PASS_ACTION + 1             # 27
NUM_TERA_CHOICES = 3
TERA_NONE = 0
TERA_SLOT_1 = 1
TERA_SLOT_2 = 2
FIRST_SLOT_MASK_ROW = 0


@dataclass(frozen=True)
class LegalActionState:
    """Per-request legality tables derived from the live Showdown request."""

    slot_action_orders: tuple[
        dict[int, SingleBattleOrder | PassBattleOrder],
        dict[int, SingleBattleOrder | PassBattleOrder],
    ]
    slot_masks: tuple[np.ndarray, np.ndarray]
    joint_mask: np.ndarray
    can_tera: np.ndarray


@dataclass(frozen=True)
class _SlotActionState:
    action_orders: dict[int, SingleBattleOrder | PassBattleOrder]
    mask: np.ndarray
    joined_orders: list[SingleBattleOrder | PassBattleOrder]
    action_by_order_id: dict[int, int]


def decode_action(action_idx: int) -> tuple[str, int, int]:
    """Decode a flat action index into (kind, move_or_switch_idx, target)."""
    if action_idx == PASS_ACTION:
        return "pass", -1, TARGET_NONE

    if action_idx >= SWITCH_OFFSET:
        return "switch", action_idx - SWITCH_OFFSET, TARGET_NONE

    move_idx = action_idx // NUM_TARGETS
    target_idx = action_idx % NUM_TARGETS
    return "move", move_idx, target_idx


def decode_tera_choice(tera_choice: int) -> tuple[bool, bool]:
    """Decode the joint tera decision into per-slot booleans."""
    if tera_choice == TERA_NONE:
        return False, False
    if tera_choice == TERA_SLOT_1:
        return True, False
    if tera_choice == TERA_SLOT_2:
        return False, True
    raise ValueError(f"Unsupported tera choice {tera_choice}.")


def build_legal_action_state(battle: DoubleBattle) -> LegalActionState:
    """Build and index all legal orders for the current live request once."""
    slot_states = (
        _build_slot_action_state(battle, slot=0),
        _build_slot_action_state(battle, slot=1),
    )
    joint_mask = _build_joint_mask(slot_states)
    return LegalActionState(
        slot_action_orders=(slot_states[0].action_orders, slot_states[1].action_orders),
        slot_masks=(slot_states[0].mask, slot_states[1].mask),
        joint_mask=joint_mask,
        can_tera=np.asarray(battle.can_tera, dtype=bool).copy(),
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


def build_tera_action_mask(
    battle: DoubleBattle,
    action_state: LegalActionState | None = None,
) -> np.ndarray:
    """Build the legal joint tera table conditioned on both chosen slot actions."""
    if action_state is None:
        action_state = build_legal_action_state(battle)
    return _build_tera_action_mask(action_state.joint_mask, action_state.can_tera)


def build_policy_action_masks(
    battle: DoubleBattle,
    action_state: LegalActionState | None = None,
) -> dict[str, np.ndarray]:
    """Build the full policy legality tables for both slots and the tera decision."""
    if action_state is None:
        action_state = build_legal_action_state(battle)
    return {
        "joint": action_state.joint_mask,
        "can_tera": action_state.can_tera,
    }


def empty_action_masks() -> dict[str, np.ndarray]:
    """Return a pass-only legality table used for finished battles."""
    joint_mask = np.zeros((ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)
    joint_mask[FIRST_SLOT_MASK_ROW, PASS_ACTION] = True
    joint_mask[PASS_ACTION + 1, PASS_ACTION] = True

    return {
        "joint": joint_mask,
        "can_tera": np.zeros(2, dtype=bool),
    }


def _build_joint_mask(
    slot_states: tuple[_SlotActionState, _SlotActionState],
) -> np.ndarray:
    joint_mask = np.zeros((ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)

    joined_orders = DoubleBattleOrder.join_orders(
        slot_states[0].joined_orders,
        slot_states[1].joined_orders,
    )
    if not joined_orders:
        if (
            set(slot_states[0].action_orders) == {PASS_ACTION}
            and set(slot_states[1].action_orders) == {PASS_ACTION}
        ):
            joint_mask[FIRST_SLOT_MASK_ROW, PASS_ACTION] = True
            joint_mask[PASS_ACTION + 1, PASS_ACTION] = True
            return joint_mask
        raise RuntimeError(
            "poke-env produced no legal joined doubles orders for the current request. "
            "This indicates the request state and action mapping are inconsistent."
        )

    for order in joined_orders:
        a1 = slot_states[0].action_by_order_id[id(order.first_order)]
        a2 = slot_states[1].action_by_order_id[id(order.second_order)]
        joint_mask[FIRST_SLOT_MASK_ROW, a1] = True
        joint_mask[a1 + 1, a2] = True

    return joint_mask


def _build_tera_action_mask(
    joint_mask: np.ndarray,
    can_tera: np.ndarray,
) -> np.ndarray:
    tera_mask = np.zeros((ACTIONS_PER_SLOT, ACTIONS_PER_SLOT, NUM_TERA_CHOICES), dtype=bool)

    for a1 in np.flatnonzero(joint_mask[FIRST_SLOT_MASK_ROW]):
        kind1, _, _ = decode_action(int(a1))
        for a2 in np.flatnonzero(joint_mask[a1 + 1]):
            kind2, _, _ = decode_action(int(a2))
            tera_mask[a1, a2, TERA_NONE] = True
            if kind1 == "move" and can_tera[0]:
                tera_mask[a1, a2, TERA_SLOT_1] = True
            if kind2 == "move" and can_tera[1]:
                tera_mask[a1, a2, TERA_SLOT_2] = True

    return tera_mask


def action_to_order(
    action_idx: int,
    battle: DoubleBattle,
    slot: int,
    terastallize: bool = False,
    action_state: LegalActionState | None = None,
) -> SingleBattleOrder | PassBattleOrder:
    """Convert a flat action index to the exact current legal order."""
    if action_state is None:
        action_state = build_legal_action_state(battle)
    try:
        base_order = action_state.slot_action_orders[slot][action_idx]
    except KeyError as exc:
        raise RuntimeError(
            f"Action {action_idx} is not legal for slot {slot} in the current request. "
            "The policy should only sample actions allowed by build_joint_action_mask()."
        ) from exc

    if not terastallize:
        return base_order

    kind, _, _ = decode_action(action_idx)
    if kind != "move":
        raise RuntimeError(
            f"Action {action_idx} for slot {slot} cannot terastallize because it is a {kind} action."
        )
    if not battle.can_tera[slot]:
        raise RuntimeError(
            f"Action {action_idx} for slot {slot} requested terastallization, but battle.can_tera[{slot}] is false."
        )
    if isinstance(base_order, PassBattleOrder):
        raise RuntimeError(f"Pass action {action_idx} cannot terastallize.")
    return SingleBattleOrder(
        base_order.order,
        terastallize=True,
        move_target=base_order.move_target,
    )


def _build_slot_action_state(battle: DoubleBattle, slot: int) -> _SlotActionState:
    trapped = getattr(battle, "trapped", [False, False])
    force_switch = battle.force_switch
    action_orders: dict[int, SingleBattleOrder | PassBattleOrder] = {}
    joined_orders: list[SingleBattleOrder | PassBattleOrder] = []
    action_by_order_id: dict[int, int] = {}
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)

    if any(force_switch) and not force_switch[slot]:
        return _build_pass_slot_state(mask)

    if not trapped[slot]:
        ordered_team = _ordered_own_team(battle)
        for pokemon in battle.available_switches[slot]:
            team_slot = _find_team_slot(ordered_team, pokemon)
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
        for move_idx, move in enumerate(battle.available_moves[slot]):
            hydrated_move, target_locs = _request_aware_move_targets(move, active_pokemon, battle, slot)
            for target_loc in target_locs:
                target_idx = _showdown_target_to_index(target_loc, slot, battle)
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
    action_orders: dict[int, SingleBattleOrder | PassBattleOrder],
    mask: np.ndarray,
    joined_orders: list[SingleBattleOrder | PassBattleOrder],
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
    action_orders: dict[int, SingleBattleOrder | PassBattleOrder],
    action_by_order_id: dict[int, int],
    joined_orders: list[SingleBattleOrder | PassBattleOrder],
    mask: np.ndarray,
    action_idx: int,
    order: SingleBattleOrder | PassBattleOrder,
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
