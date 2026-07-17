import numpy as np
import pytest
from poke_env.battle import Move, Pokemon
from poke_env.player import (
    DoubleBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
    SkippedBattleOrder,
)

from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    PASS_ACTION,
    SWITCH_OFFSET,
    _EXPLICIT_PASS,
    _SKIPPED_PASS,
    _SlotActionState,
    _build_joint_mask,
)


def _slot_state(action_orders: dict[int, object]) -> _SlotActionState:
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)
    mask[list(action_orders)] = True
    return _SlotActionState(action_orders=action_orders, mask=mask)


def _join_orders_reference(
    states: tuple[_SlotActionState, _SlotActionState],
) -> np.ndarray:
    """Reproduce the former object-based mask builder as a test oracle."""
    joint_mask = np.zeros((ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)
    materialized = []
    for state in states:
        orders = {}
        for action, entry in state.action_orders.items():
            if isinstance(entry, (Move, Pokemon)):
                orders[action] = SingleBattleOrder(entry)
            elif entry is _EXPLICIT_PASS:
                orders[action] = PassBattleOrder()
            elif entry is _SKIPPED_PASS:
                orders[action] = SkippedBattleOrder()
            else:
                raise AssertionError(f"Unsupported test order template: {entry!r}")
        materialized.append(orders)
    first_by_id = {id(order): action for action, order in materialized[0].items()}
    second_by_id = {id(order): action for action, order in materialized[1].items()}
    joined = DoubleBattleOrder.join_orders(
        list(materialized[0].values()),
        list(materialized[1].values()),
    )
    if not joined:
        if set(states[0].action_orders) == {PASS_ACTION} and set(
            states[1].action_orders
        ) == {PASS_ACTION}:
            joint_mask[FIRST_SLOT_MASK_ROW, PASS_ACTION] = True
            joint_mask[PASS_ACTION + 1, PASS_ACTION] = True
            return joint_mask
        raise RuntimeError("reference produced no legal joined orders")

    for order in joined:
        first_action = first_by_id[id(order.first_order)]
        second_action = second_by_id[id(order.second_order)]
        joint_mask[FIRST_SLOT_MASK_ROW, first_action] = True
        joint_mask[first_action + 1, second_action] = True
    return joint_mask


def test_direct_joint_mask_matches_join_orders_for_moves_switches_and_passes() -> None:
    shared_switch = Pokemon(gen=9, species="Pikachu")
    other_switch = Pokemon(gen=9, species="Raichu")
    states = (
        _slot_state(
            {
                0: Move("protect", gen=9),
                SWITCH_OFFSET: shared_switch,
                SWITCH_OFFSET + 1: other_switch,
                PASS_ACTION: _EXPLICIT_PASS,
            }
        ),
        _slot_state(
            {
                1: Move("detect", gen=9),
                SWITCH_OFFSET: shared_switch,
                SWITCH_OFFSET + 1: other_switch,
                PASS_ACTION: _SKIPPED_PASS,
            }
        ),
    )

    np.testing.assert_array_equal(
        _build_joint_mask(states), _join_orders_reference(states)
    )


def test_direct_joint_mask_preserves_pass_only_default_fallback() -> None:
    states = (
        _slot_state({PASS_ACTION: _EXPLICIT_PASS}),
        _slot_state({PASS_ACTION: _EXPLICIT_PASS}),
    )

    np.testing.assert_array_equal(
        _build_joint_mask(states), _join_orders_reference(states)
    )


def test_direct_joint_mask_rejects_an_only_duplicate_switch_pair() -> None:
    shared_switch = Pokemon(gen=9, species="Pikachu")
    states = (
        _slot_state({SWITCH_OFFSET: shared_switch}),
        _slot_state({SWITCH_OFFSET: shared_switch}),
    )

    with pytest.raises(RuntimeError, match="no legal joined doubles orders"):
        _build_joint_mask(states)
