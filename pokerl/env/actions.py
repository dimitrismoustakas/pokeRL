"""Action space helpers for doubles battles.

The model acts in a fixed discrete space. Single-slot move targets come from
``poke-env``'s request-aware Showdown target resolver, and joint legality is
derived directly from the indexed slot orders.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.player import PassBattleOrder, SingleBattleOrder, SkippedBattleOrder

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
SWITCH_OFFSET = NUM_MOVES * NUM_TARGETS  # 20
PASS_ACTION = SWITCH_OFFSET + MAX_TEAM_SLOTS  # 26
ACTIONS_PER_SLOT = PASS_ACTION + 1  # 27
NUM_TERA_CHOICES = 3
TERA_NONE = 0
TERA_SLOT_1 = 1
TERA_SLOT_2 = 2
FIRST_SLOT_MASK_ROW = 0


_BattleOrder = SingleBattleOrder | PassBattleOrder | SkippedBattleOrder


@dataclass(frozen=True, slots=True)
class _PassOrderTemplate:
    explicit: bool


_EXPLICIT_PASS = _PassOrderTemplate(explicit=True)
_SKIPPED_PASS = _PassOrderTemplate(explicit=False)
_OrderEntry = Move | Pokemon | _PassOrderTemplate


@dataclass(frozen=True)
class LegalActionState:
    """Per-request legality tables derived from the live Showdown request."""

    slot_action_orders: tuple[
        Mapping[int, _OrderEntry],
        Mapping[int, _OrderEntry],
    ]
    slot_masks: tuple[np.ndarray, np.ndarray]
    joint_mask: np.ndarray
    can_tera: np.ndarray


INTENT_NONE = 0
INTENT_PASS = 1
INTENT_MOVE = 2
INTENT_SWITCH = 3
NUM_INTENT_KINDS = 4

SLOT_INTENT_KIND = 0
SLOT_INTENT_MOVE = 1
SLOT_INTENT_TARGET = 2
SLOT_INTENT_SWITCH_SPECIES = 3
NUM_SLOT_INTENT_COMPONENTS = 4
TERA_INTENT_USE = 0
TERA_INTENT_SLOT = 1
NUM_TERA_INTENT_COMPONENTS = 2


@dataclass(frozen=True)
class SemanticSlotIntent:
    """Stable meaning of one submitted action after request-local resolution."""

    kind: int
    move_id: str | None = None
    target: int = TARGET_NONE
    switch_species: str | None = None
    terastallize: bool = False


@dataclass(frozen=True)
class SemanticActionIntent:
    """The two policy-controlled slot choices submitted for one request."""

    slots: tuple[SemanticSlotIntent, SemanticSlotIntent]


@dataclass(frozen=True)
class SemanticSlotDecisionMask:
    """Which semantic components represented a real choice for one slot."""

    kind: bool
    move: bool
    target: bool
    switch_species: bool


@dataclass(frozen=True)
class SemanticActionDecisionMask:
    """Component-specific masks for trainer-only opponent supervision."""

    slots: tuple[SemanticSlotDecisionMask, SemanticSlotDecisionMask]
    tera_use: bool
    tera_slot: bool


@dataclass(frozen=True)
class SemanticActionSupervision:
    """Chosen semantic intent plus masks for non-forced decision components."""

    intent: SemanticActionIntent
    decision_mask: SemanticActionDecisionMask


def semantic_action_request_key(battle: DoubleBattle) -> tuple[str, int]:
    """Return poke-env's backend-independent decision-request identity."""
    generation = getattr(battle, "request_generation", None)
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise RuntimeError(
            "Semantic action supervision requires poke-env request_generation."
        )
    return str(getattr(battle, "battle_tag", "")), generation


@dataclass(frozen=True)
class _SlotActionState:
    action_orders: Mapping[int, _OrderEntry]
    mask: np.ndarray


def decode_action(action_idx: int) -> tuple[str, int, int]:
    """Decode a flat action index into (kind, move_or_switch_idx, target)."""
    if action_idx == PASS_ACTION:
        return "pass", -1, TARGET_NONE

    if action_idx >= SWITCH_OFFSET:
        return "switch", action_idx - SWITCH_OFFSET, TARGET_NONE

    move_idx = action_idx // NUM_TARGETS
    target_idx = action_idx % NUM_TARGETS
    return "move", move_idx, target_idx


def request_move_slot(move: Move, *, active_slot: int) -> int:
    """Return the fixed zero-based action slot from Showdown request metadata."""
    request_index = getattr(move, "request_index", None)
    if not isinstance(request_index, int) or isinstance(request_index, bool):
        raise RuntimeError(
            f"Available move {move.id!r} for active slot {active_slot} has no valid "
            "Showdown request index."
        )
    move_slot = request_index - 1
    if not 0 <= move_slot < NUM_MOVES:
        raise RuntimeError(
            f"Available move {move.id!r} for active slot {active_slot} has request "
            f"index {request_index}; expected an index from 1 to {NUM_MOVES}."
        )
    return move_slot


def decode_tera_choice(tera_choice: int) -> tuple[bool, bool]:
    """Decode the joint tera decision into per-slot booleans."""
    if tera_choice == TERA_NONE:
        return False, False
    if tera_choice == TERA_SLOT_1:
        return True, False
    if tera_choice == TERA_SLOT_2:
        return False, True
    raise ValueError(f"Unsupported tera choice {tera_choice}.")


def semantic_action_intent(
    action: np.ndarray,
    action_state: LegalActionState,
) -> SemanticActionIntent:
    """Resolve request-local action ids to move/species semantics."""
    action = np.asarray(action, dtype=np.int64)
    if action.shape != (3,):
        raise ValueError("A semantic doubles intent requires action shape (3,).")
    tera_slots = decode_tera_choice(int(action[2]))
    slots = [
        _semantic_slot_intent(
            action_state,
            slot,
            int(action[slot]),
            terastallize=tera_slots[slot],
        )
        for slot in range(2)
    ]
    return SemanticActionIntent(slots=(slots[0], slots[1]))


def semantic_action_supervision(
    action: np.ndarray,
    action_state: LegalActionState,
) -> SemanticActionSupervision:
    """Resolve an action and mask semantic components that were forced.

    The first slot is evaluated against every legal first action. The second
    slot is evaluated against actions legal after the selected first action,
    matching the policy's autoregressive decision order. Tera is split into a
    use-now gate and a conditional slot choice.
    """
    action = np.asarray(action, dtype=np.int64)
    if action.shape != (3,):
        raise ValueError("Semantic action supervision requires action shape (3,).")

    intent = semantic_action_intent(action, action_state)
    first_action = int(action[0])
    second_action = int(action[1])
    if not action_state.joint_mask[FIRST_SLOT_MASK_ROW, first_action]:
        raise RuntimeError("Selected first-slot action is absent from the joint mask.")
    if not action_state.joint_mask[first_action + 1, second_action]:
        raise RuntimeError(
            "Selected second-slot action is absent from its conditional joint mask."
        )

    candidate_indices = (
        np.flatnonzero(action_state.joint_mask[FIRST_SLOT_MASK_ROW]),
        np.flatnonzero(action_state.joint_mask[first_action + 1]),
    )
    slot_masks: list[SemanticSlotDecisionMask] = []
    for slot, indices in enumerate(candidate_indices):
        candidates = [
            _semantic_slot_intent(
                action_state,
                slot,
                int(action_index),
                terastallize=False,
            )
            for action_index in indices
        ]
        chosen = intent.slots[slot]
        kinds = {candidate.kind for candidate in candidates}
        move_ids = {
            candidate.move_id
            for candidate in candidates
            if candidate.kind == INTENT_MOVE
        }
        chosen_move_targets = {
            candidate.target
            for candidate in candidates
            if candidate.kind == INTENT_MOVE and candidate.move_id == chosen.move_id
        }
        switch_species = {
            candidate.switch_species
            for candidate in candidates
            if candidate.kind == INTENT_SWITCH
        }
        slot_masks.append(
            SemanticSlotDecisionMask(
                kind=len(kinds) > 1,
                move=chosen.kind == INTENT_MOVE and len(move_ids) > 1,
                target=(chosen.kind == INTENT_MOVE and len(chosen_move_targets) > 1),
                switch_species=(
                    chosen.kind == INTENT_SWITCH and len(switch_species) > 1
                ),
            )
        )

    tera_mask = _build_tera_action_mask(
        action_state.joint_mask,
        action_state.can_tera,
    )[first_action, second_action]
    tera_choice = int(action[2])
    if not tera_mask[tera_choice]:
        raise RuntimeError("Selected Tera choice is absent from the legal Tera mask.")
    legal_tera_choices = np.flatnonzero(tera_mask)
    tera_use_options = {int(choice) != TERA_NONE for choice in legal_tera_choices}
    tera_slot_options = {
        int(choice) - 1 for choice in legal_tera_choices if choice != TERA_NONE
    }
    tera_used = tera_choice != TERA_NONE

    return SemanticActionSupervision(
        intent=intent,
        decision_mask=SemanticActionDecisionMask(
            slots=(slot_masks[0], slot_masks[1]),
            tera_use=len(tera_use_options) > 1,
            tera_slot=tera_used and len(tera_slot_options) > 1,
        ),
    )


def _semantic_slot_intent(
    action_state: LegalActionState,
    slot: int,
    action_index: int,
    *,
    terastallize: bool,
) -> SemanticSlotIntent:
    action_orders = action_state.slot_action_orders[slot]
    kind, _, target = decode_action(action_index)
    try:
        entry = action_orders[action_index]
    except KeyError as exc:
        raise RuntimeError(
            f"Action {action_index} is not legal for slot {slot}."
        ) from exc
    if kind == "move":
        if not isinstance(entry, Move):
            raise RuntimeError("A move action did not resolve to a Move order.")
        return SemanticSlotIntent(
            kind=INTENT_MOVE,
            move_id=entry.id,
            target=target,
            terastallize=terastallize,
        )
    if kind == "switch":
        species = getattr(entry, "species", None)
        if species is None:
            raise RuntimeError("A switch action did not resolve to a Pokemon.")
        return SemanticSlotIntent(
            kind=INTENT_SWITCH,
            switch_species=str(species),
        )
    return SemanticSlotIntent(kind=INTENT_PASS)


def build_legal_action_state(battle: DoubleBattle) -> LegalActionState:
    """Build and index the legal action templates for one live request."""
    team_slot_by_identity = {
        id(pokemon): team_slot
        for team_slot, pokemon in enumerate(_ordered_own_team(battle))
        if pokemon is not None
    }
    slot_states = (
        _build_slot_action_state(
            battle,
            slot=0,
            team_slot_by_identity=team_slot_by_identity,
        ),
        _build_slot_action_state(
            battle,
            slot=1,
            team_slot_by_identity=team_slot_by_identity,
        ),
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


def is_default_only_action_state(action_state: LegalActionState) -> bool:
    """Return true when Showdown only accepts the implicit default order.

    These requests do not represent a policy decision. Routing them through the
    neural policy adds avoidable queue pressure and can block local protocol
    dispatch behind inference that cannot affect the emitted order.
    """
    slot_one_orders, slot_two_orders = action_state.slot_action_orders
    if set(slot_one_orders) != {PASS_ACTION} or set(slot_two_orders) != {PASS_ACTION}:
        return False
    if np.any(action_state.can_tera):
        return False
    joint_mask = action_state.joint_mask
    return (
        int(joint_mask.sum()) == 2
        and bool(joint_mask[FIRST_SLOT_MASK_ROW, PASS_ACTION])
        and bool(joint_mask[PASS_ACTION + 1, PASS_ACTION])
    )


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

    # The slot builders only create ordinary moves, switches, and pass/skip
    # orders.  For those orders, DoubleBattleOrder.join_orders() is equivalent
    # to the Cartesian product below with two exclusions: both slots cannot
    # switch to the same Pokemon, and two explicit pass orders are not a real
    # joint choice. A skipped slot is distinct from an explicit pass and may be
    # paired with one. Building the boolean table directly avoids allocating a
    # DoubleBattleOrder for every candidate pair on every policy decision.
    legal_pairs = np.logical_and.outer(slot_states[0].mask, slot_states[1].mask)
    switch_actions = np.arange(SWITCH_OFFSET, PASS_ACTION)
    legal_pairs[switch_actions, switch_actions] = False
    if _has_explicit_pass(slot_states[0].action_orders) and _has_explicit_pass(
        slot_states[1].action_orders
    ):
        legal_pairs[PASS_ACTION, PASS_ACTION] = False

    if not np.any(legal_pairs):
        if set(slot_states[0].action_orders) == {PASS_ACTION} and set(
            slot_states[1].action_orders
        ) == {PASS_ACTION}:
            legal_pairs[PASS_ACTION, PASS_ACTION] = True
        else:
            raise RuntimeError(
                "poke-env produced no legal joined doubles orders for the current request. "
                "This indicates the request state and action mapping are inconsistent."
            )

    joint_mask[FIRST_SLOT_MASK_ROW] = np.any(legal_pairs, axis=1)
    joint_mask[1:] = legal_pairs

    return joint_mask


def _build_tera_action_mask(
    joint_mask: np.ndarray,
    can_tera: np.ndarray,
) -> np.ndarray:
    tera_mask = np.zeros(
        (ACTIONS_PER_SLOT, ACTIONS_PER_SLOT, NUM_TERA_CHOICES), dtype=bool
    )

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
) -> SingleBattleOrder | PassBattleOrder | SkippedBattleOrder:
    """Convert a flat action index to the exact current legal order."""
    if action_state is None:
        action_state = build_legal_action_state(battle)
    action_orders = action_state.slot_action_orders[slot]
    if action_idx not in action_orders:
        raise RuntimeError(
            f"Action {action_idx} is not legal for slot {slot} in the current request. "
            "The policy should only sample actions allowed by build_joint_action_mask()."
        )

    kind, _, _ = decode_action(action_idx)
    if terastallize:
        if kind != "move":
            raise RuntimeError(
                f"Action {action_idx} for slot {slot} cannot terastallize because it is a {kind} action."
            )
        if not battle.can_tera[slot]:
            raise RuntimeError(
                f"Action {action_idx} for slot {slot} requested terastallization, but battle.can_tera[{slot}] is false."
            )

    entry = action_orders[action_idx]
    if isinstance(entry, Move):
        return SingleBattleOrder(
            entry,
            terastallize=terastallize,
            move_target=_showdown_target_from_action(action_idx, slot, battle),
        )
    if kind == "switch" and getattr(entry, "species", None) is not None:
        return SingleBattleOrder(entry)
    if isinstance(entry, _PassOrderTemplate):
        return PassBattleOrder() if entry.explicit else SkippedBattleOrder()
    raise RuntimeError(
        f"Action {action_idx} for slot {slot} has an unsupported order template."
    )


def _build_slot_action_state(
    battle: DoubleBattle,
    slot: int,
    *,
    team_slot_by_identity: dict[int, int] | None = None,
) -> _SlotActionState:
    trapped = getattr(battle, "trapped", [False, False])
    force_switch = battle.force_switch
    action_orders: dict[int, _OrderEntry] = {}
    mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)

    if any(force_switch) and not force_switch[slot]:
        return _build_pass_slot_state(mask, explicit=True)

    if not trapped[slot]:
        if team_slot_by_identity is None:
            team_slot_by_identity = {
                id(pokemon): team_slot
                for team_slot, pokemon in enumerate(_ordered_own_team(battle))
                if pokemon is not None
            }
        for pokemon in battle.available_switches[slot]:
            team_slot = team_slot_by_identity.get(id(pokemon))
            if team_slot is None:
                raise RuntimeError(
                    f"Available switch target {pokemon.species!r} was not found in the ordered team. "
                    "This indicates battle.team and available_switches are out of sync."
                )
            _add_switch_template(
                action_orders=action_orders,
                mask=mask,
                action_idx=SWITCH_OFFSET + team_slot,
                pokemon=pokemon,
                slot=slot,
            )

    if all(force_switch) and len(battle.available_switches[0]) == 1:
        _add_pass_template(
            action_orders,
            mask,
            explicit=True,
        )
        return _SlotActionState(action_orders=action_orders, mask=mask)

    active_pokemon = battle.active_pokemon[slot]
    if active_pokemon is not None and not force_switch[slot]:
        occupied_move_slots: set[int] = set()
        for move in battle.available_moves[slot]:
            hydrated_move, target_locs = _request_aware_move_targets(
                move, active_pokemon, battle, slot
            )
            move_idx = request_move_slot(hydrated_move, active_slot=slot)
            if move_idx in occupied_move_slots:
                raise RuntimeError(
                    f"Two available moves for active slot {slot} share Showdown "
                    f"request index {move_idx + 1}."
                )
            occupied_move_slots.add(move_idx)
            for target_loc in target_locs:
                target_idx = _showdown_target_to_index(target_loc, slot, battle)
                action_idx = move_idx * NUM_TARGETS + target_idx
                _add_move_template(
                    action_orders=action_orders,
                    mask=mask,
                    action_idx=action_idx,
                    move=hydrated_move,
                    target_loc=target_loc,
                    slot=slot,
                )

    if not action_orders:
        return _build_pass_slot_state(mask, explicit=bool(force_switch[slot]))
    return _SlotActionState(action_orders=action_orders, mask=mask)


def _build_pass_slot_state(mask: np.ndarray, *, explicit: bool) -> _SlotActionState:
    action_orders: dict[int, _OrderEntry] = {}
    _add_pass_template(action_orders, mask, explicit=explicit)
    return _SlotActionState(
        action_orders=action_orders,
        mask=mask,
    )


def _add_move_template(
    *,
    action_orders: dict[int, _OrderEntry],
    mask: np.ndarray,
    action_idx: int,
    move: Move,
    target_loc: int,
    slot: int,
) -> None:
    existing = action_orders.get(action_idx)
    if existing is not None:
        if not isinstance(existing, Move):
            raise RuntimeError(
                f"Action template {action_idx} for slot {slot} has the wrong kind."
            )
        _validate_same_action_message(
            action_idx=action_idx,
            slot=slot,
            existing=SingleBattleOrder(existing, move_target=target_loc),
            candidate=SingleBattleOrder(move, move_target=target_loc),
        )
        return

    action_orders[action_idx] = move
    mask[action_idx] = True


def _add_switch_template(
    *,
    action_orders: dict[int, _OrderEntry],
    mask: np.ndarray,
    action_idx: int,
    pokemon,
    slot: int,
) -> None:
    existing = action_orders.get(action_idx)
    if existing is not None:
        if getattr(existing, "species", None) is None:
            raise RuntimeError(
                f"Action template {action_idx} for slot {slot} has the wrong kind."
            )
        _validate_same_action_message(
            action_idx=action_idx,
            slot=slot,
            existing=SingleBattleOrder(existing),
            candidate=SingleBattleOrder(pokemon),
        )
        return

    action_orders[action_idx] = pokemon
    mask[action_idx] = True


def _add_pass_template(
    action_orders: dict[int, _OrderEntry],
    mask: np.ndarray,
    *,
    explicit: bool,
) -> None:
    action_orders[PASS_ACTION] = _EXPLICIT_PASS if explicit else _SKIPPED_PASS
    mask[PASS_ACTION] = True


def _validate_same_action_message(
    *,
    action_idx: int,
    slot: int,
    existing: _BattleOrder,
    candidate: _BattleOrder,
) -> None:
    if existing.message != candidate.message:
        raise RuntimeError(
            f"Two different legal orders mapped to the same action id {action_idx} "
            f"for slot {slot}: {existing.message!r} vs {candidate.message!r}. "
            "Expand the action encoding so these orders are distinguishable."
        )


def _has_explicit_pass(action_orders: Mapping[int, _OrderEntry]) -> bool:
    return action_orders.get(PASS_ACTION) is _EXPLICIT_PASS


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
    """Return a move carrying the current request's slot and target metadata.

    Modern ``poke-env`` attaches both values while constructing
    ``DoubleBattle.available_moves`` and clears them before parsing the next
    request. Reusing that move avoids rescanning ``last_request`` and cloning
    every legal move on every policy decision. A fallback remains for tests or
    third-party battle objects that provide only partially hydrated moves.
    """
    request_index = getattr(move, "request_index", None)
    request_target = getattr(move, "request_target", None)
    if request_index is not None and request_target is not None:
        return move

    # An omitted request target is meaningful for forced/random moves. Confirm
    # that it was actually omitted before accepting a target-less hydrated move.
    move_request = _request_move_entry(move, battle, slot)
    target = None if move_request is None else move_request.get("target")
    if request_index is not None and target is None:
        return move

    hydrated = Move(move.id, gen=move.gen)
    hydrated.request_index = request_index
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


def _showdown_target_to_index(
    showdown_target: int,
    slot: int,
    battle: DoubleBattle,
) -> int:
    ally_position = (
        battle.POKEMON_2_POSITION if slot == 0 else battle.POKEMON_1_POSITION
    )
    self_position = (
        battle.POKEMON_1_POSITION if slot == 0 else battle.POKEMON_2_POSITION
    )

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


def _showdown_target_from_action(
    action_idx: int,
    slot: int,
    battle: DoubleBattle,
) -> int:
    """Recover the exact Showdown target for one selected move template."""
    target_idx = action_idx % NUM_TARGETS
    if target_idx == TARGET_OPPONENT_1:
        return battle.OPPONENT_1_POSITION
    if target_idx == TARGET_OPPONENT_2:
        return battle.OPPONENT_2_POSITION
    if target_idx == TARGET_ALLY:
        return battle.POKEMON_2_POSITION if slot == 0 else battle.POKEMON_1_POSITION
    if target_idx == TARGET_SELF:
        return battle.POKEMON_1_POSITION if slot == 0 else battle.POKEMON_2_POSITION
    if target_idx == TARGET_NONE:
        return battle.EMPTY_TARGET_POSITION
    raise RuntimeError(
        f"Unsupported action target {target_idx} for action {action_idx}."
    )


def _legal_target_locs(move: Move, active_pokemon, battle: DoubleBattle) -> list[int]:
    target_locs = battle.get_possible_showdown_targets(move, active_pokemon)
    if not target_locs:
        raise RuntimeError(
            f"poke-env produced no legal Showdown targets for move {move.id!r}. "
            "This indicates the request state and target mapping are inconsistent."
        )
    return list(target_locs)
