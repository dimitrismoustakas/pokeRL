from types import SimpleNamespace

import numpy as np
import pytest
import torch

from poke_env.battle import Move, Pokemon
from poke_env.player import PassBattleOrder, SingleBattleOrder

from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    INTENT_MOVE,
    INTENT_SWITCH,
    LegalActionState,
    PASS_ACTION,
    SLOT_INTENT_KIND,
    SLOT_INTENT_MOVE,
    SLOT_INTENT_SWITCH_SPECIES,
    SLOT_INTENT_TARGET,
    SemanticActionDecisionMask,
    SemanticActionIntent,
    SemanticActionSupervision,
    SemanticSlotDecisionMask,
    SemanticSlotIntent,
    SWITCH_OFFSET,
    TARGET_OPPONENT_1,
    TARGET_OPPONENT_2,
    TARGET_SELF,
    TERA_INTENT_SLOT,
    TERA_INTENT_USE,
    TERA_NONE,
    TERA_SLOT_1,
    TERA_SLOT_2,
    semantic_action_supervision,
    semantic_action_request_key,
)
from pokerl.env.showdown_env import _encode_opponent_intent_supervision
from pokerl.model.heads import OpponentIntentHead


def _semantic_state() -> LegalActionState:
    protect = Move("protect", gen=9)
    thunderbolt = Move("thunderbolt", gen=9)
    spore = Move("spore", gen=9)
    slot_zero_orders = {
        TARGET_SELF: SingleBattleOrder(protect, move_target=0),
        5 + TARGET_OPPONENT_1: SingleBattleOrder(thunderbolt, move_target=1),
        5 + TARGET_OPPONENT_2: SingleBattleOrder(thunderbolt, move_target=2),
        SWITCH_OFFSET: SingleBattleOrder(Pokemon(gen=9, species="incineroar")),
        SWITCH_OFFSET + 1: SingleBattleOrder(Pokemon(gen=9, species="rillaboom")),
    }
    slot_one_orders = {
        TARGET_OPPONENT_1: SingleBattleOrder(spore, move_target=1),
        TARGET_OPPONENT_2: SingleBattleOrder(spore, move_target=2),
        SWITCH_OFFSET: SingleBattleOrder(Pokemon(gen=9, species="amoonguss")),
        SWITCH_OFFSET + 1: SingleBattleOrder(Pokemon(gen=9, species="rillaboom")),
    }
    slot_masks = []
    for orders in (slot_zero_orders, slot_one_orders):
        mask = np.zeros(ACTIONS_PER_SLOT, dtype=bool)
        mask[list(orders)] = True
        slot_masks.append(mask)

    joint_mask = np.zeros(
        (ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
        dtype=bool,
    )
    joint_mask[FIRST_SLOT_MASK_ROW, list(slot_zero_orders)] = True
    for first_action in slot_zero_orders:
        joint_mask[first_action + 1, list(slot_one_orders)] = True
    return LegalActionState(
        slot_action_orders=(slot_zero_orders, slot_one_orders),
        slot_masks=(slot_masks[0], slot_masks[1]),
        joint_mask=joint_mask,
        can_tera=np.array([True, True], dtype=bool),
    )


def test_semantic_supervision_masks_only_real_component_choices() -> None:
    supervision = semantic_action_supervision(
        np.array(
            [
                5 + TARGET_OPPONENT_2,
                SWITCH_OFFSET + 1,
                TERA_SLOT_1,
            ]
        ),
        _semantic_state(),
    )

    first, second = supervision.intent.slots
    first_mask, second_mask = supervision.decision_mask.slots
    assert first.kind == INTENT_MOVE
    assert first.move_id == "thunderbolt"
    assert first.target == TARGET_OPPONENT_2
    assert first.terastallize
    assert first_mask.kind
    assert first_mask.move
    assert first_mask.target
    assert not first_mask.switch_species

    assert second.kind == INTENT_SWITCH
    assert second.switch_species == "rillaboom"
    assert second_mask.kind
    assert not second_mask.move
    assert not second_mask.target
    assert second_mask.switch_species

    assert supervision.decision_mask.tera_use
    assert not supervision.decision_mask.tera_slot


def test_tera_slot_supervision_requires_use_and_two_eligible_slots() -> None:
    state = _semantic_state()
    actions = np.array([5 + TARGET_OPPONENT_1, TARGET_OPPONENT_2, TERA_NONE])

    waited = semantic_action_supervision(actions, state)
    assert waited.decision_mask.tera_use
    assert not waited.decision_mask.tera_slot

    actions[2] = TERA_SLOT_2
    used = semantic_action_supervision(actions, state)
    assert used.decision_mask.tera_use
    assert used.decision_mask.tera_slot
    assert used.intent.slots[1].terastallize


def test_default_only_semantic_supervision_has_no_training_components() -> None:
    joint_mask = np.zeros(
        (ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
        dtype=bool,
    )
    joint_mask[FIRST_SLOT_MASK_ROW, PASS_ACTION] = True
    joint_mask[PASS_ACTION + 1, PASS_ACTION] = True
    state = LegalActionState(
        slot_action_orders=(
            {PASS_ACTION: PassBattleOrder()},
            {PASS_ACTION: PassBattleOrder()},
        ),
        slot_masks=(
            np.eye(1, ACTIONS_PER_SLOT, PASS_ACTION, dtype=bool).reshape(-1),
            np.eye(1, ACTIONS_PER_SLOT, PASS_ACTION, dtype=bool).reshape(-1),
        ),
        joint_mask=joint_mask,
        can_tera=np.zeros(2, dtype=bool),
    )

    supervision = semantic_action_supervision(
        np.array([PASS_ACTION, PASS_ACTION, TERA_NONE]),
        state,
    )

    assert not any(
        value
        for slot_mask in supervision.decision_mask.slots
        for value in (
            slot_mask.kind,
            slot_mask.move,
            slot_mask.target,
            slot_mask.switch_species,
        )
    )
    assert not supervision.decision_mask.tera_use
    assert not supervision.decision_mask.tera_slot


def test_semantic_action_request_key_uses_normalized_request_generation() -> None:
    left = SimpleNamespace(
        battle_tag="battle-request-key",
        request_generation=8,
        last_request={},
    )
    right = SimpleNamespace(
        battle_tag="battle-request-key",
        request_generation=8,
        last_request={"rqid": 999},
    )

    assert semantic_action_request_key(left) == semantic_action_request_key(right)
    assert semantic_action_request_key(left) == ("battle-request-key", 8)

    right.request_generation = 9
    assert semantic_action_request_key(left) != semantic_action_request_key(right)

    with pytest.raises(RuntimeError, match="requires poke-env request_generation"):
        semantic_action_request_key(SimpleNamespace(battle_tag="old-poke-env"))


def test_semantic_supervision_encodes_global_vocab_labels_and_masks() -> None:
    supervision = semantic_action_supervision(
        np.array(
            [
                5 + TARGET_OPPONENT_2,
                SWITCH_OFFSET + 1,
                TERA_SLOT_1,
            ]
        ),
        _semantic_state(),
    )
    game_data = SimpleNamespace(
        move_to_idx={"thunderbolt": 17},
        species_to_idx={"rillaboom": 23},
    )

    slot_labels, slot_masks, tera_labels, tera_masks = (
        _encode_opponent_intent_supervision(game_data, supervision)
    )

    assert slot_labels[0, SLOT_INTENT_KIND] == INTENT_MOVE
    assert slot_labels[0, SLOT_INTENT_MOVE] == 17
    assert slot_labels[0, SLOT_INTENT_TARGET] == TARGET_OPPONENT_2
    assert slot_labels[1, SLOT_INTENT_KIND] == INTENT_SWITCH
    assert slot_labels[1, SLOT_INTENT_SWITCH_SPECIES] == 23
    assert slot_masks.tolist() == [
        [True, True, True, False],
        [True, False, False, True],
    ]
    assert tera_labels[TERA_INTENT_USE] == 1
    assert tera_labels[TERA_INTENT_SLOT] == 0
    assert tera_masks.tolist() == [True, False]


def test_semantic_supervision_normalizes_global_vocab_ids() -> None:
    supervision = SemanticActionSupervision(
        intent=SemanticActionIntent(
            slots=(
                SemanticSlotIntent(
                    kind=INTENT_MOVE,
                    move_id="Thunder-Bolt",
                    target=TARGET_OPPONENT_1,
                ),
                SemanticSlotIntent(
                    kind=INTENT_SWITCH,
                    switch_species="Urshifu-Rapid-Strike",
                ),
            )
        ),
        decision_mask=SemanticActionDecisionMask(
            slots=(
                SemanticSlotDecisionMask(
                    kind=True,
                    move=True,
                    target=True,
                    switch_species=False,
                ),
                SemanticSlotDecisionMask(
                    kind=True,
                    move=False,
                    target=False,
                    switch_species=True,
                ),
            ),
            tera_use=False,
            tera_slot=False,
        ),
    )
    game_data = SimpleNamespace(
        move_to_idx={"thunderbolt": 29},
        species_to_idx={"urshifurapidstrike": 31},
    )

    slot_labels, slot_masks, _, _ = _encode_opponent_intent_supervision(
        game_data,
        supervision,
    )

    assert slot_labels[0, SLOT_INTENT_MOVE] == 29
    assert slot_labels[1, SLOT_INTENT_SWITCH_SPECIES] == 31
    assert slot_masks[0, SLOT_INTENT_MOVE]
    assert slot_masks[1, SLOT_INTENT_SWITCH_SPECIES]


def test_opponent_intent_head_outputs_semantic_vocabularies() -> None:
    head = OpponentIntentHead(
        context_dim=6,
        pokemon_dim=4,
        num_moves=11,
        num_species=13,
        hidden_dim=8,
    )
    context = torch.randn(3, 6)
    opponent_team = torch.randn(3, 6, 4)
    opponent_active = torch.randn(3, 2, 4)
    own_active = torch.randn(3, 2, 4)
    opponent_mask = torch.tensor(
        [
            [False, False, False, False, True, True],
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
        ]
    )

    out = head(
        context,
        opponent_team,
        opponent_active,
        own_active,
        opponent_mask,
    )

    assert out["opp_intent_kind_logits"].shape == (3, 2, 4)
    assert out["opp_intent_move_logits"].shape == (3, 2, 11)
    assert out["opp_intent_target_logits"].shape == (3, 2, 5)
    assert out["opp_intent_switch_species_logits"].shape == (3, 2, 13)
    assert out["opp_intent_tera_use_logits"].shape == (3, 2)
    assert out["opp_intent_tera_slot_logits"].shape == (3, 2)
