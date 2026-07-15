import math

import pytest
import torch

from pokerl.config import TrainingConfig
from pokerl.env.actions import ACTIONS_PER_SLOT, SWITCH_OFFSET, TERA_NONE
from pokerl.env.observation import BOOSTS_OFFSET
from pokerl.model.heads import ActionHead, _PublicConsequenceState
from pokerl.public_consequence import (
    MAX_PUBLIC_HIT_COUNT,
    NUM_MOVE_BRANCHES,
    NUM_NONMOVE_REASONS,
    NUM_TARGET_INTERACTION_REASONS,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
)
from pokerl.training.aux_metrics import public_consequence_additive_metrics
from pokerl.training.ppo import PPOTrainer


def _head_and_inputs(batch_size: int = 3) -> tuple[ActionHead, dict[str, object]]:
    torch.manual_seed(7)
    head = ActionHead(context_dim=5, pokemon_dim=4, hidden_dim=12)
    actions = torch.stack(
        [
            torch.arange(batch_size) % 4,
            (torch.arange(batch_size) + 2) % 4,
            torch.full((batch_size,), TERA_NONE),
        ],
        dim=1,
    )
    return head, {
        "context": torch.randn(batch_size, 5),
        "active_reps": torch.randn(batch_size, 2, 4),
        "own_team_reps": torch.randn(batch_size, 6, 4),
        "opp_team_reps": torch.randn(batch_size, 6, 4),
        "opp_reserve_mask": torch.zeros(batch_size, 6, dtype=torch.bool),
        "opp_active_reps": torch.randn(batch_size, 2, 4),
        "masks": {
            "joint": torch.ones(
                (batch_size, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
                dtype=torch.bool,
            ),
            "can_tera": torch.zeros(batch_size, 2, dtype=torch.bool),
        },
        "actions": actions,
    }


def _consequence_tensors(
    base_shape: tuple[int, ...] = (2, 3, 2),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    target_shape = (*base_shape, 4)
    logits = {
        "public_consequence_move_emitted_logits": torch.randn(base_shape),
        "public_consequence_nonmove_reason_logits": torch.randn(
            *base_shape, NUM_NONMOVE_REASONS
        ),
        "public_consequence_move_branch_logits": torch.randn(
            *base_shape, NUM_MOVE_BRANCHES
        ),
        "public_consequence_route_logits": torch.randn(target_shape),
        "public_consequence_interaction_logits": torch.randn(target_shape),
        "public_consequence_interaction_reason_logits": torch.randn(
            *target_shape, NUM_TARGET_INTERACTION_REASONS
        ),
        "public_consequence_hit_count_logits": torch.randn(
            *target_shape, MAX_PUBLIC_HIT_COUNT
        ),
        "public_consequence_cant_logits": torch.randn(*base_shape, 2),
        "public_consequence_field_logits": torch.randn(
            *base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM
        ),
    }
    field_targets = torch.randint(
        0, 2, (*base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM)
    ).float()
    labels = {
        "move_emitted_targets": torch.randint(0, 2, base_shape).float(),
        "move_emitted_masks": torch.ones(base_shape, dtype=torch.bool),
        "nonmove_reason_labels": torch.randint(0, NUM_NONMOVE_REASONS, base_shape),
        "nonmove_reason_masks": torch.ones(base_shape, dtype=torch.bool),
        "move_branch_labels": torch.randint(0, NUM_MOVE_BRANCHES, base_shape),
        "move_branch_masks": torch.ones(base_shape, dtype=torch.bool),
        "route_targets": torch.randint(0, 2, target_shape).float(),
        "route_masks": torch.ones(target_shape, dtype=torch.bool),
        "interaction_targets": torch.randint(0, 2, target_shape).float(),
        "interaction_masks": torch.ones(target_shape, dtype=torch.bool),
        "interaction_reason_labels": torch.randint(
            0, NUM_TARGET_INTERACTION_REASONS, target_shape
        ),
        "interaction_reason_masks": torch.ones(target_shape, dtype=torch.bool),
        "hit_count_labels": torch.randint(0, MAX_PUBLIC_HIT_COUNT, target_shape),
        "hit_count_masks": torch.ones(target_shape, dtype=torch.bool),
        "cant_targets": torch.randint(0, 2, (*base_shape, 2)).float(),
        "cant_masks": torch.ones((*base_shape, 2), dtype=torch.bool),
        "field_targets": field_targets,
        "field_masks": torch.ones(base_shape, dtype=torch.bool),
        "persistence_fields": 1.0 - field_targets,
    }
    return logits, labels


def test_zero_initialized_consequence_adapter_preserves_policy_exactly() -> None:
    head, inputs = _head_and_inputs()
    head.eval()
    with torch.no_grad():
        before = head(**inputs)
        head.public_consequence_action_head.weight.normal_(7.0, 3.0)
        head.public_consequence_target_head.weight.normal_(5.0, 2.0)
        after = head(**inputs)

    torch.testing.assert_close(before["log_prob"], after["log_prob"], rtol=0, atol=0)
    assert not torch.equal(
        before["public_consequence_move_emitted_logits"],
        after["public_consequence_move_emitted_logits"],
    )


def test_actor_gradient_cannot_train_detached_world_predictions() -> None:
    head, inputs = _head_and_inputs()
    with torch.no_grad():
        head.public_consequence_policy_adapter.weight.normal_()

    (-head(**inputs)["log_prob"].sum()).backward()

    assert head.public_consequence_action_head.weight.grad is None
    assert head.public_consequence_target_head.weight.grad is None
    assert head.public_consequence_policy_adapter.weight.grad is not None
    assert head.public_consequence_policy_adapter.weight.grad.norm() > 0


def test_selected_public_consequence_output_shapes() -> None:
    head, inputs = _head_and_inputs(batch_size=2)

    output = head(**inputs)

    assert output["public_consequence_move_emitted_logits"].shape == (2, 2)
    assert output["public_consequence_nonmove_reason_logits"].shape == (2, 2, 2)
    assert output["public_consequence_move_branch_logits"].shape == (2, 2, 4)
    assert output["public_consequence_route_logits"].shape == (2, 2, 4)
    assert output["public_consequence_interaction_logits"].shape == (2, 2, 4)
    assert output["public_consequence_interaction_reason_logits"].shape == (
        2,
        2,
        4,
        5,
    )
    assert output["public_consequence_hit_count_logits"].shape == (2, 2, 4, 10)
    assert output["public_consequence_field_logits"].shape == (2, 2, 33)


def test_target_decoder_sees_accuracy_evasion_and_ignore_evasion_exception() -> None:
    head = ActionHead(
        context_dim=3,
        pokemon_dim=2,
        hidden_dim=8,
        move_type_lookup=torch.zeros(3, dtype=torch.long),
        move_base_power_lookup=torch.zeros(3),
        move_accuracy_lookup=torch.tensor([0.0, 1.0, 1.0]),
        move_bypasses_accuracy_lookup=torch.zeros(3),
        move_ignores_evasion_lookup=torch.tensor([0.0, 0.0, 1.0]),
        move_target_scope_lookup=torch.tensor([0, 2, 2]),
        move_multihit_lookup=torch.zeros(3),
        move_priority_lookup=torch.zeros(3),
        move_category_lookup=torch.zeros(3, dtype=torch.long),
        move_target_flags=torch.zeros((3, 6)),
        type_effectiveness=torch.ones((1, 1)),
    )
    own_continuous = torch.zeros((1, 6, 44))
    own_continuous[0, 0, BOOSTS_OFFSET + 5] = -2.0 / 6.0
    opp_continuous = torch.zeros((1, 6, 44))
    opp_continuous[0, 0, BOOSTS_OFFSET + 6] = 3.0 / 6.0
    battle_obs = {
        "own_moves": torch.tensor([[[1, 2, 0, 0]] + [[0, 0, 0, 0]] * 5]),
        "own_types": torch.zeros((1, 6, 3), dtype=torch.long),
        "own_continuous": own_continuous,
        "own_active_indices": torch.tensor([[0, 1]]),
        "own_active_present": torch.tensor([[True, True]]),
        "opp_types": torch.zeros((1, 6, 3), dtype=torch.long),
        "opp_continuous": opp_continuous,
        "opp_active_indices": torch.tensor([[0, 1]]),
        "opp_active_present": torch.tensor([[True, True]]),
    }
    context = head._matchup_context(battle_obs, 1, torch.device("cpu"))
    features = head._candidate_action_features(
        matchup_context=context,
        actor_slot=0,
        can_tera_slot=torch.tensor([False]),
        batch_size=1,
        device=torch.device("cpu"),
    )

    mechanics = head._candidate_public_target_mechanics(
        context,
        actor_slot=0,
        matchup_features=features,
    )

    assert mechanics.shape == (1, SWITCH_OFFSET, 4, 6)
    # Action 0 is Tackle-like; action 5 is Sacred Sword-like. Target position
    # 2 is the first opponent in public own-own-opponent-opponent ordering.
    assert mechanics[0, 0, 2, 2] == 0.0
    assert mechanics[0, 5, 2, 2] == 1.0
    assert mechanics[0, 0, 2, 3] == pytest.approx(-2.0 / 6.0)
    assert mechanics[0, 0, 2, 4] == pytest.approx(3.0 / 6.0)


def test_amortized_target_decoder_matches_explicit_broadcast_reference() -> None:
    head, _ = _head_and_inputs(batch_size=2)
    move_hidden = torch.randn(2, SWITCH_OFFSET, 12)
    target_reps = torch.randn(2, 4, 4)
    mechanics = torch.randn(2, SWITCH_OFFSET, 4, 6)

    actual = head._public_consequence_target_logits(
        move_hidden,
        target_reps,
        mechanics,
    )
    candidate = head.public_consequence_target_candidate_proj(move_hidden)
    target = head.public_consequence_target_entity_proj(target_reps)
    position = head.public_consequence_target_position_emb.weight
    mechanic = head.public_consequence_target_mechanics_proj(mechanics)
    explicit_hidden = torch.relu(
        candidate[:, :, None, :]
        + target[:, None, :, :]
        + position[None, None, :, :]
        + mechanic
    )
    explicit = head.public_consequence_target_head(explicit_hidden)
    route, interaction, reason, hit_count = explicit.split(
        (1, 1, NUM_TARGET_INTERACTION_REASONS, MAX_PUBLIC_HIT_COUNT), dim=-1
    )

    torch.testing.assert_close(actual["route_logits"], route.squeeze(-1))
    torch.testing.assert_close(actual["interaction_logits"], interaction.squeeze(-1))
    torch.testing.assert_close(actual["interaction_reason_logits"], reason)
    torch.testing.assert_close(actual["hit_count_logits"], hit_count)


def test_all_candidate_target_logits_are_zero_for_switch_and_pass() -> None:
    head, _ = _head_and_inputs(batch_size=2)
    logits = head._all_public_consequence_logits(
        torch.randn(2, ACTIONS_PER_SLOT, 12),
        torch.randn(2, 4, 4),
        torch.randn(2, SWITCH_OFFSET, 4, 6),
    )

    for name in (
        "route_logits",
        "interaction_logits",
        "interaction_reason_logits",
        "hit_count_logits",
    ):
        assert logits[name].shape[1] == ACTIONS_PER_SLOT
        assert torch.count_nonzero(logits[name][:, SWITCH_OFFSET:]) == 0


def test_selected_recomputation_matches_detached_all_candidate_logits() -> None:
    head, _ = _head_and_inputs(batch_size=3)
    hidden = torch.randn(3, ACTIONS_PER_SLOT, 12, requires_grad=True)
    targets = torch.randn(3, 4, 4, requires_grad=True)
    mechanics = torch.randn(3, SWITCH_OFFSET, 4, 6)
    selected = torch.tensor([0, 7, SWITCH_OFFSET - 1])
    all_logits = head._all_public_consequence_logits(hidden, targets, mechanics)
    selected_logits = head._selected_public_consequence_logits(
        _PublicConsequenceState(hidden, targets, mechanics, None),
        selected,
    )

    batch_indices = torch.arange(selected.shape[0])
    for name, actual in selected_logits.items():
        torch.testing.assert_close(actual, all_logits[name][batch_indices, selected])

    sum(value.sum() for value in selected_logits.values()).backward()
    assert head.public_consequence_target_candidate_proj.weight.grad is not None
    assert head.public_consequence_target_entity_proj.weight.grad is not None


def test_conditional_auxiliary_loss_trains_action_and_target_heads() -> None:
    head, inputs = _head_and_inputs()
    output = head(**inputs)
    _, labels = _consequence_tensors(base_shape=(3, 2))
    logits = {key: output[key] for key in _consequence_tensors((3, 2))[0]}

    loss = PPOTrainer._public_consequence_loss(logits=logits, **labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert head.public_consequence_action_head.weight.grad is not None
    assert head.public_consequence_target_head.weight.grad is not None
    assert head.public_consequence_target_candidate_proj.weight.grad is not None
    assert head.public_consequence_target_entity_proj.weight.grad is not None
    assert head.public_consequence_target_mechanics_proj[0].weight.grad is not None
    assert head.score[0].weight.grad is not None


def test_conditional_losses_are_normalized_before_combining() -> None:
    logits, labels = _consequence_tensors(base_shape=(1, 2))
    for value in logits.values():
        value.zero_()
    for name, value in labels.items():
        if name.endswith("_masks"):
            value.zero_()
    labels["move_emitted_masks"][0, 0] = True
    labels["move_emitted_targets"][0, 0] = 1.0
    labels["move_branch_masks"][0, 0] = True
    labels["move_branch_labels"][0, 0] = 3

    loss = PPOTrainer._public_consequence_loss(logits=logits, **labels)

    assert loss == pytest.approx((math.log(2.0) + math.log(4.0)) / 2.0)


def test_auxiliary_loss_with_no_labels_is_finite_zero() -> None:
    logits, labels = _consequence_tensors(base_shape=(1, 2))
    for value in logits.values():
        value.requires_grad_()
    for name, value in labels.items():
        if name.endswith("_masks"):
            value.zero_()

    loss = PPOTrainer._public_consequence_loss(logits=logits, **labels)
    loss.backward()

    assert loss.item() == 0.0
    assert logits["public_consequence_move_emitted_logits"].grad is not None
    assert logits["public_consequence_move_emitted_logits"].grad.count_nonzero() == 0


def test_additive_metrics_and_finalizer_report_conditional_support() -> None:
    logits, labels = _consequence_tensors()

    metrics = public_consequence_additive_metrics(logits=logits, **labels)
    finalized = PPOTrainer._finalize_aux_diagnostics(metrics)

    assert metrics["aux_public_consequence_emitted_count"] == 12.0
    assert metrics["aux_public_consequence_route_count"] == 48.0
    assert metrics["aux_public_consequence_interaction_reason_count"] == 48.0
    assert metrics["aux_public_consequence_field_count"] == 12.0
    assert 0.0 <= finalized["aux_public_consequence_emitted_brier"] <= 1.0
    assert 0.0 <= finalized["aux_public_consequence_move_branch_top1_accuracy"] <= 1.0
    assert finalized["aux_public_consequence_interaction_reason_failed_count"] >= 0


def test_public_consequence_coefficient_is_conservative_and_validated() -> None:
    assert TrainingConfig().aux_public_consequence_coef == pytest.approx(0.1)
    for invalid in (0.0, -0.01):
        with pytest.raises(ValueError, match="aux_public_consequence_coef"):
            TrainingConfig(aux_public_consequence_coef=invalid)
