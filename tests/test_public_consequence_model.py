import math

import pytest
import torch

from pokerl.config import TrainingConfig
from pokerl.data import MOVE_MECHANICS_DIM
from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    NUM_MOVES,
    NUM_TARGETS,
    SWITCH_OFFSET,
    TERA_NONE,
)
from pokerl.env.observation import BOOSTS_OFFSET
from pokerl.field_transition import (
    FIELD_CHANGE_COMPONENT_DIM,
    SCALAR_FIELD_INDICES,
    TERRAIN_CLASS_SLICE,
    WEATHER_CLASS_SLICE,
)
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
        "public_consequence_field_change_logits": torch.randn(
            *base_shape, FIELD_CHANGE_COMPONENT_DIM
        ),
        "public_consequence_field_alternative_logits": torch.randn(
            *base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM
        ),
    }
    persistence_fields = torch.zeros(*base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM)
    persistence_fields[..., WEATHER_CLASS_SLICE.start] = 1.0
    persistence_fields[..., TERRAIN_CLASS_SLICE.start] = 1.0
    field_targets = persistence_fields.clone()
    field_targets[..., WEATHER_CLASS_SLICE] = 0.0
    field_targets[..., WEATHER_CLASS_SLICE.start + 1] = 1.0
    field_targets[..., TERRAIN_CLASS_SLICE] = 0.0
    field_targets[..., TERRAIN_CLASS_SLICE.start + 2] = 1.0
    field_targets[..., SCALAR_FIELD_INDICES] = torch.randint(
        0,
        2,
        (*base_shape, len(SCALAR_FIELD_INDICES)),
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
        "persistence_fields": persistence_fields,
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
    assert head.public_consequence_field_change_head.weight.grad is None
    assert head.public_consequence_field_alternative_head.weight.grad is None
    assert head.public_consequence_target_head.weight.grad is None
    assert head.public_consequence_policy_adapter.weight.grad is not None
    assert head.public_consequence_policy_adapter.weight.grad.norm() > 0


def test_untrained_field_transition_starts_at_advanced_persistence() -> None:
    head, inputs = _head_and_inputs(batch_size=2)

    output = head(**inputs)

    torch.testing.assert_close(
        output["public_consequence_field_probs"],
        output["public_consequence_field_persistence"],
        atol=7e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output["public_consequence_field_probs"][..., WEATHER_CLASS_SLICE].sum(dim=-1),
        torch.ones(2, 2),
    )
    torch.testing.assert_close(
        output["public_consequence_field_probs"][..., TERRAIN_CLASS_SLICE].sum(dim=-1),
        torch.ones(2, 2),
    )


def test_missing_move_semantics_keeps_projection_in_gradient_graph() -> None:
    head, inputs = _head_and_inputs()

    (-head(**inputs)["log_prob"].sum()).backward()

    projection_weight_grad = head.move_semantic_proj[0].weight.grad
    assert projection_weight_grad is not None
    assert projection_weight_grad.count_nonzero() == 0


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
    assert output["public_consequence_field_change_logits"].shape == (
        2,
        2,
        FIELD_CHANGE_COMPONENT_DIM,
    )
    assert output["public_consequence_field_alternative_logits"].shape == (2, 2, 33)
    assert output["public_consequence_field_probs"].shape == (2, 2, 33)
    assert output["public_consequence_field_persistence"].shape == (2, 2, 33)


def test_target_decoder_sees_accuracy_evasion_and_ignore_evasion_exception() -> None:
    head = ActionHead(
        context_dim=3,
        pokemon_dim=2,
        hidden_dim=8,
        move_base_power_lookup=torch.zeros(3),
        move_accuracy_lookup=torch.tensor([0.0, 1.0, 1.0]),
        move_bypasses_accuracy_lookup=torch.zeros(3),
        move_ignores_evasion_lookup=torch.tensor([0.0, 0.0, 1.0]),
        move_target_scope_lookup=torch.tensor([0, 2, 2]),
        move_multihit_lookup=torch.zeros(3),
        move_priority_lookup=torch.zeros(3),
        move_target_flags=torch.zeros((3, 6)),
        type_effectiveness=torch.ones((1, 1)),
    )
    own_continuous = torch.zeros((1, 6, 44))
    own_continuous[0, 0, BOOSTS_OFFSET + 5] = -2.0 / 6.0
    opp_continuous = torch.zeros((1, 6, 44))
    opp_continuous[0, 0, BOOSTS_OFFSET + 6] = 3.0 / 6.0
    battle_obs = {
        "own_moves": torch.tensor([[[1, 2, 0, 0]] + [[0, 0, 0, 0]] * 5]),
        "own_request_moves": torch.tensor([[[1, 2, 0, 0], [0, 0, 0, 0]]]),
        "own_request_move_metadata": torch.zeros((1, 2, 4, 2), dtype=torch.long),
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


def test_live_request_move_identity_drives_candidate_and_dependent_decisions() -> None:
    hidden_dim = 8
    num_moves = 3
    head = ActionHead(
        context_dim=3,
        pokemon_dim=2,
        hidden_dim=hidden_dim,
        move_base_power_lookup=torch.zeros(num_moves),
        move_accuracy_lookup=torch.ones(num_moves),
        move_bypasses_accuracy_lookup=torch.zeros(num_moves),
        move_ignores_evasion_lookup=torch.zeros(num_moves),
        move_target_scope_lookup=torch.full((num_moves,), 2, dtype=torch.long),
        move_multihit_lookup=torch.zeros(num_moves),
        move_priority_lookup=torch.zeros(num_moves),
        move_target_flags=torch.zeros((num_moves, 6)),
        move_mechanics_lookup=torch.zeros((num_moves, MOVE_MECHANICS_DIM)),
        move_embedding_dim=2,
        type_effectiveness=torch.ones((1, 1)),
    )
    move_embedding = torch.nn.Embedding(num_moves, 2, padding_idx=0)
    with torch.no_grad():
        move_embedding.weight.zero_()
        move_embedding.weight[1, 0] = 1.0
        move_embedding.weight[2, 0] = 2.0
        head.move_semantic_proj[0].weight.zero_()
        head.move_semantic_proj[0].bias.zero_()
        head.move_semantic_proj[0].weight[0, 0] = 1.0
        head.score[0].weight.zero_()
        head.score[0].bias.zero_()
        head.score[0].weight[0, 4 * hidden_dim] = 1.0
        head.score[2].weight.zero_()
        head.score[2].bias.zero_()
        head.score[2].weight[0, 0] = 1.0

    def battle_obs_for(move_id: int) -> dict[str, torch.Tensor]:
        own_moves = torch.zeros((1, 6, 4), dtype=torch.long)
        own_moves[0, 0, 0] = 1
        own_request_moves = torch.zeros((1, 2, 4), dtype=torch.long)
        own_request_moves[0, 0, 0] = move_id
        return {
            "own_moves": own_moves,
            "own_request_moves": own_request_moves,
            "own_request_move_metadata": torch.zeros((1, 2, 4, 2), dtype=torch.long),
            "own_types": torch.zeros((1, 6, 3), dtype=torch.long),
            "own_continuous": torch.zeros((1, 6, 44)),
            "own_active_indices": torch.tensor([[0, 1]]),
            "own_active_present": torch.tensor([[True, True]]),
            "opp_types": torch.zeros((1, 6, 3), dtype=torch.long),
            "opp_continuous": torch.zeros((1, 6, 44)),
            "opp_active_indices": torch.tensor([[0, 1]]),
            "opp_active_present": torch.tensor([[True, True]]),
        }

    def semantics_for(move_id: int) -> torch.Tensor:
        battle_obs = battle_obs_for(move_id)
        context = head._matchup_context(battle_obs, 1, torch.device("cpu"))
        return head._offered_move_semantics(
            context,
            actor_slot=0,
            move_embedding=move_embedding,
            batch_size=1,
            device=torch.device("cpu"),
        )

    common = {
        "query": torch.zeros((1, hidden_dim)),
        "actor_rep": torch.zeros((1, 2)),
        "partner_rep": torch.zeros((1, 2)),
        "own_team_reps": torch.zeros((1, 6, 2)),
        "opp_active_reps": torch.zeros((1, 2, 2)),
        "matchup_features": torch.zeros(
            (1, ACTIONS_PER_SLOT, head._MATCHUP_FEATURE_DIM)
        ),
        "action_features": torch.zeros((ACTIONS_PER_SLOT, hidden_dim)),
        "public_target_reps": torch.zeros((1, 4, 2)),
        "target_mechanics": torch.zeros((1, SWITCH_OFFSET, 4, 6)),
    }
    with torch.no_grad():
        first_logits, _ = head._score_actions(
            **common,
            move_semantics=semantics_for(1),
        )
        second_logits, _ = head._score_actions(
            **common,
            move_semantics=semantics_for(2),
        )

    assert second_logits[0, 0] > first_logits[0, 0]
    torch.testing.assert_close(
        second_logits[:, SWITCH_OFFSET:],
        first_logits[:, SWITCH_OFFSET:],
    )

    query2_inputs: list[torch.Tensor] = []
    tera_inputs: list[torch.Tensor] = []
    query2_handle = head.query2[0].register_forward_pre_hook(
        lambda _module, args: query2_inputs.append(args[0].detach().clone())
    )
    tera_handle = head.tera_query[0].register_forward_pre_hook(
        lambda _module, args: tera_inputs.append(args[0].detach().clone())
    )
    forward_inputs = {
        "context": torch.zeros((1, 3)),
        "active_reps": torch.zeros((1, 2, 2)),
        "own_team_reps": torch.zeros((1, 6, 2)),
        "opp_team_reps": torch.zeros((1, 6, 2)),
        "opp_reserve_mask": torch.zeros((1, 6), dtype=torch.bool),
        "opp_active_reps": torch.zeros((1, 2, 2)),
        "masks": {
            "joint": torch.ones(
                (1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=torch.bool
            ),
            "can_tera": torch.zeros((1, 2), dtype=torch.bool),
        },
        "actions": torch.tensor([[0, 0, TERA_NONE]]),
        "battle_obs": battle_obs_for(2),
        "move_embedding": move_embedding,
    }
    try:
        with torch.no_grad():
            head(**forward_inputs)
    finally:
        query2_handle.remove()
        tera_handle.remove()

    selected_conditioning = (
        head.action_id_emb(torch.tensor([0])) + semantics_for(2)[:, 0]
    )
    torch.testing.assert_close(query2_inputs[0][:, -hidden_dim:], selected_conditioning)
    tera_action_offset = 3 + 4 * 2
    torch.testing.assert_close(
        tera_inputs[0][:, tera_action_offset : tera_action_offset + hidden_dim],
        selected_conditioning,
    )


def test_candidate_scores_match_dense_layer_for_moves_switches_and_pass() -> None:
    head, board = _head_and_inputs(batch_size=2)
    actor, partner = board["active_reps"].unbind(1)
    opponents = board["opp_active_reps"]
    team = board["own_team_reps"]
    query = torch.randn(2, 12)
    action_features = head._static_action_features()
    matchups = torch.randn(2, ACTIONS_PER_SLOT, head._MATCHUP_FEATURE_DIM)
    moves = torch.randn(2, NUM_MOVES + 1, 12)
    moves[:, -1] = 0
    _, state = head._score_actions(
        query,
        actor,
        partner,
        team,
        opponents,
        matchups,
        moves,
        action_features,
        torch.cat((board["active_reps"], opponents), dim=1),
        torch.zeros(2, SWITCH_OFFSET, 4, 6),
    )

    blank = torch.zeros_like(actor).unsqueeze(1)
    targets = torch.cat((opponents, partner[:, None], actor[:, None], blank), dim=1)
    entities = torch.cat((targets.repeat(1, NUM_MOVES, 1), team, blank), dim=1)
    move_rows = torch.cat(
        (
            moves[:, :NUM_MOVES].repeat_interleave(NUM_TARGETS, dim=1),
            moves[:, -1:].expand(-1, ACTIONS_PER_SLOT - SWITCH_OFFSET, -1),
        ),
        dim=1,
    )
    dense_input = torch.cat(
        (
            query[:, None].expand_as(move_rows),
            action_features.expand_as(move_rows),
            head.entity_proj(entities),
            head.matchup_proj(matchups),
            move_rows,
        ),
        dim=-1,
    )
    expected = head.score[1](head.score[0](dense_input))
    torch.testing.assert_close(state.score_hidden, expected)
    weights = torch.randn_like(expected)
    params = (
        head.score[0].weight,
        head.entity_proj[0].weight,
        head.matchup_proj[0].weight,
    )
    actual_grad = torch.autograd.grad(state.score_hidden, params, weights)
    dense_grad = torch.autograd.grad(expected, params, weights)
    for actual, wanted in zip(actual_grad, dense_grad, strict=True):
        torch.testing.assert_close(actual, wanted)


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
    assert head.public_consequence_field_change_head.weight.grad is not None
    assert head.public_consequence_field_alternative_head.weight.grad is not None
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
    assert metrics["aux_public_consequence_field_change_count"] == (
        12.0 * FIELD_CHANGE_COMPONENT_DIM
    )
    assert 0.0 <= finalized["aux_public_consequence_emitted_brier"] <= 1.0
    assert 0.0 <= finalized["aux_public_consequence_move_branch_top1_accuracy"] <= 1.0
    assert finalized["aux_public_consequence_interaction_reason_failed_count"] >= 0
    assert math.isfinite(finalized["aux_public_consequence_field_typed_nll_skill"])
    assert (
        finalized["aux_public_consequence_field_weather_probability_sum_error"] >= 0.0
    )


def test_public_consequence_coefficient_is_conservative_and_validated() -> None:
    assert TrainingConfig().aux_public_consequence_coef == pytest.approx(0.1)
    for invalid in (0.0, -0.01):
        with pytest.raises(ValueError, match="aux_public_consequence_coef"):
            TrainingConfig(aux_public_consequence_coef=invalid)
