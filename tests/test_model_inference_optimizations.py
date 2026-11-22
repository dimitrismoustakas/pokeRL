import copy

import torch

from pokerl.env.actions import ACTIONS_PER_SLOT
from pokerl.env.observation import CONTINUOUS_DIM
from pokerl.model.heads import ActionHead


def _make_head_and_inputs(batch_size: int = 3):
    torch.manual_seed(7)
    context_dim = 5
    pokemon_dim = 4
    hidden_dim = 8
    num_moves = 17
    num_types = 6
    head = ActionHead(
        context_dim=context_dim,
        pokemon_dim=pokemon_dim,
        hidden_dim=hidden_dim,
        move_base_power_lookup=torch.rand(num_moves),
        move_accuracy_lookup=torch.rand(num_moves),
        move_bypasses_accuracy_lookup=torch.randint(0, 2, (num_moves,)).float(),
        move_ignores_evasion_lookup=torch.randint(0, 2, (num_moves,)).float(),
        move_target_scope_lookup=torch.randint(0, 3, (num_moves,)),
        move_multihit_lookup=torch.randint(0, 2, (num_moves,)).float(),
        move_priority_lookup=torch.rand(num_moves),
        move_target_flags=torch.randint(0, 2, (num_moves, 6)).float(),
        type_effectiveness=torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.5, 1.0, 2.0, 4.0],
                [1.0, 2.0, 1.0, 0.5, 0.0, 1.0],
                [1.0, 0.5, 2.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 2.0, 0.5, 1.0],
                [1.0, 4.0, 1.0, 0.0, 2.0, 0.5],
            ]
        ),
    )

    active_indices = torch.tensor([[0, 1]]).expand(batch_size, -1).clone()
    present = torch.ones((batch_size, 2), dtype=torch.bool)
    battle_obs = {
        "own_moves": torch.randint(0, num_moves, (batch_size, 6, 4)),
        "own_request_moves": torch.randint(0, num_moves, (batch_size, 2, 4)),
        "own_request_move_metadata": torch.stack(
            (
                torch.randint(0, num_types, (batch_size, 2, 4)),
                torch.randint(0, 3, (batch_size, 2, 4)),
            ),
            dim=-1,
        ),
        "own_types": torch.randint(0, num_types, (batch_size, 6, 3)),
        "own_continuous": torch.rand(batch_size, 6, CONTINUOUS_DIM),
        "own_active_indices": active_indices,
        "own_active_present": present,
        "opp_types": torch.randint(0, num_types, (batch_size, 6, 3)),
        "opp_continuous": torch.rand(batch_size, 6, CONTINUOUS_DIM),
        "opp_active_indices": active_indices.clone(),
        "opp_active_present": present.clone(),
    }
    joint_mask = torch.ones(
        (batch_size, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
        dtype=torch.bool,
    )
    masks = {
        "joint": joint_mask,
        "can_tera": torch.tensor([[True, False]]).expand(batch_size, -1).clone(),
    }
    actions = torch.stack(
        [
            torch.arange(batch_size) % 4,
            (torch.arange(batch_size) + 1) % 4,
            torch.arange(batch_size) % 2,
        ],
        dim=1,
    )
    inputs = {
        "context": torch.randn(batch_size, context_dim),
        "active_reps": torch.randn(batch_size, 2, pokemon_dim),
        "own_team_reps": torch.randn(batch_size, 6, pokemon_dim),
        "opp_team_reps": torch.randn(batch_size, 6, pokemon_dim),
        "opp_reserve_mask": torch.tensor(
            [[True, True, False, False, False, False]],
            dtype=torch.bool,
        ).expand(batch_size, -1),
        "opp_active_reps": torch.randn(batch_size, 2, pokemon_dim),
        "masks": masks,
        "actions": actions,
        "battle_obs": battle_obs,
    }
    return head, inputs


def _install_scalar_effectiveness_reference(head: ActionHead) -> None:
    original = head._type_effectiveness

    def scalar_effectiveness(
        move_types: torch.Tensor,
        target_type_1: torch.Tensor,
        target_type_2: torch.Tensor,
    ) -> torch.Tensor:
        if move_types.ndim == 3 and target_type_1.shape[-1] == 5:
            scalar_move_types = move_types.squeeze(-1)
            return torch.stack(
                [
                    original(
                        scalar_move_types,
                        target_type_1[..., target_index],
                        target_type_2[..., target_index],
                    )
                    for target_index in range(5)
                ],
                dim=-1,
            )
        return original(move_types, target_type_1, target_type_2)

    head._type_effectiveness = scalar_effectiveness  # type: ignore[method-assign]


def _capture_head_logits(head: ActionHead):
    captured: list[torch.Tensor] = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())

    handles = [
        head.score[2].register_forward_hook(capture),
        head.tera_score.register_forward_hook(capture),
    ]
    return captured, handles


def test_action_head_inference_optimizations_match_scalar_reference() -> None:
    optimized, inputs = _make_head_and_inputs()
    reference = copy.deepcopy(optimized)
    _install_scalar_effectiveness_reference(reference)
    optimized_logits, optimized_handles = _capture_head_logits(optimized)
    reference_logits, reference_handles = _capture_head_logits(reference)

    try:
        with torch.enable_grad():
            reference_out = reference(**inputs)
        with torch.no_grad():
            optimized_out = optimized(**inputs)
    finally:
        for handle in optimized_handles + reference_handles:
            handle.remove()

    assert len(optimized_logits) == len(reference_logits) == 3
    for optimized_value, reference_value in zip(
        optimized_logits,
        reference_logits,
        strict=True,
    ):
        assert torch.equal(optimized_value, reference_value)
    assert torch.equal(optimized_out["actions"], reference_out["actions"])
    assert torch.equal(optimized_out["log_prob"], reference_out["log_prob"])
    assert torch.equal(optimized_out["entropy"], reference_out["entropy"])


def test_vectorized_effectiveness_preserves_gradient_enabled_head_gradients() -> None:
    optimized, inputs = _make_head_and_inputs()
    reference = copy.deepcopy(optimized)
    _install_scalar_effectiveness_reference(reference)

    optimized_loss = (
        optimized(**inputs)["log_prob"] + optimized(**inputs)["entropy"]
    ).sum()
    reference_loss = (
        reference(**inputs)["log_prob"] + reference(**inputs)["entropy"]
    ).sum()
    optimized_loss.backward()
    reference_loss.backward()

    for (optimized_name, optimized_parameter), (
        reference_name,
        reference_parameter,
    ) in zip(
        optimized.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        assert optimized_name == reference_name
        if optimized_name.startswith(
            "public_consequence_"
        ) and not optimized_name.startswith("public_consequence_policy_adapter."):
            # Actor gradients deliberately stop at the detached consequence
            # probabilities; this head is grounded only by its auxiliary loss.
            assert optimized_parameter.grad is None
            assert reference_parameter.grad is None
            continue
        assert optimized_parameter.grad is not None
        assert reference_parameter.grad is not None
        assert torch.equal(optimized_parameter.grad, reference_parameter.grad)


def test_reserve_item_projections_are_reused_only_without_gradients() -> None:
    head, inputs = _make_head_and_inputs()
    counts = {"key": 0, "value": 0}
    handles = [
        head.reserve_attention.key_proj.register_forward_hook(
            lambda *_args: counts.__setitem__("key", counts["key"] + 1)
        ),
        head.reserve_attention.value_proj.register_forward_hook(
            lambda *_args: counts.__setitem__("value", counts["value"] + 1)
        ),
    ]
    try:
        with torch.enable_grad():
            head(**inputs)
        assert counts == {"key": 2, "value": 2}

        counts.update(key=0, value=0)
        with torch.no_grad():
            head(**inputs)
        assert counts == {"key": 1, "value": 1}
    finally:
        for handle in handles:
            handle.remove()


def test_static_action_feature_cache_tracks_parameter_versions() -> None:
    head, _ = _make_head_and_inputs(batch_size=1)

    with torch.inference_mode():
        first = head._static_action_features()
        second = head._static_action_features()
    assert second is first

    with torch.no_grad():
        head.action_id_emb.weight.add_(0.125)
        updated = head._static_action_features()
    assert updated is not first
    assert not torch.equal(updated, first)

    head.zero_grad(set_to_none=True)
    grad_features = head._static_action_features()
    assert grad_features is not updated
    assert grad_features.requires_grad
    grad_features.sum().backward()
    assert head.action_id_emb.weight.grad is not None

    head.to(dtype=torch.float64)
    assert head._static_action_features_cache is None
    assert head._static_action_features_cache_key is None
