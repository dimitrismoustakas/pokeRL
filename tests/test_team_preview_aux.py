import torch
import torch.nn.functional as F
import pytest

from pokerl.env.observation import (
    ABILITY_KNOWN_OFFSET,
    CONTINUOUS_DIM,
    ITEM_KNOWN_OFFSET,
    TERA_KNOWN_OFFSET,
)
from pokerl.model.heads import HiddenInfoHead, OpponentPreviewHead
from pokerl.training.ppo import PPOTrainer


def test_opponent_preview_head_masks_invalid_lead_pairs() -> None:
    head = OpponentPreviewHead(pokemon_dim=4, context_dim=6, hidden_dim=8)
    pokemon_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)
    team_mask = torch.tensor(
        [
            [False, False, False, False, True, True],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    out = head(pokemon_reps, context, team_mask)

    assert out["opponent_bring_logits"].shape == (2, 6)
    assert out["opponent_lead_pair_logits"].shape == (2, 36)

    masked_logits = out["opponent_lead_pair_logits"][0].reshape(6, 6)
    assert torch.all(masked_logits.diag() < -1e7)
    assert torch.all(masked_logits[4] < -1e7)
    assert torch.all(masked_logits[:, 5] < -1e7)


def test_hidden_info_loss_matches_dense_reference() -> None:
    torch.manual_seed(0)
    head = HiddenInfoHead(
        pokemon_dim=4,
        context_dim=6,
        num_items=7,
        num_abilities=5,
        num_types=6,
        num_moves=9,
    )
    opp_reps = torch.randn(2, 2, 3, 4)
    labels = {
        "opp_item": torch.tensor(
            [[[1, 2, 3], [4, 0, 1]], [[2, 3, 4], [1, 2, 0]]],
            dtype=torch.long,
        ),
        "opp_ability": torch.tensor(
            [[[1, 2, 3], [4, 0, 1]], [[2, 3, 4], [1, 2, 0]]],
            dtype=torch.long,
        ),
        "opp_types": torch.tensor(
            [
                [[[1, 2, 3], [2, 0, 4], [3, 0, 0]], [[1, 0, 2], [2, 0, 0], [0, 0, 0]]],
                [[[2, 0, 5], [3, 0, 0], [4, 0, 1]], [[1, 0, 0], [2, 0, 3], [0, 0, 0]]],
            ],
            dtype=torch.long,
        ),
        "opp_moves": torch.tensor(
            [
                [
                    [[1, 2, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]],
                    [[4, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
                [
                    [[2, 3, 4, 0], [0, 0, 0, 0], [6, 0, 0, 0]],
                    [[0, 0, 0, 0], [7, 8, 0, 0], [0, 0, 0, 0]],
                ],
            ],
            dtype=torch.long,
        ),
        "opp_continuous": torch.zeros((2, 2, 3, CONTINUOUS_DIM), dtype=torch.float32),
        "opp_mask": torch.tensor(
            [
                [[False, False, True], [False, True, True]],
                [[False, True, False], [True, False, True]],
            ],
            dtype=torch.bool,
        ),
    }
    labels["opp_continuous"][..., ITEM_KNOWN_OFFSET] = 1.0
    labels["opp_continuous"][..., ABILITY_KNOWN_OFFSET] = 1.0
    labels["opp_continuous"][..., TERA_KNOWN_OFFSET] = 1.0
    label_valid = torch.tensor(
        [
            [[True, True, True], [False, False, False]],
            [[True, True, True], [True, True, True]],
        ],
        dtype=torch.bool,
    )

    sparse_loss = PPOTrainer._hidden_info_loss(head(opp_reps), labels, label_valid)
    dense_loss = _dense_hidden_info_loss(head(opp_reps), labels, label_valid)

    assert torch.allclose(sparse_loss, dense_loss)


def test_opponent_intent_loss_averages_available_prediction_tasks() -> None:
    logits = {
        "opp_intent_kind_logits": torch.zeros((2, 2, 4)),
        "opp_intent_move_logits": torch.zeros((2, 2, 5)),
        "opp_intent_target_logits": torch.zeros((2, 2, 3)),
        "opp_intent_switch_species_logits": torch.zeros((2, 2, 7)),
        "opp_intent_tera_use_logits": torch.zeros((2, 2)),
        "opp_intent_tera_slot_logits": torch.zeros((2, 2)),
    }
    slot_labels = torch.tensor(
        [[[[1, 2, 0, -1], [2, -1, -1, 3]], [[1, -1, -1, -1], [1, -1, -1, -1]]]]
    )
    slot_masks = torch.zeros_like(slot_labels, dtype=torch.bool)
    slot_masks[0, 0, 0, :3] = True
    slot_masks[0, 0, 1, 3] = True
    tera_labels = torch.tensor([[[1, 0], [0, -1]]])
    tera_masks = torch.tensor([[[True, True], [False, False]]])

    loss = PPOTrainer._opponent_intent_loss(
        logits,
        slot_labels,
        slot_masks,
        tera_labels,
        tera_masks,
    )

    expected = (
        sum(torch.log(torch.tensor(float(classes))) for classes in (4, 5, 3, 7, 2, 2))
        / 6
    )
    assert loss == pytest.approx(expected.item())


def test_policy_belief_adapter_is_initially_exactly_compatible() -> None:
    torch.manual_seed(11)
    head = HiddenInfoHead(
        pokemon_dim=4,
        context_dim=6,
        num_items=7,
        num_abilities=5,
        num_types=6,
        num_moves=9,
    )
    opp_reps = torch.randn(2, 3, 4)
    context = torch.randn(2, 6, requires_grad=True)

    spatial_only = head(opp_reps)
    encoded = head.encode(opp_reps, context)
    decoded = head.decode(encoded)
    conditioned = head(opp_reps, context)

    assert torch.count_nonzero(head.policy_belief_adapter.weight) == 0
    assert torch.count_nonzero(head.policy_belief_adapter.bias) == 0
    assert torch.equal(encoded, opp_reps)
    for key in spatial_only:
        assert torch.equal(decoded[key], conditioned[key])
        assert torch.equal(spatial_only[key], conditioned[key])

    initial_loss = sum(value.square().mean() for value in conditioned.values())
    initial_loss.backward()
    assert head.policy_belief_adapter.weight.grad is not None
    assert torch.count_nonzero(head.policy_belief_adapter.weight.grad) > 0
    assert context.grad is not None
    assert torch.count_nonzero(context.grad) == 0

    with torch.no_grad():
        head.policy_belief_adapter.weight.add_(
            -0.1 * head.policy_belief_adapter.weight.grad
        )
    head.zero_grad(set_to_none=True)
    learned_context = context.detach().clone().requires_grad_(True)
    learned_loss = sum(
        value.square().mean() for value in head(opp_reps, learned_context).values()
    )
    learned_loss.backward()

    assert learned_context.grad is not None
    assert torch.count_nonzero(learned_context.grad) > 0


def test_legacy_auxiliary_adapter_state_fails_strict_loading() -> None:
    head = HiddenInfoHead(
        pokemon_dim=4,
        context_dim=6,
        num_items=7,
        num_abilities=5,
        num_types=6,
        num_moves=9,
    )
    legacy_state = {
        key.replace("policy_belief_adapter", "context_adapter"): value
        for key, value in head.state_dict().items()
    }

    with pytest.raises(RuntimeError, match="Missing key.*policy_belief_adapter"):
        head.load_state_dict(legacy_state, strict=True)


def test_hidden_info_coverage_reports_selected_slots_and_public_fraction() -> None:
    label_valid = torch.tensor(
        [
            [[True, True, False, False], [False, False, False, False]],
            [[True, False, True, False], [False, False, False, False]],
        ]
    )
    opponent_padding = torch.tensor(
        [
            [[False, False, False, False], [False, False, True, True]],
            [[False, False, False, False], [False, False, True, True]],
        ]
    )

    metrics = PPOTrainer._hidden_info_coverage_metrics(
        label_valid,
        opponent_padding,
    )

    assert metrics["aux_hidden_info_valid_selected_slots"] == 4.0
    assert metrics["aux_hidden_info_public_slots"] == 12.0
    assert metrics["aux_hidden_info_label_coverage"] == pytest.approx(1 / 3)
    assert metrics["aux_hidden_info_labeled_steps"] == 2.0
    assert metrics["aux_hidden_info_valid_selected_slots_per_labeled_step"] == 2.0
    assert metrics["aux_hidden_info_labeled_step_coverage"] == 0.5


def test_gradient_diagnostics_measure_norms_conflicts_and_preserve_graph() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unrelated = torch.nn.Parameter(torch.tensor([3.0]))
    objectives = {
        "actor": (parameter * torch.tensor([1.0, 0.0])).sum(),
        "value": (parameter * torch.tensor([0.0, 2.0])).sum(),
        "opponent_intent": (parameter * torch.tensor([-1.0, 0.0])).sum(),
        "hidden_info": unrelated.square().sum(),
        "zero_connected": (parameter * 0.0).sum(),
    }

    diagnostics = PPOTrainer._gradient_diagnostics(
        objectives,
        (parameter,),
        prefix="memory_grad",
    )
    sum(objectives.values()).backward()

    assert diagnostics["memory_grad_norm_actor"] == pytest.approx(1.0)
    assert diagnostics["memory_grad_norm_value"] == pytest.approx(2.0)
    assert diagnostics["memory_grad_norm_opponent_intent"] == pytest.approx(1.0)
    assert diagnostics["memory_grad_norm_hidden_info"] == pytest.approx(0.0)
    assert diagnostics["memory_grad_connected_hidden_info"] == pytest.approx(0.0)
    assert diagnostics["memory_grad_nonzero_hidden_info"] == pytest.approx(0.0)
    assert diagnostics["memory_grad_connected_zero_connected"] == pytest.approx(1.0)
    assert diagnostics["memory_grad_nonzero_zero_connected"] == pytest.approx(0.0)
    assert diagnostics["memory_grad_norm_combined"] == pytest.approx(2.0)
    assert diagnostics["memory_grad_cancellation_fraction"] == pytest.approx(0.5)
    assert diagnostics["memory_grad_cos_actor_value"] == pytest.approx(0.0)
    assert diagnostics["memory_grad_cos_actor_value_valid"] == pytest.approx(1.0)
    assert diagnostics["memory_grad_cos_actor_opponent_intent"] == pytest.approx(-1.0)
    assert diagnostics["memory_grad_cos_value_opponent_intent"] == pytest.approx(0.0)
    assert diagnostics["memory_grad_cos_actor_hidden_info_valid"] == pytest.approx(0.0)
    assert torch.equal(parameter.grad, torch.tensor([0.0, 2.0]))


def _dense_hidden_info_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    label_valid: torch.Tensor,
) -> torch.Tensor:
    opp_pad = labels["opp_mask"]
    opp_cont = labels["opp_continuous"]
    item_known = opp_cont[..., ITEM_KNOWN_OFFSET] > 0.5
    ability_known = opp_cont[..., ABILITY_KNOWN_OFFSET] > 0.5
    tera_known = opp_cont[..., TERA_KNOWN_OFFSET] > 0.5
    base_mask = label_valid & ~opp_pad

    item_mask = (base_mask & item_known).reshape(-1)
    ability_mask = (base_mask & ability_known).reshape(-1)
    tera_types = labels["opp_types"][..., 2].reshape(-1)
    tera_mask = (base_mask & tera_known).reshape(-1) & (tera_types > 0)
    item_logits = logits["item_logits"].reshape(-1, logits["item_logits"].shape[-1])
    ability_logits = logits["ability_logits"].reshape(
        -1, logits["ability_logits"].shape[-1]
    )
    tera_logits = logits["tera_type_logits"].reshape(
        -1, logits["tera_type_logits"].shape[-1]
    )
    item_labels = labels["opp_item"].reshape(-1)
    ability_labels = labels["opp_ability"].reshape(-1)
    component_losses = []
    if item_mask.any():
        component_losses.append(
            F.cross_entropy(item_logits[item_mask], item_labels[item_mask])
        )
    if ability_mask.any():
        component_losses.append(
            F.cross_entropy(ability_logits[ability_mask], ability_labels[ability_mask])
        )
    if tera_mask.any():
        component_losses.append(
            F.cross_entropy(tera_logits[tera_mask], tera_types[tera_mask])
        )

    move_logits = logits["move_logits"]
    move_slots = labels["opp_moves"]
    valid_slot = move_slots > 0
    move_slot_mask = base_mask & valid_slot.any(dim=-1)
    safe_slots = move_slots.clamp(min=0, max=move_logits.shape[-1] - 1)
    multihot = torch.zeros_like(move_logits)
    multihot.scatter_(-1, safe_slots, valid_slot.float())
    multihot[..., 0] = 0.0
    if move_slot_mask.any():
        move_loss_raw = F.binary_cross_entropy_with_logits(
            move_logits, multihot, reduction="none"
        )
        denom = move_slot_mask.float().sum()
        component_losses.append(
            (move_loss_raw.mean(dim=-1) * move_slot_mask.float()).sum() / denom
        )
    return torch.stack(component_losses).mean()
