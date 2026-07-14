from types import MethodType

import pytest
import torch

from pokerl.config import TrainingConfig
from pokerl.env.actions import ACTIONS_PER_SLOT, SWITCH_OFFSET, TERA_NONE
from pokerl.env.observation import FIELD_TRICK_ROOM_OFFSET
from pokerl.model.heads import ActionHead
from pokerl.public_consequence import (
    NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
)
from pokerl.training.aux_metrics import public_consequence_additive_metrics
from pokerl.training.ppo import PPOTrainer


def _head_and_inputs(batch_size: int = 3) -> tuple[ActionHead, dict[str, object]]:
    torch.manual_seed(7)
    head = ActionHead(context_dim=5, pokemon_dim=4, hidden_dim=12)
    joint = torch.ones(
        (batch_size, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
        dtype=torch.bool,
    )
    actions = torch.stack(
        [
            torch.arange(batch_size) % 4,
            (torch.arange(batch_size) + 2) % 4,
            torch.full((batch_size,), TERA_NONE),
        ],
        dim=1,
    )
    inputs: dict[str, object] = {
        "context": torch.randn(batch_size, 5),
        "active_reps": torch.randn(batch_size, 2, 4),
        "own_team_reps": torch.randn(batch_size, 6, 4),
        "opp_team_reps": torch.randn(batch_size, 6, 4),
        "opp_reserve_mask": torch.zeros(batch_size, 6, dtype=torch.bool),
        "opp_active_reps": torch.randn(batch_size, 2, 4),
        "masks": {
            "joint": joint,
            "can_tera": torch.zeros(batch_size, 2, dtype=torch.bool),
        },
        "actions": actions,
    }
    return head, inputs


def test_zero_consequence_adapter_exactly_preserves_policy() -> None:
    head, inputs = _head_and_inputs()
    head.eval()
    with torch.no_grad():
        before = head(**inputs)
        head.public_consequence_head.weight.normal_(mean=7.0, std=3.0)
        head.public_consequence_head.bias.normal_(mean=-4.0, std=2.0)
        after = head(**inputs)

    torch.testing.assert_close(
        before["log_prob"], after["log_prob"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(before["entropy"], after["entropy"], rtol=0.0, atol=0.0)
    assert not torch.equal(
        before["public_consequence_resolution_logits"],
        after["public_consequence_resolution_logits"],
    )


def test_policy_adapter_detaches_consequence_predictions_from_actor_gradient() -> None:
    head, inputs = _head_and_inputs()
    with torch.no_grad():
        head.public_consequence_policy_adapter.weight.normal_()

    actor_loss = -head(**inputs)["log_prob"].sum()
    actor_loss.backward()

    assert head.public_consequence_head.weight.grad is None
    assert head.public_consequence_head.bias.grad is None
    assert head.public_consequence_policy_adapter.weight.grad is not None
    assert head.public_consequence_policy_adapter.weight.grad.norm().item() > 0.0
    assert head.score[0].weight.grad is not None
    assert head.score[0].weight.grad.norm().item() > 0.0


def test_public_consequence_adapter_only_changes_supervised_move_candidates() -> None:
    head, _ = _head_and_inputs(batch_size=1)
    hidden_dim = head.action_id_emb.embedding_dim
    pokemon_dim = head.entity_proj[0].in_features
    query = torch.ones(1, hidden_dim)
    actor = torch.zeros(1, pokemon_dim)
    own_team = torch.zeros(1, 6, pokemon_dim)
    opponent_active = torch.zeros(1, 2, pokemon_dim)
    matchup = torch.zeros(1, ACTIONS_PER_SLOT, head._MATCHUP_FEATURE_DIM)
    action_features = head._static_action_features()

    with torch.no_grad():
        raw, _ = head._score_actions(
            query,
            actor,
            actor,
            own_team,
            opponent_active,
            matchup,
            action_features,
        )
        head.public_consequence_policy_adapter.weight.fill_(0.1)
        adapted, _ = head._score_actions(
            query,
            actor,
            actor,
            own_team,
            opponent_active,
            matchup,
            action_features,
        )

    delta = adapted - raw
    assert torch.all(delta[:, :SWITCH_OFFSET] > 0.0)
    torch.testing.assert_close(
        delta[:, SWITCH_OFFSET:],
        torch.zeros_like(delta[:, SWITCH_OFFSET:]),
    )


def test_public_consequence_outputs_gather_the_two_chosen_candidates() -> None:
    head, inputs = _head_and_inputs(batch_size=1)
    inputs["actions"] = torch.tensor([[3, 5, TERA_NONE]])
    call_count = 0

    def fake_score_actions(self, query, *args, **kwargs):
        del self, args, kwargs
        nonlocal call_count
        offset = 100.0 * call_count
        call_count += 1
        candidate = torch.arange(
            ACTIONS_PER_SLOT, dtype=query.dtype, device=query.device
        ).view(1, ACTIONS_PER_SLOT, 1)
        candidate = candidate.expand(query.shape[0], -1, -1) + offset
        return query.new_zeros((query.shape[0], ACTIONS_PER_SLOT)), {
            "resolution_logits": candidate.expand(
                -1, -1, NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES
            ),
            "cant_logits": candidate.expand(-1, -1, 2),
            "field_logits": candidate.expand(-1, -1, PUBLIC_CONSEQUENCE_FIELD_DIM),
        }

    head._score_actions = MethodType(fake_score_actions, head)
    out = head(**inputs)

    assert out["public_consequence_resolution_logits"].shape == (1, 2, 4)
    assert out["public_consequence_cant_logits"].shape == (1, 2, 2)
    assert out["public_consequence_field_logits"].shape == (1, 2, 33)
    assert out["public_consequence_resolution_logits"][0, 0].tolist() == [3.0] * 4
    assert out["public_consequence_resolution_logits"][0, 1].tolist() == [105.0] * 4


def test_public_consequence_outputs_are_omitted_without_policy_stats() -> None:
    head, inputs = _head_and_inputs(batch_size=1)
    out = head(**inputs, include_policy_stats=False)

    assert "public_consequence_resolution_logits" not in out
    assert "public_consequence_cant_logits" not in out
    assert "public_consequence_field_logits" not in out


def _consequence_tensors() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    logits = {
        "public_consequence_resolution_logits": torch.randn(2, 3, 2, 4),
        "public_consequence_cant_logits": torch.randn(2, 3, 2, 2),
        "public_consequence_field_logits": torch.randn(2, 3, 2, 33),
    }
    field_targets = torch.rand(2, 3, 2, 33)
    persistence_fields = torch.where(
        field_targets < 0.5,
        field_targets + 0.5,
        field_targets - 0.5,
    )
    labels = {
        "resolution_labels": torch.randint(0, 4, (2, 3, 2)),
        "resolution_masks": torch.ones(2, 3, 2, dtype=torch.bool),
        "cant_targets": torch.randint(0, 2, (2, 3, 2, 2)).float(),
        "cant_masks": torch.ones(2, 3, 2, 2, dtype=torch.bool),
        "field_targets": field_targets,
        "field_masks": torch.ones(2, 3, 2, dtype=torch.bool),
        "persistence_fields": persistence_fields,
    }
    return logits, labels


def test_public_consequence_aux_loss_is_finite_and_trains_shared_candidate_path() -> (
    None
):
    head, inputs = _head_and_inputs()
    out = head(**inputs)
    labels = {
        "resolution_labels": torch.randint(0, 4, (3, 2)),
        "resolution_masks": torch.ones(3, 2, dtype=torch.bool),
        "cant_targets": torch.randint(0, 2, (3, 2, 2)).float(),
        "cant_masks": torch.ones(3, 2, 2, dtype=torch.bool),
        "field_targets": torch.rand(3, 2, 33),
        "field_masks": torch.ones(3, 2, dtype=torch.bool),
        "persistence_fields": torch.rand(3, 2, 33),
    }
    logits = {
        key: out[key]
        for key in (
            "public_consequence_resolution_logits",
            "public_consequence_cant_logits",
            "public_consequence_field_logits",
        )
    }

    loss = PPOTrainer._public_consequence_loss(logits=logits, **labels)
    assert torch.isfinite(loss)
    loss.backward()

    assert head.public_consequence_head.weight.grad is not None
    assert head.public_consequence_head.weight.grad.norm().item() > 0.0
    assert head.score[0].weight.grad is not None
    assert head.score[0].weight.grad.norm().item() > 0.0


def test_public_consequence_aux_loss_handles_no_labels() -> None:
    logits, labels = _consequence_tensors()
    for tensor in logits.values():
        tensor.requires_grad_()
    labels["resolution_masks"].zero_()
    labels["cant_masks"].zero_()
    labels["field_masks"].zero_()

    loss = PPOTrainer._public_consequence_loss(logits=logits, **labels)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits["public_consequence_resolution_logits"].grad is not None
    assert logits["public_consequence_resolution_logits"].grad.count_nonzero() == 0


def test_public_consequence_aux_loss_normalizes_each_task_before_combining() -> None:
    logits = {
        "public_consequence_resolution_logits": torch.zeros(1, 2, 4),
        "public_consequence_cant_logits": torch.zeros(1, 2, 2),
        "public_consequence_field_logits": torch.zeros(1, 2, 33),
    }
    loss = PPOTrainer._public_consequence_loss(
        logits=logits,
        resolution_labels=torch.tensor([[0, 1]], dtype=torch.long),
        resolution_masks=torch.ones(1, 2, dtype=torch.bool),
        cant_targets=torch.zeros(1, 2, 2),
        cant_masks=torch.ones(1, 2, 2, dtype=torch.bool),
        field_targets=torch.zeros(1, 2, 33),
        field_masks=torch.ones(1, 2, dtype=torch.bool),
        persistence_fields=torch.zeros(1, 2, 33),
    )

    execution_loss = (
        -torch.log(torch.tensor(0.25)) - torch.log(torch.tensor(0.75))
    ) / 2
    resolution_loss = (execution_loss + torch.log(torch.tensor(3.0))) / 2
    expected = (resolution_loss + torch.log(torch.tensor(2.0)) + 0.25) / 3
    torch.testing.assert_close(loss, expected)


def test_public_consequence_resolution_loss_is_proper_hierarchical_likelihood() -> None:
    logits = torch.tensor([[[1.5, -0.2, 0.3, -1.0], [-0.5, 1.0, 0.1, -0.4]]])
    labels = torch.tensor([[0, 2]])
    loss = PPOTrainer._public_consequence_loss(
        logits={
            "public_consequence_resolution_logits": logits,
            "public_consequence_cant_logits": torch.zeros(1, 2, 2),
            "public_consequence_field_logits": torch.zeros(1, 2, 33),
        },
        resolution_labels=labels,
        resolution_masks=torch.ones(1, 2, dtype=torch.bool),
        cant_targets=torch.zeros(1, 2, 2),
        cant_masks=torch.zeros(1, 2, 2, dtype=torch.bool),
        field_targets=torch.zeros(1, 2, 33),
        field_masks=torch.zeros(1, 2, dtype=torch.bool),
        persistence_fields=torch.zeros(1, 2, 33),
    )

    execution_logits = torch.stack(
        (logits[..., 0], torch.logsumexp(logits[..., 1:], dim=-1)),
        dim=-1,
    )
    expected_execution = torch.nn.functional.cross_entropy(
        execution_logits.reshape(-1, 2),
        torch.tensor([0, 1]),
    )
    expected_type = torch.nn.functional.cross_entropy(
        logits[0, 1, 1:].unsqueeze(0),
        torch.tensor([1]),
    )
    torch.testing.assert_close(loss, (expected_execution + expected_type) / 2)


def test_public_consequence_field_loss_balances_changed_and_unchanged_features() -> (
    None
):
    field_logits = torch.full((1, 2, 33), -20.0)
    field_targets = torch.zeros(1, 2, 33)
    persistence = torch.zeros_like(field_targets)
    field_targets[0, 0, FIELD_TRICK_ROOM_OFFSET] = 1.0
    loss = PPOTrainer._public_consequence_loss(
        logits={
            "public_consequence_resolution_logits": torch.zeros(1, 2, 4),
            "public_consequence_cant_logits": torch.zeros(1, 2, 2),
            "public_consequence_field_logits": field_logits,
        },
        resolution_labels=torch.full((1, 2), -1, dtype=torch.long),
        resolution_masks=torch.zeros(1, 2, dtype=torch.bool),
        cant_targets=torch.zeros(1, 2, 2),
        cant_masks=torch.zeros(1, 2, 2, dtype=torch.bool),
        field_targets=field_targets,
        field_masks=torch.tensor([[True, False]]),
        persistence_fields=persistence,
    )

    probabilities = field_logits.sigmoid()
    changed_mse = (probabilities[0, 0, FIELD_TRICK_ROOM_OFFSET] - 1.0).square()
    unchanged = torch.ones(33, dtype=torch.bool)
    unchanged[FIELD_TRICK_ROOM_OFFSET] = False
    unchanged_mse = probabilities[0, 0, unchanged].square().mean()
    torch.testing.assert_close(loss, (changed_mse + unchanged_mse) / 2)


def test_public_consequence_field_loss_falls_back_when_only_one_group_exists() -> None:
    field_logits = torch.zeros(1, 2, 33)
    field_targets = torch.ones(1, 2, 33)
    loss = PPOTrainer._public_consequence_loss(
        logits={
            "public_consequence_resolution_logits": torch.zeros(1, 2, 4),
            "public_consequence_cant_logits": torch.zeros(1, 2, 2),
            "public_consequence_field_logits": field_logits,
        },
        resolution_labels=torch.full((1, 2), -1, dtype=torch.long),
        resolution_masks=torch.zeros(1, 2, dtype=torch.bool),
        cant_targets=torch.zeros(1, 2, 2),
        cant_masks=torch.zeros(1, 2, 2, dtype=torch.bool),
        field_targets=field_targets,
        field_masks=torch.ones(1, 2, dtype=torch.bool),
        persistence_fields=torch.zeros_like(field_targets),
    )

    torch.testing.assert_close(loss, torch.tensor(0.25))


def test_public_consequence_metrics_report_support_and_persistence_baseline() -> None:
    logits, labels = _consequence_tensors()
    persistence = labels["persistence_fields"]
    metrics = public_consequence_additive_metrics(
        logits,
        labels["resolution_labels"],
        labels["resolution_masks"],
        labels["cant_targets"],
        labels["cant_masks"],
        labels["field_targets"],
        labels["field_masks"],
        persistence,
    )

    assert metrics["aux_public_consequence_resolution_count"] == 12.0
    assert metrics["aux_public_consequence_cant_count"] == 24.0
    assert metrics["aux_public_consequence_field_count"] == 12.0
    assert metrics["aux_public_consequence_field_changed_count"] == 396.0
    assert metrics["aux_public_consequence_field_trick_room_count"] == 12.0
    assert metrics["aux_public_consequence_cant_positive"] == pytest.approx(
        labels["cant_targets"].sum().item()
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_public_consequence_metrics_batch_device_scalar_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, labels = _consequence_tensors()

    def fail_scalar_extraction(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "per-metric tensor scalar extraction synchronized the device"
        )

    monkeypatch.setattr(torch.Tensor, "item", fail_scalar_extraction)
    monkeypatch.setattr(torch.Tensor, "__bool__", fail_scalar_extraction)
    metrics = public_consequence_additive_metrics(
        logits,
        labels["resolution_labels"],
        labels["resolution_masks"],
        labels["cant_targets"],
        labels["cant_masks"],
        labels["field_targets"],
        labels["field_masks"],
        labels["persistence_fields"],
    )

    assert metrics["aux_public_consequence_resolution_count"] == 12.0
    assert all(isinstance(value, float) for value in metrics.values())


def test_public_consequence_metrics_report_resolution_classes_and_field_changes() -> (
    None
):
    resolution_logits = torch.tensor(
        [
            [8.0, 0.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 8.0, 0.0, 0.0],
        ]
    ).reshape(1, 2, 2, 4)
    field_targets = torch.zeros(1, 2, 2, 33)
    persistence = torch.zeros_like(field_targets)
    field_targets[0, 0, 0, FIELD_TRICK_ROOM_OFFSET] = 1.0
    persistence[0, 1, 0, FIELD_TRICK_ROOM_OFFSET] = 1.0
    field_targets[0, 1, 1, FIELD_TRICK_ROOM_OFFSET] = 1.0
    persistence[0, 1, 1, FIELD_TRICK_ROOM_OFFSET] = 1.0
    metrics = public_consequence_additive_metrics(
        {
            "public_consequence_resolution_logits": resolution_logits,
            "public_consequence_cant_logits": torch.zeros(1, 2, 2, 2),
            "public_consequence_field_logits": torch.zeros(1, 2, 2, 33),
        },
        torch.tensor([[[0, 1], [2, 3]]]),
        torch.ones(1, 2, 2, dtype=torch.bool),
        torch.zeros(1, 2, 2, 2),
        torch.zeros(1, 2, 2, 2, dtype=torch.bool),
        field_targets,
        torch.ones(1, 2, 2, dtype=torch.bool),
        persistence,
    )

    for class_name in ("executed", "failed", "missed", "cant"):
        assert metrics[f"aux_public_consequence_resolution_{class_name}_count"] == 1.0
    assert metrics["aux_public_consequence_resolution_executed_correct"] == 1.0
    assert metrics["aux_public_consequence_resolution_failed_correct"] == 0.0
    assert metrics["aux_public_consequence_resolution_missed_correct"] == 1.0
    assert metrics["aux_public_consequence_resolution_cant_correct"] == 0.0
    assert metrics["aux_public_consequence_field_changed_count"] == 2.0
    assert metrics["aux_public_consequence_field_changed_mse_sum"] == pytest.approx(0.5)
    assert metrics[
        "aux_public_consequence_field_changed_persistence_mse_sum"
    ] == pytest.approx(2.0)
    assert metrics["aux_public_consequence_field_trick_room_count"] == 4.0
    assert metrics["aux_public_consequence_field_trick_room_mse_sum"] == pytest.approx(
        1.0
    )
    assert metrics[
        "aux_public_consequence_field_trick_room_persistence_mse_sum"
    ] == pytest.approx(2.0)
    assert metrics["aux_public_consequence_field_trick_room_value_changed_count"] == 2.0
    assert (
        metrics["aux_public_consequence_field_trick_room_presence_changed_count"] == 2.0
    )

    finalized = PPOTrainer._finalize_aux_diagnostics(metrics)
    assert finalized["aux_public_consequence_resolution_executed_recall"] == 1.0
    assert finalized["aux_public_consequence_resolution_failed_recall"] == 0.0
    assert finalized["aux_public_consequence_resolution_missed_recall"] == 1.0
    assert finalized["aux_public_consequence_resolution_cant_recall"] == 0.0
    assert finalized["aux_public_consequence_resolution_executed_nll"] > 0.0
    assert finalized["aux_public_consequence_field_changed_mse"] == pytest.approx(0.25)
    assert finalized[
        "aux_public_consequence_field_changed_persistence_mse"
    ] == pytest.approx(1.0)
    assert finalized["aux_public_consequence_field_changed_mse_skill"] == pytest.approx(
        0.75
    )
    assert finalized["aux_public_consequence_field_trick_room_mse"] == pytest.approx(
        0.25
    )
    assert finalized[
        "aux_public_consequence_field_trick_room_persistence_mse"
    ] == pytest.approx(0.5)
    assert finalized[
        "aux_public_consequence_field_trick_room_value_changed_mse_skill"
    ] == pytest.approx(0.75)
    assert finalized[
        "aux_public_consequence_field_trick_room_presence_changed_mse_skill"
    ] == pytest.approx(0.75)


def test_public_consequence_cant_metrics_separate_positive_and_negative_recall() -> (
    None
):
    metrics = public_consequence_additive_metrics(
        {
            "public_consequence_resolution_logits": torch.zeros(1, 1, 2, 4),
            "public_consequence_cant_logits": torch.tensor(
                [[[[2.0, -2.0], [2.0, -2.0]]]]
            ),
            "public_consequence_field_logits": torch.zeros(1, 1, 2, 33),
        },
        torch.full((1, 1, 2), -1, dtype=torch.long),
        torch.zeros(1, 1, 2, dtype=torch.bool),
        torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        torch.ones(1, 1, 2, 2, dtype=torch.bool),
        torch.zeros(1, 1, 2, 33),
        torch.zeros(1, 1, 2, dtype=torch.bool),
        torch.zeros(1, 1, 2, 33),
    )

    finalized = PPOTrainer._finalize_aux_diagnostics(metrics)
    assert finalized["aux_public_consequence_cant_prevalence"] == pytest.approx(0.5)
    assert finalized[
        "aux_public_consequence_cant_predicted_positive_rate"
    ] == pytest.approx(0.5)
    assert finalized["aux_public_consequence_cant_positive_recall"] == pytest.approx(
        0.5
    )
    assert finalized[
        "aux_public_consequence_cant_negative_specificity"
    ] == pytest.approx(0.5)


def test_public_consequence_coefficient_is_conservative_and_validated() -> None:
    assert TrainingConfig().aux_public_consequence_coef == pytest.approx(0.1)
    for invalid in (0.0, -0.01):
        with pytest.raises(ValueError, match="aux_public_consequence_coef"):
            TrainingConfig(aux_public_consequence_coef=invalid)
