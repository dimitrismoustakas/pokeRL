import pytest
import torch

from pokerl.env.observation import (
    ABILITY_KNOWN_OFFSET,
    ITEM_KNOWN_OFFSET,
    TERA_KNOWN_OFFSET,
)
from pokerl.training.aux_metrics import (
    hidden_info_belief_additive_metrics,
    opponent_action_additive_metrics,
)
from pokerl.training.ppo import PPOTrainer


def _belief_fixture():
    base_shape = (1, 1, 2)
    continuous_width = (
        max(
            ABILITY_KNOWN_OFFSET,
            ITEM_KNOWN_OFFSET,
            TERA_KNOWN_OFFSET,
        )
        + 1
    )
    terminal_continuous = torch.zeros((*base_shape, continuous_width))
    terminal_continuous[..., ITEM_KNOWN_OFFSET] = 1.0
    terminal_continuous[..., ABILITY_KNOWN_OFFSET] = 1.0
    terminal_continuous[..., TERA_KNOWN_OFFSET] = 1.0
    public_continuous = torch.zeros_like(terminal_continuous)
    public_continuous[..., 0, ITEM_KNOWN_OFFSET] = 0.0
    public_continuous[..., 1, ITEM_KNOWN_OFFSET] = 1.0
    public_continuous[..., 0, ABILITY_KNOWN_OFFSET] = 1.0
    public_continuous[..., 1, ABILITY_KNOWN_OFFSET] = 0.0

    item_logits = torch.tensor(
        [[[[0.0, 1.0, 3.0], [4.0, 0.0, 0.0]]]], requires_grad=True
    )
    ability_logits = torch.tensor(
        [[[[2.0, 0.0, 0.0], [3.0, 1.0, 0.0]]]], requires_grad=True
    )
    tera_logits = torch.tensor(
        [[[[0.0, 0.0, 3.0, 1.0], [4.0, 0.0, 0.0, 0.0]]]], requires_grad=True
    )
    move_logits = torch.zeros((*base_shape, 8), requires_grad=True)
    with torch.no_grad():
        # Revealed moves 1 and 2 have the highest scores but must be excluded
        # from the candidate ranking. Of unknown true moves {3, 4}, only 3 is
        # then present in the top four {3, 5, 6, 7}.
        move_logits[..., 0, 1] = 20.0
        move_logits[..., 0, 2] = 19.0
        move_logits[..., 0, 3] = 10.0
        move_logits[..., 0, 5] = 9.0
        move_logits[..., 0, 6] = 8.0
        move_logits[..., 0, 7] = 7.0
        move_logits[..., 0, 4] = 1.0

    logits = {
        "item_logits": item_logits,
        "ability_logits": ability_logits,
        "tera_type_logits": tera_logits,
        "move_logits": move_logits,
    }
    terminal_labels = {
        "opp_mask": torch.zeros(base_shape, dtype=torch.bool),
        "opp_continuous": terminal_continuous,
        "opp_types": torch.tensor([[[[1, 0, 2], [1, 0, 0]]]]),
        "opp_item": torch.tensor([[[2, 0]]]),
        "opp_ability": torch.tensor([[[0, 1]]]),
        "opp_moves": torch.tensor([[[[1, 2, 3, 4], [1, 2, 3, 4]]]]),
    }
    public_obs = {
        "opp_mask": torch.zeros(base_shape, dtype=torch.bool),
        "opp_continuous": public_continuous,
        "opp_moves": torch.tensor([[[[1, 2, 0, 0], [0, 0, 0, 0]]]]),
    }
    label_valid = torch.tensor([[[True, False]]])
    return logits, terminal_labels, label_valid, public_obs


def test_hidden_info_metrics_are_unknown_only_additive_and_gradient_free() -> None:
    logits, terminal_labels, label_valid, public_obs = _belief_fixture()

    metrics = hidden_info_belief_additive_metrics(
        logits,
        terminal_labels,
        label_valid,
        public_obs,
    )

    item_log_probs = torch.log_softmax(logits["item_logits"][0, 0, 0], dim=-1)
    item_probs = item_log_probs.exp()
    expected_item_brier = item_probs.square().sum() - 2 * item_probs[2] + 1
    tera_log_probs = torch.log_softmax(logits["tera_type_logits"][0, 0, 0], dim=-1)
    tera_probs = tera_log_probs.exp()
    expected_tera_brier = tera_probs.square().sum() - 2 * tera_probs[2] + 1

    assert metrics["aux_item_unknown_top1_correct"] == 1.0
    assert metrics["aux_item_unknown_count"] == 1.0
    assert metrics["aux_item_unknown_nll_sum"] == pytest.approx(
        -item_log_probs[2].item()
    )
    assert metrics["aux_item_unknown_brier_sum"] == pytest.approx(
        expected_item_brier.item()
    )
    # The only valid slot's ability is already public, so it is excluded.
    assert metrics["aux_ability_unknown_count"] == 0.0
    assert metrics["aux_ability_unknown_top1_correct"] == 0.0
    assert metrics["aux_tera_unknown_top1_correct"] == 1.0
    assert metrics["aux_tera_unknown_count"] == 1.0
    assert metrics["aux_tera_unknown_nll_sum"] == pytest.approx(
        -tera_log_probs[2].item()
    )
    assert metrics["aux_tera_unknown_brier_sum"] == pytest.approx(
        expected_tera_brier.item()
    )
    assert metrics["aux_move_unknown_recall_at_4_correct"] == 1.0
    assert metrics["aux_move_unknown_recall_at_4_count"] == 2.0
    assert all(tensor.grad is None for tensor in logits.values())


def test_hidden_info_metrics_require_exact_terminal_targets() -> None:
    logits, terminal_labels, label_valid, public_obs = _belief_fixture()
    terminal_labels["opp_continuous"][..., ITEM_KNOWN_OFFSET] = 0.0
    terminal_labels["opp_continuous"][..., TERA_KNOWN_OFFSET] = 0.0

    metrics = hidden_info_belief_additive_metrics(
        logits,
        terminal_labels,
        label_valid,
        public_obs,
    )

    assert metrics["aux_item_unknown_count"] == 0.0
    assert metrics["aux_tera_unknown_count"] == 0.0


def test_hidden_info_move_recall_excludes_all_revealed_targets() -> None:
    logits, terminal_labels, label_valid, public_obs = _belief_fixture()
    public_obs["opp_moves"][..., 0, :] = torch.tensor([1, 2, 3, 4])

    metrics = hidden_info_belief_additive_metrics(
        logits,
        terminal_labels,
        label_valid,
        public_obs,
    )

    assert metrics["aux_move_unknown_recall_at_4_correct"] == 0.0
    assert metrics["aux_move_unknown_recall_at_4_count"] == 0.0


def test_opponent_action_metrics_report_each_raw_component() -> None:
    logits = {
        "opp_a1_logits": torch.tensor([[0.0, 3.0, 1.0], [4.0, 0.0, 0.0]]),
        "opp_a2_logits": torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 4.0]]),
        "opp_tera_logits": torch.tensor([[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]]),
    }
    labels = torch.tensor([[1, 0, 2], [-1, 1, 0]])
    valid = torch.tensor([True, True])

    metrics = opponent_action_additive_metrics(logits, labels, valid)

    assert metrics == {
        "opp_action_a1_top1_correct": 1.0,
        "opp_action_a1_count": 1.0,
        "opp_action_a2_top1_correct": 1.0,
        "opp_action_a2_count": 2.0,
        "opp_action_tera_top1_correct": 2.0,
        "opp_action_tera_count": 2.0,
    }


def test_aux_metrics_reject_out_of_range_targets() -> None:
    logits = {
        "opp_a1_logits": torch.zeros((1, 2)),
        "opp_a2_logits": torch.zeros((1, 2)),
        "opp_tera_logits": torch.zeros((1, 3)),
    }
    with pytest.raises(ValueError, match="out-of-range"):
        opponent_action_additive_metrics(
            logits,
            torch.tensor([[2, 0, 0]]),
            torch.tensor([True]),
        )


def test_ppo_finalizes_additive_belief_metrics_as_rates() -> None:
    metrics = PPOTrainer._finalize_aux_diagnostics(
        {
            "aux_item_unknown_top1_correct": 3.0,
            "aux_item_unknown_count": 4.0,
            "aux_item_unknown_nll_sum": 2.0,
            "aux_item_unknown_brier_sum": 1.0,
            "aux_move_unknown_recall_at_4_correct": 5.0,
            "aux_move_unknown_recall_at_4_count": 8.0,
            "opp_action_a1_top1_correct": 7.0,
            "opp_action_a1_count": 10.0,
        }
    )

    assert metrics["aux_item_unknown_top1_accuracy"] == pytest.approx(0.75)
    assert metrics["aux_item_unknown_nll"] == pytest.approx(0.5)
    assert metrics["aux_item_unknown_brier"] == pytest.approx(0.25)
    assert metrics["aux_move_unknown_recall_at_4"] == pytest.approx(0.625)
    assert metrics["opp_action_a1_top1_accuracy"] == pytest.approx(0.7)
    assert metrics["aux_ability_unknown_count"] == 0.0
