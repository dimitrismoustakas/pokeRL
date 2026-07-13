import math

import torch

from pokerl.model.heads import MaskedSetAttention, TeamPreviewHead


def test_masked_set_attention_supports_structured_candidate_queries() -> None:
    attention = MaskedSetAttention(query_dim=3, item_dim=2, hidden_dim=4)
    queries = torch.randn(2, 3, 5, 3)
    items = torch.randn(2, 6, 2)
    mask = torch.tensor(
        [
            [False, False, True, True, True, True],
            [True, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )

    summaries = attention(queries, items, mask)

    assert summaries.shape == (2, 3, 5, 2)
    assert torch.isfinite(summaries).all()
    assert torch.equal(summaries[1], torch.zeros_like(summaries[1]))


def test_team_preview_head_uses_candidate_conditioned_opponent_matchups() -> None:
    head = TeamPreviewHead(pokemon_dim=1, context_dim=1, pick_size=2)

    # Construct a transparent policy. Each candidate attends toward the
    # opponent value matching the sign of its own representation, and the
    # score prefers an attended matchup nearest zero. The scorer itself has no
    # direct own-pokemon input, so the decision can differ only through the
    # candidate-conditioned opponent summary.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()

        for attention, own_offset in (
            (head.bring_opponent_attention, 1),
            (head.lead_opponent_attention, 1),
        ):
            attention.query_proj.weight[0, own_offset] = math.sqrt(128.0)
            attention.key_proj.weight[0, 0] = 1.0
            attention.value_proj.weight[0, 0] = 1.0

        # bring_score input: [own, context, candidate-opponent-summary]
        head.bring_score[0].weight[0, 2] = 1.0
        head.bring_score[0].weight[1, 2] = -1.0
        head.bring_score[2].weight[0, 0] = -1.0
        head.bring_score[2].weight[0, 1] = -1.0

        # lead_pair_score input: [left, right, bring, pair-opponent-summary, context]
        head.lead_pair_score[0].weight[0, 3] = 1.0
        head.lead_pair_score[0].weight[1, 3] = -1.0
        head.lead_pair_score[2].weight[0, 0] = -1.0
        head.lead_pair_score[2].weight[0, 1] = -1.0

    own_reps = torch.tensor([[[1.0], [-1.0], [0.0]]])
    context = torch.zeros(1, 1)

    positive_opponents = torch.tensor([[[0.0], [2.0]]])
    negative_opponents = torch.tensor([[[-2.0], [0.0]]])
    positive_result = head(own_reps, positive_opponents, context, greedy=True)
    negative_result = head(own_reps, negative_opponents, context, greedy=True)

    assert torch.equal(positive_result["bring_picks"], torch.tensor([[1, 2]]))
    assert torch.equal(negative_result["bring_picks"], torch.tensor([[0, 2]]))
    assert torch.equal(positive_result["lead_pair"], torch.tensor([[1, 2]]))
    assert torch.equal(negative_result["lead_pair"], torch.tensor([[0, 2]]))


def test_team_preview_candidate_attention_preserves_legality_and_policy_stats() -> None:
    torch.manual_seed(7)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=4)
    own_reps = torch.randn(2, 6, 4)
    opponent_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)
    team_mask = torch.tensor(
        [
            [False, False, False, False, False, True],
            [False, False, False, False, True, False],
        ],
        dtype=torch.bool,
    )
    opponent_mask = torch.tensor(
        [
            [False, False, False, True, True, True],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    picks = torch.tensor([[3, 1, 4, 0], [5, 2, 3, 0]], dtype=torch.long)

    result = head(
        own_reps,
        opponent_reps,
        context,
        team_mask=team_mask,
        opponent_mask=opponent_mask,
        picks=picks,
    )

    assert result["picks"].shape == (2, 4)
    assert result["bring_picks"].shape == (2, 4)
    assert result["lead_pair"].shape == (2, 2)
    assert torch.equal(result["lead_pair"], picks[:, :2])
    assert torch.all(result["bring_picks"][:, 1:] > result["bring_picks"][:, :-1])
    assert torch.equal(result["picks"][:, 2:], result["picks"][:, 2:].sort(1).values)
    assert not team_mask.gather(1, result["picks"]).any()

    for name in (
        "bring_log_prob",
        "lead_log_prob",
        "log_prob",
        "bring_entropy",
        "lead_entropy",
        "entropy",
    ):
        assert result[name].shape == (2,)
        assert torch.isfinite(result[name]).all()
    assert (result["entropy"] >= 0).all()
