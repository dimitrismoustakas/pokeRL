import math
from pathlib import Path

import pytest
import torch

from pokerl.config import ModelConfig
from pokerl.data import GameData
from pokerl.env.vector_inputs import preview_observations
from pokerl.model.agent import PokeAgent
from pokerl.model.heads import MaskedSetAttention, TeamPreviewHead
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool


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


def test_masked_set_attention_never_assigns_mass_to_extremely_better_masked_item() -> (
    None
):
    attention = MaskedSetAttention(query_dim=3, item_dim=2, hidden_dim=4)
    summaries = attention._pool_projected_queries(
        torch.tensor([[[1.0e10, 0.0, 0.0, 0.0]]]),
        torch.tensor([[[-1.0e10, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]),
        torch.tensor([[[7.0, 11.0], [101.0, 103.0]]]),
        torch.tensor([[False, True]]),
    )

    torch.testing.assert_close(summaries, torch.tensor([[[7.0, 11.0]]]))


def test_team_preview_head_uses_candidate_conditioned_opponent_matchups() -> None:
    head = TeamPreviewHead(
        pokemon_dim=1,
        context_dim=1,
        pick_size=2,
        team_size=3,
    )

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


def test_team_preview_zero_logits_are_uniform_over_exact_joint_actions() -> None:
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    )
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()

    result = head(
        torch.randn(2, 6, 4),
        torch.randn(2, 6, 4),
        torch.randn(2, 6),
        greedy=True,
    )

    probabilities = result["joint_log_probs"].exp()
    expected = torch.full_like(probabilities, 1.0 / 180.0)
    torch.testing.assert_close(probabilities, expected, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(
        result["entropy"],
        torch.full((2,), math.log(180.0)),
        atol=1e-6,
        rtol=1e-6,
    )

    action_table = head._joint_picks
    slot_inclusion = torch.stack(
        [probabilities[0, (action_table == slot).any(dim=1)].sum() for slot in range(6)]
    )
    torch.testing.assert_close(
        slot_inclusion,
        torch.full((6,), 4.0 / 6.0),
        atol=1e-6,
        rtol=1e-6,
    )

    masked = head(
        torch.randn(1, 6, 4),
        torch.randn(1, 6, 4),
        torch.randn(1, 6),
        team_mask=torch.tensor([[False, False, False, False, False, True]]),
        opponent_mask=torch.ones((1, 6), dtype=torch.bool),
        greedy=True,
    )["joint_log_probs"].exp()[0]
    contains_masked_slot = (action_table == 5).any(dim=1)
    assert torch.count_nonzero(masked[contains_masked_slot]) == 0
    torch.testing.assert_close(
        masked[~contains_masked_slot],
        torch.full((60,), 1.0 / 60.0),
        atol=1e-7,
        rtol=1e-6,
    )


def test_team_preview_extreme_logits_cannot_resurrect_masked_actions() -> None:
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    )
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.bring_score[2].bias.fill_(-1.0e9)

    team_mask = torch.tensor([[False, False, False, False, False, True]])
    own_reps = torch.zeros(1, 6, 4)
    opponent_reps = torch.zeros(1, 6, 4)
    context = torch.zeros(1, 6)
    result = head(
        own_reps,
        opponent_reps,
        context,
        team_mask=team_mask,
        greedy=True,
    )

    contains_masked_slot = (head.action_table == 5).any(dim=1)
    joint_log_probs = result["joint_log_probs"][0]
    assert torch.isneginf(joint_log_probs[contains_masked_slot]).all()
    assert torch.isfinite(joint_log_probs[~contains_masked_slot]).all()
    assert torch.count_nonzero(joint_log_probs[contains_masked_slot].exp()) == 0
    torch.testing.assert_close(
        joint_log_probs[~contains_masked_slot].exp().sum(),
        torch.tensor(1.0),
    )
    assert 5 not in result["picks"][0]
    assert torch.isfinite(result["log_prob"]).all()
    assert torch.isfinite(result["entropy"]).all()

    batch_size = 64
    sampled = head(
        own_reps.expand(batch_size, -1, -1),
        opponent_reps.expand(batch_size, -1, -1),
        context.expand(batch_size, -1),
        team_mask=team_mask.expand(batch_size, -1),
        include_policy_stats=False,
    )
    assert not (sampled["picks"] == 5).any()


def test_team_preview_fast_and_full_lead_scorers_are_equivalent() -> None:
    torch.manual_seed(17)
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    )
    own_reps = torch.randn(3, 6, 4)
    opponent_reps = torch.randn(3, 6, 4)
    context = torch.randn(3, 6)
    opponent_mask = torch.tensor(
        [
            [False, False, False, True, True, True],
            [False, True, False, True, False, True],
            [True, True, True, True, True, True],
        ]
    )

    _, subset_members, bring_summaries = head._score_bring_subsets(
        own_reps,
        opponent_reps,
        context,
        opponent_mask,
    )
    keys, values = head.lead_opponent_attention._project_items(opponent_reps)
    full_logits = head._score_all_lead_orders(
        own_reps,
        bring_summaries,
        context,
        keys,
        values,
        opponent_mask,
    )
    fast_logits = torch.stack(
        [
            head._score_lead_orders(
                subset_members[:, subset_index],
                bring_summaries[:, subset_index],
                context,
                keys,
                values,
                opponent_mask,
            )
            for subset_index in range(head._bring_subsets.shape[0])
        ],
        dim=1,
    )

    torch.testing.assert_close(fast_logits, full_logits, atol=1e-6, rtol=1e-6)


def test_team_preview_fast_sample_log_prob_matches_exact_reevaluation() -> None:
    torch.manual_seed(18)
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    ).eval()
    own_reps = torch.randn(4, 6, 4)
    opponent_reps = torch.randn(4, 6, 4)
    context = torch.randn(4, 6)
    team_mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, False, False, True],
            [True, False, False, False, False, False],
            [False, False, False, False, False, False],
        ]
    )
    opponent_mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
            [True, False, True, False, True, False],
            [True, True, True, True, True, True],
        ]
    )

    with torch.inference_mode():
        sampled = head(
            own_reps,
            opponent_reps,
            context,
            team_mask=team_mask,
            opponent_mask=opponent_mask,
            include_policy_stats=False,
        )
        reevaluated = head(
            own_reps,
            opponent_reps,
            context,
            team_mask=team_mask,
            opponent_mask=opponent_mask,
            picks=sampled["picks"],
        )

    torch.testing.assert_close(
        sampled["log_prob"],
        reevaluated["log_prob"],
        atol=2e-6,
        rtol=2e-6,
    )


def test_uniform_migrated_preview_actor_learns_on_its_first_step() -> None:
    torch.manual_seed(18)
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    )
    with torch.no_grad():
        head.bring_score[2].weight.zero_()
        head.bring_score[2].bias.zero_()
        head.lead_pair_score[2].weight.zero_()
        head.lead_pair_score[2].bias.zero_()

    own_reps = torch.randn(2, 6, 4)
    opponent_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)
    target = head.action_table[0].unsqueeze(0).expand(2, -1)
    before = head(
        own_reps,
        opponent_reps,
        context,
        picks=target,
    )["joint_log_probs"].detach()

    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    loss = -head(
        own_reps,
        opponent_reps,
        context,
        picks=target,
    )["log_prob"].mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    after = head(
        own_reps,
        opponent_reps,
        context,
        picks=target,
    )["joint_log_probs"].detach()
    assert not torch.equal(before, after)
    assert (after[:, 0] > before[:, 0]).all()


def test_team_preview_joint_distribution_is_roster_permutation_equivariant() -> None:
    torch.manual_seed(19)
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    ).eval()
    own_reps = torch.randn(2, 6, 4)
    opponent_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)
    team_mask = torch.tensor(
        [
            [False, False, False, False, False, True],
            [False, False, False, False, False, False],
        ]
    )
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(6)

    original = head(
        own_reps,
        opponent_reps,
        context,
        team_mask=team_mask,
        greedy=True,
    )["joint_log_probs"]
    reordered = head(
        own_reps[:, permutation],
        opponent_reps,
        context,
        team_mask=team_mask[:, permutation],
        greedy=True,
    )["joint_log_probs"]

    remapped_actions = inverse_permutation[head._joint_picks]
    remapped_actions = torch.cat(
        [
            remapped_actions[:, :2],
            remapped_actions[:, 2:].sort(dim=1).values,
        ],
        dim=1,
    )
    action_matches = (
        remapped_actions.unsqueeze(1).eq(head._joint_picks.unsqueeze(0)).sum(dim=2)
        == head.pick_size
    )
    assert torch.equal(
        action_matches.sum(dim=1),
        torch.ones(head.legal_action_count, dtype=torch.long),
    )
    remapped_indices = action_matches.to(torch.int64).argmax(dim=1)
    torch.testing.assert_close(
        original,
        reordered[:, remapped_indices],
        atol=2e-6,
        rtol=2e-6,
    )


def test_team_preview_distribution_is_invariant_to_opponent_roster_order() -> None:
    torch.manual_seed(20)
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    ).eval()
    own_reps = torch.randn(2, 6, 4)
    opponent_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)
    opponent_mask = torch.tensor(
        [
            [False, False, False, True, True, True],
            [False, False, False, False, False, False],
        ]
    )
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])

    original = head(
        own_reps,
        opponent_reps,
        context,
        opponent_mask=opponent_mask,
        greedy=True,
    )["joint_log_probs"]
    reordered = head(
        own_reps,
        opponent_reps[:, permutation],
        context,
        opponent_mask=opponent_mask[:, permutation],
        greedy=True,
    )["joint_log_probs"]

    torch.testing.assert_close(original, reordered, atol=2e-6, rtol=2e-6)


def test_team_preview_entropy_log_prob_and_greedy_action_are_exact() -> None:
    torch.manual_seed(23)
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=3,
        team_size=5,
    ).eval()
    own_reps = torch.randn(3, 5, 4)
    opponent_reps = torch.randn(3, 5, 4)
    context = torch.randn(3, 6)

    result = head(
        own_reps,
        opponent_reps,
        context,
        greedy=True,
    )
    joint_log_probs = result["joint_log_probs"]
    probabilities = joint_log_probs.exp()
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(3),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        result["entropy"],
        -(probabilities * joint_log_probs).sum(dim=1),
        atol=1e-6,
        rtol=1e-6,
    )

    greedy_indices = joint_log_probs.argmax(dim=1)
    assert torch.equal(result["picks"], head._joint_picks[greedy_indices])
    replay = head(
        own_reps,
        opponent_reps,
        context,
        picks=result["picks"],
    )
    torch.testing.assert_close(
        replay["log_prob"],
        joint_log_probs.gather(1, greedy_indices.unsqueeze(1)).squeeze(1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_team_preview_action_tables_are_not_checkpoint_state() -> None:
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    )

    assert not any(key.startswith("_") for key in head.state_dict())
    exported = head.action_table
    exported.zero_()
    assert torch.count_nonzero(head.action_table) > 0


def test_team_preview_rejects_broadcasting_team_mask() -> None:
    head = TeamPreviewHead(
        pokemon_dim=4,
        context_dim=6,
        pick_size=4,
        team_size=6,
    )

    with pytest.raises(ValueError, match="expected team_mask shape"):
        head(
            torch.randn(2, 6, 4),
            torch.randn(2, 6, 4),
            torch.randn(2, 6),
            team_mask=torch.zeros(1, 6, dtype=torch.bool),
        )


def test_team_preview_rejects_combinatorial_action_spaces_before_materializing() -> (
    None
):
    with pytest.raises(ValueError, match="exact action space is too large"):
        TeamPreviewHead(
            pokemon_dim=4,
            context_dim=6,
            pick_size=7,
            team_size=14,
        )


def test_full_preview_policy_is_equivariant_to_own_roster_permutation() -> None:
    game_data = GameData(gen=9)
    team_pool = load_team_pool(
        default_team_pool_manifest_path(Path.cwd(), "gen9vgc2026regf"),
        subset="curated",
        seed=0,
    )
    own_team, opponent_team = team_pool.sample_packed_pair()
    observation = preview_observations([(own_team, opponent_team)], game_data)
    obs = {key: torch.from_numpy(value) for key, value in observation.items()}
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(6)
    reordered = {key: value.clone() for key, value in obs.items()}
    for suffix in (
        "species",
        "moves",
        "last_move",
        "ability",
        "item",
        "types",
        "continuous",
        "roles",
        "mask",
    ):
        key = f"own_{suffix}"
        reordered[key] = reordered[key][:, permutation]

    torch.manual_seed(29)
    agent = PokeAgent(ModelConfig(), game_data).eval()
    with torch.inference_mode():
        original_log_probs = agent.team_preview(
            obs,
            greedy=True,
            include_value=False,
            include_opponent_preview=False,
        )["joint_log_probs"]
        reordered_log_probs = agent.team_preview(
            reordered,
            greedy=True,
            include_value=False,
            include_opponent_preview=False,
        )["joint_log_probs"]

    action_table = agent.team_preview_head.action_table
    remapped_actions = inverse_permutation[action_table]
    remapped_actions = torch.cat(
        [
            remapped_actions[:, :2],
            remapped_actions[:, 2:].sort(dim=1).values,
        ],
        dim=1,
    )
    matches = (
        remapped_actions.unsqueeze(1).eq(action_table.unsqueeze(0)).sum(dim=2)
        == agent.config.pick_size
    )
    remapped_indices = matches.to(torch.int64).argmax(dim=1)
    torch.testing.assert_close(
        original_log_probs,
        reordered_log_probs[:, remapped_indices],
        atol=2e-5,
        rtol=2e-5,
    )
