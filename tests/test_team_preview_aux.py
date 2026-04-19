import torch

from pokerl.model.heads import OpponentPreviewHead


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
