import torch

from pokerl.model.heads import OpponentIntentHead


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
