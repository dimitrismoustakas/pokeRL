import torch
import torch.nn.functional as F

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
        pokemon_dim=4, num_items=7, num_abilities=5, num_types=6, num_moves=9
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
    zero = torch.zeros(())

    item_loss = (
        F.cross_entropy(item_logits[item_mask], item_labels[item_mask])
        if item_mask.any()
        else zero
    )
    ability_loss = (
        F.cross_entropy(ability_logits[ability_mask], ability_labels[ability_mask])
        if ability_mask.any()
        else zero
    )
    tera_loss = (
        F.cross_entropy(tera_logits[tera_mask], tera_types[tera_mask])
        if tera_mask.any()
        else zero
    )

    move_logits = logits["move_logits"]
    move_slots = labels["opp_moves"]
    valid_slot = move_slots > 0
    move_slot_mask = base_mask & valid_slot.any(dim=-1)
    safe_slots = move_slots.clamp(min=0, max=move_logits.shape[-1] - 1)
    multihot = torch.zeros_like(move_logits)
    multihot.scatter_(-1, safe_slots, valid_slot.float())
    multihot[..., 0] = 0.0
    move_loss_raw = F.binary_cross_entropy_with_logits(
        move_logits, multihot, reduction="none"
    )
    denom = move_slot_mask.float().sum().clamp(min=1.0)
    move_loss = (move_loss_raw.mean(dim=-1) * move_slot_mask.float()).sum() / denom
    return item_loss + ability_loss + tera_loss + move_loss
