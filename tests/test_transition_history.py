from pokerl.env.vector_inputs import preview_observations

import torch
from pokerl.config import ModelConfig
from pokerl.data import GameData
from pokerl.env.observation import (
    NUM_PUBLIC_EVENT_FLAGS,
    POSITION_IMPLICIT,
    POSITION_OPP_2,
)
from pokerl.model.agent import PokeAgent
from pokerl.model.encoders import RecentEventEncoder


PACKED_TEAM = "]".join(
    [
        "Pikachu||lightball|static|thunderbolt,protect|Timid|||||50|",
        "Raichu||lifeorb|static|thunderbolt,protect|Timid|||||50|",
        "Bulbasaur||eviolite|overgrow|gigadrain,protect|Bold|||||50|",
        "Charmander||charcoal|blaze|flamethrower,protect|Timid|||||50|",
        "Squirtle||leftovers|torrent|surf,protect|Bold|||||50|",
        "Eevee||choicescarf|runaway|quickattack,protect|Jolly|||||50|",
    ]
)


def test_recent_event_encoder_is_zero_without_history_and_receives_gradient() -> None:
    encoder = RecentEventEncoder(
        num_moves=16,
        num_species=16,
        num_types=8,
        num_event_kinds=7,
        num_positions=6,
        num_intent_kinds=4,
    )
    obs = {
        "field": torch.zeros(2, 36),
        "recent_event_kind": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_actor": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_target": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_move": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_species": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_tera_type": torch.zeros(2, 12, dtype=torch.long),
        "recent_event_flags": torch.zeros(2, 12, NUM_PUBLIC_EVENT_FLAGS),
        "recent_event_mask": torch.ones(2, 12, dtype=torch.bool),
        "recent_event_meta": torch.zeros(2, 3),
        "prev_own_kind": torch.zeros(2, 2, dtype=torch.long),
        "prev_own_move": torch.zeros(2, 2, dtype=torch.long),
        "prev_own_target": torch.full((2, 2), POSITION_IMPLICIT, dtype=torch.long),
        "prev_own_switch_species": torch.zeros(2, 2, dtype=torch.long),
        "prev_own_tera": torch.zeros(2, 2, dtype=torch.bool),
        "prev_own_valid": torch.zeros(2, 2, dtype=torch.bool),
    }
    assert torch.equal(encoder(obs), torch.zeros(2, encoder.out_dim))
    assert torch.equal(
        encoder({"field": obs["field"]}),
        encoder(obs),
    )

    obs["recent_event_kind"][0, 0] = 1
    obs["recent_event_actor"][0, 0] = POSITION_OPP_2
    obs["recent_event_move"][0, 0] = 3
    obs["recent_event_mask"][0, 0] = False
    obs["recent_event_meta"][0, 0] = 1.0
    output = encoder(obs)
    output[0].sum().backward()

    assert output[0].abs().sum() > 0
    assert encoder.event_projection[0].weight.grad is not None


def test_team_preview_bypasses_zero_recent_event_history() -> None:
    game_data = GameData(gen=9)
    preview_obs = preview_observations([(PACKED_TEAM, PACKED_TEAM)] * 2, game_data)
    obs_batch = {key: torch.from_numpy(value) for key, value in preview_obs.items()}
    agent = PokeAgent(ModelConfig(), game_data).eval()

    def fail_if_called(_obs):
        raise AssertionError("team preview evaluated zero recent-event history")

    agent.spatial.recent_event_encoder.forward = fail_if_called
    with torch.inference_mode():
        result = agent.team_preview(
            obs_batch,
            include_value=False,
            include_policy_stats=False,
            include_opponent_preview=False,
        )

    assert result["picks"].shape == (2, 4)
