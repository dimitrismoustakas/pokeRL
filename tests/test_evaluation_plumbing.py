from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pokerl.config import ModelConfig
from pokerl.model.agent import PokeAgent, load_policy, policy_weights
import pokerl.model.agent as agent_module


class _SpatialStub(nn.Module):
    def forward(self, obs):
        batch_size = obs["opp_roles"].shape[0]
        context = torch.zeros(batch_size, 3)
        active = torch.zeros(batch_size, 2, 2)
        own = torch.zeros(batch_size, 6, 2)
        opp = torch.zeros(batch_size, 6, 2)
        opp_active = torch.zeros(batch_size, 2, 2)
        return context, active, own, opp, opp_active


class _MemoryStub(nn.Module):
    def forward(self, context, hidden):
        return context, hidden


class _BeliefStub(nn.Module):
    def context_residual(self, context):
        return torch.zeros((*context.shape[:-1], 2), device=context.device)

    def encode_with_residual(self, opponent_reps, residual):
        return opponent_reps + residual.unsqueeze(-2)


class _RecordingActionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.args = None
        self.kwargs = None

    def forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        batch_size = args[0].shape[0]
        return {
            "actions": torch.zeros(batch_size, 3, dtype=torch.long),
            "entropy": torch.zeros(batch_size),
            "log_prob": torch.zeros(batch_size),
        }


def test_poke_agent_forward_passes_greedy_to_action_head() -> None:
    agent = PokeAgent.__new__(PokeAgent)
    nn.Module.__init__(agent)
    agent.spatial = _SpatialStub()
    agent.memory = _MemoryStub()
    agent.hidden_info_head = _BeliefStub()
    action_head = _RecordingActionHead()
    agent.action_head = action_head

    action_masks = {
        "joint": torch.ones(1, 2, 2, dtype=torch.bool),
        "can_tera": torch.ones(1, 2, dtype=torch.bool),
    }
    agent(
        {
            "opp_roles": torch.zeros(1, 6, dtype=torch.long),
            "opp_mask": torch.zeros(1, 6, dtype=torch.bool),
        },
        action_masks=action_masks,
        greedy=True,
        include_value=False,
    )

    assert action_head.kwargs is not None
    assert action_head.kwargs["greedy"] is True
    assert action_head.args is not None
    assert action_head.args[6] is action_masks


class _TinyCheckpointAgent(nn.Module):
    def __init__(self, config, game_data) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(2, 2)


def _patch_tiny_checkpoint_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "PokeAgent", _TinyCheckpointAgent)


def test_evaluation_loader_accepts_exact_league_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_checkpoint_agent(monkeypatch)
    game_data = object()
    source = _TinyCheckpointAgent(ModelConfig(move_dim=48), game_data)
    checkpoint_path = tmp_path / "step_1382400.pt"
    torch.save(
        policy_weights(source),
        checkpoint_path,
    )

    loaded = load_policy(checkpoint_path, game_data, torch.device("cpu"))

    assert loaded.config == source.config
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
