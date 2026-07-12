from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import ModelConfig
from pokerl.eval.fixed_bot import ModelPolicyController, load_agent_for_evaluation
from pokerl.model.agent import PokeAgent
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


class _ControllerAgent:
    def __init__(self) -> None:
        self.calls = []
        self.initial_hidden_calls = []

    def initial_hidden(self, batch_size: int):
        self.initial_hidden_calls.append(batch_size)
        return (torch.ones(1, batch_size, 2), torch.ones(1, batch_size, 2))

    def __call__(self, obs, hidden, **kwargs):
        self.calls.append((obs, hidden, kwargs))
        return {
            "actions": torch.tensor([[4, 7, 1]]),
            "entropy": torch.tensor([0.25]),
            "log_prob": torch.tensor([-0.5]),
            "hidden": tuple(state + 1 for state in hidden),
        }


def test_model_policy_controller_uses_agent_forward_with_eval_options() -> None:
    agent = _ControllerAgent()
    controller = ModelPolicyController(
        agent,
        device=torch.device("cpu"),
        action_mode="argmax",
    )
    masks = {
        "joint": np.ones((2, 2), dtype=bool),
        "can_tera": np.ones(2, dtype=bool),
    }

    result = controller.choose_action(
        {"feature": np.array([1.0, 2.0], dtype=np.float32)},
        masks,
    )

    assert agent.initial_hidden_calls == [1]
    assert len(agent.calls) == 1
    obs, hidden, kwargs = agent.calls[0]
    assert obs["feature"].shape == (1, 2)
    assert hidden is not None
    assert kwargs["greedy"] is True
    assert kwargs["include_value"] is False
    assert set(kwargs["action_masks"]) == set(masks)
    assert all(mask.dtype == torch.bool for mask in kwargs["action_masks"].values())
    np.testing.assert_array_equal(result["actions"], np.array([4, 7, 1]))
    assert result["entropy"] == pytest.approx(0.25)
    assert result["log_prob"] == pytest.approx(-0.5)


class _GameDataStub:
    @staticmethod
    def vocab_signature() -> str:
        return "evaluation-test-vocabulary"


class _TinyCheckpointAgent(nn.Module):
    def __init__(self, config, game_data) -> None:
        super().__init__()
        del config
        self.projection = nn.Linear(2, 2)
        self.game_data_signature = game_data.vocab_signature()


def _patch_tiny_checkpoint_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module, "PokeAgent", _TinyCheckpointAgent)


def test_evaluation_loader_accepts_exact_league_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_checkpoint_agent(monkeypatch)
    game_data = _GameDataStub()
    source = _TinyCheckpointAgent(ModelConfig(), game_data)
    checkpoint_path = tmp_path / "step_1382400.pt"
    torch.save(
        {"checkpoint_type": "league", "agent": source.state_dict()},
        checkpoint_path,
    )

    loaded = load_agent_for_evaluation(
        checkpoint_path=str(checkpoint_path),
        device=torch.device("cpu"),
        model_config=ModelConfig(),
        game_data=game_data,
    )

    for key, expected in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], expected)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_evaluation_loader_rejects_missing_state_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_checkpoint_agent(monkeypatch)
    game_data = _GameDataStub()
    source = _TinyCheckpointAgent(ModelConfig(), game_data)
    incomplete_state = dict(source.state_dict())
    incomplete_state.pop("projection.bias")
    checkpoint_path = tmp_path / "incomplete.pt"
    torch.save({"agent": incomplete_state}, checkpoint_path)

    with pytest.raises(ValueError, match="does not exactly match"):
        load_agent_for_evaluation(
            checkpoint_path=str(checkpoint_path),
            device=torch.device("cpu"),
            model_config=ModelConfig(),
            game_data=game_data,
        )
