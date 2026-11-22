import copy
from dataclasses import dataclass
import random

import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.training.league import League, LeagueMember
from pokerl.training.ppo import (
    PPOTrainer,
)


@dataclass(frozen=True)
class TinyModelConfig:
    width: int = 2


class TinyAgent(nn.Module):
    def __init__(
        self,
        width: int = 2,
        *,
        config: TinyModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(2, width)
        self.config = TinyModelConfig(width) if config is None else config


class OptimizerGroupingAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_matrix = nn.Parameter(torch.ones(2, 2))
        self.frozen_matrix = nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.embedding = nn.Embedding(8, 3)
        self.norm = nn.LayerNorm(3)
        self.hidden = nn.Linear(3, 4)
        self.memory = nn.LSTM(4, 4)
        self.output = nn.Linear(4, 1)


def _assert_nested_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_value, expected_value)
        return
    assert actual == expected


def test_adam_settings_preserve_current_optimizer_defaults() -> None:
    trainer = PPOTrainer(
        TinyAgent(),
        TrainingConfig(optimizer_name="adam", lr=2.5e-4),
        torch.device("cpu"),
    )

    assert type(trainer.optimizer) is torch.optim.Adam
    assert len(trainer.optimizer.param_groups) == 1
    assert trainer.optimizer_settings() == {
        "name": "adam",
        "lr": 2.5e-4,
        "betas": (0.9, 0.999),
        "eps": 1e-5,
        "weight_decay": 0.0,
        "decay_grouping": "none",
    }


def test_adamw_groups_every_trainable_parameter_exactly_once() -> None:
    agent = OptimizerGroupingAgent()
    trainer = PPOTrainer(
        agent,
        TrainingConfig(
            optimizer_name="adamw",
            lr=3e-4,
            weight_decay=0.02,
        ),
        torch.device("cpu"),
    )

    assert type(trainer.optimizer) is torch.optim.AdamW
    assert len(trainer.optimizer.param_groups) == 2
    decay_group, no_decay_group = trainer.optimizer.param_groups
    assert decay_group["weight_decay"] == pytest.approx(0.02)
    assert no_decay_group["weight_decay"] == 0.0

    decay_ids = {id(param) for param in decay_group["params"]}
    no_decay_ids = {id(param) for param in no_decay_group["params"]}
    assert decay_ids == {
        id(agent.root_matrix),
        id(agent.hidden.weight),
        id(agent.memory.weight_ih_l0),
        id(agent.memory.weight_hh_l0),
        id(agent.output.weight),
    }
    assert no_decay_ids == {
        id(agent.embedding.weight),
        id(agent.norm.weight),
        id(agent.norm.bias),
        id(agent.hidden.bias),
        id(agent.memory.bias_ih_l0),
        id(agent.memory.bias_hh_l0),
        id(agent.output.bias),
    }
    assert decay_ids.isdisjoint(no_decay_ids)
    assert decay_ids | no_decay_ids == {
        id(param) for param in agent.parameters() if param.requires_grad
    }
    assert id(agent.frozen_matrix) not in decay_ids | no_decay_ids


def test_zero_decay_adamw_matches_one_adam_step_exactly_on_cpu() -> None:
    torch.manual_seed(47)
    adam_agent = TinyAgent()
    adamw_agent = copy.deepcopy(adam_agent)
    adam = PPOTrainer(
        adam_agent,
        TrainingConfig(optimizer_name="adam", lr=2.5e-4),
        torch.device("cpu"),
    )
    adamw = PPOTrainer(
        adamw_agent,
        TrainingConfig(
            optimizer_name="adamw",
            lr=2.5e-4,
            weight_decay=0.0,
        ),
        torch.device("cpu"),
    )
    features = torch.tensor([[0.5, -1.0], [1.25, 0.75]])

    assert (
        adam._apply_update_step(adam.agent.linear(features).square().sum()) is not None
    )
    assert (
        adamw._apply_update_step(adamw.agent.linear(features).square().sum())
        is not None
    )

    for adam_param, adamw_param in zip(
        adam.agent.parameters(),
        adamw.agent.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(adam_param, adamw_param, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("saved_family", ["adam", "adamw"])
def test_trainer_checkpoint_roundtrips(tmp_path, saved_family) -> None:
    trainer = PPOTrainer(
        TinyAgent(),
        TrainingConfig(
            optimizer_name=saved_family,
            weight_decay=0.05 if saved_family == "adamw" else 0.0,
            lr=0.0002,
            tera_exploration_start=0.3,
            tera_exploration_end=0.05,
            tera_exploration_decay_updates=113,
        ),
        torch.device("cpu"),
    )
    loss = trainer.agent.linear(torch.tensor([[0.5, -1.0]])).square().sum()
    assert trainer._apply_update_step(loss)
    trainer.global_step = 123
    trainer.completed_updates = 7
    checkpoint_path = tmp_path / "trainer.pt"
    expected_agent = copy.deepcopy(trainer.agent.state_dict())
    expected_optimizer = copy.deepcopy(trainer.optimizer.state_dict())

    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(404)

    trainer.save(str(checkpoint_path))
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()
    expected_python = [random.random() for _ in range(4)]
    expected_numpy = np.random.random(4)
    expected_torch = torch.rand(4)
    expected_cuda = (
        torch.rand(4, device="cuda").cpu() if torch.cuda.is_available() else None
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)

    other_family = "adamw" if saved_family == "adam" else "adam"
    loaded = PPOTrainer(
        TinyAgent(), TrainingConfig(optimizer_name=other_family), torch.device("cpu")
    )
    loaded.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    )

    assert loaded.global_step == 123
    assert loaded.completed_updates == 7
    assert loaded.config.tera_exploration_start == pytest.approx(0.3)
    assert loaded.config.tera_exploration_end == pytest.approx(0.05)
    assert loaded.config.tera_exploration_decay_updates == 113
    _assert_nested_equal(loaded.agent.state_dict(), expected_agent)
    _assert_nested_equal(loaded.optimizer.state_dict(), expected_optimizer)
    assert type(loaded.optimizer) is type(trainer.optimizer)
    assert loaded.optimizer_settings() == trainer.optimizer_settings()
    assert [random.random() for _ in range(4)] == expected_python
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0.0, atol=0.0)
    if expected_cuda is not None:
        torch.testing.assert_close(
            torch.rand(4, device="cuda").cpu(),
            expected_cuda,
            rtol=0.0,
            atol=0.0,
        )


def test_league_checkpoint_refuses_to_overwrite_existing_archive_bytes(
    tmp_path,
) -> None:
    target = tmp_path / "league" / "step_7.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"immutable prior run")
    league = object.__new__(League)
    league.config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league.checkpoints = []
    league._members = {}
    league._current_update = 0

    with pytest.raises(FileExistsError, match="already exists on disk"):
        league.add_checkpoint(TinyAgent(), step=7)

    assert target.read_bytes() == b"immutable prior run"


def test_league_sampling_uses_its_own_reproducible_rng() -> None:
    left = object.__new__(League)
    right = object.__new__(League)
    left.config = TrainingConfig()
    right.config = TrainingConfig()
    left.checkpoints = [f"step_{index}.pt" for index in range(12)]
    right.checkpoints = list(left.checkpoints)
    left._members = {
        path: LeagueMember(
            path=path,
            step=index,
            created_update=0,
        )
        for index, path in enumerate(left.checkpoints)
    }
    right._members = {
        path: LeagueMember(
            path=path,
            step=index,
            created_update=0,
        )
        for index, path in enumerate(right.checkpoints)
    }
    left._current_update = 0
    right._current_update = 0
    left._rng = random.Random(37)
    right._rng = random.Random(37)

    left_samples = [left._draw_checkpoint_paths(1, 0)[0] for _ in range(20)]
    random.seed(999)
    _ = [random.random() for _ in range(100)]
    right_samples = [right._draw_checkpoint_paths(1, 0)[0] for _ in range(20)]

    assert left_samples == right_samples
