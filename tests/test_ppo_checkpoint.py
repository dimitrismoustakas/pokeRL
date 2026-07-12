from collections import deque
import copy
from dataclasses import dataclass
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.training.league import League
from pokerl.training.ppo import (
    TRAINER_CHECKPOINT_FORMAT_VERSION,
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
        if config is not None:
            self.config = config


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


def test_trainer_load_rejects_league_only_checkpoint(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "league_step.pt"
    torch.save(
        {"checkpoint_type": "league", "agent": trainer.agent.state_dict()},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="checkpoint_type='trainer'"):
        trainer.load(str(checkpoint_path))


def test_trainer_load_rejects_legacy_agent_only_checkpoint(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "legacy_league_step.pt"
    torch.save({"agent": trainer.agent.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="Legacy and league checkpoints"):
        trainer.load(str(checkpoint_path))


def test_trainer_checkpoint_roundtrips(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    loss = trainer.agent.linear(torch.tensor([[0.5, -1.0]])).square().sum()
    assert trainer._apply_update_step(loss)
    trainer.global_step = 123
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
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert raw_checkpoint["checkpoint_type"] == "trainer"
    assert raw_checkpoint["format_version"] == TRAINER_CHECKPOINT_FORMAT_VERSION
    assert isinstance(raw_checkpoint["training_state_id"], str)
    assert raw_checkpoint["model_metadata"]["state_schema"]
    assert raw_checkpoint["rng_state"]["torch_cpu"].dtype == torch.uint8

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

    loaded = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    loaded.load(str(checkpoint_path))

    assert loaded.global_step == 123
    assert loaded.loaded_training_state_id == raw_checkpoint["training_state_id"]
    _assert_nested_equal(loaded.agent.state_dict(), expected_agent)
    _assert_nested_equal(loaded.optimizer.state_dict(), expected_optimizer)
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


def test_trainer_load_rejects_legacy_trainer_checkpoint_without_version(
    tmp_path,
) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "legacy_trainer.pt"
    torch.save(
        {
            "checkpoint_type": "trainer",
            "agent": trainer.agent.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "global_step": 0,
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="unsupported format_version=None"):
        trainer.load(str(checkpoint_path))


def test_trainer_load_rejects_wrong_checkpoint_format_version(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "trainer.pt"
    trainer.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["format_version"] = TRAINER_CHECKPOINT_FORMAT_VERSION + 1
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="unsupported format_version"):
        trainer.load(str(checkpoint_path))


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_trainer_load_rejects_model_state_that_disagrees_with_declared_schema(
    tmp_path,
    corruption: str,
) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / f"trainer_{corruption}.pt"
    trainer.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if corruption == "missing":
        checkpoint["agent"].pop("linear.bias")
    elif corruption == "unexpected":
        checkpoint["agent"]["unexpected"] = torch.zeros(1)
    else:
        checkpoint["agent"]["linear.weight"] = checkpoint["agent"]["linear.weight"][:1]
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="declared model-state schema"):
        trainer.load(str(checkpoint_path))


def test_trainer_load_rejects_current_model_architecture_mismatch(tmp_path) -> None:
    source = PPOTrainer(TinyAgent(width=2), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "trainer.pt"
    source.save(str(checkpoint_path))
    target = PPOTrainer(TinyAgent(width=3), TrainingConfig(), torch.device("cpu"))
    expected_target_state = copy.deepcopy(target.agent.state_dict())

    with pytest.raises(ValueError, match="model-state schema mismatch"):
        target.load(str(checkpoint_path))
    _assert_nested_equal(target.agent.state_dict(), expected_target_state)


def test_trainer_load_rejects_model_config_mismatch(tmp_path) -> None:
    source = PPOTrainer(
        TinyAgent(config=TinyModelConfig(width=2)),
        TrainingConfig(),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer.pt"
    source.save(str(checkpoint_path))
    target = PPOTrainer(
        TinyAgent(config=TinyModelConfig(width=3)),
        TrainingConfig(),
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="model config mismatch"):
        target.load(str(checkpoint_path))


def test_trainer_load_rejects_game_data_vocab_mismatch(tmp_path) -> None:
    source_agent = TinyAgent()
    source_agent.game_data_signature = "vocab-a"
    source = PPOTrainer(source_agent, TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "trainer.pt"
    source.save(str(checkpoint_path))

    target_agent = TinyAgent()
    target_agent.game_data_signature = "vocab-b"
    target = PPOTrainer(target_agent, TrainingConfig(), torch.device("cpu"))

    with pytest.raises(ValueError, match="game-data vocabulary mismatch"):
        target.load(str(checkpoint_path))


def test_league_checkpoint_is_marked_as_league(tmp_path) -> None:
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league = object.__new__(League)
    league.config = config
    league.checkpoints = deque(maxlen=config.league_size)
    league._inference = SimpleNamespace(
        prune_checkpoints=lambda keep_paths: None,
        prefetch_checkpoint=lambda checkpoint_path: None,
    )

    league.add_checkpoint(TinyAgent(), step=7)

    checkpoint = torch.load(
        tmp_path / "league" / "step_7.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["checkpoint_type"] == "league"


def test_league_sampling_uses_its_own_reproducible_rng() -> None:
    left = object.__new__(League)
    right = object.__new__(League)
    left.checkpoints = deque((f"step_{index}.pt" for index in range(12)), maxlen=12)
    right.checkpoints = deque(left.checkpoints, maxlen=12)
    left._rng = random.Random(37)
    right._rng = random.Random(37)

    left_samples = [left._sample_checkpoint() for _ in range(20)]
    random.seed(999)
    _ = [random.random() for _ in range(100)]
    right_samples = [right._sample_checkpoint() for _ in range(20)]

    assert left_samples == right_samples
