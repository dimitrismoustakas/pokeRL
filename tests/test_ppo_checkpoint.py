from collections import deque
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.training.league import League
from pokerl.training.ppo import PPOTrainer


class TinyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)


def test_trainer_load_rejects_league_only_checkpoint(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "league_step.pt"
    torch.save({"checkpoint_type": "league", "agent": trainer.agent.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint_type='trainer'"):
        trainer.load(str(checkpoint_path))


def test_trainer_load_rejects_legacy_agent_only_checkpoint(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "legacy_league_step.pt"
    torch.save({"agent": trainer.agent.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="League checkpoints"):
        trainer.load(str(checkpoint_path))


def test_trainer_checkpoint_roundtrips(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    trainer.global_step = 123
    checkpoint_path = tmp_path / "trainer.pt"

    trainer.save(str(checkpoint_path))
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert raw_checkpoint["checkpoint_type"] == "trainer"

    loaded = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    loaded.load(str(checkpoint_path))

    assert loaded.global_step == 123


def test_league_checkpoint_is_marked_as_league(tmp_path) -> None:
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league = object.__new__(League)
    league.config = config
    league.checkpoints = deque(maxlen=config.league_size)
    league._inference = SimpleNamespace(prune_checkpoints=lambda keep_paths: None)

    league.add_checkpoint(TinyAgent(), step=7)

    checkpoint = torch.load(
        tmp_path / "league" / "step_7.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["checkpoint_type"] == "league"
