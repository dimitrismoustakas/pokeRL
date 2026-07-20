import copy
from dataclasses import dataclass
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.model.checkpoint import (
    LEAGUE_MODEL_CHECKPOINT_FORMAT_VERSION,
    MODEL_CHECKPOINT_METADATA_FIELDS,
    model_checkpoint_metadata,
)
from pokerl.training.league import League, LeagueMember, _LeagueInferenceService
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
    trainer = PPOTrainer(
        TinyAgent(),
        TrainingConfig(
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
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert raw_checkpoint["checkpoint_type"] == "trainer"
    assert raw_checkpoint["format_version"] == TRAINER_CHECKPOINT_FORMAT_VERSION
    assert raw_checkpoint["tera_exploration_schedule"] == {
        "start": 0.3,
        "end": 0.05,
        "decay_updates": 113,
    }
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
    assert loaded.completed_updates == 7
    assert loaded.config.tera_exploration_start == pytest.approx(0.3)
    assert loaded.config.tera_exploration_end == pytest.approx(0.05)
    assert loaded.config.tera_exploration_decay_updates == 113
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


def test_trainer_checkpoint_requires_resolved_tera_schedule(tmp_path) -> None:
    trainer = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))

    with pytest.raises(ValueError, match="tera_exploration_decay_updates"):
        trainer.save(str(tmp_path / "trainer.pt"))


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_trainer_resume_rejects_invalid_tera_schedule(tmp_path, value) -> None:
    source = PPOTrainer(
        TinyAgent(),
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer.pt"
    source.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["tera_exploration_schedule"]["decay_updates"] = value
    torch.save(checkpoint, checkpoint_path)

    target = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    with pytest.raises(ValueError, match="tera_exploration_schedule"):
        target.load(str(checkpoint_path))


def test_weight_only_initializer_loads_exact_model_with_fresh_training_state(
    tmp_path,
) -> None:
    source = PPOTrainer(
        TinyAgent(),
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    loss = source.agent.linear(torch.tensor([[0.5, -1.0]])).square().sum()
    assert source._apply_update_step(loss)
    source.global_step = 1234
    source.completed_updates = 19
    expected_agent = copy.deepcopy(source.agent.state_dict())
    checkpoint_path = tmp_path / "source.pt"
    source.save(str(checkpoint_path), training_state_id="source-state")

    target = PPOTrainer(
        TinyAgent(),
        TrainingConfig(
            tera_exploration_start=0.2,
            tera_exploration_end=0.01,
            tera_exploration_decay_updates=40,
        ),
        torch.device("cpu"),
    )
    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    target.initialize_from_trainer_weights(str(checkpoint_path))
    actual_random = [random.random() for _ in range(3)]
    actual_numpy = np.random.random(3)
    actual_torch = torch.rand(3)

    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    assert actual_random == [random.random() for _ in range(3)]
    np.testing.assert_array_equal(actual_numpy, np.random.random(3))
    torch.testing.assert_close(actual_torch, torch.rand(3), rtol=0.0, atol=0.0)
    _assert_nested_equal(target.agent.state_dict(), expected_agent)
    assert target.optimizer.state_dict()["state"] == {}
    assert target.global_step == 0
    assert target.completed_updates == 0
    assert target.config.tera_exploration_start == pytest.approx(0.2)
    assert target.config.tera_exploration_end == pytest.approx(0.01)
    assert target.config.tera_exploration_decay_updates == 40
    assert target.loaded_training_state_id is None
    assert target.initialized_from_training_state_id == "source-state"


def test_weight_only_initializer_rejects_league_checkpoint(tmp_path) -> None:
    target = PPOTrainer(TinyAgent(), TrainingConfig(), torch.device("cpu"))
    checkpoint_path = tmp_path / "league.pt"
    torch.save(
        {"checkpoint_type": "league", "agent": target.agent.state_dict()},
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="versioned trainer checkpoint"):
        target.initialize_from_trainer_weights(str(checkpoint_path))


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
    trainer = PPOTrainer(
        TinyAgent(),
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer.pt"
    trainer.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["format_version"] = TRAINER_CHECKPOINT_FORMAT_VERSION + 1
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="unsupported format_version"):
        trainer.load(str(checkpoint_path))


def test_trainer_load_rejects_previous_v4_checkpoint_schema(tmp_path) -> None:
    trainer = PPOTrainer(
        TinyAgent(),
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer_v4.pt"
    trainer.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["format_version"] = 4
    checkpoint.pop("completed_updates")
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="unsupported format_version=4"):
        trainer.load(str(checkpoint_path))


@pytest.mark.parametrize("completed_updates", [True, -1, 1.5])
def test_trainer_load_rejects_invalid_completed_updates(
    tmp_path,
    completed_updates,
) -> None:
    trainer = PPOTrainer(
        TinyAgent(),
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer.pt"
    trainer.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint["completed_updates"] = completed_updates
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="completed_updates"):
        trainer.load(str(checkpoint_path))


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_trainer_load_rejects_model_state_that_disagrees_with_declared_schema(
    tmp_path,
    corruption: str,
) -> None:
    trainer = PPOTrainer(
        TinyAgent(),
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
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
    source = PPOTrainer(
        TinyAgent(width=2),
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
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
        TrainingConfig(tera_exploration_decay_updates=113),
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
    source = PPOTrainer(
        source_agent,
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer.pt"
    source.save(str(checkpoint_path))

    target_agent = TinyAgent()
    target_agent.game_data_signature = "vocab-b"
    target = PPOTrainer(target_agent, TrainingConfig(), torch.device("cpu"))

    with pytest.raises(ValueError, match="game-data vocabulary mismatch"):
        target.load(str(checkpoint_path))


def test_trainer_load_rejects_architecture_signature_mismatch(tmp_path) -> None:
    source_agent = TinyAgent()
    source_agent.architecture_signature = "architecture-a"
    source_agent.model_signature = "model-a"
    source = PPOTrainer(
        source_agent,
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer.pt"
    source.save(str(checkpoint_path))

    target_agent = TinyAgent()
    target_agent.architecture_signature = "architecture-b"
    target_agent.model_signature = "model-b"
    target = PPOTrainer(target_agent, TrainingConfig(), torch.device("cpu"))

    with pytest.raises(ValueError, match="architecture signature mismatch"):
        target.load(str(checkpoint_path))


def test_trainer_load_rejects_complete_model_signature_mismatch(tmp_path) -> None:
    source_agent = TinyAgent()
    source_agent.architecture_signature = "same-architecture"
    source_agent.model_signature = "model-a"
    source = PPOTrainer(
        source_agent,
        TrainingConfig(tera_exploration_decay_updates=113),
        torch.device("cpu"),
    )
    checkpoint_path = tmp_path / "trainer.pt"
    source.save(str(checkpoint_path))

    target_agent = TinyAgent()
    target_agent.architecture_signature = "same-architecture"
    target_agent.model_signature = "model-b"
    target = PPOTrainer(target_agent, TrainingConfig(), torch.device("cpu"))

    with pytest.raises(ValueError, match="model signature mismatch"):
        target.load(str(checkpoint_path))


def test_league_checkpoint_is_marked_as_league(tmp_path) -> None:
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league = object.__new__(League)
    league.config = config
    league.checkpoints = []
    league._members = {}
    league._current_update = 0
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
    assert checkpoint["format_version"] == LEAGUE_MODEL_CHECKPOINT_FORMAT_VERSION
    assert set(checkpoint["model_metadata"]) == MODEL_CHECKPOINT_METADATA_FIELDS


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


def test_league_inference_rejects_model_signature_mismatch(tmp_path) -> None:
    source = TinyAgent()
    source.architecture_signature = "same-architecture"
    source.model_signature = "source-model"
    target = TinyAgent()
    target.architecture_signature = "same-architecture"
    target.model_signature = "target-model"
    checkpoint_path = tmp_path / "league.pt"
    torch.save(
        {
            "checkpoint_type": "league",
            "format_version": LEAGUE_MODEL_CHECKPOINT_FORMAT_VERSION,
            "model_metadata": model_checkpoint_metadata(source),
            "agent": source.state_dict(),
        },
        checkpoint_path,
    )
    service = object.__new__(_LeagueInferenceService)
    service._agent_template = target
    service._device = torch.device("cpu")

    with pytest.raises(ValueError, match="model signature mismatch"):
        service._load_agent(str(checkpoint_path))


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
            checkpoint_sha256=f"{index:064x}",
            archive_source="self_play_snapshot",
            step=index,
            created_update=0,
        )
        for index, path in enumerate(left.checkpoints)
    }
    right._members = {
        path: LeagueMember(
            path=path,
            checkpoint_sha256=f"{index:064x}",
            archive_source="self_play_snapshot",
            step=index,
            created_update=0,
        )
        for index, path in enumerate(right.checkpoints)
    }
    left._current_update = 0
    right._current_update = 0
    left._rng = random.Random(37)
    right._rng = random.Random(37)

    left_samples = [left._sample_checkpoint() for _ in range(20)]
    random.seed(999)
    _ = [random.random() for _ in range(100)]
    right_samples = [right._sample_checkpoint() for _ in range(20)]

    assert left_samples == right_samples
