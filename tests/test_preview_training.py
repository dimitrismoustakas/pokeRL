from __future__ import annotations

import copy
import sys

import numpy as np
import pytest
import torch

from pokerl.config import TrainingConfig
from pokerl.training.preview import (
    VersionedTeamPreviewTrainingSample,
    filter_current_team_preview_training_samples,
    select_team_preview_training_samples,
    update_preview_then_battle,
)


def _samples(count: int):
    return [
        (
            {"sample_id": np.asarray(index, dtype=np.int64)},
            np.asarray([0, 1, 2, 3], dtype=np.int64),
            float(index % 2),
            np.asarray([0, 1, 2, 3], dtype=np.int64),
        )
        for index in range(count)
    ]


def _sample_ids(selection) -> tuple[int, ...]:
    return tuple(int(sample[0]["sample_id"]) for sample in selection.samples)


def test_preview_selection_is_deterministic_for_seed_and_update() -> None:
    samples = _samples(20)

    first = select_team_preview_training_samples(
        samples,
        7,
        seed=123,
        update=41,
    )
    second = select_team_preview_training_samples(
        samples,
        7,
        seed=123,
        update=41,
    )

    assert _sample_ids(first) == _sample_ids(second)
    assert first.available_count == 20
    assert first.used_count == 7
    assert tuple(sorted(_sample_ids(first))) == _sample_ids(first)


def test_preview_selection_uses_every_current_rollout_sample_below_cap() -> None:
    samples = _samples(5)

    selection = select_team_preview_training_samples(
        samples,
        8,
        seed=7,
        update=3,
    )

    assert selection.samples == tuple(samples)
    assert selection.available_count == 5
    assert selection.used_count == 5


def test_preview_selection_uses_every_sample_without_a_cap() -> None:
    samples = _samples(12)

    selection = select_team_preview_training_samples(
        samples,
        None,
        seed=7,
        update=3,
    )

    assert selection.samples == tuple(samples)
    assert selection.available_count == selection.used_count == 12


def test_preview_selection_is_uniform_without_replacement() -> None:
    samples = _samples(8)
    inclusion_counts = np.zeros(len(samples), dtype=np.int64)
    draws = 4_000
    sample_size = 3

    for seed in range(draws):
        selection = select_team_preview_training_samples(
            samples,
            sample_size,
            seed=seed,
            update=11,
        )
        selected_ids = _sample_ids(selection)
        assert len(selected_ids) == len(set(selected_ids)) == sample_size
        inclusion_counts[list(selected_ids)] += 1

    expected = draws * sample_size / len(samples)
    # A fixed generous bound catches biased truncation while keeping the test
    # deterministic and insensitive to ordinary pseudorandom variation.
    assert np.max(np.abs(inclusion_counts - expected)) < expected * 0.08


@pytest.mark.parametrize("max_samples", [0, -1])
def test_preview_selection_rejects_non_positive_cap(max_samples: int) -> None:
    with pytest.raises(ValueError, match="max_samples"):
        select_team_preview_training_samples(
            _samples(1),
            max_samples,
            seed=0,
            update=0,
        )


def test_preview_policy_filter_excludes_battles_spanning_update_boundary() -> None:
    samples = _samples(4)
    versioned = [
        VersionedTeamPreviewTrainingSample(samples[0], policy_version=6),
        VersionedTeamPreviewTrainingSample(samples[1], policy_version=7),
        VersionedTeamPreviewTrainingSample(samples[2], policy_version=5),
        VersionedTeamPreviewTrainingSample(samples[3], policy_version=7),
    ]

    filtered = filter_current_team_preview_training_samples(
        versioned,
        policy_version=7,
    )

    assert tuple(int(sample[0]["sample_id"]) for sample in filtered.samples) == (
        1,
        3,
    )
    assert filtered.stale_excluded_count == 2


def test_preview_policy_filter_rejects_future_version() -> None:
    versioned = [VersionedTeamPreviewTrainingSample(_samples(1)[0], policy_version=9)]

    with pytest.raises(ValueError, match="newer"):
        filter_current_team_preview_training_samples(
            versioned,
            policy_version=8,
        )


@pytest.mark.parametrize("max_samples", [0, -1])
def test_training_config_rejects_non_positive_preview_cap(max_samples: int) -> None:
    with pytest.raises(ValueError, match="team_preview_max_samples"):
        TrainingConfig(team_preview_max_samples=max_samples)


@pytest.mark.parametrize("minibatch_size", [0, -1])
def test_training_config_rejects_non_positive_preview_minibatch(
    minibatch_size: int,
) -> None:
    with pytest.raises(ValueError, match="preview_update_minibatch_size"):
        TrainingConfig(preview_update_minibatch_size=minibatch_size)


def test_training_config_uses_every_preview_sample_by_default() -> None:
    config = TrainingConfig()

    assert config.team_preview_max_samples is None
    assert config.preview_update_minibatch_size == 512


def test_preview_cli_plumbs_optional_cap_and_update_minibatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.train import build_configs, parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--team-preview-max-samples",
            "77",
            "--preview-update-minibatch-size",
            "19",
        ],
    )

    _, _, config = build_configs(parse_args())

    assert config.team_preview_max_samples == 77
    assert config.preview_update_minibatch_size == 19


def test_preview_update_uses_current_data_before_battle_ppo() -> None:
    events = []

    class _Trainer:
        def update_team_preview(self, data):
            events.append(
                ("preview", tuple(int(item[0]["sample_id"]) for item in data))
            )
            return {"preview_loss": 1.0}

        def update(self, batch):
            events.append(("battle", batch))
            return {"battle_loss": 2.0}

    current_rollout = _samples(12)
    metrics = update_preview_then_battle(
        _Trainer(),
        battle_batch="current-battle-rollout",
        preview_samples=current_rollout,
        max_preview_samples=5,
        seed=9,
        update=17,
    )

    assert [event[0] for event in events] == ["preview", "battle"]
    assert set(events[0][1]).issubset(set(range(len(current_rollout))))
    assert events[1] == ("battle", "current-battle-rollout")
    assert metrics == {
        "preview_loss": 1.0,
        "battle_loss": 2.0,
        "team_preview_samples_available_current": 12.0,
        "team_preview_samples_used": 5.0,
    }


def test_preview_update_records_empty_current_rollout_without_stale_reuse() -> None:
    events = []

    class _Trainer:
        def update_team_preview(self, data):
            raise AssertionError(f"empty rollout must not update preview: {data}")

        def update(self, batch):
            events.append(batch)
            return {}

    metrics = update_preview_then_battle(
        _Trainer(),
        battle_batch="battle",
        preview_samples=[],
        max_preview_samples=10,
        seed=0,
        update=4,
    )

    assert events == ["battle"]
    assert metrics == {
        "team_preview_samples_available_current": 0.0,
        "team_preview_samples_used": 0.0,
    }


def test_preview_update_uses_all_current_samples_without_a_cap() -> None:
    seen = []

    class _Trainer:
        def update_team_preview(self, data):
            seen.extend(int(item[0]["sample_id"]) for item in data)
            return {}

        def update(self, batch):
            return {}

    samples = _samples(13)
    metrics = update_preview_then_battle(
        _Trainer(),
        battle_batch="battle",
        preview_samples=samples,
        max_preview_samples=None,
        seed=0,
        update=4,
    )

    assert seen == list(range(13))
    assert metrics["team_preview_samples_available_current"] == 13.0
    assert metrics["team_preview_samples_used"] == 13.0


class _TinyPreviewAgent(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.score = torch.nn.Linear(2, 1)
        self.bring_scale = torch.nn.Parameter(torch.linspace(-0.4, 0.5, 6))
        self.lead_scale = torch.nn.Parameter(torch.linspace(-0.3, 0.4, 36))
        self.batch_sizes: list[int] = []

    def team_preview(self, obs, *, picks):
        self.batch_sizes.append(int(picks.shape[0]))
        score = self.score(obs["features"]).squeeze(-1)
        pick_signal = picks.to(score.dtype).sum(dim=-1) * 0.01
        return {
            "log_prob": -(score - pick_signal).square(),
            "entropy": torch.sigmoid(score),
            "preview_value": score * 0.7,
            "opponent_bring_logits": score.unsqueeze(-1) * self.bring_scale,
            "opponent_lead_pair_logits": score.unsqueeze(-1) * self.lead_scale,
        }


def _preview_samples(count: int):
    rng = np.random.default_rng(8)
    samples = []
    for index in range(count):
        opponent_picks = np.roll(np.arange(6, dtype=np.int64), index)[:4]
        opponent_mask = np.zeros(6, dtype=np.bool_)
        if index % 3 == 0:
            opponent_mask[-1] = True
        samples.append(
            (
                {
                    "features": rng.normal(size=2).astype(np.float32),
                    "opp_mask": opponent_mask,
                },
                np.asarray([0, 1, 2, 3], dtype=np.int64),
                1.0 if index % 2 else -1.0,
                opponent_picks,
            )
        )
    return samples


def test_chunked_preview_update_matches_full_batch_and_steps_once() -> None:
    from pokerl.training.ppo import PPOTrainer

    torch.manual_seed(13)
    full_agent = _TinyPreviewAgent()
    chunked_agent = copy.deepcopy(full_agent)
    full_trainer = PPOTrainer(
        full_agent,
        TrainingConfig(
            lr=3e-4,
            max_grad_norm=0.05,
            preview_update_minibatch_size=64,
        ),
        torch.device("cpu"),
    )
    chunked_trainer = PPOTrainer(
        chunked_agent,
        TrainingConfig(
            lr=3e-4,
            max_grad_norm=0.05,
            preview_update_minibatch_size=3,
        ),
        torch.device("cpu"),
    )
    samples = _preview_samples(7)

    full_metrics = full_trainer.update_team_preview(samples)
    chunked_metrics = chunked_trainer.update_team_preview(samples)

    assert full_agent.batch_sizes == [7]
    assert chunked_agent.batch_sizes == [3, 3, 1]
    assert chunked_metrics == pytest.approx(full_metrics, rel=2e-6, abs=2e-7)
    for full_parameter, chunked_parameter in zip(
        full_agent.parameters(),
        chunked_agent.parameters(),
        strict=True,
    ):
        assert torch.allclose(
            chunked_parameter,
            full_parameter,
            rtol=2e-6,
            atol=2e-7,
        )
    optimizer_steps = {
        int(state["step"].item()) for state in chunked_trainer.optimizer.state.values()
    }
    assert optimizer_steps == {1}


def test_nonfinite_preview_chunk_discards_accumulated_gradients() -> None:
    from pokerl.training.ppo import PPOTrainer

    torch.manual_seed(19)
    agent = _TinyPreviewAgent()
    trainer = PPOTrainer(
        agent,
        TrainingConfig(preview_update_minibatch_size=2),
        torch.device("cpu"),
    )
    samples = _preview_samples(4)
    samples[2][0]["features"][0] = np.nan
    initial_parameters = [
        parameter.detach().clone() for parameter in agent.parameters()
    ]

    metrics = trainer.update_team_preview(samples)

    assert metrics == {"team_preview_skipped_nonfinite": 1.0}
    assert trainer.optimizer.state == {}
    for parameter, initial in zip(
        agent.parameters(),
        initial_parameters,
        strict=True,
    ):
        assert torch.equal(parameter, initial)
        assert parameter.grad is None
