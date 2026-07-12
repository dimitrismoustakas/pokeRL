from __future__ import annotations

import numpy as np
import pytest

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
