from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

import pokerl.env.showdown_env as showdown_env_module
from pokerl.env.showdown_env import BattleEnv, ParallelBattleEnv
from pokerl.training.preview import TeamPreviewEpisodeStart, TeamPreviewResult


def _episode_start(sample_id: int = 0) -> TeamPreviewEpisodeStart:
    return TeamPreviewEpisodeStart(
        obs={"sample_id": np.asarray([sample_id], dtype=np.int64)},
        picks=np.asarray([0, 1, 2, 3], dtype=np.int64),
        log_prob=-2.0,
        value=0.25,
        policy_version=7,
        opponent_picks=np.asarray([3, 2, 1, 0], dtype=np.int64),
        opponent_aux_labels={"opp_item": np.asarray([1, 2, 3, 4], dtype=np.int64)},
    )


def test_team_preview_episode_start_owns_copied_payload() -> None:
    obs_value = np.asarray([4, 5], dtype=np.int64)
    picks = np.asarray([0, 1, 2, 3], dtype=np.int64)
    record = TeamPreviewEpisodeStart(
        obs={"context": obs_value},
        picks=picks,
        log_prob=-1.25,
        value=0.5,
        policy_version=3,
        opponent_picks=np.asarray([3, 2, 1, 0], dtype=np.int64),
        opponent_aux_labels={"opp_item": np.asarray([9], dtype=np.int64)},
    )

    obs_value[0] = 99
    picks[0] = 5
    assert record.obs["context"].tolist() == [4, 5]
    assert record.picks.tolist() == [0, 1, 2, 3]
    with pytest.raises(FrozenInstanceError):
        record.value = 1.0


def test_battle_env_finalizes_complete_training_preview_after_both_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = {"context": np.asarray([1, 2], dtype=np.int64)}
    own_picks = np.asarray([0, 1, 2, 3], dtype=np.int64)
    opponent_picks = np.asarray([3, 1, 2, 0], dtype=np.int64)
    result = TeamPreviewResult(
        picks=own_picks,
        policy_version=11,
        log_prob=-2.5,
        value=0.125,
    )
    label_calls = []

    def build_labels(game_data, team_size, packed_team, *, selected_indices):
        label_calls.append((game_data, team_size, packed_team, selected_indices.copy()))
        return {"opp_item": np.asarray([7, 8, 9, 10], dtype=np.int64)}

    monkeypatch.setattr(
        showdown_env_module,
        "build_packed_team_aux_labels",
        build_labels,
    )
    env = SimpleNamespace(
        _episode_team_preview=(obs, result),
        _pending_team_preview_start=None,
        _episode_opponent_preview_picks=opponent_picks,
        _episode_opponent_packed_team="packed-opponent",
        game_data=object(),
        config=SimpleNamespace(team_size=6),
    )

    BattleEnv._finalize_team_preview_start(env)
    record = BattleEnv.take_team_preview_start(env)

    assert record is not None
    assert record.policy_version == 11
    assert record.log_prob == -2.5
    assert record.value == 0.125
    assert record.opponent_picks.tolist() == opponent_picks.tolist()
    assert record.opponent_aux_labels["opp_item"].tolist() == [7, 8, 9, 10]
    assert len(label_calls) == 1
    assert label_calls[0][:3] == (env.game_data, 6, "packed-opponent")
    np.testing.assert_array_equal(label_calls[0][3], opponent_picks)
    assert env._episode_team_preview is None
    assert BattleEnv.take_team_preview_start(env) is None


def test_battle_env_omits_training_record_for_pick_only_callback() -> None:
    env = SimpleNamespace(
        _episode_team_preview=(
            {"context": np.asarray([1], dtype=np.int64)},
            TeamPreviewResult(
                picks=np.asarray([0, 1, 2, 3], dtype=np.int64),
                policy_version=None,
            ),
        ),
        _pending_team_preview_start=None,
    )

    BattleEnv._finalize_team_preview_start(env)

    assert BattleEnv.take_team_preview_start(env) is None


def test_battle_env_keeps_preview_record_without_exact_packed_team() -> None:
    env = SimpleNamespace(
        _episode_team_preview=(
            {"context": np.asarray([1], dtype=np.int64)},
            TeamPreviewResult(
                picks=np.asarray([0, 1, 2, 3], dtype=np.int64),
                policy_version=4,
                log_prob=-2.0,
                value=0.5,
            ),
        ),
        _pending_team_preview_start=None,
        _episode_opponent_preview_picks=np.asarray(
            [3, 2, 1, 0],
            dtype=np.int64,
        ),
        _episode_opponent_packed_team=None,
    )

    BattleEnv._finalize_team_preview_start(env)
    record = BattleEnv.take_team_preview_start(env)

    assert record is not None
    assert record.opponent_aux_labels is None


def test_battle_env_rejects_partial_training_preview_metadata() -> None:
    env = SimpleNamespace(
        _episode_team_preview=(
            {"context": np.asarray([1], dtype=np.int64)},
            TeamPreviewResult(
                picks=np.asarray([0, 1, 2, 3], dtype=np.int64),
                policy_version=4,
            ),
        ),
        _pending_team_preview_start=None,
    )

    with pytest.raises(RuntimeError, match="partial behavior metadata"):
        BattleEnv._finalize_team_preview_start(env)


class _ResetWithPreview:
    def __init__(self, sample_id: int):
        self.sample_id = sample_id
        self.pending = None
        self.config = SimpleNamespace(reset_concurrency=1)

    def reset(self):
        self.pending = _episode_start(self.sample_id)
        return (
            {"obs": np.asarray([self.sample_id], dtype=np.int64)},
            {"mask": np.asarray([True])},
        )

    def step(self, action):
        del action
        return (
            {"obs": np.asarray([-1], dtype=np.int64)},
            {"mask": np.asarray([False])},
            1.0,
            True,
            {},
        )

    def take_team_preview_start(self):
        record = self.pending
        self.pending = None
        return record

    def snapshot_timing(self):
        return {}

    def reset_timing(self):
        return None

    def close(self):
        return None


def test_parallel_env_drains_initial_and_auto_reset_starts_once() -> None:
    parallel = ParallelBattleEnv(
        [_ResetWithPreview(0), _ResetWithPreview(1)],
        reset_concurrency=2,
    )
    try:
        parallel.reset()
        initial = parallel.drain_team_preview_starts()
        assert [env_index for env_index, _record in initial] == [0, 1]
        assert [int(record.obs["sample_id"][0]) for _env_index, record in initial] == [
            0,
            1,
        ]
        assert parallel.drain_team_preview_starts() == []

        parallel.step(np.zeros((2, 3), dtype=np.int64))
        auto_reset = parallel.drain_team_preview_starts()
        assert [env_index for env_index, _record in auto_reset] == [0, 1]
        assert parallel.drain_team_preview_starts() == []
    finally:
        parallel.close()
