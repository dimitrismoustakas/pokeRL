from collections import deque
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

import scripts.train as train_module
from pokerl.config import EnvConfig, TrainingConfig
from pokerl.training.league import League
from scripts.train import (
    _league_state_path_for_resume,
    _maintain_league_after_update,
    _record_opponent_outcome,
    _sample_refresh_assignments,
    _summarize_opponent_outcomes,
    make_league_opponent_factory,
    make_envs,
)


def test_sample_refresh_assignments_groups_by_opponent_group() -> None:
    envs = [
        SimpleNamespace(config=SimpleNamespace(opponent_group=0)),
        SimpleNamespace(config=SimpleNamespace(opponent_group=1)),
        SimpleNamespace(config=SimpleNamespace(opponent_group=0)),
        SimpleNamespace(config=SimpleNamespace(opponent_group=1)),
    ]
    parallel_env = SimpleNamespace(envs=envs)
    samples = iter(["step_a.pt", "step_b.pt"])
    league = SimpleNamespace(sample_checkpoint_path=lambda: next(samples))

    assignments = _sample_refresh_assignments(parallel_env, league)

    assert assignments == [
        (envs[0], "step_a.pt"),
        (envs[1], "step_b.pt"),
        (envs[2], "step_a.pt"),
        (envs[3], "step_b.pt"),
    ]


def test_make_envs_groups_initial_opponent_checkpoints(monkeypatch) -> None:
    created_envs = []

    class FakeLoopThread:
        created = []

        def __init__(self, index: int):
            self.loop = f"loop-{index}"
            self.closed = False

        @classmethod
        def start(cls):
            loop_thread = cls(len(cls.created))
            cls.created.append(loop_thread)
            return loop_thread

        def close(self):
            self.closed = True

    class FakeBattleEnv:
        def __init__(self, config, game_data, opponent_factory, **kwargs):
            self.config = config
            self.loop_thread = kwargs["loop_thread"]
            self.opponent = opponent_factory(self.loop_thread.loop)
            created_envs.append(self)

    class FakeParallelBattleEnv:
        def __init__(self, envs, collect_timing=False, loop_threads=None):
            self.envs = envs
            self.loop_threads = loop_threads

    monkeypatch.setattr(train_module, "LoopThread", FakeLoopThread)
    monkeypatch.setattr(train_module, "BattleEnv", FakeBattleEnv)
    monkeypatch.setattr(train_module, "ParallelBattleEnv", FakeParallelBattleEnv)

    samples = iter(["step_a.pt", "step_b.pt"])
    make_opponent_calls: list[str] = []

    def make_opponent(loop, checkpoint_path=None):
        make_opponent_calls.append(checkpoint_path)
        return checkpoint_path

    league = SimpleNamespace(
        sample_checkpoint_path=lambda: next(samples),
        make_opponent=make_opponent,
    )

    parallel_env = make_envs(
        EnvConfig(num_envs=4, battle_loop_count=2),
        game_data=object(),
        league=league,
        opponent_groups=2,
        team_preview_fn=lambda obs: None,
    )

    assert parallel_env.envs == created_envs
    assert parallel_env.loop_threads == FakeLoopThread.created
    assert [env.loop_thread for env in created_envs] == [
        FakeLoopThread.created[0],
        FakeLoopThread.created[1],
        FakeLoopThread.created[0],
        FakeLoopThread.created[1],
    ]
    assert [env.config.opponent_group for env in created_envs] == [0, 1, 0, 1]
    assert make_opponent_calls == ["step_a.pt", "step_b.pt", "step_a.pt", "step_b.pt"]


def test_league_snapshot_is_added_before_refresh_sampling() -> None:
    events: list[str] = []

    class FakeLeague:
        checkpoints = ["old.pt"]

        def add_checkpoint(self, _agent, step: int) -> None:
            events.append(f"add:{step}")
            self.checkpoints.append("new.pt")

        def sample_checkpoint_path(self) -> str:
            events.append("sample")
            return self.checkpoints[-1]

    class FakeEnv:
        config = SimpleNamespace(opponent_group=0)

        def stage_next_opponent_checkpoint(self, checkpoint_path: str) -> bool:
            events.append(f"stage:{checkpoint_path}")
            return True

    maintenance = _maintain_league_after_update(
        update=10,
        num_updates=20,
        global_step=123,
        agent=object(),
        parallel_env=SimpleNamespace(envs=[FakeEnv()]),
        league=FakeLeague(),
        config=SimpleNamespace(
            league_update_interval=10,
            league_refresh_interval=1,
        ),
    )

    assert events == ["add:123", "sample", "stage:new.pt"]
    assert maintenance.checkpoint_added is True
    assert maintenance.assignments_by_group == {0: "new.pt"}


def test_refresh_cadence_is_independent_of_snapshot_cadence() -> None:
    events: list[str] = []

    class FakeLeague:
        def add_checkpoint(self, _agent, step: int) -> None:
            events.append(f"add:{step}")

        def sample_checkpoint_path(self) -> str:
            events.append("sample")
            return "existing.pt"

    class FakeEnv:
        config = SimpleNamespace(opponent_group=0)

        def stage_next_opponent_checkpoint(self, checkpoint_path: str) -> bool:
            events.append(f"stage:{checkpoint_path}")
            return True

    maintenance = _maintain_league_after_update(
        update=2,
        num_updates=20,
        global_step=123,
        agent=object(),
        parallel_env=SimpleNamespace(envs=[FakeEnv()]),
        league=FakeLeague(),
        config=SimpleNamespace(
            league_update_interval=10,
            league_refresh_interval=1,
        ),
    )

    assert events == ["sample", "stage:existing.pt"]
    assert maintenance.checkpoint_added is False
    assert maintenance.assignments_by_group == {0: "existing.pt"}


def test_opponent_outcomes_include_draws_unknown_paths_and_totals() -> None:
    counts = {}
    _record_opponent_outcome(counts, "step_a.pt", 1.0)
    _record_opponent_outcome(counts, "step_a.pt", -1.0)
    _record_opponent_outcome(counts, "step_b.pt", 0.0)
    _record_opponent_outcome(counts, None, 1.0)

    summary = _summarize_opponent_outcomes(counts)

    assert summary["games"] == 4
    assert summary["wins"] == 2
    assert summary["losses"] == 1
    assert summary["draws"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["unattributed_games"] == 1
    assert summary["by_checkpoint"] == [
        {
            "checkpoint_path": "step_a.pt",
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "win_rate": 0.5,
        },
        {
            "checkpoint_path": "step_b.pt",
            "games": 1,
            "wins": 0,
            "losses": 0,
            "draws": 1,
            "win_rate": 0.0,
        },
        {
            "checkpoint_path": None,
            "games": 1,
            "wins": 1,
            "losses": 0,
            "draws": 0,
            "win_rate": 1.0,
        },
    ]


def test_make_envs_closes_partial_envs_and_all_loops_after_failure(monkeypatch) -> None:
    events: list[str] = []

    class FakeLoopThread:
        created = []

        def __init__(self, index: int):
            self.index = index
            self.loop = object()

        @classmethod
        def start(cls):
            loop_thread = cls(len(cls.created))
            cls.created.append(loop_thread)
            return loop_thread

        def close(self):
            events.append(f"loop-{self.index}")

    class FakeBattleEnv:
        created = 0

        def __init__(self, *_args, **_kwargs):
            self.index = type(self).created
            type(self).created += 1
            if self.index == 2:
                raise RuntimeError("injected environment construction failure")

        def close(self):
            events.append(f"env-{self.index}")

    monkeypatch.setattr(train_module, "LoopThread", FakeLoopThread)
    monkeypatch.setattr(train_module, "BattleEnv", FakeBattleEnv)
    league = SimpleNamespace(sample_checkpoint_path=lambda: "step_a.pt")

    with pytest.raises(RuntimeError, match="injected environment"):
        make_envs(
            EnvConfig(num_envs=4, battle_loop_count=2),
            game_data=object(),
            league=league,
            opponent_groups=1,
            team_preview_fn=lambda obs: None,
        )

    assert events == ["env-1", "env-0", "loop-0", "loop-1"]


def test_make_envs_closes_loops_created_before_loop_start_failure(monkeypatch) -> None:
    events: list[str] = []

    class FakeLoopThread:
        calls = 0

        @classmethod
        def start(cls):
            cls.calls += 1
            if cls.calls == 2:
                raise RuntimeError("injected loop construction failure")
            return SimpleNamespace(close=lambda: events.append("loop-0"))

    monkeypatch.setattr(train_module, "LoopThread", FakeLoopThread)
    league = SimpleNamespace(sample_checkpoint_path=lambda: "step_a.pt")

    with pytest.raises(RuntimeError, match="injected loop"):
        make_envs(
            EnvConfig(num_envs=4, battle_loop_count=2),
            game_data=object(),
            league=league,
            opponent_groups=1,
            team_preview_fn=lambda obs: None,
        )

    assert events == ["loop-0"]


def test_make_league_opponent_factory_passes_forced_checkpoint() -> None:
    calls: list[tuple[object, str]] = []

    def make_opponent(loop, checkpoint_path=None):
        calls.append((loop, checkpoint_path))
        return object()

    league = SimpleNamespace(make_opponent=make_opponent)

    factory = make_league_opponent_factory(league, checkpoint_path="step_123.pt")
    factory("loop-token")

    assert calls == [("loop-token", "step_123.pt")]


def test_league_state_path_for_resume_is_paired_with_exact_checkpoint() -> None:
    resume_path = r"C:\tmp\run_a\final.pt"

    assert _league_state_path_for_resume(resume_path) == Path(resume_path).with_suffix(
        ".league_state.json"
    )


def test_league_state_roundtrips_membership_rng_and_extra_state(tmp_path) -> None:
    checkpoint_paths = [tmp_path / f"step_{index}.pt" for index in range(3)]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.touch()

    source = object.__new__(League)
    source.config = TrainingConfig(league_size=3)
    source.checkpoints = deque(
        (str(path) for path in checkpoint_paths),
        maxlen=3,
    )
    source._rng = random.Random(37)
    source._inference = SimpleNamespace(prune_checkpoints=lambda paths: None)
    state_path = tmp_path / "step_3.league_state.json"
    source.save_state(
        str(state_path),
        extra_state={"team_pool_rng_state": random.Random(91).getstate()},
    )
    expected_samples = [source._sample_checkpoint() for _ in range(12)]

    target = object.__new__(League)
    target.config = TrainingConfig(league_size=3)
    target.checkpoints = deque(maxlen=3)
    target._rng = random.Random(999)
    target._inference = SimpleNamespace(prune_checkpoints=lambda paths: None)
    extra_state = target.load_state(str(state_path))

    assert list(target.checkpoints) == [str(path) for path in checkpoint_paths]
    assert [target._sample_checkpoint() for _ in range(12)] == expected_samples
    assert isinstance(extra_state["team_pool_rng_state"], tuple)
    assert not state_path.with_suffix(".json.tmp").exists()
