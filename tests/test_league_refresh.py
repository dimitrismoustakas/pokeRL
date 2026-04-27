from pathlib import Path
from types import SimpleNamespace

import scripts.train as train_module
from pokerl.config import EnvConfig
from scripts.train import (
    _league_state_path_for_resume,
    _sample_refresh_assignments,
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

    class FakeBattleEnv:
        def __init__(self, config, game_data, opponent_factory, **kwargs):
            self.config = config
            self.opponent = opponent_factory("loop-token")
            created_envs.append(self)

    class FakeParallelBattleEnv:
        def __init__(self, envs, collect_timing=False):
            self.envs = envs

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
        EnvConfig(num_envs=4),
        game_data=object(),
        league=league,
        opponent_groups=2,
        team_preview_fn=lambda obs: None,
    )

    assert parallel_env.envs == created_envs
    assert [env.config.opponent_group for env in created_envs] == [0, 1, 0, 1]
    assert make_opponent_calls == ["step_a.pt", "step_b.pt", "step_a.pt", "step_b.pt"]


def test_make_league_opponent_factory_passes_forced_checkpoint() -> None:
    calls: list[tuple[object, str]] = []

    def make_opponent(loop, checkpoint_path=None):
        calls.append((loop, checkpoint_path))
        return object()

    league = SimpleNamespace(make_opponent=make_opponent)

    factory = make_league_opponent_factory(league, checkpoint_path="step_123.pt")
    factory("loop-token")

    assert calls == [("loop-token", "step_123.pt")]


def test_league_state_path_for_resume_uses_checkpoint_parent() -> None:
    resume_path = r"C:\tmp\run_a\final.pt"

    assert _league_state_path_for_resume(resume_path) == Path(resume_path).with_name("league_state.json")
