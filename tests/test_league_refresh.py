from types import SimpleNamespace

from scripts.train import _sample_refresh_assignments, make_league_opponent_factory


def test_sample_refresh_assignments_groups_by_server() -> None:
    envs = [
        SimpleNamespace(config=SimpleNamespace(server_url="localhost:8000")),
        SimpleNamespace(config=SimpleNamespace(server_url="localhost:8001")),
        SimpleNamespace(config=SimpleNamespace(server_url="localhost:8000")),
        SimpleNamespace(config=SimpleNamespace(server_url="localhost:8001")),
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


def test_make_league_opponent_factory_passes_forced_checkpoint() -> None:
    calls: list[tuple[object, str, str]] = []

    def make_opponent(loop, server_url=None, checkpoint_path=None):
        calls.append((loop, server_url, checkpoint_path))
        return object()

    env_config = SimpleNamespace(server_url="localhost:8000")
    league = SimpleNamespace(make_opponent=make_opponent)

    factory = make_league_opponent_factory(env_config, league, checkpoint_path="step_123.pt")
    factory("loop-token")

    assert calls == [("loop-token", "localhost:8000", "step_123.pt")]
