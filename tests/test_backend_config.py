from pathlib import Path

import pytest
from poke_env import LocalBattleStreamConfiguration

import pokerl.env.showdown_env as showdown_env
from pokerl.config import EnvConfig
from pokerl.env.showdown_env import battlestream_summary, make_battlestream_configuration
from scripts.train import resolve_opponent_groups


def test_make_battlestream_configuration_uses_configured_workers(tmp_path: Path) -> None:
    config = EnvConfig(
        num_envs=72,
        showdown_dir=str(tmp_path),
        worker_count=16,
        battles_per_worker=9,
        pool_mode="shared",
        runtime_loop_count=4,
        battlestream_startup_timeout=55.0,
    )

    battlestream_config = make_battlestream_configuration(config)

    assert isinstance(battlestream_config, LocalBattleStreamConfiguration)
    assert battlestream_config.showdown_dir == tmp_path
    assert battlestream_config.worker_count == 16
    assert battlestream_config.max_battles_per_worker == 9
    assert battlestream_config.pool_mode == "shared"
    assert battlestream_config.runtime_loop_count == 4
    assert battlestream_config.startup_timeout == 55.0


def test_battlestream_defaults_use_bounded_shared_workers(tmp_path: Path) -> None:
    config = EnvConfig(num_envs=72, showdown_dir=str(tmp_path))

    summary = battlestream_summary(config)

    assert summary["worker_count"] == 24
    assert summary["max_battles_per_worker"] == 3
    assert summary["pool_mode"] == "shared"
    assert summary["runtime_loop_count"] == 2
    assert summary["startup_timeout"] == 30.0


def test_battlestream_single_mode_defaults_isolate_workers(tmp_path: Path) -> None:
    config = EnvConfig(num_envs=72, showdown_dir=str(tmp_path), pool_mode="single")

    summary = battlestream_summary(config)

    assert summary["worker_count"] == 72
    assert summary["max_battles_per_worker"] == 1
    assert summary["pool_mode"] == "single"
    assert summary["runtime_loop_count"] == 2


def test_battlestream_explicit_shared_worker_count_defaults_to_required_capacity(
    tmp_path: Path,
) -> None:
    config = EnvConfig(num_envs=72, showdown_dir=str(tmp_path), worker_count=16)

    summary = battlestream_summary(config)

    assert summary["worker_count"] == 16
    assert summary["max_battles_per_worker"] == 5
    assert summary["pool_mode"] == "shared"


def test_battlestream_configuration_requires_showdown_dir() -> None:
    with pytest.raises(ValueError, match="showdown_dir"):
        make_battlestream_configuration(EnvConfig())


def test_battlestream_configuration_requires_local_poke_env_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(showdown_env, "LocalBattleStreamConfiguration", None)
    monkeypatch.setattr(showdown_env, "_LOCAL_BATTLESTREAM_IMPORT_ERROR", ImportError("missing"))

    with pytest.raises(RuntimeError, match="LocalBattleStreamConfiguration"):
        make_battlestream_configuration(EnvConfig(showdown_dir=str(tmp_path)))


def test_resolve_opponent_groups_defaults_to_two_when_possible() -> None:
    assert resolve_opponent_groups(None, num_envs=72) == 2
    assert resolve_opponent_groups(None, num_envs=1) == 1
    assert resolve_opponent_groups(4, num_envs=2) == 2
