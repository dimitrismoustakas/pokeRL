import argparse
from pathlib import Path

import pytest
from poke_env import LocalBattleStreamConfiguration

from pokerl.config import EnvConfig
from pokerl.env.showdown_env import (
    battlestream_summary,
    make_battlestream_configuration,
    resolve_battle_loop_count,
    resolve_reset_concurrency,
)
from scripts.train import (
    apply_torch_runtime_config,
    build_configs,
    parse_args,
    resolve_opponent_groups,
)


def test_make_battlestream_configuration_uses_configured_workers(
    tmp_path: Path,
) -> None:
    config = EnvConfig(
        num_envs=72,
        showdown_dir=str(tmp_path),
        worker_count=16,
        battles_per_worker=9,
        runtime_loop_count=4,
        battlestream_startup_timeout=55.0,
    )

    battlestream_config = make_battlestream_configuration(config)

    assert isinstance(battlestream_config, LocalBattleStreamConfiguration)
    assert battlestream_config.showdown_dir == tmp_path
    assert battlestream_config.worker_count == 16
    assert battlestream_config.max_battles_per_worker == 9
    assert battlestream_config.runtime_loop_count == 4
    assert battlestream_config.startup_timeout == 55.0


def test_battlestream_defaults_use_bounded_shared_workers(tmp_path: Path) -> None:
    config = EnvConfig(num_envs=72, showdown_dir=str(tmp_path))

    summary = battlestream_summary(config)

    assert summary["worker_count"] == 24
    assert summary["max_battles_per_worker"] == 3
    assert summary["runtime_loop_count"] == 2
    assert summary["battle_loop_count"] == 4
    assert summary["startup_timeout"] == 30.0
    assert summary["reset_concurrency"] == 72


def test_battlestream_explicit_shared_worker_count_defaults_to_required_capacity(
    tmp_path: Path,
) -> None:
    config = EnvConfig(num_envs=72, showdown_dir=str(tmp_path), worker_count=16)

    summary = battlestream_summary(config)

    assert summary["worker_count"] == 16
    assert summary["max_battles_per_worker"] == 5
    assert summary["reset_concurrency"] == 72


def test_battlestream_summary_reports_explicit_reset_concurrency(
    tmp_path: Path,
) -> None:
    config = EnvConfig(
        num_envs=72,
        showdown_dir=str(tmp_path),
        worker_count=16,
        reset_concurrency=32,
    )

    summary = battlestream_summary(config)

    assert summary["worker_count"] == 16
    assert summary["reset_concurrency"] == 32


def test_battlestream_configuration_requires_showdown_dir() -> None:
    with pytest.raises(ValueError, match="showdown_dir"):
        make_battlestream_configuration(EnvConfig())


def test_battle_loop_count_defaults_and_explicit_value() -> None:
    assert resolve_battle_loop_count(None, num_envs=1) == 1
    assert resolve_battle_loop_count(None, num_envs=72) == 4
    assert resolve_battle_loop_count(7, num_envs=72) == 7


@pytest.mark.parametrize("value", [0, -1, True])
def test_battle_loop_count_rejects_non_positive_integers(value) -> None:
    with pytest.raises(ValueError, match="battle_loop_count"):
        resolve_battle_loop_count(value, num_envs=72)


def test_battle_loop_count_cli_override(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--num-envs", "8", "--battle-loop-count", "3"],
    )

    _, env_config, _ = build_configs(parse_args())

    assert env_config.battle_loop_count == 3


def test_battle_loop_count_cli_rejects_zero(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train", "--battle-loop-count", "0"])

    with pytest.raises(ValueError, match="battle_loop_count"):
        build_configs(parse_args())


def test_reset_concurrency_defaults_clamps_and_accepts_explicit_value() -> None:
    assert resolve_reset_concurrency(None, num_envs=72) == 72
    assert resolve_reset_concurrency(32, num_envs=72) == 32
    assert resolve_reset_concurrency(100, num_envs=72) == 72


@pytest.mark.parametrize("value", [0, -1, True])
def test_reset_concurrency_rejects_non_positive_integers(value) -> None:
    with pytest.raises(ValueError, match="reset_concurrency"):
        resolve_reset_concurrency(value, num_envs=72)


def test_reset_concurrency_cli_rejects_zero(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train", "--reset-concurrency", "0"])

    with pytest.raises(ValueError, match="reset_concurrency"):
        build_configs(parse_args())


def test_float32_matmul_precision_defaults_to_highest(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train"])

    assert parse_args().float32_matmul_precision == "highest"


def test_resolve_opponent_groups_defaults_to_two_when_possible() -> None:
    assert resolve_opponent_groups(None, num_envs=72) == 2
    assert resolve_opponent_groups(None, num_envs=1) == 1
    assert resolve_opponent_groups(4, num_envs=2) == 2


def test_apply_torch_runtime_config_rejects_non_positive_threads() -> None:
    args = argparse.Namespace(
        float32_matmul_precision="high",
        torch_num_threads=0,
        torch_num_interop_threads=None,
    )

    with pytest.raises(ValueError, match="torch-num-threads"):
        apply_torch_runtime_config(args)
