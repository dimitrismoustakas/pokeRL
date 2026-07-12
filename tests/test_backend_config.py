import argparse
from pathlib import Path

import pytest
from poke_env import LocalBattleStreamConfiguration

from pokerl.config import EnvConfig, TrainingConfig
from pokerl.env.showdown_env import (
    battlestream_summary,
    make_battlestream_configuration,
    resolve_battle_loop_count,
    resolve_reset_concurrency,
)
from scripts.train import (
    apply_torch_runtime_config,
    build_configs,
    configure_training_randomness,
    parse_args,
    resolve_opponent_groups,
    resolve_tera_exploration_decay_updates,
    resolve_update_bounds,
    tera_exploration_probability,
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


def test_memory_gradient_diagnostics_are_opt_in(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train", "--diagnose-memory-gradients"])

    _, _, train_config = build_configs(parse_args())

    assert train_config.diagnose_memory_gradients is True


def test_league_refresh_defaults_to_every_update(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train"])

    _, _, train_config = build_configs(parse_args())

    assert train_config.league_update_interval == 10
    assert train_config.league_refresh_interval == 1


def test_league_refresh_cli_is_independent_of_snapshot_interval(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--league-update-interval",
            "12",
            "--league-refresh-interval",
            "3",
        ],
    )

    _, _, train_config = build_configs(parse_args())

    assert train_config.league_update_interval == 12
    assert train_config.league_refresh_interval == 3


def test_factorized_tera_cli_overrides(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--tera-exploration-end",
            "0.01",
            "--tera-exploration-decay-fraction",
            "0.4",
            "--tera-gate-entropy-coef",
            "0.02",
            "--tera-slot-entropy-coef",
            "0.03",
        ],
    )

    _, _, config = build_configs(parse_args())

    assert config.tera_exploration_end == pytest.approx(0.01)
    assert config.tera_exploration_decay_fraction == pytest.approx(0.4)
    assert config.tera_gate_entropy_coef == pytest.approx(0.02)
    assert config.tera_slot_entropy_coef == pytest.approx(0.03)


def test_factorized_tera_defaults_leave_unmodified_training_tail() -> None:
    config = TrainingConfig()

    assert config.tera_exploration_start == pytest.approx(0.15)
    assert config.tera_exploration_end == 0.0
    assert config.tera_exploration_decay_fraction == pytest.approx(0.25)


def test_tera_exploration_decay_uses_planned_update_horizon() -> None:
    decay_updates = resolve_tera_exploration_decay_updates(100, 0.25)

    assert decay_updates == 25
    assert tera_exploration_probability(
        start=0.15,
        end=0.0,
        update=1,
        decay_updates=decay_updates,
    ) == pytest.approx(0.15)
    assert tera_exploration_probability(
        start=0.15,
        end=0.0,
        update=13,
        decay_updates=decay_updates,
    ) == pytest.approx(0.075)
    assert tera_exploration_probability(
        start=0.15,
        end=0.0,
        update=25,
        decay_updates=decay_updates,
    ) == pytest.approx(0.0)
    assert tera_exploration_probability(
        start=0.15,
        end=0.0,
        update=90,
        decay_updates=decay_updates,
    ) == pytest.approx(0.0)


def test_zero_tera_decay_fraction_applies_end_immediately() -> None:
    decay_updates = resolve_tera_exploration_decay_updates(100, 0.0)

    assert decay_updates == 0
    assert (
        tera_exploration_probability(
            start=0.15,
            end=0.0,
            update=1,
            decay_updates=decay_updates,
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "field_name", ["league_update_interval", "league_refresh_interval"]
)
def test_league_intervals_must_be_positive(field_name: str) -> None:
    values = {field_name: 0}

    with pytest.raises(ValueError, match=field_name):
        TrainingConfig(**values)


def test_training_seed_streams_are_recorded_and_separate() -> None:
    streams = configure_training_randomness(41, team_pool_seed=None)

    assert streams == {
        "master": 41,
        "torch": 41,
        "python": 42,
        "numpy": 43,
        "league": 44,
        "team_pool": 45,
    }


def test_explicit_team_pool_seed_overrides_derived_stream() -> None:
    streams = configure_training_randomness(41, team_pool_seed=123)

    assert streams["team_pool"] == 123


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_training_seed_rejects_values_outside_uint32(seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(seed=seed)


def test_num_updates_are_additional_when_resuming() -> None:
    steps_per_update = 9_216

    assert resolve_update_bounds(
        total_timesteps=steps_per_update,
        steps_per_update=steps_per_update,
        global_step=158 * steps_per_update,
        additional_updates=True,
    ) == (158, 159)


def test_total_timestep_target_remains_absolute_when_resuming() -> None:
    steps_per_update = 9_216

    assert resolve_update_bounds(
        total_timesteps=200 * steps_per_update,
        steps_per_update=steps_per_update,
        global_step=158 * steps_per_update,
        additional_updates=False,
    ) == (158, 200)


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
