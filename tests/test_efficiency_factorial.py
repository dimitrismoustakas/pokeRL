import json
import math
from collections import Counter

import pytest

import scripts.benchmark_efficiency_factorial as efficiency_factorial
from scripts.benchmark_efficiency_factorial import (
    VARIANT_FACTORS,
    build_retention_decision,
    build_production_schedule,
    build_simulator_schedule,
    deterministic_package_tree_identity,
    ensure_experiment_manifest,
    factorial_effects,
    git_identity,
    planned_run_payload,
    require_clean_git_identity,
    simulator_pair_effects,
    summarize_simulator_summary,
    summarize_training_summary,
    validated_success_marker,
)


def test_williams_schedule_balances_positions_and_carryover() -> None:
    schedule = build_production_schedule(4)

    by_block = {
        block: [
            item.variant
            for item in sorted(
                (row for row in schedule if row.block == block),
                key=lambda row: row.position,
            )
        ]
        for block in range(1, 5)
    }
    positions = Counter((row.variant, row.position) for row in schedule)
    carryover = Counter(
        pair for order in by_block.values() for pair in zip(order, order[1:])
    )

    assert len(schedule) == 16
    assert set(by_block) == {1, 2, 3, 4}
    assert all(
        positions[(variant, position)] == 1
        for variant in VARIANT_FACTORS
        for position in range(1, 5)
    )
    assert len(carryover) == 12
    assert set(carryover.values()) == {1}
    assert all(row.seed is None for row in schedule)
    assert all("seed" not in planned_run_payload(row) for row in schedule)


def test_simulator_schedule_alternates_pair_order() -> None:
    schedule = build_simulator_schedule(6)

    orders = [
        [row.variant for row in schedule if row.block == round_index]
        for round_index in range(1, 7)
    ]

    assert orders == [
        ["Q", "L"],
        ["L", "Q"],
        ["Q", "L"],
        ["L", "Q"],
        ["Q", "L"],
        ["L", "Q"],
    ]
    assert all(isinstance(row.seed, int) for row in schedule)
    assert all(planned_run_payload(row)["seed"] == row.seed for row in schedule)


def test_factorial_effects_recover_multiplicative_main_effects() -> None:
    rows = []
    values = {"Q": 100.0, "L": 110.0, "C": 120.0, "B": 132.0}
    for block in range(1, 5):
        for position, variant in enumerate(("Q", "L", "C", "B"), start=1):
            rows.append(
                {
                    "phase": "production",
                    "block": block,
                    "position": position,
                    "variant": variant,
                    "wall_steps_per_second": values[variant],
                }
            )

    result = factorial_effects(rows, metric="wall_steps_per_second")

    assert result["complete_blocks"] == 4
    assert result["lazy"]["effect_percent"] == pytest.approx(10.0)
    assert result["coalescing"]["effect_percent"] == pytest.approx(20.0)
    assert result["interaction"]["effect_percent"] == pytest.approx(0.0, abs=1e-12)
    assert result["lazy"]["positive_constituents"] == 8
    assert result["coalescing"]["positive_constituents"] == 8


def test_simulator_pair_effects_pair_by_round_not_run_position() -> None:
    rows = [
        {
            "phase": "simulator",
            "block": 1,
            "variant": "Q",
            "env_steps_per_second": 100.0,
        },
        {
            "phase": "simulator",
            "block": 1,
            "variant": "L",
            "env_steps_per_second": 105.0,
        },
        {
            "phase": "simulator",
            "block": 2,
            "variant": "L",
            "env_steps_per_second": 210.0,
        },
        {
            "phase": "simulator",
            "block": 2,
            "variant": "Q",
            "env_steps_per_second": 200.0,
        },
    ]

    result = simulator_pair_effects(rows)

    assert result["complete_pairs"] == 2
    assert result["lazy"]["effect_percent"] == pytest.approx(5.0)
    assert result["lazy"]["positive_pairs"] == 2


def test_simulator_summary_requires_explicit_lazy_order_mode() -> None:
    summary = {
        "config": {},
        "results": {
            "total_env_steps": 2,
            "run_sec": 1.0,
            "env_steps_per_sec": 2.0,
        },
    }

    with pytest.raises(ValueError, match="config.lazy_action_orders"):
        summarize_simulator_summary(summary)

    summary["config"]["lazy_action_orders"] = True
    metrics = summarize_simulator_summary(summary)
    assert metrics["lazy_action_orders"] is True


def _training_update(index: int, *, wakeups: int) -> dict:
    return {
        "update": index,
        "global_step": index * 10,
        "update_wall_sec": 10.0,
        "rollout_sec": 8.0,
        "ppo_sec": 1.0,
        "preview_refresh_sec": 0.25,
        "update_overhead_sec": 0.5,
        "completed_battles": 20,
        "retained_player_battles": 2,
        "retained_opponent_battles": 2,
        "cuda_memory_allocated_bytes": 100,
        "cuda_memory_reserved_bytes": 200,
        "league": {
            "inference_cache": {
                "action": {
                    "cuda_graphs_enabled": True,
                    "coalesce_target_batch_size": 16,
                    "coalesce_target_wakeups": wakeups,
                    "cuda_graph_replays": index * 10,
                    "cuda_graph_fallbacks": 0,
                    "cuda_graph_captures": index,
                    "cuda_graph_real_rows": index * 20,
                    "cuda_graph_padded_rows": index * 2,
                    "cuda_graph_last_error": None,
                    "resident_cuda_graphs": 1,
                }
            }
        },
    }


def test_training_summary_uses_only_post_warmup_rows_and_cache_deltas() -> None:
    summary = {
        "status": "completed",
        "config": {"num_envs": 2, "rollout_steps": 5},
        "updates": [
            _training_update(index, wakeups=index * 3) for index in range(1, 6)
        ],
    }

    metrics = summarize_training_summary(
        summary,
        expected_invocation_updates=5,
        discard_updates=2,
    )

    assert metrics["measured_steps"] == 30
    assert metrics["wall_steps_per_second"] == pytest.approx(1.0)
    assert metrics["rollout_steps_per_second"] == pytest.approx(1.25)
    assert metrics["rollout_battles_per_second"] == pytest.approx(2.5)
    assert metrics["coalesce_target_wakeups"] == 9
    assert metrics["cuda_graph_replays"] == 30
    assert metrics["cuda_graph_captures"] == 3
    assert metrics["cuda_graph_real_rows"] == 60
    assert metrics["cuda_graph_padded_rows"] == 6


def test_training_summary_validates_exact_resume_boundary() -> None:
    summary = {
        "status": "completed",
        "config": {"num_envs": 2, "rollout_steps": 5},
        "updates": [_training_update(index, wakeups=0) for index in range(2, 5)],
        "update_count": 4,
        "global_step": 40,
        "final": {"steps_this_run": 30, "global_step": 40},
    }

    metrics = summarize_training_summary(
        summary,
        expected_invocation_updates=3,
        discard_updates=1,
        expected_initial_completed_updates=1,
        expected_initial_global_step=10,
    )
    assert metrics["final_update"] == 4
    assert metrics["final_global_step"] == 40

    summary["updates"][0]["update"] = 3
    with pytest.raises(ValueError, match="update boundary mismatch"):
        summarize_training_summary(
            summary,
            expected_invocation_updates=3,
            discard_updates=1,
            expected_initial_completed_updates=1,
            expected_initial_global_step=10,
        )


def test_success_marker_verifies_every_raw_artifact_hash(tmp_path) -> None:
    artifact = tmp_path / "attempt-001" / "summary.json"
    artifact.parent.mkdir()
    artifact.write_text("{}", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "SUCCESS.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {"attempt-001/summary.json": digest},
            }
        ),
        encoding="utf-8",
    )

    assert validated_success_marker(tmp_path) is not None

    artifact.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        validated_success_marker(tmp_path)


def test_existing_experiment_manifest_must_match_exactly(tmp_path) -> None:
    path = tmp_path / "experiment.json"
    expected = {"schema_version": 1, "config": {"blocks": 4}}

    ensure_experiment_manifest(path, expected)
    ensure_experiment_manifest(path, expected)

    with pytest.raises(ValueError, match="manifest mismatch"):
        ensure_experiment_manifest(
            path,
            {"schema_version": 1, "config": {"blocks": 8}},
        )


def test_factorial_effects_reject_non_positive_throughput() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        factorial_effects(
            [
                {
                    "phase": "production",
                    "block": 1,
                    "variant": "Q",
                    "metric": math.nan,
                }
            ],
            metric="metric",
        )


def test_git_identity_fails_closed_when_status_cannot_be_verified(
    tmp_path,
    monkeypatch,
) -> None:
    responses = iter(((True, "a" * 40), (False, "")))
    monkeypatch.setattr(
        efficiency_factorial,
        "run_capture_result",
        lambda command, cwd=None: next(responses),
    )

    identity = git_identity(tmp_path)

    assert identity["git_sha"] == "a" * 40
    assert identity["status_verified"] is False
    assert identity["worktree_dirty"] is None
    with pytest.raises(RuntimeError, match="Could not verify"):
        require_clean_git_identity(identity, name="test checkout")


def test_git_identity_checks_untracked_files_and_rejects_dirty_tree(
    tmp_path,
    monkeypatch,
) -> None:
    commands = []

    def fake_capture(command, cwd=None):
        commands.append(command)
        if "rev-parse" in command:
            return True, "b" * 40
        return True, "?? untracked.py"

    monkeypatch.setattr(
        efficiency_factorial,
        "run_capture_result",
        fake_capture,
    )

    identity = git_identity(tmp_path)

    assert identity["status_verified"] is True
    assert identity["worktree_dirty"] is True
    assert any("--untracked-files=all" in command for command in commands)
    with pytest.raises(RuntimeError, match="tracked or untracked changes"):
        require_clean_git_identity(identity, name="test checkout")


def test_installed_package_tree_digest_tracks_importable_bytes(tmp_path) -> None:
    package = tmp_path / "poke_env"
    package.mkdir()
    module = package / "module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"ignored")

    before = deterministic_package_tree_identity(package)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    after = deterministic_package_tree_identity(package)

    assert before["file_count"] == 1
    assert before["sha256"] != after["sha256"]


def test_git_source_attribution_requires_origin_to_be_tracked(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    origin = repo / ".venv" / "Lib" / "site-packages" / "poke_env" / "__init__.py"
    origin.parent.mkdir(parents=True)
    origin.write_text("", encoding="utf-8")
    responses = iter(((True, str(repo)), (False, "")))
    monkeypatch.setattr(
        efficiency_factorial,
        "run_capture_result",
        lambda command, cwd=None: next(responses),
    )

    assert efficiency_factorial._tracked_git_root(origin) is None


def test_retention_decision_requires_valid_runtime_and_prespecified_evidence() -> None:
    production = [{"config_fingerprint": "same"}] * 16
    effects = {
        "lazy": {
            "effect_percent": 2.0,
            "ci_low_percent": 0.5,
            "positive_constituents": 7,
            "constituent_count": 8,
        },
        "coalescing": {
            "effect_percent": 0.5,
            "ci_low_percent": 0.1,
            "positive_constituents": 8,
            "constituent_count": 8,
        },
    }

    decision = build_retention_decision(
        operationally_complete=True,
        production=production,
        primary_effects=effects,
        warnings=[],
    )
    assert decision["scientifically_valid"] is True
    assert decision["factors"]["lazy"]["retain_optimization"] is True
    assert decision["factors"]["coalescing"]["retain_optimization"] is False

    invalid = build_retention_decision(
        operationally_complete=True,
        production=production,
        primary_effects=effects,
        warnings=["graph fallback"],
    )
    assert invalid["scientifically_valid"] is False
    assert invalid["factors"]["lazy"]["retain_optimization"] is False
