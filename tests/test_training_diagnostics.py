import pytest

from scripts.train import build_rollout_diagnostics


def test_build_rollout_diagnostics_computes_wait_and_latency_shares() -> None:
    diagnostics = build_rollout_diagnostics(
        rollout_profile={
            "rollout_sec": 100.0,
            "policy_forward_sec": 5.0,
            "obs_to_device_sec": 1.0,
            "mask_to_device_sec": 2.0,
            "feedback_to_device_sec": 3.0,
        },
        env_profile={
            "env_step_total_sec": 80.0,
            "env_step_calls": 40.0,
            "env_queue_wait_sec": 60.0,
            "parallel_step_wall_sec": 10.0,
            "parallel_step_calls": 20.0,
            "opponent_choice_sec": 3.0,
            "opponent_choice_calls": 12.0,
        },
    )

    assert diagnostics["env_queue_wait_share"] == pytest.approx(0.75)
    assert diagnostics["avg_env_step_ms"] == pytest.approx(2000.0)
    assert diagnostics["avg_parallel_step_wall_ms"] == pytest.approx(500.0)
    assert diagnostics["avg_opponent_choice_ms"] == pytest.approx(250.0)
    assert diagnostics["policy_forward_share"] == pytest.approx(0.05)
    assert diagnostics["device_transfer_share"] == pytest.approx(0.06)


def test_build_rollout_diagnostics_handles_missing_counters() -> None:
    diagnostics = build_rollout_diagnostics({}, {})

    assert all(value == 0.0 for value in diagnostics.values())
