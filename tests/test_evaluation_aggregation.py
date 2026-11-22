import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_anchor_evaluations import (
    aggregate_anchor_summaries,
    hierarchical_paired_bootstrap_ci,
)


def _summary(seed: int, rewards: list[tuple[float, float]]) -> dict:
    digest_a = "a" * 64
    digest_b = "b" * 64
    forward = []
    reverse = []
    for index, (forward_reward, reverse_external_reward) in enumerate(rewards):
        forward.append(
            {
                "reward": forward_reward,
                "external_team_file": f"left-{index}",
                "opponent_team_file": f"right-{index}",
            }
        )
        reverse.append(
            {
                "reward": reverse_external_reward,
                "external_team_file": f"right-{index}",
                "opponent_team_file": f"left-{index}",
            }
        )
    return {
        "config": {
            "evaluation_mode": "checkpoint_comparison",
            "checkpoint_policy_id": "candidate",
            "anchor_policy_id": "anchor",
            "checkpoint": {"path": "candidate.pt", "sha256": digest_a},
            "anchor": {"path": "anchor.pt", "sha256": digest_b},
            "battle_format": "test-format",
            "action_mode": "sample",
            "team_pool_signature": "c" * 64,
            "model_signature": "d" * 64,
            "mirrored": True,
            "seed": seed,
            "simulator_rng_seeded": False,
            "team_pair_schedule": {"schedule_sha256": f"{seed:064x}"},
        },
        "results": {
            "directions": [
                {"games_detail": forward},
                {"games_detail": reverse},
            ]
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_hierarchical_bootstrap_constant_groups_is_exact() -> None:
    interval = hierarchical_paired_bootstrap_ci(
        [np.ones(5), np.ones(7)], samples=200, seed=9
    )
    assert interval == [1.0, 1.0]


def test_hierarchical_bootstrap_includes_between_seed_uncertainty() -> None:
    # A fixed-strata bootstrap is degenerate at 0.5 here because there is no
    # within-seed variation. Resampling the two seeds produces {0, 0.5, 1}.
    groups = [np.zeros(8), np.ones(8)]

    interval = hierarchical_paired_bootstrap_ci(groups, samples=2_000, seed=17)

    assert interval == [0.0, 1.0]
    assert interval == hierarchical_paired_bootstrap_ci(
        groups,
        samples=2_000,
        seed=17,
    )


def test_aggregate_inverts_reverse_external_outcomes_and_preserves_pairs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "seed-1.json"
    second = tmp_path / "seed-2.json"
    # Candidate outcomes: W/W, L/L, W/L, D/D -> 3.5/8 score.
    _write(first, _summary(1, [(1.0, -1.0), (-1.0, 1.0)]))
    _write(second, _summary(2, [(1.0, 1.0), (0.0, 0.0)]))

    aggregate = aggregate_anchor_summaries(
        [first, second], bootstrap_samples=500, bootstrap_seed=3
    )
    reversed_aggregate = aggregate_anchor_summaries(
        [second, first], bootstrap_samples=500, bootstrap_seed=3
    )

    results = aggregate["results"]
    assert results["games"] == 8
    assert results["wins"] == 3
    assert results["losses"] == 3
    assert results["draws"] == 2
    assert results["paired_team_matchups"] == 4
    assert results["paired_score_rate"] == pytest.approx(0.5)
    assert len(results["paired_score_rate_bootstrap_ci95"]) == 2
    assert aggregate["config"]["bootstrap_method"] == (
        "hierarchical_seed_then_paired_matchup"
    )
    assert aggregate["config"]["bootstrap_seed_strata"] == 2
    assert (
        results["paired_score_rate_bootstrap_ci95"]
        == reversed_aggregate["results"]["paired_score_rate_bootstrap_ci95"]
    )
    assert (
        results["paired_role_score_difference_bootstrap_ci95"]
        == reversed_aggregate["results"]["paired_role_score_difference_bootstrap_ci95"]
    )
    assert len(aggregate["runs"]) == 2


def test_aggregate_combines_schedule_partitions_for_the_same_seed(
    tmp_path: Path,
) -> None:
    first = tmp_path / "seed-1-a.json"
    second = tmp_path / "seed-1-b.json"
    first_summary = _summary(1, [(1.0, -1.0)])
    second_summary = _summary(1, [(-1.0, 1.0)])
    second_summary["config"]["team_pair_schedule"]["schedule_sha256"] = "e" * 64
    _write(first, first_summary)
    _write(second, second_summary)

    aggregate = aggregate_anchor_summaries(
        [first, second], bootstrap_samples=200, bootstrap_seed=4
    )

    assert aggregate["config"]["bootstrap_seed_strata"] == 1
    assert aggregate["results"]["between_seed_score_std"] == 0.0


def test_aggregate_keeps_historical_showdown_and_pikapro_separate(
    tmp_path: Path,
) -> None:
    historical, current = tmp_path / "old.json", tmp_path / "current.json"
    _write(historical, _summary(1, [(1.0, -1.0)]))
    run = _summary(2, [(1.0, -1.0)])
    run["config"]["simulator"] = "showdown"
    _write(current, run)
    aggregate = aggregate_anchor_summaries(
        [historical, current], bootstrap_samples=50, bootstrap_seed=0
    )
    assert aggregate["config"]["simulator"] == "showdown"
    run["config"]["simulator"] = "pikapro"
    _write(current, run)
    with pytest.raises(ValueError, match="Incompatible evaluation identity"):
        aggregate_anchor_summaries(
            [historical, current], bootstrap_samples=50, bootstrap_seed=0
        )


def test_aggregate_rejects_duplicate_artifact(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    _write(path, _summary(1, [(1.0, -1.0)]))

    with pytest.raises(ValueError, match="same evaluation artifact"):
        aggregate_anchor_summaries([path, path], bootstrap_samples=10, bootstrap_seed=0)


def test_aggregate_same_policy_reports_external_role_advantage(tmp_path: Path) -> None:
    path = tmp_path / "same-policy.json"
    summary = _summary(4, [(1.0, 1.0), (1.0, -1.0)])
    summary["config"]["evaluation_mode"] = "same_policy_role_bias"
    summary["config"]["anchor"]["sha256"] = summary["config"]["checkpoint"]["sha256"]
    _write(path, summary)

    aggregate = aggregate_anchor_summaries(
        [path], bootstrap_samples=100, bootstrap_seed=0
    )

    assert aggregate["results"]["paired_score_rate"] == pytest.approx(0.75)
    assert aggregate["results"]["same_policy_external_role_advantage"] == pytest.approx(
        0.25
    )
