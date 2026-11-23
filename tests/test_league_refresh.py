from itertools import permutations
import random
from types import SimpleNamespace

import pytest

from pokerl.config import TrainingConfig
from pokerl.training.league import League, LeagueMember, _expected_group_counts
from scripts.train import (
    _maintain_league_after_update,
    _record_opponent_outcome,
    _summarize_opponent_outcomes,
)


def _league_state_stub(
    paths: list[str],
    *,
    config: TrainingConfig,
    seed: int,
) -> League:
    league = object.__new__(League)
    league.config = config
    league.checkpoints = list(paths)
    league._members = {
        path: LeagueMember(
            path=path,
            step=index,
            created_update=index,
        )
        for index, path in enumerate(paths)
    }
    league._current_update = len(paths)
    league._rng = random.Random(seed)
    league._inference = SimpleNamespace(
        prune_checkpoints=lambda paths: None,
        prefetch_checkpoint=lambda path: None,
        cache_info=lambda: {
            "capacity": config.league_size,
            "resident_models": 0,
            "hits": 0,
            "misses": 0,
            "loads": 0,
            "evictions": 0,
        },
    )
    return league


def test_league_snapshot_is_added_before_refresh_sampling() -> None:
    events: list[str] = []

    class FakeLeague:
        checkpoints = ["old.pt"]

        def add_checkpoint(self, _agent, step: int, *, update: int) -> None:
            events.append(f"add:{step}")
            self.checkpoints.append("new.pt")

        def sample_checkpoint_paths(self, count: int) -> list[str]:
            events.append("sample")
            return [self.checkpoints[-1]] * count

    maintenance = _maintain_league_after_update(
        update=10,
        num_updates=20,
        global_step=123,
        agent=object(),
        parallel_env=SimpleNamespace(assignments={0: "old.pt"}),
        league=FakeLeague(),
        config=SimpleNamespace(
            league_update_interval=10,
            league_refresh_interval=1,
        ),
    )

    assert events == ["add:123", "sample"]
    assert maintenance.checkpoint_added is True
    assert maintenance.assignments_by_group == {0: "new.pt"}


def test_refresh_cadence_is_independent_of_snapshot_cadence() -> None:
    events: list[str] = []

    class FakeLeague:
        def add_checkpoint(self, _agent, step: int, *, update: int) -> None:
            events.append(f"add:{step}")

        def sample_checkpoint_paths(self, count: int) -> list[str]:
            events.append("sample")
            return ["existing.pt"] * count

    maintenance = _maintain_league_after_update(
        update=2,
        num_updates=20,
        global_step=123,
        agent=object(),
        parallel_env=SimpleNamespace(assignments={0: "old.pt"}),
        league=FakeLeague(),
        config=SimpleNamespace(
            league_update_interval=10,
            league_refresh_interval=1,
        ),
    )

    assert events == ["sample"]
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


def test_discounted_payoff_treats_each_update_as_one_evidence_unit() -> None:
    path = "step_a.pt"
    league = _league_state_stub(
        [path],
        config=TrainingConfig(league_payoff_half_life_updates=20.0),
        seed=7,
    )
    league._current_update = 0

    league.record_outcomes(
        {path: {"wins": 800, "losses": 200, "draws": 0}},
        update=1,
    )

    member = league._members[path]
    assert member.discounted_wins == pytest.approx(0.8)
    assert member.discounted_losses == pytest.approx(0.2)
    assert member.lifetime_wins == 800
    assert member.lifetime_losses == 200
    row = league.sampling_distribution(update=1)[0]
    assert row["discounted_effective_update_equivalents"] == pytest.approx(1.0)
    assert row["current_policy_win_probability"] == pytest.approx(0.6)


def test_stale_payoff_evidence_decays_toward_neutral() -> None:
    path = "step_a.pt"
    league = _league_state_stub(
        [path],
        config=TrainingConfig(league_payoff_half_life_updates=20.0),
        seed=7,
    )
    league._current_update = 0
    league.record_outcomes(
        {path: {"wins": 0, "losses": 1000, "draws": 0}},
        update=1,
    )

    fresh = league.sampling_distribution(update=1)[0]
    stale = league.sampling_distribution(update=21)[0]

    assert fresh["current_policy_win_probability"] == pytest.approx(1.0 / 3.0)
    assert stale["discounted_effective_update_equivalents"] == pytest.approx(0.5)
    assert stale["current_policy_win_probability"] == pytest.approx(0.4)


def test_hard_pfsp_distribution_has_uniform_floor_and_compact_summary() -> None:
    paths = ["easy.pt", "neutral.pt", "hard.pt"]
    league = _league_state_stub(
        paths,
        config=TrainingConfig(league_pfsp_uniform_fraction=0.1),
        seed=11,
    )
    league._current_update = 0
    league.record_outcomes(
        {
            paths[0]: {"wins": 100, "losses": 0, "draws": 0},
            paths[2]: {"wins": 0, "losses": 100, "draws": 0},
        },
        update=1,
    )

    rows = league.sampling_distribution(update=1)
    probabilities = {
        row["checkpoint_path"]: row["first_draw_probability"] for row in rows
    }

    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(probability >= 0.1 / 3 for probability in probabilities.values())
    assert probabilities["hard.pt"] > probabilities["neutral.pt"]
    assert probabilities["neutral.pt"] > probabilities["easy.pt"]
    compact = league.describe(update=1, include_members=False)
    assert "members" not in compact
    assert compact["hardest_checkpoint_path"] == "hard.pt"
    assert 1.0 <= compact["first_draw_effective_support"] <= 3.0


def test_group_sampling_is_distinct_when_archive_allows_it() -> None:
    paths = ["step_a.pt", "step_b.pt", "step_c.pt"]
    league = _league_state_stub(paths, config=TrainingConfig(), seed=13)

    for _ in range(50):
        sampled = league.sample_checkpoint_paths(2)
        assert len(sampled) == len(set(sampled)) == 2

    single = _league_state_stub(["only.pt"], config=TrainingConfig(), seed=13)
    assert single.sample_checkpoint_paths(2) == ["only.pt", "only.pt"]


@pytest.mark.parametrize("sampler", ["random", "balanced"])
def test_restore_preserves_league_draws_and_payoffs_with_smaller_cache(
    tmp_path, sampler
):
    paths = [tmp_path / f"step_{i}.pt" for i in range(3)]
    for path in paths:
        path.touch()
    saved = _league_state_stub(
        list(map(str, paths)),
        config=TrainingConfig(league_size=3, league_sampler=sampler),
        seed=37,
    )
    saved.record_outcomes(
        {str(paths[0]): {"wins": 3, "losses": 1, "draws": 0}}, update=5
    )
    saved.sample_checkpoint_paths(2)
    restored = _league_state_stub([], config=TrainingConfig(league_size=1), seed=999)
    restored.load_state_dict(saved.state_dict())

    assert restored.config.league_size == 1
    assert restored.sampling_distribution() == saved.sampling_distribution()
    assert [restored.sample_checkpoint_paths(2) for _ in range(12)] == [
        saved.sample_checkpoint_paths(2) for _ in range(12)
    ]


@pytest.mark.parametrize("groups", [1, 2, 4, 6])
def test_balanced_sampler_tracks_expected_weighted_draws(groups):
    weights = [0.45, 0.25, 0.15, 0.10, 0.05]
    expected = [max(0, groups - len(weights)) * p for p in weights]
    for order in permutations(range(len(weights)), min(groups, len(weights))):
        remaining = 1.0
        probability = 1.0
        for i in order:
            probability *= weights[i] / remaining
            remaining -= weights[i]
        for i in order:
            expected[i] += probability
    assert _expected_group_counts(weights, groups) == pytest.approx(expected, abs=1e-11)

    paths = [str(i) for i in range(len(weights))]
    league = _league_state_stub(
        paths, config=TrainingConfig(league_sampler="balanced"), seed=7
    )
    league.sampling_distribution = lambda **_: [
        {"checkpoint_path": path, "first_draw_probability": p}
        for path, p in zip(paths, weights, strict=True)
    ]
    counts = [0] * len(paths)
    for _ in range(60):
        picked = league.sample_checkpoint_paths(groups)
        assert len(set(picked)) == min(groups, len(paths))
        for path in picked:
            counts[int(path)] += 1
    assert (
        max(abs(count - 60 * p) for count, p in zip(counts, expected, strict=True)) < 1
    )
