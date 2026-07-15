import hashlib
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

import scripts.train as train_module
from pokerl.config import EnvConfig, TrainingConfig
from pokerl.training.league import League, LeagueMember
from pokerl.training.phase import FrozenMixtureSampler, MixtureWeight, PhaseRunState
from scripts.train import (
    FrozenPhaseOpponentSource,
    _league_state_path_for_resume,
    _maintain_best_response_after_update,
    _maintain_league_after_update,
    _phase_state_path_for_resume,
    _record_opponent_outcome,
    _restore_opponent_assignments,
    _save_best_response_training_state,
    _sample_refresh_assignments,
    _serialize_opponent_assignments,
    _summarize_opponent_outcomes,
    make_league_opponent_factory,
    make_envs,
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
            checkpoint_sha256=hashlib.sha256(
                (
                    Path(path).read_bytes()
                    if Path(path).is_file()
                    else path.encode("utf-8")
                )
            ).hexdigest(),
            archive_source="self_play_snapshot",
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


def test_sample_refresh_assignments_groups_by_opponent_group() -> None:
    envs = [
        SimpleNamespace(config=SimpleNamespace(opponent_group=0)),
        SimpleNamespace(config=SimpleNamespace(opponent_group=1)),
        SimpleNamespace(config=SimpleNamespace(opponent_group=0)),
        SimpleNamespace(config=SimpleNamespace(opponent_group=1)),
    ]
    parallel_env = SimpleNamespace(envs=envs)
    samples = iter(["step_a.pt", "step_b.pt"])
    league = SimpleNamespace(
        sample_checkpoint_paths=lambda count: [next(samples) for _ in range(count)]
    )

    assignments = _sample_refresh_assignments(parallel_env, league)

    assert assignments == [
        (envs[0], "step_a.pt"),
        (envs[1], "step_b.pt"),
        (envs[2], "step_a.pt"),
        (envs[3], "step_b.pt"),
    ]


def test_frozen_refresh_sampler_never_falls_back_to_adaptive_league() -> None:
    envs = [
        SimpleNamespace(config=SimpleNamespace(opponent_group=0)),
        SimpleNamespace(config=SimpleNamespace(opponent_group=1)),
    ]
    parallel_env = SimpleNamespace(envs=envs)
    league = SimpleNamespace(
        sample_checkpoint_paths=lambda count: pytest.fail(
            f"adaptive league sampled {count} target(s)"
        )
    )
    draws: list[int] = []

    assignments = _sample_refresh_assignments(
        parallel_env,
        league,
        lambda count: draws.append(count) or ["pinned-a.pt", "pinned-b.pt"],
    )

    assert draws == [2]
    assert assignments == [
        (envs[0], "pinned-a.pt"),
        (envs[1], "pinned-b.pt"),
    ]


def test_frozen_phase_opponent_source_maps_only_exact_mixture_support() -> None:
    sampler = FrozenMixtureSampler(
        [MixtureWeight("main-a", 0.5), MixtureWeight("main-b", 0.5)],
        seed=7,
    )
    source = FrozenPhaseOpponentSource(
        sampler=sampler,
        checkpoint_by_policy_id={
            "main-a": "a.pt",
            "main-b": "b.pt",
        },
    )

    assert set(source.sample_checkpoint_paths(20)) == {"a.pt", "b.pt"}
    assert sampler.draw_count == 20
    with pytest.raises(ValueError, match="exactly match"):
        FrozenPhaseOpponentSource(
            sampler=sampler,
            checkpoint_by_policy_id={"main-a": "a.pt"},
        )


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
        sample_checkpoint_paths=lambda count: [next(samples) for _ in range(count)],
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


def test_make_envs_reuses_restored_assignments_without_pfsp_draw(monkeypatch) -> None:
    class FakeLoopThread:
        def __init__(self):
            self.loop = "loop"

        @classmethod
        def start(cls):
            return cls()

        def close(self):
            return None

    class FakeBattleEnv:
        def __init__(self, config, game_data, opponent_factory, **kwargs):
            del game_data, kwargs
            self.config = config
            self.opponent = opponent_factory("loop")

    class FakeParallelBattleEnv:
        def __init__(self, envs, collect_timing=False, loop_threads=None):
            del collect_timing, loop_threads
            self.envs = envs

    monkeypatch.setattr(train_module, "LoopThread", FakeLoopThread)
    monkeypatch.setattr(train_module, "BattleEnv", FakeBattleEnv)
    monkeypatch.setattr(train_module, "ParallelBattleEnv", FakeParallelBattleEnv)
    make_opponent_calls: list[str] = []

    def make_opponent(loop, checkpoint_path=None):
        del loop
        make_opponent_calls.append(checkpoint_path)
        return checkpoint_path

    league = SimpleNamespace(
        checkpoints=["step_a.pt", "step_b.pt"],
        sample_checkpoint_paths=lambda count: pytest.fail(
            f"strict resume must not redraw {count} opponent(s)"
        ),
        make_opponent=make_opponent,
    )

    parallel_env = make_envs(
        EnvConfig(num_envs=2, battle_loop_count=1),
        game_data=object(),
        league=league,
        opponent_groups=2,
        team_preview_fn=lambda obs: None,
        initial_checkpoint_by_group={0: "step_b.pt", 1: "step_a.pt"},
    )

    assert [env.opponent for env in parallel_env.envs] == ["step_b.pt", "step_a.pt"]
    assert make_opponent_calls == ["step_b.pt", "step_a.pt"]


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
        def add_checkpoint(self, _agent, step: int, *, update: int) -> None:
            events.append(f"add:{step}")

        def sample_checkpoint_paths(self, count: int) -> list[str]:
            events.append("sample")
            return ["existing.pt"] * count

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


def test_best_response_maintenance_only_refreshes_from_pinned_sampler() -> None:
    events: list[str] = []

    class FrozenLeague:
        def sample_checkpoint_paths(self, _count: int) -> list[str]:
            pytest.fail("best-response maintenance must not call adaptive PFSP")

        def add_checkpoint(self, *_args, **_kwargs) -> None:
            pytest.fail("best-response learner must not snapshot itself")

        def record_outcomes(self, *_args, **_kwargs) -> None:
            pytest.fail("best-response target payoffs must not adapt")

    class FakeEnv:
        config = SimpleNamespace(opponent_group=0)

        def stage_next_opponent_checkpoint(self, checkpoint_path: str) -> bool:
            events.append(f"stage:{checkpoint_path}")
            return True

    def sample_pinned(count: int) -> list[str]:
        events.append(f"sample:{count}")
        return ["pinned.pt"] * count

    maintenance = _maintain_best_response_after_update(
        update=2,
        num_updates=10,
        parallel_env=SimpleNamespace(envs=[FakeEnv()]),
        league=FrozenLeague(),
        config=SimpleNamespace(league_refresh_interval=2),
        checkpoint_sampler=sample_pinned,
    )

    assert events == ["sample:1", "stage:pinned.pt"]
    assert maintenance.checkpoint_added is False
    assert maintenance.assignments_by_group == {0: "pinned.pt"}


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


def test_opponent_assignment_state_roundtrips_without_sampling() -> None:
    paths = ["step_a.pt", "step_b.pt"]
    league = _league_state_stub(paths, config=TrainingConfig(), seed=17)
    assignments = {0: paths[1], 1: paths[0]}

    serialized = _serialize_opponent_assignments(assignments, league)
    restored = _restore_opponent_assignments(
        serialized,
        opponent_groups=2,
        league=league,
    )

    assert serialized == {"0": paths[1], "1": paths[0]}
    assert restored == assignments


def test_restored_assignment_rejects_non_archive_checkpoint() -> None:
    league = _league_state_stub(["step_a.pt"], config=TrainingConfig(), seed=17)

    with pytest.raises(ValueError, match="outside the archive"):
        _restore_opponent_assignments(
            {"0": "missing.pt"},
            opponent_groups=1,
            league=league,
        )


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
    league = SimpleNamespace(
        sample_checkpoint_paths=lambda count: ["step_a.pt"] * count
    )

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
    league = SimpleNamespace(
        sample_checkpoint_paths=lambda count: ["step_a.pt"] * count
    )

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


def test_best_response_state_pairs_phase_hash_sampler_and_trainer_id(tmp_path) -> None:
    checkpoint_path = tmp_path / "step_123.pt"
    target_path = str((tmp_path / "target.pt").resolve())
    sampler = FrozenMixtureSampler([MixtureWeight("main", 1.0)], seed=11)
    sampler.sample_policy_ids(5)
    team_rng = random.Random(17).getstate()
    team_pool = SimpleNamespace(
        signature=lambda: "3" * 64,
        rng_state=lambda: team_rng,
    )
    league = SimpleNamespace(checkpoints=[target_path])

    _save_best_response_training_state(
        checkpoint_path=str(checkpoint_path),
        phase=SimpleNamespace(
            phase_hash="1" * 64,
            target_population_revision_id="2" * 64,
        ),
        sampler=sampler,
        team_pool=team_pool,
        training_state_id="trainer-id",
        completed_updates=9,
        opponent_assignments={0: target_path},
        league=league,
    )

    state_path = _phase_state_path_for_resume(str(checkpoint_path))
    state = PhaseRunState.load(state_path)
    assert state.phase_hash == "1" * 64
    assert state.target_population_revision_id == "2" * 64
    assert state.training_state_id == "trainer-id"
    assert state.completed_updates == 9
    assert state.sampler_state["draw_count"] == 5
    assert state.team_pool_signature == "3" * 64
    assert state.opponent_assignments == ((0, target_path),)
    assert not _league_state_path_for_resume(str(checkpoint_path)).exists()


def test_frozen_population_mode_disables_every_adaptive_archive_operation(
    tmp_path,
) -> None:
    checkpoint_paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.touch()
    league = object.__new__(League)
    league.config = TrainingConfig()
    league.checkpoints = []
    league._members = {}
    league._current_update = 0
    league._frozen_population = False
    league._inference = SimpleNamespace(
        cache_info=lambda: {"resident_models": 0},
    )

    league.install_frozen_population([str(path) for path in checkpoint_paths])

    assert league.checkpoints == [str(path.resolve()) for path in checkpoint_paths]
    assert league.describe(include_members=True)["mode"] == (
        "frozen_population_inference"
    )
    with pytest.raises(RuntimeError, match="add a checkpoint"):
        league.add_checkpoint(object(), step=1)
    with pytest.raises(RuntimeError, match="record outcomes"):
        league.record_outcomes({}, update=1)
    with pytest.raises(RuntimeError, match="PFSP-sample"):
        league.sample_checkpoint_paths(1)
    with pytest.raises(RuntimeError, match="save adaptive league state"):
        league.save_state(str(tmp_path / "league.json"))
    with pytest.raises(RuntimeError, match="add an admitted checkpoint"):
        league.add_admitted_checkpoint(
            checkpoint_paths[0],
            hashlib.sha256(checkpoint_paths[0].read_bytes()).hexdigest(),
            step=1,
        )


def test_admitted_checkpoint_joins_adaptive_pfsp_and_exact_state(
    tmp_path,
) -> None:
    snapshot_path = tmp_path / "snapshot.pt"
    snapshot_path.write_bytes(b"main snapshot")
    admitted_path = tmp_path / "exploiter.pt"
    admitted_path.write_bytes(b"evaluated exploiter")
    admitted_sha256 = hashlib.sha256(admitted_path.read_bytes()).hexdigest()
    config = TrainingConfig(league_size=1)
    league = _league_state_stub(
        [str(snapshot_path)],
        config=config,
        seed=37,
    )

    canonical_path = league.add_admitted_checkpoint(
        admitted_path,
        admitted_sha256,
        step=91,
        update=2,
    )

    assert canonical_path == str(admitted_path.resolve())
    assert league.checkpoints == [str(snapshot_path), canonical_path]
    member = league._members[canonical_path]
    assert member.checkpoint_sha256 == admitted_sha256
    assert member.archive_source == "admitted_external"
    assert member.created_update == 2
    neutral_rows = {
        row["checkpoint_path"]: row for row in league.sampling_distribution(update=2)
    }
    assert neutral_rows[canonical_path]["first_draw_probability"] == pytest.approx(0.5)
    assert neutral_rows[canonical_path]["checkpoint_sha256"] == admitted_sha256
    assert neutral_rows[canonical_path]["archive_source"] == "admitted_external"

    league.record_outcomes(
        {canonical_path: {"wins": 0, "losses": 20, "draws": 0}},
        update=3,
    )
    rows = {
        row["checkpoint_path"]: row for row in league.sampling_distribution(update=3)
    }
    assert (
        rows[canonical_path]["current_policy_win_probability"]
        < rows[str(snapshot_path)]["current_policy_win_probability"]
    )
    assert (
        rows[canonical_path]["first_draw_probability"]
        > rows[str(snapshot_path)]["first_draw_probability"]
    )

    state_path = tmp_path / "main.league_state.json"
    league.save_state(str(state_path), extra_state={"marker": "exact"})
    restored = _league_state_stub([], config=config, seed=999)
    extra_state = restored.load_state(str(state_path))

    assert extra_state == {"marker": "exact"}
    assert restored.checkpoints == league.checkpoints
    assert restored._members[canonical_path] == member
    assert restored.current_update == 3
    # The permanent archive may exceed the bounded resident inference cache.
    assert len(restored.checkpoints) == 2
    assert restored.config.league_size == 1


def test_admitted_checkpoint_rejects_hash_and_duplicate_identity(
    tmp_path,
) -> None:
    first_path = tmp_path / "first.pt"
    first_path.write_bytes(b"same immutable bytes")
    alias_path = tmp_path / "second.pt"
    alias_path.write_bytes(first_path.read_bytes())
    checkpoint_sha256 = hashlib.sha256(first_path.read_bytes()).hexdigest()
    league = _league_state_stub([], config=TrainingConfig(), seed=7)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        league.add_admitted_checkpoint(
            tmp_path / "missing.pt",
            checkpoint_sha256,
            step=1,
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        league.add_admitted_checkpoint(
            first_path,
            "0" * 64,
            step=1,
        )
    assert league.checkpoints == []

    canonical_path = league.add_admitted_checkpoint(
        first_path,
        checkpoint_sha256,
        step=1,
    )
    with pytest.raises(ValueError, match="already contains admitted checkpoint path"):
        league.add_admitted_checkpoint(
            first_path.parent / "." / first_path.name,
            checkpoint_sha256,
            step=2,
        )
    with pytest.raises(ValueError, match="already contains checkpoint bytes"):
        league.add_admitted_checkpoint(
            alias_path,
            checkpoint_sha256,
            step=2,
        )
    assert league.checkpoints == [canonical_path]


def test_league_state_rejects_mutated_checkpoint_bytes(tmp_path) -> None:
    checkpoint_path = tmp_path / "admitted.pt"
    checkpoint_path.write_bytes(b"admitted bytes")
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    source = _league_state_stub([], config=TrainingConfig(), seed=37)
    source.add_admitted_checkpoint(
        checkpoint_path,
        checkpoint_sha256,
        step=4,
    )
    state_path = tmp_path / "main.league_state.json"
    source.save_state(str(state_path))
    checkpoint_path.write_bytes(b"silently replaced bytes")
    target = _league_state_stub([], config=TrainingConfig(), seed=999)

    with pytest.raises(ValueError, match="exact saved league checkpoint bytes"):
        target.load_state(str(state_path))


def test_league_state_roundtrips_membership_rng_and_extra_state(tmp_path) -> None:
    checkpoint_paths = [tmp_path / f"step_{index}.pt" for index in range(3)]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.touch()

    source = _league_state_stub(
        [str(path) for path in checkpoint_paths],
        config=TrainingConfig(league_size=3),
        seed=37,
    )
    state_path = tmp_path / "step_3.league_state.json"
    source.save_state(
        str(state_path),
        extra_state={"team_pool_rng_state": random.Random(91).getstate()},
    )
    expected_samples = [source._sample_checkpoint() for _ in range(12)]

    target = _league_state_stub(
        [],
        config=TrainingConfig(league_size=3),
        seed=999,
    )
    extra_state = target.load_state(str(state_path))

    assert list(target.checkpoints) == [str(path) for path in checkpoint_paths]
    assert [target._sample_checkpoint() for _ in range(12)] == expected_samples
    assert isinstance(extra_state["team_pool_rng_state"], tuple)
    assert not state_path.with_suffix(".json.tmp").exists()


def test_permanent_archive_state_can_exceed_resident_cache_capacity(tmp_path) -> None:
    checkpoint_paths = [tmp_path / f"step_{index}.pt" for index in range(5)]
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.touch()
    paths = [str(path) for path in checkpoint_paths]
    config = TrainingConfig(league_size=2)
    source = _league_state_stub(paths, config=config, seed=23)
    state_path = tmp_path / "archive.league_state.json"
    source.save_state(str(state_path))
    target = _league_state_stub([], config=config, seed=999)

    target.load_state(str(state_path))

    assert target.checkpoints == paths
    assert len(target.checkpoints) == 5
    assert target.config.league_size == 2


def test_league_state_rejects_missing_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "step_1.pt"
    checkpoint_path.touch()
    source = _league_state_stub(
        [str(checkpoint_path)],
        config=TrainingConfig(league_size=3),
        seed=37,
    )
    state_path = tmp_path / "step_1.league_state.json"
    source.save_state(str(state_path))
    checkpoint_path.unlink()

    target = _league_state_stub(
        [],
        config=TrainingConfig(league_size=3),
        seed=999,
    )

    with pytest.raises(FileNotFoundError, match="every saved league checkpoint"):
        target.load_state(str(state_path))


def test_league_state_rejects_changed_league_size(tmp_path) -> None:
    checkpoint_path = tmp_path / "step_1.pt"
    checkpoint_path.touch()
    source = _league_state_stub(
        [str(checkpoint_path)],
        config=TrainingConfig(league_size=3),
        seed=37,
    )
    state_path = tmp_path / "step_1.league_state.json"
    source.save_state(str(state_path))

    target = _league_state_stub(
        [],
        config=TrainingConfig(league_size=4),
        seed=999,
    )

    with pytest.raises(ValueError, match="league schedule does not match"):
        target.load_state(str(state_path))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("league_update_interval", 11),
        ("league_refresh_interval", 2),
    ],
)
def test_league_state_rejects_changed_schedule(tmp_path, field, value) -> None:
    checkpoint_path = tmp_path / "step_1.pt"
    checkpoint_path.touch()
    source = _league_state_stub(
        [str(checkpoint_path)],
        config=TrainingConfig(league_size=3),
        seed=37,
    )
    state_path = tmp_path / "step_1.league_state.json"
    source.save_state(str(state_path))

    config_kwargs = {"league_size": 3, field: value}
    target = _league_state_stub(
        [],
        config=TrainingConfig(**config_kwargs),
        seed=999,
    )

    with pytest.raises(ValueError, match="league schedule does not match"):
        target.load_state(str(state_path))
