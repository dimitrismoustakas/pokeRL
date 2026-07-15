import json
import random

import pytest

from pokerl.training.phase import (
    ArchivePolicy,
    FrozenMixtureSampler,
    InitializationMode,
    MixtureWeight,
    OpponentSamplerKind,
    PhaseInitialization,
    PhaseKind,
    PhaseRunState,
    PhaseSpec,
)
from pokerl.training.population import (
    CheckpointRef,
    PolicyMember,
    PolicyRole,
    PopulationRevision,
)
from scripts.train import load_best_response_phase


TEAM_POOL_SIGNATURE = "1" * 64
MODEL_SIGNATURE = "2" * 64


def _population(tmp_path) -> PopulationRevision:
    checkpoints = []
    for policy_id in ("main-a", "main-b"):
        path = tmp_path / f"{policy_id}.pt"
        path.write_bytes(policy_id.encode())
        checkpoints.append(CheckpointRef.from_file(path, relative_to=tmp_path))
    return PopulationRevision.create(
        parent_revision_id=None,
        battle_format="gen9vgc2026regf",
        team_pool_signature=TEAM_POOL_SIGNATURE,
        model_signature=MODEL_SIGNATURE,
        evaluation_action_mode="sample",
        policies=[
            PolicyMember(
                policy_id="main-a",
                role=PolicyRole.MAIN,
                checkpoint=checkpoints[0],
                lineage_id="main",
                parent_policy_id=None,
                generation=0,
            ),
            PolicyMember(
                policy_id="main-b",
                role=PolicyRole.MAIN,
                checkpoint=checkpoints[1],
                lineage_id="main",
                parent_policy_id="main-a",
                generation=1,
            ),
        ],
    )


def _best_response_phase(
    population: PopulationRevision,
    *,
    distribution=None,
    initialization_mode=InitializationMode.WEIGHTS_ONLY,
    opponent_sampler=OpponentSamplerKind.FROZEN_MIXTURE,
    archive_policy=ArchivePolicy.NONE,
    target_population_path="population.json",
) -> PhaseSpec:
    if distribution is None:
        distribution = [
            MixtureWeight("main-a", 0.75),
            MixtureWeight("main-b", 0.25),
        ]
    return PhaseSpec.create(
        phase_id="league-exploiter-1",
        learner_policy_id="league-exploiter-1-final",
        kind=PhaseKind.BEST_RESPONSE,
        learner_role=PolicyRole.LEAGUE_EXPLOITER,
        initialization=PhaseInitialization(
            mode=initialization_mode,
            source_policy_id="main-b",
            checkpoint=population.policies[1].checkpoint,
        ),
        target_population_path=target_population_path,
        target_population_revision_id=population.revision_id,
        target_distribution=distribution,
        opponent_sampler=opponent_sampler,
        archive_policy=archive_policy,
        battle_format=population.battle_format,
        team_pool_signature=population.team_pool_signature,
        model_signature=population.model_signature,
    )


def test_phase_hash_is_semantic_and_mixture_order_independent(tmp_path) -> None:
    population = _population(tmp_path)
    left = _best_response_phase(population)
    right = _best_response_phase(
        population,
        distribution=[
            MixtureWeight("main-b", 0.25),
            MixtureWeight("main-a", 0.75),
        ],
    )

    assert left == right
    assert left.phase_hash == right.phase_hash
    assert [item.policy_id for item in left.target_distribution] == [
        "main-a",
        "main-b",
    ]


def test_phase_roundtrips_and_validates_population_and_initializer(tmp_path) -> None:
    population = _population(tmp_path)
    population.save(tmp_path / "population.json")
    phase = _best_response_phase(population)
    phase.validate_against_population(population)
    target = tmp_path / "phase.json"

    phase.save(target)
    loaded = PhaseSpec.load(target)

    assert loaded == phase
    assert loaded.verify_initializer(tmp_path) == (tmp_path / "main-b.pt").resolve()
    assert loaded.load_target_population(tmp_path) == population


def test_phase_target_population_path_is_resolved_from_phase_not_cwd(
    tmp_path,
    monkeypatch,
) -> None:
    population_dir = tmp_path / "population"
    population_dir.mkdir()
    phase_dir = tmp_path / "plans"
    phase_dir.mkdir()
    population = _population(population_dir)
    population.save(population_dir / "population.json")
    phase = _best_response_phase(
        population,
        target_population_path="../population/population.json",
    )
    phase_path = phase.save(phase_dir / "phase.json")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    loaded = PhaseSpec.load(phase_path)
    bundle = load_best_response_phase(str(phase_path))

    assert loaded.load_target_population(phase_path.parent) == population
    assert (
        loaded.verify_initializer(phase_path.parent)
        == (population_dir / "main-b.pt").resolve()
    )
    assert bundle.population_path == (population_dir / "population.json").resolve()
    assert bundle.checkpoint_by_policy_id["main-b"] == str(
        (population_dir / "main-b.pt").resolve()
    )


def test_training_phase_loader_returns_verified_absolute_target_mapping(
    tmp_path,
) -> None:
    population = _population(tmp_path)
    population_path = population.save(tmp_path / "population.json")
    phase_path = _best_response_phase(population).save(tmp_path / "phase.json")

    bundle = load_best_response_phase(str(phase_path))

    assert bundle.spec_path == phase_path
    assert bundle.population_path == population_path
    assert bundle.population == population
    assert bundle.checkpoint_by_policy_id == {
        "main-a": str((tmp_path / "main-a.pt").resolve()),
        "main-b": str((tmp_path / "main-b.pt").resolve()),
    }


def test_phase_rejects_same_bytes_under_a_different_checkpoint_reference(
    tmp_path,
) -> None:
    population = _population(tmp_path)
    copied_checkpoint = tmp_path / "copied-main-b.pt"
    copied_checkpoint.write_bytes((tmp_path / "main-b.pt").read_bytes())
    phase = _best_response_phase(population)
    mismatched = PhaseSpec.create(
        phase_id=phase.phase_id,
        learner_policy_id=phase.learner_policy_id,
        kind=phase.kind,
        learner_role=phase.learner_role,
        initialization=PhaseInitialization(
            mode=phase.initialization.mode,
            source_policy_id=phase.initialization.source_policy_id,
            checkpoint=CheckpointRef.from_file(
                copied_checkpoint,
                relative_to=tmp_path,
            ),
        ),
        target_population_path=phase.target_population_path,
        target_population_revision_id=phase.target_population_revision_id,
        target_distribution=phase.target_distribution,
        opponent_sampler=phase.opponent_sampler,
        archive_policy=phase.archive_policy,
        battle_format=phase.battle_format,
        team_pool_signature=phase.team_pool_signature,
        model_signature=phase.model_signature,
    )

    with pytest.raises(ValueError, match="does not exactly match"):
        mismatched.validate_against_population(population)


def test_phase_rejects_semantic_tampering(tmp_path) -> None:
    phase = _best_response_phase(_population(tmp_path))
    payload = phase.to_dict()
    payload["target_distribution"][0]["probability"] = 0.7
    payload["target_distribution"][1]["probability"] = 0.3

    with pytest.raises(ValueError, match="Phase hash does not match"):
        PhaseSpec.from_dict(payload)


@pytest.mark.parametrize(
    ("initialization_mode", "opponent_sampler", "archive_policy", "match"),
    [
        (
            InitializationMode.EXACT_TRAINER,
            OpponentSamplerKind.FROZEN_MIXTURE,
            ArchivePolicy.NONE,
            "weight-only initialization",
        ),
        (
            InitializationMode.WEIGHTS_ONLY,
            OpponentSamplerKind.ADAPTIVE_PFSP,
            ArchivePolicy.NONE,
            "frozen opponent mixture",
        ),
        (
            InitializationMode.WEIGHTS_ONLY,
            OpponentSamplerKind.FROZEN_MIXTURE,
            ArchivePolicy.MAIN_SNAPSHOTS,
            "must not enter its own target archive",
        ),
    ],
)
def test_best_response_phase_enforces_stationary_target_invariants(
    tmp_path,
    initialization_mode,
    opponent_sampler,
    archive_policy,
    match,
) -> None:
    population = _population(tmp_path)

    with pytest.raises(ValueError, match=match):
        _best_response_phase(
            population,
            initialization_mode=initialization_mode,
            opponent_sampler=opponent_sampler,
            archive_policy=archive_policy,
        )


def test_main_phase_requires_exact_resume_adaptive_pfsp_and_snapshots(tmp_path) -> None:
    population = _population(tmp_path)
    phase = PhaseSpec.create(
        phase_id="main-2",
        learner_policy_id="main-2-final",
        kind=PhaseKind.MAIN,
        learner_role=PolicyRole.MAIN,
        initialization=PhaseInitialization(
            mode=InitializationMode.EXACT_TRAINER,
            source_policy_id="main-b",
            checkpoint=population.policies[1].checkpoint,
        ),
        target_population_path="population.json",
        target_population_revision_id=population.revision_id,
        target_distribution=[],
        opponent_sampler=OpponentSamplerKind.ADAPTIVE_PFSP,
        archive_policy=ArchivePolicy.MAIN_SNAPSHOTS,
        battle_format=population.battle_format,
        team_pool_signature=population.team_pool_signature,
        model_signature=population.model_signature,
    )

    phase.validate_against_population(population)


def test_phase_population_validation_rejects_wrong_revision(tmp_path) -> None:
    population = _population(tmp_path)
    phase = _best_response_phase(population)
    other_checkpoint = tmp_path / "other.pt"
    other_checkpoint.write_bytes(b"other")
    other_population = PopulationRevision.create(
        parent_revision_id=population.revision_id,
        battle_format=population.battle_format,
        team_pool_signature=population.team_pool_signature,
        model_signature=population.model_signature,
        evaluation_action_mode="sample",
        policies=[
            *population.policies,
            PolicyMember(
                policy_id="other",
                role=PolicyRole.MAIN_EXPLOITER,
                checkpoint=CheckpointRef.from_file(
                    other_checkpoint,
                    relative_to=tmp_path,
                ),
                lineage_id="other",
                parent_policy_id="main-b",
                generation=0,
            ),
        ],
    )

    with pytest.raises(ValueError, match="target population hash"):
        phase.validate_against_population(other_population)


def test_best_response_rejects_greedy_population_payoffs(tmp_path) -> None:
    population = _population(tmp_path)
    phase = _best_response_phase(population)
    greedy_population = PopulationRevision.create(
        parent_revision_id=None,
        battle_format=population.battle_format,
        team_pool_signature=population.team_pool_signature,
        model_signature=population.model_signature,
        evaluation_action_mode="argmax",
        policies=population.policies,
    )
    greedy_phase = PhaseSpec.create(
        phase_id=phase.phase_id,
        learner_policy_id=phase.learner_policy_id,
        kind=phase.kind,
        learner_role=phase.learner_role,
        initialization=phase.initialization,
        target_population_path=phase.target_population_path,
        target_population_revision_id=greedy_population.revision_id,
        target_distribution=phase.target_distribution,
        opponent_sampler=phase.opponent_sampler,
        archive_policy=phase.archive_policy,
        battle_format=phase.battle_format,
        team_pool_signature=phase.team_pool_signature,
        model_signature=phase.model_signature,
    )

    with pytest.raises(ValueError, match="action_mode='sample'"):
        greedy_phase.validate_against_population(greedy_population)


def test_frozen_mixture_sampler_rng_roundtrips_through_json() -> None:
    weights = [MixtureWeight("a", 0.2), MixtureWeight("b", 0.8)]
    source = FrozenMixtureSampler(weights, seed=1234)
    source.sample_policy_ids(17)
    state = json.loads(json.dumps(source.state_dict()))
    expected = source.sample_policy_ids(100)

    restored = FrozenMixtureSampler(list(reversed(weights)), seed=9999)
    restored.load_state_dict(state)

    assert restored.draw_count == 17
    assert restored.sample_policy_ids(100) == expected
    assert restored.draw_count == 117


def test_frozen_sampler_rejects_state_from_another_mixture_without_mutating() -> None:
    source = FrozenMixtureSampler(
        [MixtureWeight("a", 0.2), MixtureWeight("b", 0.8)],
        seed=1,
    )
    target = FrozenMixtureSampler(
        [MixtureWeight("a", 0.3), MixtureWeight("b", 0.7)],
        seed=2,
    )
    before = target.state_dict()

    with pytest.raises(ValueError, match="different target mixture"):
        target.load_state_dict(source.state_dict())

    assert target.state_dict() == before


def test_phase_run_state_roundtrips_and_restores_exact_sampler_rng(tmp_path) -> None:
    population = _population(tmp_path)
    phase = _best_response_phase(population)
    sampler = FrozenMixtureSampler(phase.target_distribution, seed=17)
    sampler.sample_policy_ids(13)
    state = PhaseRunState(
        phase_hash=phase.phase_hash,
        target_population_revision_id=population.revision_id,
        training_state_id="trainer-state",
        completed_updates=9,
        sampler_state=sampler.state_dict(),
        team_pool_signature=TEAM_POOL_SIGNATURE,
        team_pool_rng_state=random.Random(31).getstate(),
        opponent_assignments=((0, "a.pt"), (1, "b.pt")),
    )
    state_path = state.save(tmp_path / "checkpoint.phase_state.json")
    expected = sampler.sample_policy_ids(50)
    restored_state = PhaseRunState.load(state_path)
    restored_sampler = FrozenMixtureSampler(phase.target_distribution, seed=999)

    restored_state.validate_resume(
        phase=phase,
        training_state_id="trainer-state",
        trainer_completed_updates=9,
        sampler=restored_sampler,
    )

    assert restored_state == state
    assert restored_state.assignments_dict() == {0: "a.pt", 1: "b.pt"}
    assert restored_sampler.sample_policy_ids(50) == expected


def test_phase_run_state_rejects_changed_phase_before_sampler_mutation(
    tmp_path,
) -> None:
    population = _population(tmp_path)
    phase = _best_response_phase(population)
    changed = PhaseSpec.create(
        phase_id="different-phase",
        learner_policy_id="different-final",
        kind=phase.kind,
        learner_role=phase.learner_role,
        initialization=phase.initialization,
        target_population_path=phase.target_population_path,
        target_population_revision_id=phase.target_population_revision_id,
        target_distribution=phase.target_distribution,
        opponent_sampler=phase.opponent_sampler,
        archive_policy=phase.archive_policy,
        battle_format=phase.battle_format,
        team_pool_signature=phase.team_pool_signature,
        model_signature=phase.model_signature,
    )
    source_sampler = FrozenMixtureSampler(phase.target_distribution, seed=17)
    source_sampler.sample_policy_ids(4)
    state = PhaseRunState(
        phase_hash=phase.phase_hash,
        target_population_revision_id=population.revision_id,
        training_state_id="trainer-state",
        completed_updates=3,
        sampler_state=source_sampler.state_dict(),
        team_pool_signature=None,
        team_pool_rng_state=None,
        opponent_assignments=((0, "a.pt"),),
    )
    target_sampler = FrozenMixtureSampler(phase.target_distribution, seed=999)
    before = target_sampler.state_dict()

    with pytest.raises(ValueError, match="phase hash"):
        state.validate_resume(
            phase=changed,
            training_state_id="trainer-state",
            trainer_completed_updates=3,
            sampler=target_sampler,
        )

    assert target_sampler.state_dict() == before
