import json

import numpy as np
import pytest

from pokerl.training.population import (
    CheckpointRef,
    PairwisePayoff,
    PolicyMember,
    PolicyRole,
    PopulationRevision,
    solve_symmetric_zero_sum,
    symmetric_exploitability,
)


TEAM_POOL_SIGNATURE = "1" * 64
MODEL_SIGNATURE = "2" * 64


def _member(tmp_path, policy_id: str, *, parent: str | None = None) -> PolicyMember:
    checkpoint = tmp_path / f"{policy_id}.pt"
    checkpoint.write_bytes(f"checkpoint:{policy_id}".encode())
    return PolicyMember(
        policy_id=policy_id,
        role=PolicyRole.MAIN,
        checkpoint=CheckpointRef.from_file(checkpoint, relative_to=tmp_path),
        lineage_id="main-lineage",
        parent_policy_id=parent,
        generation=0 if parent is None else 1,
        source_training_state_id=f"state-{policy_id}",
    )


def _revision(policies, payoffs=()) -> PopulationRevision:
    return PopulationRevision.create(
        parent_revision_id=None,
        battle_format="gen9vgc2026regf",
        team_pool_signature=TEAM_POOL_SIGNATURE,
        model_signature=MODEL_SIGNATURE,
        evaluation_action_mode="sample",
        policies=policies,
        payoffs=payoffs,
    )


def test_population_revision_roundtrips_and_verifies_checkpoint_bytes(tmp_path) -> None:
    member = _member(tmp_path, "main-a")
    revision = _revision([member])
    path = tmp_path / "population.json"

    revision.save(path)
    loaded = PopulationRevision.load(path)

    assert loaded == revision
    assert loaded.verify_checkpoints(tmp_path) == {
        "main-a": (tmp_path / "main-a.pt").resolve()
    }

    (tmp_path / "main-a.pt").write_bytes(b"mutated checkpoint")
    with pytest.raises(ValueError, match="Checkpoint hash mismatch"):
        PopulationRevision.load(path)


def test_population_revision_rejects_semantic_tampering() -> None:
    # A real-looking checkpoint hash is sufficient here because from_dict does
    # not touch the filesystem; load(..., verify_checkpoints=True) does.
    member = PolicyMember(
        policy_id="main-a",
        role=PolicyRole.MAIN,
        checkpoint=CheckpointRef(path="main-a.pt", sha256="3" * 64),
        lineage_id="main",
        parent_policy_id=None,
        generation=0,
    )
    revision = _revision([member])
    payload = revision.to_dict()
    payload["battle_format"] = "tampered-format"

    with pytest.raises(ValueError, match="revision id does not match"):
        PopulationRevision.from_dict(payload)


def test_population_revision_is_not_overwritten_in_place(tmp_path) -> None:
    first = _revision([_member(tmp_path, "main-a")])
    target = tmp_path / "population.json"
    first.save(target)
    # Idempotently writing identical immutable content is harmless.
    assert first.save(target) == target.resolve()

    second = PopulationRevision.create(
        parent_revision_id=first.revision_id,
        battle_format=first.battle_format,
        team_pool_signature=first.team_pool_signature,
        model_signature=first.model_signature,
        evaluation_action_mode=first.evaluation_action_mode,
        policies=[
            first.policies[0],
            _member(tmp_path, "main-b", parent="main-a"),
        ],
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        second.save(target)


def test_population_json_has_a_content_addressed_revision_id(tmp_path) -> None:
    revision = _revision([_member(tmp_path, "main-a")])
    target = revision.save(tmp_path / "population.json")
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["revision_id"] == revision.revision_id
    assert len(payload["revision_id"]) == 64


def test_payoff_matrix_is_complete_and_exactly_skew_symmetric(tmp_path) -> None:
    policies = [_member(tmp_path, policy_id) for policy_id in ("a", "b", "c")]
    payoffs = [
        PairwisePayoff(
            "a",
            "b",
            wins=75,
            losses=25,
            draws=0,
            evaluation_sha256="a" * 64,
            evaluation_path="a-b.json",
        ),
        PairwisePayoff(
            "a",
            "c",
            wins=0,
            losses=100,
            draws=0,
            evaluation_sha256="b" * 64,
            evaluation_path="a-c.json",
        ),
        PairwisePayoff(
            "b",
            "c",
            wins=75,
            losses=25,
            draws=0,
            evaluation_sha256="c" * 64,
            evaluation_path="b-c.json",
        ),
    ]
    revision = _revision(policies, payoffs)

    policy_ids, matrix = revision.payoff_matrix()

    assert policy_ids == ("a", "b", "c")
    np.testing.assert_allclose(
        matrix,
        np.asarray(
            [
                [0.0, 0.5, -1.0],
                [-0.5, 0.0, 0.5],
                [1.0, -0.5, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(matrix, -matrix.T)


def test_incomplete_payoff_matrix_is_not_silently_solved(tmp_path) -> None:
    policies = [_member(tmp_path, policy_id) for policy_id in ("a", "b", "c")]
    revision = _revision(
        policies,
        [PairwisePayoff("a", "b", 5, 5, 0, "a" * 64, "a-b.json")],
    )

    with pytest.raises(ValueError, match="missing pair.*a/c.*b/c"):
        revision.payoff_matrix()


def test_no_regret_solver_finds_nonuniform_symmetric_equilibrium(tmp_path) -> None:
    policies = [_member(tmp_path, policy_id) for policy_id in ("a", "b", "c")]
    revision = _revision(
        policies,
        [
            PairwisePayoff("a", "b", 75, 25, 0, "a" * 64, "a-b.json"),
            PairwisePayoff("a", "c", 0, 100, 0, "b" * 64, "a-c.json"),
            PairwisePayoff("b", "c", 75, 25, 0, "c" * 64, "b-c.json"),
        ],
    )
    _, matrix = revision.payoff_matrix()

    solution = solve_symmetric_zero_sum(
        matrix,
        tolerance=4e-3,
        max_iterations=200_000,
        check_every=100,
    )

    assert solution.converged is True
    assert solution.exploitability_gap <= 4e-3
    np.testing.assert_allclose(solution.probabilities, [0.25, 0.5, 0.25], atol=0.012)
    assert symmetric_exploitability(matrix, solution.probabilities) == pytest.approx(
        solution.exploitability_gap
    )


def test_solver_reports_an_unmet_certificate_instead_of_claiming_convergence() -> None:
    matrix = np.asarray(
        [
            [0.0, 0.5, -1.0],
            [-0.5, 0.0, 0.5],
            [1.0, -0.5, 0.0],
        ]
    )

    solution = solve_symmetric_zero_sum(
        matrix,
        tolerance=1e-12,
        max_iterations=1,
        check_every=1,
    )

    assert solution.converged is False
    assert solution.exploitability_gap > 1e-12
