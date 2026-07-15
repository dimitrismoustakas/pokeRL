import json
from pathlib import Path

import pytest

from pokerl.eval.metrics import mirrored_candidate_summary, outcome_summary
from pokerl.training.population import (
    CheckpointRef,
    PolicyMember,
    PolicyRole,
    PopulationRevision,
)
from pokerl.training.population_evidence import (
    FrozenPairwiseEvaluation,
    build_child_population_revision,
    build_pairwise_evidence_block,
)


TEAM_POOL_SIGNATURE = "1" * 64
MODEL_SIGNATURE = "2" * 64
BATTLE_FORMAT = "gen9vgc2026regf"


def _member(
    tmp_path: Path,
    policy_id: str,
    *,
    parent: str | None = None,
) -> PolicyMember:
    checkpoint = tmp_path / f"{policy_id}.pt"
    checkpoint.write_bytes(f"checkpoint:{policy_id}".encode())
    return PolicyMember(
        policy_id=policy_id,
        role=PolicyRole.MAIN if parent is None else PolicyRole.LEAGUE_EXPLOITER,
        checkpoint=CheckpointRef.from_file(checkpoint, relative_to=tmp_path),
        lineage_id=f"lineage-{policy_id}",
        parent_policy_id=parent,
        generation=0 if parent is None else 1,
        source_training_state_id=f"state-{policy_id}",
    )


def _direction(
    *,
    label: str,
    outcomes: list[str],
    team_pairs: list[tuple[str, str]],
    swapped: bool,
) -> dict[str, object]:
    rewards = {"win": 1.0, "loss": -1.0, "draw": 0.0}
    summary = outcome_summary(rewards[outcome] for outcome in outcomes)
    return {
        "label": label,
        "teams_swapped_from_forward_schedule": swapped,
        **summary,
        "games_detail": [
            {
                "game": index + 1,
                "reward": rewards[outcome],
                "outcome": outcome,
                "external_team_file": external,
                "opponent_team_file": opponent,
            }
            for index, (outcome, (external, opponent)) in enumerate(
                zip(outcomes, team_pairs, strict=True)
            )
        ],
    }


def _write_evidence(
    tmp_path: Path,
    *,
    checkpoint_member: PolicyMember,
    anchor_member: PolicyMember,
    name: str,
    action_mode: str = "sample",
) -> Path:
    forward_pairs = [("teams/a.txt", "teams/b.txt"), ("teams/c.txt", "teams/d.txt")]
    reverse_pairs = [(right, left) for left, right in forward_pairs]
    forward = _direction(
        label="candidate_external",
        outcomes=["win", "draw"],
        team_pairs=forward_pairs,
        swapped=False,
    )
    reverse = _direction(
        label="anchor_external_candidate_opponent",
        outcomes=["loss", "win"],
        team_pairs=reverse_pairs,
        swapped=True,
    )
    candidate = mirrored_candidate_summary(forward, reverse)
    checkpoint_ref = CheckpointRef.from_file(
        tmp_path / f"{checkpoint_member.policy_id}.pt"
    )
    anchor_ref = CheckpointRef.from_file(tmp_path / f"{anchor_member.policy_id}.pt")
    summary = {
        "config": {"diagnostic": "retained but not trusted as evidence"},
        "results": {
            "candidate_perspective": candidate,
            "directions": [forward, reverse],
        },
        "pairwise_evidence": build_pairwise_evidence_block(
            checkpoint_policy_id=checkpoint_member.policy_id,
            anchor_policy_id=anchor_member.policy_id,
            checkpoint=checkpoint_ref,
            anchor=anchor_ref,
            battle_format=BATTLE_FORMAT,
            team_pool_signature=TEAM_POOL_SIGNATURE,
            model_signature=MODEL_SIGNATURE,
            action_mode=action_mode,
            games_per_role=2,
            directions=[forward, reverse],
            candidate_perspective=candidate,
        ),
    }
    path = tmp_path / name
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_member_spec(tmp_path: Path, member: PolicyMember) -> Path:
    path = tmp_path / f"{member.policy_id}.member.json"
    path.write_text(
        json.dumps(member.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _initial_population(tmp_path: Path, member: PolicyMember) -> Path:
    revision = PopulationRevision.create(
        parent_revision_id=None,
        battle_format=BATTLE_FORMAT,
        team_pool_signature=TEAM_POOL_SIGNATURE,
        model_signature=MODEL_SIGNATURE,
        evaluation_action_mode="sample",
        policies=[member],
    )
    return revision.save(tmp_path / "population-0.json")


def test_evidence_verifies_mirrored_schedule_checkpoint_bytes_and_counts(
    tmp_path: Path,
) -> None:
    anchor = _member(tmp_path, "a")
    candidate = _member(tmp_path, "b", parent="a")
    path = _write_evidence(
        tmp_path,
        checkpoint_member=candidate,
        anchor_member=anchor,
        name="b-v-a.json",
    )

    evidence = FrozenPairwiseEvaluation.load(path)

    assert evidence.policy_pair == ("a", "b")
    assert (evidence.wins, evidence.losses, evidence.draws) == (2, 1, 1)
    payoff = evidence.to_payoff(relative_to=tmp_path)
    # Stored rows are lexical; the evaluated candidate was b, so W/L invert.
    assert (payoff.row_policy_id, payoff.column_policy_id) == ("a", "b")
    assert (payoff.wins, payoff.losses, payoff.draws) == (1, 2, 1)
    assert payoff.verify_evaluation(tmp_path) == path.resolve()


def test_builder_creates_only_complete_child_and_inherits_verified_evidence(
    tmp_path: Path,
) -> None:
    a = _member(tmp_path, "a")
    b = _member(tmp_path, "b", parent="a")
    parent_path = _initial_population(tmp_path, a)
    ab = _write_evidence(
        tmp_path,
        checkpoint_member=b,
        anchor_member=a,
        name="b-v-a.json",
    )
    child_path = tmp_path / "revisions" / "population-1.json"

    first_child = build_child_population_revision(
        parent_path=parent_path,
        member_spec_paths=[_write_member_spec(tmp_path, b)],
        evidence_paths=[ab],
        output_path=child_path,
    )

    assert (
        first_child.parent_revision_id
        == PopulationRevision.load(parent_path).revision_id
    )
    assert [member.policy_id for member in first_child.policies] == ["a", "b"]
    assert first_child.payoff_matrix()[0] == ("a", "b")
    assert PopulationRevision.load(child_path) == first_child

    c = _member(tmp_path, "c", parent="a")
    ac = _write_evidence(
        tmp_path,
        checkpoint_member=c,
        anchor_member=a,
        name="c-v-a.json",
    )
    bc = _write_evidence(
        tmp_path,
        checkpoint_member=c,
        anchor_member=b,
        name="c-v-b.json",
    )
    second_path = tmp_path / "next" / "population-2.json"
    second_child = build_child_population_revision(
        parent_path=child_path,
        member_spec_paths=[_write_member_spec(tmp_path, c)],
        evidence_paths=[ac, bc],
        output_path=second_path,
    )

    assert len(second_child.payoffs) == 3
    second_child.payoff_matrix(require_complete=True)
    assert PopulationRevision.load(second_path) == second_child


def test_builder_rejects_missing_duplicate_or_greedy_pairwise_evidence(
    tmp_path: Path,
) -> None:
    a = _member(tmp_path, "a")
    b = _member(tmp_path, "b", parent="a")
    parent = _initial_population(tmp_path, a)
    member_spec = _write_member_spec(tmp_path, b)
    sampled = _write_evidence(
        tmp_path,
        checkpoint_member=b,
        anchor_member=a,
        name="sampled.json",
    )
    greedy = _write_evidence(
        tmp_path,
        checkpoint_member=b,
        anchor_member=a,
        name="greedy.json",
        action_mode="argmax",
    )

    with pytest.raises(ValueError, match="exactly one evaluation.*missing"):
        build_child_population_revision(
            parent_path=parent,
            member_spec_paths=[member_spec],
            evidence_paths=[],
            output_path=tmp_path / "missing.json",
        )
    with pytest.raises(ValueError, match="Duplicate pairwise evidence"):
        build_child_population_revision(
            parent_path=parent,
            member_spec_paths=[member_spec],
            evidence_paths=[sampled, sampled],
            output_path=tmp_path / "duplicate.json",
        )
    with pytest.raises(ValueError, match="action_mode='sample'"):
        build_child_population_revision(
            parent_path=parent,
            member_spec_paths=[member_spec],
            evidence_paths=[greedy],
            output_path=tmp_path / "greedy-child.json",
        )
    assert not (tmp_path / "missing.json").exists()
    assert not (tmp_path / "duplicate.json").exists()
    assert not (tmp_path / "greedy-child.json").exists()


def test_evidence_and_population_fail_closed_after_tampering(tmp_path: Path) -> None:
    a = _member(tmp_path, "a")
    b = _member(tmp_path, "b", parent="a")
    evidence_path = _write_evidence(
        tmp_path,
        checkpoint_member=b,
        anchor_member=a,
        name="b-v-a.json",
    )
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["pairwise_evidence"]["candidate_perspective"]["wins"] += 1
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match mirrored directions"):
        FrozenPairwiseEvaluation.load(evidence_path)

    mirrored_path = _write_evidence(
        tmp_path,
        checkpoint_member=b,
        anchor_member=a,
        name="bad-mirror.json",
    )
    mirrored_payload = json.loads(mirrored_path.read_text(encoding="utf-8"))
    mirrored_payload["results"]["directions"][1]["games_detail"][0][
        "external_team_file"
    ] = "teams/not-the-forward-opponent.txt"
    mirrored_path.write_text(json.dumps(mirrored_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not swap the planned team roles"):
        FrozenPairwiseEvaluation.load(mirrored_path)

    valid_path = _write_evidence(
        tmp_path,
        checkpoint_member=b,
        anchor_member=a,
        name="valid.json",
    )
    population_path = tmp_path / "child.json"
    build_child_population_revision(
        parent_path=_initial_population(tmp_path, a),
        member_spec_paths=[_write_member_spec(tmp_path, b)],
        evidence_paths=[valid_path],
        output_path=population_path,
    )
    valid_path.write_text(
        valid_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="evaluation hash mismatch"):
        PopulationRevision.load(population_path)
