"""Frozen cross-play evidence and fail-closed population revision building.

Pairwise training meta-games are meaningful only when they describe the same
game the best-response trainer will face.  This module therefore accepts only
mirrored, sampled-policy evaluations with exact checkpoint bytes and matching
format, team-pool, and categorical-vocabulary signatures.  It never applies an
admission score or changes rewards: completed cross-play is evidence, not a
hand-authored skill gate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from pokerl.training.population import (
    CheckpointRef,
    PairwisePayoff,
    PolicyMember,
    PopulationRevision,
    canonical_sha256,
    file_sha256,
)


PAIRWISE_EVIDENCE_FORMAT_VERSION = 1
_SHA256_LENGTH = 64


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _require_exact_keys(
    value: object,
    expected: set[str],
    name: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name)
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{name} has invalid fields: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    return mapping


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_sha256(value: object, name: str) -> str:
    digest = _require_string(value, name)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _require_count(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return value


def _load_json_object(path: Path, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        duplicates: list[str] = []
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        if duplicates:
            raise ValueError(
                f"{name} contains duplicate JSON key(s): "
                + ", ".join(sorted(set(duplicates)))
            )
        return result

    def reject_non_finite(value: str) -> object:
        raise ValueError(f"{name} contains non-finite JSON number {value!r}.")

    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path!s}") from exc
    return _require_mapping(raw, name)


def _direction_counts(
    raw: object,
    name: str,
) -> tuple[Mapping[str, Any], tuple[int, int, int, int]]:
    value = _require_exact_keys(
        raw,
        {
            "external_policy_id",
            "opponent_policy_id",
            "external_checkpoint_sha256",
            "opponent_checkpoint_sha256",
            "teams_swapped_from_forward_schedule",
            "games",
            "wins",
            "losses",
            "draws",
        },
        name,
    )
    games = _require_count(value["games"], f"{name} games", positive=True)
    wins = _require_count(value["wins"], f"{name} wins")
    losses = _require_count(value["losses"], f"{name} losses")
    draws = _require_count(value["draws"], f"{name} draws")
    if wins + losses + draws != games:
        raise ValueError(f"{name} W/L/D counts do not sum to games.")
    if type(value["teams_swapped_from_forward_schedule"]) is not bool:
        raise ValueError(f"{name} teams-swapped flag must be boolean.")
    _require_string(value["external_policy_id"], f"{name} external policy id")
    _require_string(value["opponent_policy_id"], f"{name} opponent policy id")
    _require_sha256(
        value["external_checkpoint_sha256"],
        f"{name} external checkpoint hash",
    )
    _require_sha256(
        value["opponent_checkpoint_sha256"],
        f"{name} opponent checkpoint hash",
    )
    return value, (games, wins, losses, draws)


def _detail_outcome_counts(
    raw_details: object,
    *,
    expected_games: int,
    name: str,
) -> tuple[int, int, int]:
    if not isinstance(raw_details, list) or len(raw_details) != expected_games:
        raise ValueError(
            f"{name} games_detail must contain exactly {expected_games} games."
        )
    wins = losses = draws = 0
    for index, raw_detail in enumerate(raw_details):
        detail = _require_mapping(raw_detail, f"{name} game {index + 1}")
        outcome = detail.get("outcome")
        reward = detail.get("reward")
        if outcome not in {"win", "loss", "draw"}:
            raise ValueError(f"{name} game {index + 1} has invalid outcome.")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError(f"{name} game {index + 1} has invalid reward.")
        if not math.isfinite(float(reward)):
            raise ValueError(f"{name} game {index + 1} has non-finite reward.")
        expected_outcome = "win" if reward > 0 else ("loss" if reward < 0 else "draw")
        if outcome != expected_outcome:
            raise ValueError(
                f"{name} game {index + 1} outcome disagrees with reward sign."
            )
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            draws += 1
        for field in ("external_team_file", "opponent_team_file"):
            _require_string(
                detail.get(field),
                f"{name} game {index + 1} {field}",
            )
    return wins, losses, draws


def _role_schedule_payload(directions: Sequence[Mapping[str, Any]]) -> list[object]:
    payload: list[object] = []
    for direction in directions:
        details = direction["games_detail"]
        assert isinstance(details, list)
        payload.append(
            {
                "label": direction.get("label"),
                "teams_swapped_from_forward_schedule": direction.get(
                    "teams_swapped_from_forward_schedule"
                ),
                "team_roles": [
                    {
                        "external_team_file": detail["external_team_file"],
                        "opponent_team_file": detail["opponent_team_file"],
                    }
                    for detail in details
                ],
            }
        )
    return payload


def role_schedule_sha256(directions: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the exact two-role team schedule retained in a summary."""
    return canonical_sha256(_role_schedule_payload(directions))


def build_pairwise_evidence_block(
    *,
    checkpoint_policy_id: str,
    anchor_policy_id: str,
    checkpoint: CheckpointRef,
    anchor: CheckpointRef,
    battle_format: str,
    team_pool_signature: str,
    model_signature: str,
    action_mode: str,
    games_per_role: int,
    directions: Sequence[Mapping[str, Any]],
    candidate_perspective: Mapping[str, Any],
) -> dict[str, object]:
    """Create the strict evidence section embedded by ``evaluate_anchor``."""
    if len(directions) != 2:
        raise ValueError("Pairwise population evidence requires two mirrored roles.")
    direction_evidence: list[dict[str, object]] = []
    role_specs = (
        (checkpoint_policy_id, anchor_policy_id, checkpoint, anchor, False),
        (anchor_policy_id, checkpoint_policy_id, anchor, checkpoint, True),
    )
    for direction, (
        external_id,
        opponent_id,
        external_ref,
        opponent_ref,
        swapped,
    ) in zip(
        directions,
        role_specs,
        strict=True,
    ):
        direction_evidence.append(
            {
                "external_policy_id": external_id,
                "opponent_policy_id": opponent_id,
                "external_checkpoint_sha256": external_ref.sha256,
                "opponent_checkpoint_sha256": opponent_ref.sha256,
                "teams_swapped_from_forward_schedule": swapped,
                "games": direction["games"],
                "wins": direction["wins"],
                "losses": direction["losses"],
                "draws": direction["draws"],
            }
        )
    return {
        "format_version": PAIRWISE_EVIDENCE_FORMAT_VERSION,
        "complete": True,
        "checkpoint_policy": {
            "policy_id": checkpoint_policy_id,
            "checkpoint": checkpoint.to_dict(),
        },
        "anchor_policy": {
            "policy_id": anchor_policy_id,
            "checkpoint": anchor.to_dict(),
        },
        "battle_format": battle_format,
        "team_pool_signature": team_pool_signature,
        "model_signature": model_signature,
        "action_mode": action_mode,
        "mirrored_role_schedule": {
            "games_per_role": games_per_role,
            "schedule_sha256": role_schedule_sha256(directions),
            "directions": direction_evidence,
        },
        "candidate_perspective": {
            key: candidate_perspective[key]
            for key in ("games", "wins", "losses", "draws")
        },
    }


@dataclass(frozen=True)
class FrozenPairwiseEvaluation:
    """One verified, immutable mirrored cross-play summary."""

    checkpoint_policy_id: str
    anchor_policy_id: str
    checkpoint: CheckpointRef
    anchor: CheckpointRef
    battle_format: str
    team_pool_signature: str
    model_signature: str
    action_mode: str
    games_per_role: int
    wins: int
    losses: int
    draws: int
    source_path: Path
    source_sha256: str

    @property
    def policy_pair(self) -> tuple[str, str]:
        return tuple(sorted((self.checkpoint_policy_id, self.anchor_policy_id)))  # type: ignore[return-value]

    def to_payoff(self, *, relative_to: str | Path) -> PairwisePayoff:
        row, column = self.policy_pair
        wins, losses = self.wins, self.losses
        if self.checkpoint_policy_id != row:
            wins, losses = losses, wins
        try:
            path = self.source_path.relative_to(Path(relative_to).resolve()).as_posix()
        except ValueError:
            path = self.source_path.as_posix()
        return PairwisePayoff(
            row_policy_id=row,
            column_policy_id=column,
            wins=wins,
            losses=losses,
            draws=self.draws,
            evaluation_sha256=self.source_sha256,
            evaluation_path=path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FrozenPairwiseEvaluation":
        source = Path(path).expanduser().resolve()
        summary = _load_json_object(source, "Pairwise evaluation summary")
        evidence = _require_exact_keys(
            summary.get("pairwise_evidence"),
            {
                "format_version",
                "complete",
                "checkpoint_policy",
                "anchor_policy",
                "battle_format",
                "team_pool_signature",
                "model_signature",
                "action_mode",
                "mirrored_role_schedule",
                "candidate_perspective",
            },
            "Pairwise evidence",
        )
        if evidence["format_version"] != PAIRWISE_EVIDENCE_FORMAT_VERSION:
            raise ValueError("Unsupported pairwise evidence format version.")
        if evidence["complete"] is not True:
            raise ValueError("Pairwise evaluation evidence is not complete.")

        checkpoint_policy = _require_exact_keys(
            evidence["checkpoint_policy"],
            {"policy_id", "checkpoint"},
            "Checkpoint policy evidence",
        )
        anchor_policy = _require_exact_keys(
            evidence["anchor_policy"],
            {"policy_id", "checkpoint"},
            "Anchor policy evidence",
        )
        checkpoint_policy_id = _require_string(
            checkpoint_policy["policy_id"], "Checkpoint policy id"
        )
        anchor_policy_id = _require_string(
            anchor_policy["policy_id"], "Anchor policy id"
        )
        if checkpoint_policy_id == anchor_policy_id:
            raise ValueError("Pairwise evidence requires two distinct policy ids.")
        checkpoint_ref = CheckpointRef.from_dict(checkpoint_policy["checkpoint"])
        anchor_ref = CheckpointRef.from_dict(anchor_policy["checkpoint"])
        checkpoint_ref.verify(source.parent)
        anchor_ref.verify(source.parent)

        battle_format = _require_string(
            evidence["battle_format"], "Evidence battle format"
        )
        team_pool_signature = _require_sha256(
            evidence["team_pool_signature"], "Evidence team-pool signature"
        )
        model_signature = _require_sha256(
            evidence["model_signature"], "Evidence model signature"
        )
        action_mode = evidence["action_mode"]
        if action_mode not in {"sample", "argmax"}:
            raise ValueError("Evidence action mode must be 'sample' or 'argmax'.")

        role_schedule = _require_exact_keys(
            evidence["mirrored_role_schedule"],
            {"games_per_role", "schedule_sha256", "directions"},
            "Mirrored role schedule",
        )
        games_per_role = _require_count(
            role_schedule["games_per_role"],
            "Mirrored games per role",
            positive=True,
        )
        schedule_hash = _require_sha256(
            role_schedule["schedule_sha256"], "Role-schedule hash"
        )
        raw_directions = role_schedule["directions"]
        if not isinstance(raw_directions, list) or len(raw_directions) != 2:
            raise ValueError("Mirrored evidence must contain exactly two directions.")
        forward, forward_counts = _direction_counts(
            raw_directions[0], "Forward direction"
        )
        reverse, reverse_counts = _direction_counts(
            raw_directions[1], "Reverse direction"
        )
        expected_roles = (
            (
                forward,
                checkpoint_policy_id,
                anchor_policy_id,
                checkpoint_ref.sha256,
                anchor_ref.sha256,
                False,
            ),
            (
                reverse,
                anchor_policy_id,
                checkpoint_policy_id,
                anchor_ref.sha256,
                checkpoint_ref.sha256,
                True,
            ),
        )
        for (
            direction,
            external_id,
            opponent_id,
            external_hash,
            opponent_hash,
            swapped,
        ) in expected_roles:
            if (
                direction["external_policy_id"] != external_id
                or direction["opponent_policy_id"] != opponent_id
                or direction["external_checkpoint_sha256"] != external_hash
                or direction["opponent_checkpoint_sha256"] != opponent_hash
                or direction["teams_swapped_from_forward_schedule"] is not swapped
            ):
                raise ValueError(
                    "Mirrored direction roles or checkpoints are inconsistent."
                )
        if forward_counts[0] != games_per_role or reverse_counts[0] != games_per_role:
            raise ValueError(
                "Each mirrored direction must contain games_per_role games."
            )

        candidate = _require_exact_keys(
            evidence["candidate_perspective"],
            {"games", "wins", "losses", "draws"},
            "Candidate-perspective result",
        )
        games = _require_count(candidate["games"], "Candidate games", positive=True)
        wins = _require_count(candidate["wins"], "Candidate wins")
        losses = _require_count(candidate["losses"], "Candidate losses")
        draws = _require_count(candidate["draws"], "Candidate draws")
        expected_counts = (
            forward_counts[1] + reverse_counts[2],
            forward_counts[2] + reverse_counts[1],
            forward_counts[3] + reverse_counts[3],
        )
        if (wins, losses, draws) != expected_counts or games != sum(expected_counts):
            raise ValueError(
                "Candidate-perspective W/L/D does not match mirrored directions."
            )
        if games != 2 * games_per_role:
            raise ValueError("Candidate games must equal twice games_per_role.")

        results = _require_mapping(summary.get("results"), "Evaluation results")
        detailed_directions = results.get("directions")
        if not isinstance(detailed_directions, list) or len(detailed_directions) != 2:
            raise ValueError("Evaluation results must retain two detailed directions.")
        for index, (detailed, counts) in enumerate(
            zip(detailed_directions, (forward_counts, reverse_counts), strict=True)
        ):
            detailed_map = _require_mapping(detailed, f"Detailed direction {index + 1}")
            expected_swapped = bool(index)
            if (
                detailed_map.get("teams_swapped_from_forward_schedule")
                is not expected_swapped
            ):
                raise ValueError(
                    f"Detailed direction {index + 1} has an inconsistent role-swap flag."
                )
            detail_counts = _detail_outcome_counts(
                detailed_map.get("games_detail"),
                expected_games=games_per_role,
                name=f"Detailed direction {index + 1}",
            )
            if detail_counts != counts[1:]:
                raise ValueError(
                    f"Detailed direction {index + 1} W/L/D disagrees with evidence."
                )
            if (
                detailed_map.get("games") != counts[0]
                or detailed_map.get("wins") != counts[1]
                or detailed_map.get("losses") != counts[2]
                or detailed_map.get("draws") != counts[3]
            ):
                raise ValueError(
                    f"Detailed direction {index + 1} summary disagrees with evidence."
                )

        forward_details = detailed_directions[0]["games_detail"]
        reverse_details = detailed_directions[1]["games_detail"]
        for index, (forward_detail, reverse_detail) in enumerate(
            zip(forward_details, reverse_details, strict=True)
        ):
            if (
                forward_detail["external_team_file"]
                != reverse_detail["opponent_team_file"]
                or forward_detail["opponent_team_file"]
                != reverse_detail["external_team_file"]
            ):
                raise ValueError(
                    f"Mirrored game {index + 1} does not swap the planned team roles."
                )
        if role_schedule_sha256(detailed_directions) != schedule_hash:
            raise ValueError("Detailed role schedule does not match its evidence hash.")

        return cls(
            checkpoint_policy_id=checkpoint_policy_id,
            anchor_policy_id=anchor_policy_id,
            checkpoint=checkpoint_ref,
            anchor=anchor_ref,
            battle_format=battle_format,
            team_pool_signature=team_pool_signature,
            model_signature=model_signature,
            action_mode=action_mode,
            games_per_role=games_per_role,
            wins=wins,
            losses=losses,
            draws=draws,
            source_path=source,
            source_sha256=file_sha256(source),
        )


def _portable_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _rebase_member(
    member: PolicyMember,
    *,
    source_base: Path,
    destination_base: Path,
) -> PolicyMember:
    checkpoint_path = member.checkpoint.verify(source_base)
    return PolicyMember(
        policy_id=member.policy_id,
        role=member.role,
        checkpoint=CheckpointRef(
            path=_portable_path(checkpoint_path, destination_base),
            sha256=member.checkpoint.sha256,
        ),
        lineage_id=member.lineage_id,
        parent_policy_id=member.parent_policy_id,
        generation=member.generation,
        source_training_state_id=member.source_training_state_id,
    )


def _rebase_payoff(
    payoff: PairwisePayoff,
    *,
    source_base: Path,
    destination_base: Path,
) -> PairwisePayoff:
    evidence_path = payoff.verify_evaluation(source_base)
    return PairwisePayoff(
        row_policy_id=payoff.row_policy_id,
        column_policy_id=payoff.column_policy_id,
        wins=payoff.wins,
        losses=payoff.losses,
        draws=payoff.draws,
        evaluation_sha256=payoff.evaluation_sha256,
        evaluation_path=_portable_path(evidence_path, destination_base),
    )


def load_policy_member_spec(path: str | Path) -> PolicyMember:
    """Load an exact ``PolicyMember`` object and verify its checkpoint bytes."""
    source = Path(path).expanduser().resolve()
    member = PolicyMember.from_dict(
        _load_json_object(source, "Policy-member specification")
    )
    member.checkpoint.verify(source.parent)
    return member


def build_child_population_revision(
    *,
    parent_path: str | Path,
    member_spec_paths: Sequence[str | Path],
    evidence_paths: Sequence[str | Path],
    output_path: str | Path,
) -> PopulationRevision:
    """Create one immutable, fully evaluated child population revision.

    Existing parent evidence is inherited.  New summaries may fill only
    previously absent unordered pairs.  The union must cover every pair in the
    child exactly once; otherwise no file is written.
    """
    parent_source = Path(parent_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if target == parent_source:
        raise ValueError("Child population output must differ from its parent file.")
    if not member_spec_paths:
        raise ValueError("A child population requires at least one new member spec.")
    parent = PopulationRevision.load(
        parent_source,
        verify_checkpoints=True,
        verify_evaluations=True,
    )
    if parent.evaluation_action_mode != "sample":
        raise ValueError(
            "Training meta-games require a parent evaluated with action_mode='sample'."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    members = [
        _rebase_member(
            member,
            source_base=parent_source.parent,
            destination_base=target.parent,
        )
        for member in parent.policies
    ]
    for member_path in member_spec_paths:
        member_source = Path(member_path).expanduser().resolve()
        member = load_policy_member_spec(member_source)
        members.append(
            _rebase_member(
                member,
                source_base=member_source.parent,
                destination_base=target.parent,
            )
        )

    member_by_id: dict[str, PolicyMember] = {}
    checkpoint_hashes: set[str] = set()
    for member in members:
        if member.policy_id in member_by_id:
            raise ValueError(f"Duplicate child policy id: {member.policy_id!r}.")
        if member.checkpoint.sha256 in checkpoint_hashes:
            raise ValueError("Child population contains duplicate checkpoint bytes.")
        member_by_id[member.policy_id] = member
        checkpoint_hashes.add(member.checkpoint.sha256)

    payoffs = [
        _rebase_payoff(
            payoff,
            source_base=parent_source.parent,
            destination_base=target.parent,
        )
        for payoff in parent.payoffs
    ]
    observed_pairs = {
        (payoff.row_policy_id, payoff.column_policy_id) for payoff in payoffs
    }
    observed_sources = {payoff.evaluation_sha256 for payoff in payoffs}
    for evidence_path in evidence_paths:
        evidence = FrozenPairwiseEvaluation.load(evidence_path)
        if evidence.action_mode != "sample":
            raise ValueError(
                "Training meta-game evidence must use action_mode='sample'."
            )
        for name, actual, expected in (
            ("battle format", evidence.battle_format, parent.battle_format),
            (
                "team-pool signature",
                evidence.team_pool_signature,
                parent.team_pool_signature,
            ),
            ("model signature", evidence.model_signature, parent.model_signature),
        ):
            if actual != expected:
                raise ValueError(
                    f"Evidence {name} mismatch: evidence={actual!r}, "
                    f"population={expected!r}."
                )
        if not set(evidence.policy_pair).issubset(member_by_id):
            raise ValueError(
                "Evidence references policy ids outside the child population: "
                f"{evidence.policy_pair!r}."
            )
        evidence_refs = {
            evidence.checkpoint_policy_id: evidence.checkpoint,
            evidence.anchor_policy_id: evidence.anchor,
        }
        for policy_id, evidence_ref in evidence_refs.items():
            expected_ref = member_by_id[policy_id].checkpoint
            if evidence_ref.sha256 != expected_ref.sha256:
                raise ValueError(
                    f"Evidence checkpoint hash for {policy_id!r} does not match "
                    "the child policy member."
                )
        pair = evidence.policy_pair
        if pair in observed_pairs:
            raise ValueError(f"Duplicate pairwise evidence for pair {pair!r}.")
        if evidence.source_sha256 in observed_sources:
            raise ValueError("One evaluation summary cannot evidence multiple pairs.")
        payoffs.append(evidence.to_payoff(relative_to=target.parent))
        observed_pairs.add(pair)
        observed_sources.add(evidence.source_sha256)

    policy_ids = sorted(member_by_id)
    expected_pairs = set(combinations(policy_ids, 2))
    missing = sorted(expected_pairs - observed_pairs)
    unexpected = sorted(observed_pairs - expected_pairs)
    if missing or unexpected:
        raise ValueError(
            "Child population requires exactly one evaluation per unordered pair: "
            f"missing={missing}, unexpected={unexpected}."
        )

    revision = PopulationRevision.create(
        parent_revision_id=parent.revision_id,
        battle_format=parent.battle_format,
        team_pool_signature=parent.team_pool_signature,
        model_signature=parent.model_signature,
        evaluation_action_mode="sample",
        policies=tuple(member_by_id.values()),
        payoffs=payoffs,
    )
    revision.payoff_matrix(require_complete=True)
    revision.save(target)
    verified = PopulationRevision.load(
        target,
        verify_checkpoints=True,
        verify_evaluations=True,
    )
    if verified != revision:
        raise RuntimeError("Saved child population did not round-trip exactly.")
    return revision
