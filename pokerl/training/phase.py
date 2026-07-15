"""Immutable training-phase specifications and frozen opponent sampling.

Population-based training has two importantly different update regimes:

* a main phase may update online PFSP evidence and add historical snapshots;
* a best-response phase must train against a pinned population mixture without
  adding the learner or changing its target.

This module encodes those invariants before either regime is connected to the
trainer.  It also provides the exact-resume RNG primitive for a frozen mixture.
"""

from __future__ import annotations

import bisect
import copy
import json
import math
import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .population import (
    CheckpointRef,
    PolicyRole,
    PopulationRevision,
    canonical_sha256,
)


PHASE_FORMAT_VERSION = 1
FROZEN_SAMPLER_STATE_VERSION = 1
PHASE_RUN_STATE_VERSION = 1


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_exact_keys(
    value: object,
    expected: set[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} has invalid fields: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}."
        )
    return value


def _tuples_to_lists(value: object) -> object:
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, list):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: _tuples_to_lists(item) for key, item in value.items()}
    return value


def _lists_to_tuples(value: object) -> object:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    return value


class PhaseKind(StrEnum):
    MAIN = "main"
    BEST_RESPONSE = "best_response"


class InitializationMode(StrEnum):
    """How a new phase consumes its initializer checkpoint."""

    EXACT_TRAINER = "exact_trainer"
    WEIGHTS_ONLY = "weights_only"


class OpponentSamplerKind(StrEnum):
    ADAPTIVE_PFSP = "adaptive_pfsp"
    FROZEN_MIXTURE = "frozen_mixture"


class ArchivePolicy(StrEnum):
    MAIN_SNAPSHOTS = "main_snapshots"
    NONE = "none"


@dataclass(frozen=True)
class PhaseInitialization:
    mode: InitializationMode
    source_policy_id: str
    checkpoint: CheckpointRef

    def __post_init__(self) -> None:
        if not isinstance(self.mode, InitializationMode):
            raise ValueError("Initialization mode must be an InitializationMode.")
        _require_nonempty_string(self.source_policy_id, "Initialization policy id")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "source_policy_id": self.source_policy_id,
            "checkpoint": self.checkpoint.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: object) -> "PhaseInitialization":
        value = _require_exact_keys(
            raw,
            {"mode", "source_policy_id", "checkpoint"},
            "Phase initialization",
        )
        try:
            mode = InitializationMode(value["mode"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Unsupported phase initialization mode: {value['mode']!r}."
            ) from exc
        return cls(
            mode=mode,
            source_policy_id=value["source_policy_id"],
            checkpoint=CheckpointRef.from_dict(value["checkpoint"]),
        )


@dataclass(frozen=True)
class MixtureWeight:
    policy_id: str
    probability: float

    def __post_init__(self) -> None:
        _require_nonempty_string(self.policy_id, "Mixture policy id")
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not math.isfinite(self.probability)
            or self.probability <= 0.0
            or self.probability > 1.0
        ):
            raise ValueError("Mixture probability must be finite and within (0, 1].")
        object.__setattr__(self, "probability", float(self.probability))

    def to_dict(self) -> dict[str, object]:
        return {"policy_id": self.policy_id, "probability": self.probability}

    @classmethod
    def from_dict(cls, raw: object) -> "MixtureWeight":
        value = _require_exact_keys(
            raw,
            {"policy_id", "probability"},
            "Mixture weight",
        )
        return cls(
            policy_id=value["policy_id"],
            probability=value["probability"],
        )


def _canonical_mixture(
    weights: Sequence[MixtureWeight],
    *,
    require_nonempty: bool,
) -> tuple[MixtureWeight, ...]:
    canonical = tuple(sorted(weights, key=lambda item: item.policy_id))
    if require_nonempty and not canonical:
        raise ValueError("Frozen opponent mixture must be non-empty.")
    policy_ids = [item.policy_id for item in canonical]
    if len(set(policy_ids)) != len(policy_ids):
        raise ValueError("Opponent mixture contains duplicate policy ids.")
    probability_sum = math.fsum(item.probability for item in canonical)
    if canonical and not math.isclose(probability_sum, 1.0, abs_tol=1e-12):
        raise ValueError(
            f"Opponent mixture probabilities must sum to one (got {probability_sum!r})."
        )
    return canonical


@dataclass(frozen=True)
class PhaseSpec:
    """A content-addressed semantic training boundary."""

    phase_hash: str
    phase_id: str
    learner_policy_id: str
    kind: PhaseKind
    learner_role: PolicyRole
    initialization: PhaseInitialization
    target_population_path: str
    target_population_revision_id: str
    target_distribution: tuple[MixtureWeight, ...]
    opponent_sampler: OpponentSamplerKind
    archive_policy: ArchivePolicy
    battle_format: str
    team_pool_signature: str
    model_signature: str
    format_version: int = PHASE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != PHASE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported phase format version: {self.format_version!r}."
            )
        _require_nonempty_string(self.phase_id, "Phase id")
        _require_nonempty_string(self.learner_policy_id, "Learner policy id")
        if not isinstance(self.kind, PhaseKind):
            raise ValueError("Phase kind must be a PhaseKind.")
        if not isinstance(self.learner_role, PolicyRole):
            raise ValueError("Learner role must be a PolicyRole.")
        _require_nonempty_string(
            self.target_population_path,
            "Target population path",
        )
        if not _is_sha256(self.target_population_revision_id):
            raise ValueError("Target population revision id must be a SHA-256 hash.")
        _require_nonempty_string(self.battle_format, "Phase battle format")
        if not _is_sha256(self.team_pool_signature):
            raise ValueError("Phase team-pool signature must be a SHA-256 hash.")
        if not _is_sha256(self.model_signature):
            raise ValueError("Phase model signature must be a SHA-256 hash.")
        if not isinstance(self.opponent_sampler, OpponentSamplerKind):
            raise ValueError("Opponent sampler must be an OpponentSamplerKind.")
        if not isinstance(self.archive_policy, ArchivePolicy):
            raise ValueError("Archive policy must be an ArchivePolicy.")

        require_mixture = self.kind is PhaseKind.BEST_RESPONSE
        canonical_distribution = _canonical_mixture(
            self.target_distribution,
            require_nonempty=require_mixture,
        )
        object.__setattr__(self, "target_distribution", canonical_distribution)

        if self.kind is PhaseKind.MAIN:
            if self.learner_role is not PolicyRole.MAIN:
                raise ValueError("A main phase requires learner_role='main'.")
            if self.initialization.mode is not InitializationMode.EXACT_TRAINER:
                raise ValueError(
                    "A main phase must preserve an exact trainer initializer."
                )
            if self.opponent_sampler is not OpponentSamplerKind.ADAPTIVE_PFSP:
                raise ValueError("A main phase requires adaptive PFSP sampling.")
            if self.archive_policy is not ArchivePolicy.MAIN_SNAPSHOTS:
                raise ValueError("A main phase must archive main snapshots.")
            if self.target_distribution:
                raise ValueError(
                    "A main phase's opponent weights are maintained by adaptive PFSP, "
                    "so target_distribution must be empty."
                )
        else:
            if self.learner_role is PolicyRole.MAIN:
                raise ValueError("A best-response phase requires an exploiter role.")
            if self.initialization.mode is not InitializationMode.WEIGHTS_ONLY:
                raise ValueError(
                    "A best-response phase must use weight-only initialization with "
                    "a fresh optimizer."
                )
            if self.opponent_sampler is not OpponentSamplerKind.FROZEN_MIXTURE:
                raise ValueError(
                    "A best-response phase requires a frozen opponent mixture."
                )
            if self.archive_policy is not ArchivePolicy.NONE:
                raise ValueError(
                    "A best-response learner must not enter its own target archive."
                )
            if self.learner_policy_id in {
                item.policy_id for item in self.target_distribution
            }:
                raise ValueError(
                    "A best-response learner cannot be a member of its target mixture."
                )

        expected_hash = canonical_sha256(self._semantic_payload())
        if self.phase_hash != expected_hash:
            raise ValueError(
                "Phase hash does not match its semantic content: "
                f"declared={self.phase_hash!r}, expected={expected_hash!r}."
            )

    @classmethod
    def create(
        cls,
        *,
        phase_id: str,
        learner_policy_id: str,
        kind: PhaseKind,
        learner_role: PolicyRole,
        initialization: PhaseInitialization,
        target_population_path: str,
        target_population_revision_id: str,
        target_distribution: Sequence[MixtureWeight],
        opponent_sampler: OpponentSamplerKind,
        archive_policy: ArchivePolicy,
        battle_format: str,
        team_pool_signature: str,
        model_signature: str,
    ) -> "PhaseSpec":
        canonical_distribution = _canonical_mixture(
            target_distribution,
            require_nonempty=kind is PhaseKind.BEST_RESPONSE,
        )
        semantic_payload = cls._build_semantic_payload(
            phase_id=phase_id,
            learner_policy_id=learner_policy_id,
            kind=kind,
            learner_role=learner_role,
            initialization=initialization,
            target_population_path=target_population_path,
            target_population_revision_id=target_population_revision_id,
            target_distribution=canonical_distribution,
            opponent_sampler=opponent_sampler,
            archive_policy=archive_policy,
            battle_format=battle_format,
            team_pool_signature=team_pool_signature,
            model_signature=model_signature,
        )
        return cls(
            phase_hash=canonical_sha256(semantic_payload),
            phase_id=phase_id,
            learner_policy_id=learner_policy_id,
            kind=kind,
            learner_role=learner_role,
            initialization=initialization,
            target_population_path=target_population_path,
            target_population_revision_id=target_population_revision_id,
            target_distribution=canonical_distribution,
            opponent_sampler=opponent_sampler,
            archive_policy=archive_policy,
            battle_format=battle_format,
            team_pool_signature=team_pool_signature,
            model_signature=model_signature,
        )

    @staticmethod
    def _build_semantic_payload(
        *,
        phase_id: str,
        learner_policy_id: str,
        kind: PhaseKind,
        learner_role: PolicyRole,
        initialization: PhaseInitialization,
        target_population_path: str,
        target_population_revision_id: str,
        target_distribution: Sequence[MixtureWeight],
        opponent_sampler: OpponentSamplerKind,
        archive_policy: ArchivePolicy,
        battle_format: str,
        team_pool_signature: str,
        model_signature: str,
    ) -> dict[str, object]:
        return {
            "format_version": PHASE_FORMAT_VERSION,
            "phase_id": phase_id,
            "learner_policy_id": learner_policy_id,
            "kind": kind.value,
            "learner_role": learner_role.value,
            "initialization": initialization.to_dict(),
            "target_population_path": target_population_path,
            "target_population_revision_id": target_population_revision_id,
            "target_distribution": [item.to_dict() for item in target_distribution],
            "opponent_sampler": opponent_sampler.value,
            "archive_policy": archive_policy.value,
            "battle_format": battle_format,
            "team_pool_signature": team_pool_signature,
            "model_signature": model_signature,
        }

    def _semantic_payload(self) -> dict[str, object]:
        return self._build_semantic_payload(
            phase_id=self.phase_id,
            learner_policy_id=self.learner_policy_id,
            kind=self.kind,
            learner_role=self.learner_role,
            initialization=self.initialization,
            target_population_path=self.target_population_path,
            target_population_revision_id=self.target_population_revision_id,
            target_distribution=self.target_distribution,
            opponent_sampler=self.opponent_sampler,
            archive_policy=self.archive_policy,
            battle_format=self.battle_format,
            team_pool_signature=self.team_pool_signature,
            model_signature=self.model_signature,
        )

    def to_dict(self) -> dict[str, object]:
        return {"phase_hash": self.phase_hash, **self._semantic_payload()}

    @classmethod
    def from_dict(cls, raw: object) -> "PhaseSpec":
        value = _require_exact_keys(
            raw,
            {
                "format_version",
                "phase_hash",
                "phase_id",
                "learner_policy_id",
                "kind",
                "learner_role",
                "initialization",
                "target_population_path",
                "target_population_revision_id",
                "target_distribution",
                "opponent_sampler",
                "archive_policy",
                "battle_format",
                "team_pool_signature",
                "model_signature",
            },
            "Phase specification",
        )
        raw_distribution = value["target_distribution"]
        if not isinstance(raw_distribution, list):
            raise ValueError("Phase target_distribution must be a list.")
        try:
            kind = PhaseKind(value["kind"])
            learner_role = PolicyRole(value["learner_role"])
            opponent_sampler = OpponentSamplerKind(value["opponent_sampler"])
            archive_policy = ArchivePolicy(value["archive_policy"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Phase specification contains an unsupported enum value."
            ) from exc
        return cls(
            phase_hash=value["phase_hash"],
            phase_id=value["phase_id"],
            learner_policy_id=value["learner_policy_id"],
            kind=kind,
            learner_role=learner_role,
            initialization=PhaseInitialization.from_dict(value["initialization"]),
            target_population_path=value["target_population_path"],
            target_population_revision_id=value["target_population_revision_id"],
            target_distribution=tuple(
                MixtureWeight.from_dict(item) for item in raw_distribution
            ),
            opponent_sampler=opponent_sampler,
            archive_policy=archive_policy,
            battle_format=value["battle_format"],
            team_pool_signature=value["team_pool_signature"],
            model_signature=value["model_signature"],
            format_version=value["format_version"],
        )

    def verify_initializer(self, phase_base_dir: str | Path) -> Path:
        """Verify the initializer relative to its pinned population file.

        ``CheckpointRef.path`` is copied verbatim from the corresponding
        ``PolicyMember``.  Relative paths therefore belong to the population
        revision's directory, even when the phase specification is stored in a
        separate plans directory.
        """
        population_dir = self.resolve_target_population_path(phase_base_dir).parent
        return self.initialization.checkpoint.verify(population_dir)

    def resolve_target_population_path(self, base_dir: str | Path) -> Path:
        path = Path(self.target_population_path).expanduser()
        if not path.is_absolute():
            path = Path(base_dir).expanduser().resolve() / path
        return path.resolve()

    def load_target_population(self, base_dir: str | Path) -> PopulationRevision:
        population = PopulationRevision.load(
            self.resolve_target_population_path(base_dir),
            verify_checkpoints=True,
        )
        self.validate_against_population(population)
        return population

    def validate_against_population(self, population: PopulationRevision) -> None:
        if population.revision_id != self.target_population_revision_id:
            raise ValueError(
                "Phase target population hash does not match the loaded revision."
            )
        if (
            self.kind is PhaseKind.BEST_RESPONSE
            and population.evaluation_action_mode != "sample"
        ):
            raise ValueError(
                "Best-response training samples frozen opponent policies, so its "
                "population must be evaluated with action_mode='sample'."
            )
        for name, phase_value, population_value in (
            ("battle format", self.battle_format, population.battle_format),
            (
                "team-pool signature",
                self.team_pool_signature,
                population.team_pool_signature,
            ),
            ("model signature", self.model_signature, population.model_signature),
        ):
            if phase_value != population_value:
                raise ValueError(
                    f"Phase {name} does not match the target population: "
                    f"phase={phase_value!r}, population={population_value!r}."
                )
        members = {member.policy_id: member for member in population.policies}
        source = members.get(self.initialization.source_policy_id)
        if source is None:
            raise ValueError(
                "Phase initializer policy is absent from the target population."
            )
        if source.checkpoint != self.initialization.checkpoint:
            raise ValueError(
                "Phase initializer checkpoint reference does not exactly match its "
                "population policy."
            )
        missing_targets = sorted(
            item.policy_id
            for item in self.target_distribution
            if item.policy_id not in members
        )
        if missing_targets:
            raise ValueError(
                "Phase mixture references policy ids outside the target population: "
                + ", ".join(missing_targets)
            )
        if self.learner_policy_id in members:
            raise ValueError(
                "Phase learner policy id already exists in the target population."
            )

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = (
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        if target.exists():
            if target.read_text(encoding="utf-8") == serialized:
                return target
            raise FileExistsError(
                f"Refusing to overwrite immutable phase specification: {target!s}"
            )
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_initializer: bool = True,
    ) -> "PhaseSpec":
        source = Path(path).expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Phase specification is not valid JSON: {source!s}"
            ) from exc
        spec = cls.from_dict(raw)
        if verify_initializer:
            spec.verify_initializer(source.parent)
        return spec


class FrozenMixtureSampler:
    """Independent-with-replacement sampling from one immutable mixture."""

    def __init__(
        self,
        weights: Sequence[MixtureWeight],
        *,
        seed: int,
    ) -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Frozen-mixture seed must be an integer in [0, 2**32).")
        self._weights = _canonical_mixture(weights, require_nonempty=True)
        self._mixture_id = canonical_sha256(
            [weight.to_dict() for weight in self._weights]
        )
        self._policy_ids = tuple(weight.policy_id for weight in self._weights)
        cumulative: list[float] = []
        running = 0.0
        for weight in self._weights:
            running += weight.probability
            cumulative.append(running)
        # Exact one avoids a floating endpoint edge in ``bisect``.
        cumulative[-1] = 1.0
        self._cumulative = tuple(cumulative)
        self._rng = random.Random(seed)
        self._draw_count = 0

    @property
    def mixture_id(self) -> str:
        return self._mixture_id

    @property
    def distribution(self) -> tuple[MixtureWeight, ...]:
        return self._weights

    @property
    def draw_count(self) -> int:
        return self._draw_count

    def sample_policy_id(self) -> str:
        threshold = self._rng.random()
        index = bisect.bisect_right(self._cumulative, threshold)
        if index >= len(self._policy_ids):
            index = len(self._policy_ids) - 1
        self._draw_count += 1
        return self._policy_ids[index]

    def sample_policy_ids(self, count: int) -> list[str]:
        if type(count) is not int or count < 0:
            raise ValueError(
                "Frozen-mixture sample count must be non-negative integer."
            )
        return [self.sample_policy_id() for _ in range(count)]

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": FROZEN_SAMPLER_STATE_VERSION,
            "mixture_id": self._mixture_id,
            "draw_count": self._draw_count,
            "rng_state": _tuples_to_lists(self._rng.getstate()),
        }

    def load_state_dict(self, raw: object) -> None:
        state = _require_exact_keys(
            raw,
            {"format_version", "mixture_id", "draw_count", "rng_state"},
            "Frozen-mixture sampler state",
        )
        if state["format_version"] != FROZEN_SAMPLER_STATE_VERSION:
            raise ValueError(
                "Unsupported frozen-mixture sampler state version: "
                f"{state['format_version']!r}."
            )
        if state["mixture_id"] != self._mixture_id:
            raise ValueError(
                "Frozen-mixture sampler state belongs to a different target mixture."
            )
        draw_count = state["draw_count"]
        if type(draw_count) is not int or draw_count < 0:
            raise ValueError("Frozen-mixture draw_count must be non-negative integer.")
        rng_state = _lists_to_tuples(state["rng_state"])
        validator = random.Random()
        try:
            validator.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("Frozen-mixture sampler RNG state is invalid.") from exc

        # Mutate only after the complete state has been validated.
        self._rng.setstate(rng_state)
        self._draw_count = draw_count


@dataclass(frozen=True)
class PhaseRunState:
    """Exact mutable boundary paired with one best-response trainer checkpoint."""

    phase_hash: str
    target_population_revision_id: str
    training_state_id: str
    completed_updates: int
    sampler_state: dict[str, object]
    team_pool_signature: str | None
    team_pool_rng_state: tuple | None
    opponent_assignments: tuple[tuple[int, str], ...]
    format_version: int = PHASE_RUN_STATE_VERSION

    def __post_init__(self) -> None:
        if self.format_version != PHASE_RUN_STATE_VERSION:
            raise ValueError(
                f"Unsupported phase-run-state format version: {self.format_version!r}."
            )
        if not _is_sha256(self.phase_hash):
            raise ValueError("Phase-run-state phase_hash must be a SHA-256 hash.")
        if not _is_sha256(self.target_population_revision_id):
            raise ValueError(
                "Phase-run-state target population id must be a SHA-256 hash."
            )
        _require_nonempty_string(self.training_state_id, "Training state id")
        if type(self.completed_updates) is not int or self.completed_updates < 0:
            raise ValueError("Phase completed_updates must be non-negative integer.")
        if not isinstance(self.sampler_state, dict):
            raise ValueError("Phase sampler_state must be an object.")
        object.__setattr__(self, "sampler_state", copy.deepcopy(self.sampler_state))

        if (self.team_pool_signature is None) != (self.team_pool_rng_state is None):
            raise ValueError(
                "Phase team-pool signature and RNG state must either both be present "
                "or both be absent."
            )
        if self.team_pool_signature is not None and not _is_sha256(
            self.team_pool_signature
        ):
            raise ValueError("Phase team-pool signature must be a SHA-256 hash.")

        canonical_assignments = tuple(sorted(self.opponent_assignments))
        object.__setattr__(self, "opponent_assignments", canonical_assignments)
        if not canonical_assignments:
            raise ValueError("Phase run state requires opponent assignments.")
        groups = [group for group, _ in canonical_assignments]
        if groups != list(range(len(groups))):
            raise ValueError(
                "Phase opponent assignments must use contiguous groups from zero."
            )
        for group, checkpoint_path in canonical_assignments:
            if type(group) is not int:
                raise ValueError("Phase opponent group ids must be integers.")
            _require_nonempty_string(
                checkpoint_path,
                "Phase opponent checkpoint path",
            )

    def validate_resume(
        self,
        *,
        phase: PhaseSpec,
        training_state_id: str | None,
        trainer_completed_updates: int,
        sampler: FrozenMixtureSampler,
    ) -> None:
        if phase.phase_hash != self.phase_hash:
            raise ValueError(
                "Resume phase hash does not match the paired phase run state."
            )
        if phase.target_population_revision_id != self.target_population_revision_id:
            raise ValueError(
                "Resume target population does not match the paired phase run state."
            )
        if training_state_id != self.training_state_id:
            raise ValueError(
                "Trainer checkpoint and paired phase state have different IDs."
            )
        if trainer_completed_updates != self.completed_updates:
            raise ValueError(
                "Trainer checkpoint and paired phase state disagree on completed "
                f"updates: trainer={trainer_completed_updates}, "
                f"phase={self.completed_updates}."
            )
        # ``FrozenMixtureSampler`` validates the pinned mixture id and the full
        # Python RNG state before mutating itself.
        sampler.load_state_dict(self.sampler_state)

    def assignments_dict(self) -> dict[int, str]:
        return dict(self.opponent_assignments)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "phase_hash": self.phase_hash,
            "target_population_revision_id": self.target_population_revision_id,
            "training_state_id": self.training_state_id,
            "completed_updates": self.completed_updates,
            "sampler_state": copy.deepcopy(self.sampler_state),
            "team_pool_signature": self.team_pool_signature,
            "team_pool_rng_state": _tuples_to_lists(self.team_pool_rng_state),
            "opponent_assignments": {
                str(group): checkpoint_path
                for group, checkpoint_path in self.opponent_assignments
            },
        }

    @classmethod
    def from_dict(cls, raw: object) -> "PhaseRunState":
        value = _require_exact_keys(
            raw,
            {
                "format_version",
                "phase_hash",
                "target_population_revision_id",
                "training_state_id",
                "completed_updates",
                "sampler_state",
                "team_pool_signature",
                "team_pool_rng_state",
                "opponent_assignments",
            },
            "Phase run state",
        )
        raw_assignments = value["opponent_assignments"]
        if not isinstance(raw_assignments, Mapping):
            raise ValueError("Phase opponent_assignments must be an object.")
        assignments: list[tuple[int, str]] = []
        for raw_group, checkpoint_path in raw_assignments.items():
            if not isinstance(raw_group, str) or not raw_group.isdecimal():
                raise ValueError("Phase opponent-assignment keys must be integers.")
            assignments.append((int(raw_group), checkpoint_path))
        sampler_state = value["sampler_state"]
        if not isinstance(sampler_state, dict):
            raise ValueError("Phase sampler_state must be an object.")
        team_pool_rng_state = _lists_to_tuples(value["team_pool_rng_state"])
        if team_pool_rng_state is not None and not isinstance(
            team_pool_rng_state,
            tuple,
        ):
            raise ValueError("Phase team-pool RNG state must be a tuple or null.")
        return cls(
            phase_hash=value["phase_hash"],
            target_population_revision_id=value["target_population_revision_id"],
            training_state_id=value["training_state_id"],
            completed_updates=value["completed_updates"],
            sampler_state=sampler_state,
            team_pool_signature=value["team_pool_signature"],
            team_pool_rng_state=team_pool_rng_state,
            opponent_assignments=tuple(assignments),
            format_version=value["format_version"],
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        serialized = (
            json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "PhaseRunState":
        source = Path(path).expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Phase run state is not valid JSON: {source!s}") from exc
        return cls.from_dict(raw)
