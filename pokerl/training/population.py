"""Immutable evaluated-policy populations for population-based self-play.

This module deliberately does not know how to train a policy.  It provides the
content-addressed, frozen objects that a future training phase can safely pin:

* immutable references to checkpoint bytes,
* selected policy metadata and pairwise frozen-policy evaluations, and
* a certified approximate meta-strategy for a finite symmetric zero-sum game.

The dense rolling archive used for ordinary self-play is a different concept.
Only policies with completed frozen-versus-frozen payoff rows belong in an
``PopulationRevision`` used by a meta-game solver.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


POPULATION_FORMAT_VERSION = 2
_SHA256_HEX_LENGTH = 64


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
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


def canonical_sha256(payload: object) -> str:
    """Return a stable SHA-256 over a JSON-compatible semantic payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without retaining checkpoint-sized buffers in memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {resolved!s}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_checkpoint_path(path: Path, relative_to: Path | None) -> str:
    resolved = path.expanduser().resolve()
    if relative_to is not None:
        base = relative_to.expanduser().resolve()
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


class PolicyRole(StrEnum):
    """A policy's functional role in the evaluated population."""

    MAIN = "main"
    MAIN_EXPLOITER = "main_exploiter"
    LEAGUE_EXPLOITER = "league_exploiter"


@dataclass(frozen=True)
class CheckpointRef:
    """An immutable reference to exact checkpoint bytes."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.path, "Checkpoint path")
        if not _is_sha256(self.sha256):
            raise ValueError(
                "Checkpoint sha256 must be 64 lowercase hexadecimal digits."
            )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        relative_to: str | Path | None = None,
    ) -> "CheckpointRef":
        checkpoint_path = Path(path)
        base = Path(relative_to) if relative_to is not None else None
        return cls(
            path=_portable_checkpoint_path(checkpoint_path, base),
            sha256=file_sha256(checkpoint_path),
        )

    def resolve(self, base_dir: str | Path) -> Path:
        path = Path(self.path).expanduser()
        if not path.is_absolute():
            path = Path(base_dir).expanduser().resolve() / path
        return path.resolve()

    def verify(self, base_dir: str | Path) -> Path:
        resolved = self.resolve(base_dir)
        actual = file_sha256(resolved)
        if actual != self.sha256:
            raise ValueError(
                "Checkpoint hash mismatch for "
                f"{resolved!s}: expected={self.sha256}, actual={actual}."
            )
        return resolved

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, raw: object) -> "CheckpointRef":
        value = _require_exact_keys(raw, {"path", "sha256"}, "Checkpoint reference")
        return cls(path=value["path"], sha256=value["sha256"])


@dataclass(frozen=True)
class PolicyMember:
    """One selected frozen policy in an empirical population."""

    policy_id: str
    role: PolicyRole
    checkpoint: CheckpointRef
    lineage_id: str
    parent_policy_id: str | None
    generation: int
    source_training_state_id: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.policy_id, "Policy id")
        _require_nonempty_string(self.lineage_id, "Policy lineage id")
        if not isinstance(self.role, PolicyRole):
            raise ValueError("Policy role must be a PolicyRole.")
        if self.parent_policy_id is not None:
            _require_nonempty_string(self.parent_policy_id, "Parent policy id")
            if self.parent_policy_id == self.policy_id:
                raise ValueError("A policy cannot be its own parent.")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("Policy generation must be a non-negative integer.")
        if self.source_training_state_id is not None:
            _require_nonempty_string(
                self.source_training_state_id,
                "Source training state id",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "role": self.role.value,
            "checkpoint": self.checkpoint.to_dict(),
            "lineage_id": self.lineage_id,
            "parent_policy_id": self.parent_policy_id,
            "generation": self.generation,
            "source_training_state_id": self.source_training_state_id,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "PolicyMember":
        value = _require_exact_keys(
            raw,
            {
                "policy_id",
                "role",
                "checkpoint",
                "lineage_id",
                "parent_policy_id",
                "generation",
                "source_training_state_id",
            },
            "Policy member",
        )
        try:
            role = PolicyRole(value["role"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported policy role: {value['role']!r}.") from exc
        return cls(
            policy_id=value["policy_id"],
            role=role,
            checkpoint=CheckpointRef.from_dict(value["checkpoint"]),
            lineage_id=value["lineage_id"],
            parent_policy_id=value["parent_policy_id"],
            generation=value["generation"],
            source_training_state_id=value["source_training_state_id"],
        )


@dataclass(frozen=True)
class PairwisePayoff:
    """Frozen mirrored-match evidence for one canonical unordered policy pair.

    ``wins`` and ``losses`` are from ``row_policy_id``'s perspective.  Policy
    ids are stored in lexical order so the opposite matrix entry has exactly
    the negated utility rather than a separately noisy estimate.
    """

    row_policy_id: str
    column_policy_id: str
    wins: int
    losses: int
    draws: int
    evaluation_sha256: str
    evaluation_path: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.row_policy_id, "Payoff row policy id")
        _require_nonempty_string(self.column_policy_id, "Payoff column policy id")
        if self.row_policy_id >= self.column_policy_id:
            raise ValueError(
                "Pairwise payoff policy ids must be distinct and lexically ordered."
            )
        for name, value in (
            ("wins", self.wins),
            ("losses", self.losses),
            ("draws", self.draws),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"Pairwise payoff {name} must be non-negative integer."
                )
        if self.games <= 0:
            raise ValueError("Pairwise payoff requires at least one evaluated game.")
        if not _is_sha256(self.evaluation_sha256):
            raise ValueError(
                "Payoff evaluation_sha256 must be 64 lowercase hexadecimal digits."
            )
        _require_nonempty_string(self.evaluation_path, "Payoff evaluation path")

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def score_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    @property
    def zero_sum_utility(self) -> float:
        return (self.wins - self.losses) / self.games

    def to_dict(self) -> dict[str, object]:
        return {
            "row_policy_id": self.row_policy_id,
            "column_policy_id": self.column_policy_id,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "evaluation_sha256": self.evaluation_sha256,
            "evaluation_path": self.evaluation_path,
        }

    @classmethod
    def from_dict(cls, raw: object) -> "PairwisePayoff":
        value = _require_exact_keys(
            raw,
            {
                "row_policy_id",
                "column_policy_id",
                "wins",
                "losses",
                "draws",
                "evaluation_sha256",
                "evaluation_path",
            },
            "Pairwise payoff",
        )
        return cls(
            row_policy_id=value["row_policy_id"],
            column_policy_id=value["column_policy_id"],
            wins=value["wins"],
            losses=value["losses"],
            draws=value["draws"],
            evaluation_sha256=value["evaluation_sha256"],
            evaluation_path=value["evaluation_path"],
        )

    def resolve_evaluation(self, base_dir: str | Path) -> Path:
        path = Path(self.evaluation_path).expanduser()
        if not path.is_absolute():
            path = Path(base_dir).expanduser().resolve() / path
        return path.resolve()

    def verify_evaluation(self, base_dir: str | Path) -> Path:
        resolved = self.resolve_evaluation(base_dir)
        actual = file_sha256(resolved)
        if actual != self.evaluation_sha256:
            raise ValueError(
                "Pairwise evaluation hash mismatch for "
                f"{resolved!s}: expected={self.evaluation_sha256}, actual={actual}."
            )
        return resolved


@dataclass(frozen=True)
class PopulationRevision:
    """A content-addressed immutable empirical-game revision."""

    revision_id: str
    parent_revision_id: str | None
    battle_format: str
    team_pool_signature: str
    model_signature: str
    evaluation_action_mode: str
    policies: tuple[PolicyMember, ...]
    payoffs: tuple[PairwisePayoff, ...]
    format_version: int = POPULATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != POPULATION_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported population format version: {self.format_version!r}."
            )
        if self.parent_revision_id is not None and not _is_sha256(
            self.parent_revision_id
        ):
            raise ValueError("Parent population revision id must be a SHA-256 hash.")
        _require_nonempty_string(self.battle_format, "Population battle format")
        if not _is_sha256(self.team_pool_signature):
            raise ValueError("Team-pool signature must be a SHA-256 hash.")
        if not _is_sha256(self.model_signature):
            raise ValueError("Model signature must be a SHA-256 hash.")
        if self.evaluation_action_mode not in {"sample", "argmax"}:
            raise ValueError(
                "Population evaluation action mode must be 'sample' or 'argmax'."
            )
        if not self.policies:
            raise ValueError("Population revision requires at least one policy.")

        canonical_policies = tuple(
            sorted(self.policies, key=lambda item: item.policy_id)
        )
        canonical_payoffs = tuple(
            sorted(
                self.payoffs,
                key=lambda item: (item.row_policy_id, item.column_policy_id),
            )
        )
        object.__setattr__(self, "policies", canonical_policies)
        object.__setattr__(self, "payoffs", canonical_payoffs)

        policy_ids = [member.policy_id for member in self.policies]
        if len(set(policy_ids)) != len(policy_ids):
            raise ValueError("Population revision contains duplicate policy ids.")
        checkpoint_hashes = [member.checkpoint.sha256 for member in self.policies]
        if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
            raise ValueError(
                "Population revision contains duplicate checkpoint contents."
            )
        known_policy_ids = set(policy_ids)
        for member in self.policies:
            if (
                member.parent_policy_id is not None
                and member.parent_policy_id not in known_policy_ids
            ):
                raise ValueError(
                    f"Policy {member.policy_id!r} references missing parent "
                    f"{member.parent_policy_id!r}."
                )
        payoff_pairs: set[tuple[str, str]] = set()
        for payoff in self.payoffs:
            pair = (payoff.row_policy_id, payoff.column_policy_id)
            if pair in payoff_pairs:
                raise ValueError(f"Population contains duplicate payoff pair {pair!r}.")
            payoff_pairs.add(pair)
            if not set(pair).issubset(known_policy_ids):
                raise ValueError(
                    f"Population payoff references unknown policy: {pair!r}."
                )

        expected_revision_id = canonical_sha256(self._semantic_payload())
        if self.revision_id != expected_revision_id:
            raise ValueError(
                "Population revision id does not match its semantic content: "
                f"declared={self.revision_id!r}, expected={expected_revision_id!r}."
            )

    @classmethod
    def create(
        cls,
        *,
        parent_revision_id: str | None,
        battle_format: str,
        team_pool_signature: str,
        model_signature: str,
        evaluation_action_mode: str,
        policies: Sequence[PolicyMember],
        payoffs: Sequence[PairwisePayoff] = (),
    ) -> "PopulationRevision":
        canonical_policies = tuple(sorted(policies, key=lambda item: item.policy_id))
        canonical_payoffs = tuple(
            sorted(
                payoffs,
                key=lambda item: (item.row_policy_id, item.column_policy_id),
            )
        )
        semantic_payload = cls._build_semantic_payload(
            parent_revision_id=parent_revision_id,
            battle_format=battle_format,
            team_pool_signature=team_pool_signature,
            model_signature=model_signature,
            evaluation_action_mode=evaluation_action_mode,
            policies=canonical_policies,
            payoffs=canonical_payoffs,
        )
        return cls(
            revision_id=canonical_sha256(semantic_payload),
            parent_revision_id=parent_revision_id,
            battle_format=battle_format,
            team_pool_signature=team_pool_signature,
            model_signature=model_signature,
            evaluation_action_mode=evaluation_action_mode,
            policies=canonical_policies,
            payoffs=canonical_payoffs,
        )

    @staticmethod
    def _build_semantic_payload(
        *,
        parent_revision_id: str | None,
        battle_format: str,
        team_pool_signature: str,
        model_signature: str,
        evaluation_action_mode: str,
        policies: Sequence[PolicyMember],
        payoffs: Sequence[PairwisePayoff],
    ) -> dict[str, object]:
        return {
            "format_version": POPULATION_FORMAT_VERSION,
            "parent_revision_id": parent_revision_id,
            "battle_format": battle_format,
            "team_pool_signature": team_pool_signature,
            "model_signature": model_signature,
            "evaluation_action_mode": evaluation_action_mode,
            "policies": [member.to_dict() for member in policies],
            "payoffs": [payoff.to_dict() for payoff in payoffs],
        }

    def _semantic_payload(self) -> dict[str, object]:
        return self._build_semantic_payload(
            parent_revision_id=self.parent_revision_id,
            battle_format=self.battle_format,
            team_pool_signature=self.team_pool_signature,
            model_signature=self.model_signature,
            evaluation_action_mode=self.evaluation_action_mode,
            policies=self.policies,
            payoffs=self.payoffs,
        )

    def to_dict(self) -> dict[str, object]:
        return {"revision_id": self.revision_id, **self._semantic_payload()}

    @classmethod
    def from_dict(cls, raw: object) -> "PopulationRevision":
        value = _require_exact_keys(
            raw,
            {
                "format_version",
                "revision_id",
                "parent_revision_id",
                "battle_format",
                "team_pool_signature",
                "model_signature",
                "evaluation_action_mode",
                "policies",
                "payoffs",
            },
            "Population revision",
        )
        raw_policies = value["policies"]
        raw_payoffs = value["payoffs"]
        if not isinstance(raw_policies, list):
            raise ValueError("Population policies must be a list.")
        if not isinstance(raw_payoffs, list):
            raise ValueError("Population payoffs must be a list.")
        return cls(
            revision_id=value["revision_id"],
            parent_revision_id=value["parent_revision_id"],
            battle_format=value["battle_format"],
            team_pool_signature=value["team_pool_signature"],
            model_signature=value["model_signature"],
            evaluation_action_mode=value["evaluation_action_mode"],
            policies=tuple(PolicyMember.from_dict(item) for item in raw_policies),
            payoffs=tuple(PairwisePayoff.from_dict(item) for item in raw_payoffs),
            format_version=value["format_version"],
        )

    def verify_checkpoints(self, base_dir: str | Path) -> dict[str, Path]:
        return {
            member.policy_id: member.checkpoint.verify(base_dir)
            for member in self.policies
        }

    def verify_evaluations(
        self,
        base_dir: str | Path,
    ) -> dict[tuple[str, str], Path]:
        """Verify every frozen summary artifact referenced by this revision."""
        return {
            (payoff.row_policy_id, payoff.column_policy_id): payoff.verify_evaluation(
                base_dir
            )
            for payoff in self.payoffs
        }

    def save(self, path: str | Path) -> Path:
        """Atomically create this revision without permitting mutation in place."""
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
                f"Refusing to overwrite immutable population revision: {target!s}"
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
        verify_checkpoints: bool = True,
        verify_evaluations: bool = True,
    ) -> "PopulationRevision":
        source = Path(path).expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Population revision is not valid JSON: {source!s}"
            ) from exc
        revision = cls.from_dict(raw)
        if verify_checkpoints:
            revision.verify_checkpoints(source.parent)
        if verify_evaluations:
            revision.verify_evaluations(source.parent)
        return revision

    def payoff_matrix(
        self,
        *,
        require_complete: bool = True,
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Build the empirical symmetric zero-sum payoff matrix.

        Mirrored evaluation evidence is stored once per unordered pair.  The
        returned matrix is exactly skew-symmetric by construction, avoiding a
        spurious non-zero self-play game value caused by independent noise in
        two directed estimates.
        """
        policy_ids = tuple(member.policy_id for member in self.policies)
        index = {policy_id: position for position, policy_id in enumerate(policy_ids)}
        matrix = np.zeros((len(policy_ids), len(policy_ids)), dtype=np.float64)
        observed: set[tuple[int, int]] = set()
        for payoff in self.payoffs:
            row = index[payoff.row_policy_id]
            column = index[payoff.column_policy_id]
            utility = payoff.zero_sum_utility
            matrix[row, column] = utility
            matrix[column, row] = -utility
            observed.add((min(row, column), max(row, column)))

        if require_complete:
            missing = [
                (policy_ids[row], policy_ids[column])
                for row in range(len(policy_ids))
                for column in range(row + 1, len(policy_ids))
                if (row, column) not in observed
            ]
            if missing:
                raise ValueError(
                    "Population payoff matrix is incomplete; missing pair(s): "
                    + ", ".join(f"{left}/{right}" for left, right in missing)
                )
        return policy_ids, matrix


@dataclass(frozen=True)
class ZeroSumSolution:
    """A finite-game meta-strategy with an explicit optimality certificate."""

    probabilities: tuple[float, ...]
    exploitability_gap: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("Zero-sum solution requires a non-empty strategy.")
        if any(not math.isfinite(value) or value < 0.0 for value in self.probabilities):
            raise ValueError("Zero-sum probabilities must be finite and non-negative.")
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("Zero-sum probabilities must sum to one.")
        if not math.isfinite(self.exploitability_gap) or self.exploitability_gap < 0.0:
            raise ValueError("Exploitability gap must be finite and non-negative.")
        if type(self.iterations) is not int or self.iterations < 0:
            raise ValueError("Solver iterations must be a non-negative integer.")


def _softmax_from_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / weights.sum()


def symmetric_exploitability(
    payoff_matrix: np.ndarray,
    probabilities: Sequence[float],
) -> float:
    """Return the exact pure-best-response gap of one symmetric strategy."""
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    strategy = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Payoff matrix must be square.")
    if strategy.shape != (matrix.shape[0],):
        raise ValueError("Strategy length must match the payoff matrix.")
    if not np.isfinite(matrix).all() or not np.isfinite(strategy).all():
        raise ValueError("Payoff matrix and strategy must be finite.")
    if (strategy < 0.0).any() or not math.isclose(
        float(strategy.sum()), 1.0, abs_tol=1e-9
    ):
        raise ValueError("Strategy must be a probability distribution.")
    if not np.allclose(matrix, -matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError("Symmetric zero-sum payoff matrix must be skew-symmetric.")
    # q^T A q is zero for skew-symmetric A, so max_i (Aq)_i is
    # exactly the gain available to a pure unilateral best response.
    return max(0.0, float(np.max(matrix @ strategy)))


def solve_symmetric_zero_sum(
    payoff_matrix: np.ndarray,
    *,
    tolerance: float = 5e-3,
    max_iterations: int = 200_000,
    check_every: int = 100,
) -> ZeroSumSolution:
    """Solve a finite symmetric zero-sum game by averaged no-regret play.

    The implementation uses multiplicative-weights learners for both sides and
    returns their averaged symmetric strategy.  The reported exploitability is
    not an optimizer heuristic: it is the exact pure-best-response gap measured
    against the returned strategy.  Callers can therefore reject a result when
    ``converged`` is false rather than treating an unverified approximation as
    a Nash mixture.

    This keeps population planning independent of an optional linear-program
    dependency.  Runtime is acceptable for the small selected populations this
    module is intended to represent; dense training archives should not be fed
    to the empirical-game solver.
    """
    matrix = np.asarray(payoff_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Payoff matrix must be square.")
    if matrix.shape[0] == 0:
        raise ValueError("Payoff matrix must be non-empty.")
    if not np.isfinite(matrix).all():
        raise ValueError("Payoff matrix must be finite.")
    if not np.allclose(matrix, -matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError("Symmetric zero-sum payoff matrix must be skew-symmetric.")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive.")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer.")
    if type(check_every) is not int or check_every <= 0:
        raise ValueError("check_every must be a positive integer.")

    size = matrix.shape[0]
    if size == 1:
        return ZeroSumSolution((1.0,), 0.0, 0, True)

    scale = max(1.0, float(np.max(np.abs(matrix))))
    normalized = matrix / scale
    # Constant-horizon Hedge has O(sqrt(log(n) / T)) average external regret.
    learning_rate = math.sqrt(2.0 * math.log(size) / max_iterations)
    row_log_weights = np.zeros(size, dtype=np.float64)
    column_log_weights = np.zeros(size, dtype=np.float64)
    row_sum = np.zeros(size, dtype=np.float64)
    column_sum = np.zeros(size, dtype=np.float64)

    best_strategy = np.full(size, 1.0 / size, dtype=np.float64)
    best_gap = symmetric_exploitability(matrix, best_strategy)
    if best_gap <= tolerance:
        return ZeroSumSolution(
            tuple(float(value) for value in best_strategy),
            best_gap,
            0,
            True,
        )

    final_iteration = 0
    for iteration in range(1, max_iterations + 1):
        row = _softmax_from_log_weights(row_log_weights)
        column = _softmax_from_log_weights(column_log_weights)
        row_sum += row
        column_sum += column

        row_log_weights += learning_rate * (normalized @ column)
        column_log_weights += learning_rate * (-(row @ normalized))
        # Log weights are invariant to a common offset; rebasing prevents a
        # very long planning run from accumulating unnecessary magnitudes.
        row_log_weights -= np.max(row_log_weights)
        column_log_weights -= np.max(column_log_weights)

        if iteration % check_every != 0 and iteration != max_iterations:
            continue
        candidate = (row_sum + column_sum) / (2.0 * iteration)
        candidate /= candidate.sum()
        gap = symmetric_exploitability(matrix, candidate)
        if gap < best_gap:
            best_strategy = candidate.copy()
            best_gap = gap
        final_iteration = iteration
        if gap <= tolerance:
            return ZeroSumSolution(
                tuple(float(value) for value in candidate),
                gap,
                iteration,
                True,
            )

    return ZeroSumSolution(
        tuple(float(value) for value in best_strategy),
        best_gap,
        final_iteration,
        False,
    )
