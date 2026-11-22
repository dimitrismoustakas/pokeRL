"""Aggregate mirrored evaluations with hierarchical paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pokerl.eval.metrics import outcome_summary
from pokerl.training.population import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _PairedRun:
    source: Path
    source_sha256: str
    seed: int
    schedule_sha256: str
    pair_scores: np.ndarray
    role_differences: np.ndarray
    rewards: tuple[float, ...]


def _load_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        duplicates: list[str] = []
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        if duplicates:
            raise ValueError(
                f"{path} contains duplicate JSON keys: {sorted(set(duplicates))}"
            )
        return result

    def reject_non_finite(value: str) -> object:
        raise ValueError(f"{path} contains non-finite JSON number {value!r}.")

    loaded = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_non_finite,
    )
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{path} must contain a JSON object.")
    return loaded


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object.")
    return value


def _score(reward: float) -> float:
    if not math.isfinite(reward):
        raise ValueError("Battle rewards must be finite.")
    return 1.0 if reward > 0.0 else (0.5 if reward == 0.0 else 0.0)


def _checkpoint_sha(config: Mapping[str, Any], key: str) -> str:
    reference = _mapping(config.get(key), f"config.{key}")
    digest = reference.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"config.{key}.sha256 must be a SHA-256 digest.")
    return digest


def _identity(summary: Mapping[str, Any]) -> tuple[object, ...]:
    config = _mapping(summary.get("config"), "config")
    return (
        config.get("evaluation_mode", "checkpoint_comparison"),
        config.get("checkpoint_policy_id"),
        config.get("anchor_policy_id"),
        _checkpoint_sha(config, "checkpoint"),
        _checkpoint_sha(config, "anchor"),
        config.get("battle_format"),
        config.get("simulator", "showdown"),
        config.get("action_mode"),
        config.get("team_pool_signature"),
        config.get("model_signature"),
    )


def _parse_run(path: Path, summary: Mapping[str, Any]) -> _PairedRun:
    config = _mapping(summary.get("config"), "config")
    if config.get("mirrored") is not True:
        raise ValueError(f"{path} is not a mirrored evaluation.")
    seed = config.get("seed")
    if type(seed) is not int:
        raise ValueError(f"{path} config.seed must be an integer.")
    schedule = _mapping(config.get("team_pair_schedule"), "team_pair_schedule")
    schedule_sha256 = schedule.get("schedule_sha256")
    if not isinstance(schedule_sha256, str) or len(schedule_sha256) != 64:
        raise ValueError(f"{path} has no valid team-pair schedule hash.")

    results = _mapping(summary.get("results"), "results")
    directions = results.get("directions")
    if not isinstance(directions, list) or len(directions) != 2:
        raise ValueError(f"{path} must contain exactly two mirrored directions.")
    details: list[list[Mapping[str, Any]]] = []
    for direction_index, direction in enumerate(directions):
        direction_map = _mapping(direction, f"direction {direction_index + 1}")
        raw_games = direction_map.get("games_detail")
        if not isinstance(raw_games, list) or not raw_games:
            raise ValueError(f"{path} direction {direction_index + 1} has no games.")
        details.append(
            [
                _mapping(game, f"direction {direction_index + 1} game {game_index + 1}")
                for game_index, game in enumerate(raw_games)
            ]
        )
    if len(details[0]) != len(details[1]):
        raise ValueError(f"{path} mirrored directions have different game counts.")

    same_policy = config.get("evaluation_mode") == "same_policy_role_bias"
    pair_scores: list[float] = []
    role_differences: list[float] = []
    rewards: list[float] = []
    for pair_index, (forward, reverse) in enumerate(
        zip(details[0], details[1], strict=True)
    ):
        if forward.get("external_team_file") != reverse.get(
            "opponent_team_file"
        ) or forward.get("opponent_team_file") != reverse.get("external_team_file"):
            raise ValueError(
                f"{path} mirrored pair {pair_index + 1} is not role-swapped."
            )
        forward_reward = float(forward.get("reward"))
        reverse_external_reward = float(reverse.get("reward"))
        if same_policy:
            forward_score = _score(forward_reward)
            reverse_score = _score(reverse_external_reward)
            rewards.extend((forward_reward, reverse_external_reward))
        else:
            forward_score = _score(forward_reward)
            reverse_score = _score(-reverse_external_reward)
            rewards.extend((forward_reward, -reverse_external_reward))
        pair_scores.append(0.5 * (forward_score + reverse_score))
        role_differences.append(forward_score - reverse_score)

    return _PairedRun(
        source=path,
        source_sha256=file_sha256(path),
        seed=seed,
        schedule_sha256=schedule_sha256,
        pair_scores=np.asarray(pair_scores, dtype=np.float64),
        role_differences=np.asarray(role_differences, dtype=np.float64),
        rewards=tuple(rewards),
    )


def hierarchical_paired_bootstrap_ci(
    groups: list[np.ndarray],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    """Resample seed strata, then whole mirrored pairs within selected seeds.

    The outer resampling makes the interval unconditional on the particular
    evaluation seeds that happened to be run. Each selected seed retains its
    original number of paired matchups, so the bootstrap targets the pooled
    paired-game estimator even when seed schedules have unequal sizes.
    """
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")
    if not groups or any(group.ndim != 1 or len(group) == 0 for group in groups):
        raise ValueError("Bootstrap groups must be non-empty one-dimensional arrays.")
    if any(not np.isfinite(group).all() for group in groups):
        raise ValueError("Bootstrap groups must contain only finite values.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    strata = len(groups)
    for sample_index in range(samples):
        score_sum = 0.0
        paired_games = 0
        sampled_strata = rng.integers(0, strata, size=strata)
        for stratum_index in sampled_strata:
            group = groups[int(stratum_index)]
            indices = rng.integers(0, len(group), size=len(group))
            score_sum += float(group[indices].sum())
            paired_games += len(group)
        estimates[sample_index] = score_sum / paired_games
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return [float(lower), float(upper)]


def _values_grouped_by_seed(
    runs: list[_PairedRun],
    attribute: str,
) -> list[np.ndarray]:
    """Combine schedule partitions without treating one seed as independent twice."""
    grouped: dict[int, list[np.ndarray]] = {}
    for run in sorted(runs, key=lambda item: (item.seed, item.schedule_sha256)):
        value = getattr(run, attribute)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"_PairedRun.{attribute} must be an ndarray.")
        grouped.setdefault(run.seed, []).append(value)
    return [np.concatenate(grouped[seed]) for seed in sorted(grouped)]


def aggregate_anchor_summaries(
    paths: list[Path],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    if not paths:
        raise ValueError("At least one summary is required.")
    summaries = [_load_json(path) for path in paths]
    expected_identity = _identity(summaries[0])
    for path, summary in zip(paths[1:], summaries[1:], strict=True):
        if _identity(summary) != expected_identity:
            raise ValueError(f"Incompatible evaluation identity in {path}.")
    runs = [_parse_run(path, summary) for path, summary in zip(paths, summaries)]
    source_hashes = [run.source_sha256 for run in runs]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("The same evaluation artifact was supplied more than once.")
    seed_schedules = [(run.seed, run.schedule_sha256) for run in runs]
    if len(set(seed_schedules)) != len(seed_schedules):
        raise ValueError("Duplicate seed/schedule evaluation strata are not allowed.")

    all_rewards = [reward for run in runs for reward in run.rewards]
    outcomes = outcome_summary(all_rewards)
    score_groups = _values_grouped_by_seed(runs, "pair_scores")
    role_groups = _values_grouped_by_seed(runs, "role_differences")
    paired_score = float(
        sum(float(group.sum()) for group in score_groups)
        / sum(len(group) for group in score_groups)
    )
    role_difference = float(
        sum(float(group.sum()) for group in role_groups)
        / sum(len(group) for group in role_groups)
    )
    per_seed_scores = [float(group.mean()) for group in score_groups]
    config = _mapping(summaries[0].get("config"), "config")
    same_policy = config.get("evaluation_mode") == "same_policy_role_bias"
    return {
        "config": {
            "simulator": config.get("simulator", "showdown"),
            "evaluation_mode": config.get("evaluation_mode", "checkpoint_comparison"),
            "checkpoint_policy_id": config.get("checkpoint_policy_id"),
            "anchor_policy_id": config.get("anchor_policy_id"),
            "checkpoint_sha256": _checkpoint_sha(config, "checkpoint"),
            "anchor_sha256": _checkpoint_sha(config, "anchor"),
            "battle_format": config.get("battle_format"),
            "action_mode": config.get("action_mode"),
            "team_pool_signature": config.get("team_pool_signature"),
            "model_signature": config.get("model_signature"),
            "simulator_rng_seeded": all(
                _mapping(summary.get("config"), "config").get("simulator_rng_seeded")
                is True
                for summary in summaries
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_method": "hierarchical_seed_then_paired_matchup",
            "bootstrap_seed_strata": len(score_groups),
        },
        "results": {
            **outcomes,
            "paired_team_matchups": sum(len(run.pair_scores) for run in runs),
            "paired_score_rate": paired_score,
            "paired_score_rate_bootstrap_ci95": hierarchical_paired_bootstrap_ci(
                score_groups,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            ),
            "paired_role_score_difference": role_difference,
            "paired_role_score_difference_bootstrap_ci95": (
                hierarchical_paired_bootstrap_ci(
                    role_groups,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 1,
                )
            ),
            "same_policy_external_role_advantage": (
                paired_score - 0.5 if same_policy else None
            ),
            "between_seed_score_std": float(np.std(per_seed_scores, ddof=1))
            if len(per_seed_scores) > 1
            else 0.0,
        },
        "runs": [
            {
                "source": str(run.source),
                "source_sha256": run.source_sha256,
                "seed": run.seed,
                "schedule_sha256": run.schedule_sha256,
                "paired_team_matchups": len(run.pair_scores),
                "paired_score_rate": float(run.pair_scores.mean()),
                "paired_role_score_difference": float(run.role_differences.mean()),
                **outcome_summary(run.rewards),
            }
            for run in runs
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return path.resolve()
        raise FileExistsError(f"Refusing to overwrite immutable aggregate: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate compatible mirrored anchor evaluations across seeds."
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Mirrored evaluate_anchor JSON; repeat once per seed.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--output", required=True, help="Immutable aggregate JSON path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path).expanduser().resolve() for path in args.summary]
    summary = aggregate_anchor_summaries(
        paths,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = _write_json(Path(args.output), summary)
    print(f"Wrote aggregate to {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
