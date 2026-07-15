"""Exactly enumerate a checkpoint's 6-to-4 team-preview policy."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from pokerl.config import ModelConfig
from pokerl.data import GameData
from pokerl.env.observation import build_team_preview_observation
from pokerl.eval.fixed_bot import (
    FixedTeamSpec,
    build_species_bst_lookup,
    load_agent_for_evaluation,
    plan_opponent_team_specs,
    select_max_bst_team_spec,
    select_team_spec_by_title,
)
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_SUM_TOLERANCE = 1e-4


def enumerate_preview_actions(*, team_size: int = 6, pick_size: int = 4) -> np.ndarray:
    """Return every bring set and ordered lead in TeamPreviewHead pick format.

    Each row contains the ordered lead pair followed by the remaining brought
    Pokemon in ascending team-slot order.  TeamPreviewHead canonicalizes the
    full row to score the bring set, while retaining the first two positions as
    the ordered lead pair.
    """
    if team_size < pick_size:
        raise ValueError("team_size must be at least pick_size.")
    if pick_size < 2:
        raise ValueError("pick_size must be at least two.")

    actions: list[tuple[int, ...]] = []
    for bring in itertools.combinations(range(team_size), pick_size):
        for lead in itertools.permutations(bring, 2):
            lead_set = set(lead)
            bench_tail = tuple(slot for slot in bring if slot not in lead_set)
            actions.append((*lead, *bench_tail))
    return np.asarray(actions, dtype=np.int64)


def summarize_preview_distributions(
    probabilities: np.ndarray,
    actions: np.ndarray,
) -> dict[str, object]:
    """Summarize exact conditional policies for uniformly weighted opponents."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.int64)
    if probabilities.ndim != 2 or min(probabilities.shape) <= 0:
        raise ValueError("probabilities must be a non-empty two-dimensional array.")
    if actions.ndim != 2 or actions.shape[0] != probabilities.shape[1]:
        raise ValueError("actions must have one row per probability column.")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")

    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(
        probability_sums,
        1.0,
        rtol=PROBABILITY_SUM_TOLERANCE,
        atol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError("each opponent policy must sum to one.")

    normalized = probabilities / probability_sums[:, None]
    log_probabilities = np.zeros_like(normalized)
    positive = normalized > 0.0
    log_probabilities[positive] = np.log(normalized[positive])
    entropies = -(normalized * log_probabilities).sum(axis=1)

    mean_policy = normalized.mean(axis=0)
    mean_log_policy = np.zeros_like(mean_policy)
    mean_positive = mean_policy > 0.0
    mean_log_policy[mean_positive] = np.log(mean_policy[mean_positive])
    mean_policy_entropy = float(-(mean_policy * mean_log_policy).sum())
    mean_conditional_entropy = float(entropies.mean())
    mutual_information = mean_policy_entropy - mean_conditional_entropy
    if abs(mutual_information) < 1e-12:
        mutual_information = 0.0

    greedy_indices = normalized.argmax(axis=1)
    greedy_actions = actions[greedy_indices]
    full_choices = {tuple(int(slot) for slot in row) for row in greedy_actions}
    bring_choices = {tuple(sorted(int(slot) for slot in row)) for row in greedy_actions}
    lead_choices = {(int(row[0]), int(row[1])) for row in greedy_actions}
    total_variation = 0.5 * np.abs(normalized - mean_policy).sum(axis=1)

    return {
        "opponent_count": int(normalized.shape[0]),
        "legal_action_count": int(normalized.shape[1]),
        "distinct_greedy_full_choices": len(full_choices),
        "distinct_greedy_bring_choices": len(bring_choices),
        "distinct_greedy_lead_choices": len(lead_choices),
        "exact_policy_entropy_mean_nats": mean_conditional_entropy,
        "exact_policy_entropy_mean_bits": mean_conditional_entropy / math.log(2.0),
        "max_probability_mean": float(normalized.max(axis=1).mean()),
        "mutual_information_nats": mutual_information,
        "mutual_information_bits": mutual_information / math.log(2.0),
        "mean_total_variation_to_mean_policy": float(total_variation.mean()),
        "greedy_action_indices": [int(index) for index in greedy_indices],
        "per_opponent_entropy_nats": [float(value) for value in entropies],
        "per_opponent_max_probability": [
            float(value) for value in normalized.max(axis=1)
        ],
    }


def evaluate_preview_distributions(
    *,
    agent,
    device: torch.device,
    game_data: GameData,
    model_team_spec: FixedTeamSpec,
    opponent_specs: list[FixedTeamSpec],
    actions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Score every supplied legal action for every opponent roster."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not opponent_specs:
        raise ValueError("at least one opponent roster is required.")

    observations = [
        build_team_preview_observation(
            game_data,
            team_size=len(model_team_spec.species),
            own_packed_team=model_team_spec.packed_team,
            opponent_packed_team=opponent_spec.packed_team,
        )
        for opponent_spec in opponent_specs
    ]
    stacked_observations = {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }
    action_count = len(actions)
    total_rows = len(observations) * action_count
    log_probabilities = np.empty(total_rows, dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, total_rows, batch_size):
            stop = min(start + batch_size, total_rows)
            flat_indices = np.arange(start, stop, dtype=np.int64)
            opponent_indices = flat_indices // action_count
            action_indices = flat_indices % action_count
            obs_batch = {
                key: torch.from_numpy(values[opponent_indices]).to(device)
                for key, values in stacked_observations.items()
            }
            picks = torch.from_numpy(actions[action_indices]).to(device)
            output = agent.team_preview(
                obs_batch,
                picks=picks,
                include_value=False,
                include_policy_stats=True,
                include_opponent_preview=False,
            )
            log_probabilities[start:stop] = (
                output["log_prob"].detach().cpu().numpy().astype(np.float64)
            )

    raw_probabilities = np.exp(
        log_probabilities.reshape(len(observations), action_count)
    )
    raw_sums = raw_probabilities.sum(axis=1)
    max_sum_error = float(np.max(np.abs(raw_sums - 1.0)))
    if not np.allclose(
        raw_sums,
        1.0,
        rtol=PROBABILITY_SUM_TOLERANCE,
        atol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise RuntimeError(
            "Enumerated preview probabilities did not sum to one: "
            f"range={raw_sums.min():.9f}..{raw_sums.max():.9f}, "
            f"max_error={max_sum_error:.3g}."
        )
    probabilities = raw_probabilities / raw_sums[:, None]
    return probabilities, {
        "raw_probability_sum_min": float(raw_sums.min()),
        "raw_probability_sum_max": float(raw_sums.max()),
        "raw_probability_sum_max_abs_error": max_sum_error,
        "normalized_probability_sum_min": float(probabilities.sum(axis=1).min()),
        "normalized_probability_sum_max": float(probabilities.sum(axis=1).max()),
    }


def _opponent_rows(
    *,
    probabilities: np.ndarray,
    actions: np.ndarray,
    opponent_specs: list[FixedTeamSpec],
    model_team_spec: FixedTeamSpec,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (policy, opponent_spec) in enumerate(
        zip(probabilities, opponent_specs, strict=True)
    ):
        action_index = int(policy.argmax())
        picks = tuple(int(slot) for slot in actions[action_index])
        bring_indices = tuple(sorted(picks))
        positive = policy > 0.0
        entropy = float(-(policy[positive] * np.log(policy[positive])).sum())
        rows.append(
            {
                "opponent_index": index,
                "opponent_title": opponent_spec.title,
                "opponent_species": list(opponent_spec.species),
                "policy_entropy_nats": entropy,
                "top_action_index": action_index,
                "top_action_probability": float(policy[action_index]),
                "top_action_picks": list(picks),
                "top_action_species": [model_team_spec.species[slot] for slot in picks],
                "top_action_bring_indices": list(bring_indices),
                "top_action_bring_species": [
                    model_team_spec.species[slot] for slot in bring_indices
                ],
                "top_action_lead_indices": list(picks[:2]),
                "top_action_lead_species": [
                    model_team_spec.species[slot] for slot in picks[:2]
                ],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exactly enumerate a trained policy's 180 legal 6-to-4 preview "
            "actions across a team-pool opponent panel."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto.")
    parser.add_argument(
        "--team-pool-format",
        default="gen9vgc2026regf",
        help="Load data/team_pools/<format>/manifest.json.",
    )
    parser.add_argument(
        "--team-pool-manifest",
        default=None,
        help="Optional explicit manifest path; overrides --team-pool-format.",
    )
    parser.add_argument(
        "--team-pool-subset",
        choices=["curated", "generated", "pool"],
        default="pool",
    )
    parser.add_argument(
        "--team-pool-seed",
        type=int,
        default=0,
        help="Reproducible ordering seed for opponent rosters.",
    )
    parser.add_argument(
        "--model-team-title",
        default=None,
        help="Exact fixed own-team title; defaults to the max-BST pool team.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Inference rows per batch."
    )
    parser.add_argument(
        "--summary-path", default=None, help="Optional JSON output path."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    game_data = GameData(gen=9)
    team_size = 6
    pick_size = 4
    model_config = ModelConfig(pick_size=pick_size)
    species_bst_lookup = build_species_bst_lookup(gen=game_data.gen)
    manifest_path = (
        Path(args.team_pool_manifest)
        if args.team_pool_manifest
        else default_team_pool_manifest_path(REPO_ROOT, args.team_pool_format)
    )
    team_pool = load_team_pool(
        manifest_path,
        subset=args.team_pool_subset,
        seed=args.team_pool_seed,
        usage="evaluation",
    )
    if args.model_team_title:
        model_team_spec = select_team_spec_by_title(
            team_pool,
            title=args.model_team_title,
            species_bst_lookup=species_bst_lookup,
            pick_size=pick_size,
        )
    else:
        model_team_spec = select_max_bst_team_spec(
            team_pool,
            species_bst_lookup=species_bst_lookup,
            pick_size=pick_size,
        )

    opponent_count = len(
        [entry for entry in team_pool.entries if entry.file != model_team_spec.file]
    )
    opponent_specs = plan_opponent_team_specs(
        team_pool,
        model_team_spec=model_team_spec,
        species_bst_lookup=species_bst_lookup,
        pick_size=pick_size,
        num_games=max(1, opponent_count),
        seed=args.team_pool_seed,
    )
    agent = load_agent_for_evaluation(
        checkpoint_path=args.checkpoint,
        device=device,
        model_config=model_config,
        game_data=game_data,
    )
    actions = enumerate_preview_actions(team_size=team_size, pick_size=pick_size)
    if actions.shape != (180, pick_size):
        raise RuntimeError(f"Expected 180 legal preview actions, got {len(actions)}.")
    probabilities, probability_sanity = evaluate_preview_distributions(
        agent=agent,
        device=device,
        game_data=game_data,
        model_team_spec=model_team_spec,
        opponent_specs=opponent_specs,
        actions=actions,
        batch_size=args.batch_size,
    )
    distribution_summary = summarize_preview_distributions(probabilities, actions)
    per_opponent = _opponent_rows(
        probabilities=probabilities,
        actions=actions,
        opponent_specs=opponent_specs,
        model_team_spec=model_team_spec,
    )
    distribution_summary.pop("greedy_action_indices")
    distribution_summary.pop("per_opponent_entropy_nats")
    distribution_summary.pop("per_opponent_max_probability")

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    summary = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "batch_size": args.batch_size,
            "team_size": team_size,
            "pick_size": pick_size,
            "team_pool_manifest": str(Path(manifest_path).expanduser().resolve()),
            "team_pool_subset": args.team_pool_subset,
            "team_pool_size": team_pool.size,
            "team_pool_signature": team_pool.signature(),
            "team_pool_seed": args.team_pool_seed,
        },
        "model_team": model_team_spec.to_summary(),
        "probability_sanity": probability_sanity,
        "distribution": distribution_summary,
        "opponents": per_opponent,
    }
    rendered = json.dumps(summary, indent=2)
    if args.summary_path:
        summary_path = Path(args.summary_path).expanduser()
        if not summary_path.is_absolute():
            summary_path = REPO_ROOT / summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote preview-policy summary to {summary_path.resolve()}")
    print(rendered)


if __name__ == "__main__":
    main()
