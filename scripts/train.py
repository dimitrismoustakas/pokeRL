"""Training entry point.

Usage:
    python -m scripts.train
    python -m scripts.train --device cuda
    python -m scripts.train --num-envs 1 --rollout-steps 8 --total-timesteps 24
    python -m scripts.train --resume checkpoints/step_204800.pt
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from contextlib import ExitStack

import numpy as np
import torch

from pokerl.config import REGULATION_F, ModelConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.env.actions import (
    NUM_SLOT_INTENT_COMPONENTS,
    NUM_TERA_INTENT_COMPONENTS,
)
from pokerl.env.vector import PikaproVectorEnv
from pokerl.model.agent import PokeAgent, load_policy
from pokerl.public_consequence import (
    PUBLIC_CONSEQUENCE_OBSERVATION_KEYS,
    empty_consequence_batch,
    encode_consequence_batch,
)
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool
from pokerl.training.rollout import RolloutBuffer
from pokerl.training.ppo import PPOTrainer
from pokerl.training.league import League
from pokerl.training.recurrent_history import RecurrentObservationHistory
from pokerl.training.cuda_execution import cuda_model_execution
from pokerl.training.inference_batch import InputTransfer
from pokerl.training.rollout_policy import RolloutPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_MODULUS = 2**32


def _derived_seed(seed: int, offset: int) -> int:
    return (seed + offset) % _SEED_MODULUS


def configure_training_randomness(
    seed: int,
    *,
    team_pool_seed: int | None,
) -> dict[str, int]:
    """Seed independent team, league and model sampling streams."""
    if not 0 <= seed < _SEED_MODULUS:
        raise ValueError(f"--seed must be in [0, 2**32) (got {seed}).")
    if team_pool_seed is not None and not 0 <= team_pool_seed < _SEED_MODULUS:
        raise ValueError(
            f"--team-pool-seed must be in [0, 2**32) (got {team_pool_seed})."
        )

    streams = {
        "master": seed,
        "torch": seed,
        "python": _derived_seed(seed, 1),
        "numpy": _derived_seed(seed, 2),
        "league": _derived_seed(seed, 3),
        "team_pool": (
            team_pool_seed if team_pool_seed is not None else _derived_seed(seed, 4)
        ),
    }
    random.seed(streams["python"])
    np.random.seed(streams["numpy"])
    torch.manual_seed(streams["torch"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(streams["torch"])
    return streams


OpponentOutcomeCounts = dict[str | None, dict[str, int]]


def _record_opponent_outcome(
    counts: OpponentOutcomeCounts,
    checkpoint_path: object,
    reward: float,
) -> str | None:
    path = (
        checkpoint_path
        if isinstance(checkpoint_path, str) and checkpoint_path
        else None
    )
    bucket = counts.setdefault(path, {"wins": 0, "losses": 0, "draws": 0})
    if reward > 0:
        bucket["wins"] += 1
    elif reward < 0:
        bucket["losses"] += 1
    else:
        bucket["draws"] += 1
    return path


def _summarize_opponent_outcomes(
    counts: OpponentOutcomeCounts,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    total_wins = 0
    total_losses = 0
    total_draws = 0
    for checkpoint_path in sorted(
        counts,
        key=lambda value: (value is None, "" if value is None else value),
    ):
        bucket = counts[checkpoint_path]
        wins = int(bucket["wins"])
        losses = int(bucket["losses"])
        draws = int(bucket["draws"])
        games = wins + losses + draws
        total_wins += wins
        total_losses += losses
        total_draws += draws
        rows.append(
            {
                "checkpoint_path": checkpoint_path,
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "win_rate": wins / games if games else 0.0,
            }
        )

    total_games = total_wins + total_losses + total_draws
    return {
        "games": total_games,
        "wins": total_wins,
        "losses": total_losses,
        "draws": total_draws,
        "win_rate": total_wins / total_games if total_games else 0.0,
        "unattributed_games": sum(
            counts.get(None, {"wins": 0, "losses": 0, "draws": 0}).values()
        ),
        "by_checkpoint": rows,
    }


def _sorted_assignment_paths(assignments: dict[int, str]) -> list[str]:
    return sorted(set(assignments.values()))


@dataclass(frozen=True)
class LeagueMaintenance:
    checkpoint_added: bool
    assignments_by_group: dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pokeRL with PPO.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Master training seed. Separate deterministic streams are derived for "
            "league and team sampling."
        ),
    )
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help=(
            "PyTorch float32 matmul precision. Defaults to highest to preserve full "
            "float32 precision; high and medium are explicit throughput tradeoffs."
        ),
    )
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=None,
        help="Optional torch intra-op CPU thread limit for loaded simulator benchmarks.",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        default=None,
        help="Optional torch inter-op CPU thread limit for loaded simulator benchmarks.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override the number of parallel battle environments.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Override rollout steps collected per update.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=None,
        help=(
            "Number of rollout/PPO updates to execute. When resuming, these are "
            "additional updates rather than an absolute update ceiling."
        ),
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override the total environment steps to train for.",
    )
    parser.add_argument(
        "--collect-timing",
        action="store_true",
        help="Collect detailed rollout/env micro-profiles even without --summary-path.",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=None, help="Override PPO epochs per update."
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adamw"),
        default=None,
        help="Optimizer family. AdamW uses explicit decay/no-decay parameter groups.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="AdamW decoupled weight decay for non-embedding matrix parameters.",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=None,
        help="Override PPO minibatches per update.",
    )
    parser.add_argument(
        "--bptt-chunk-steps",
        type=int,
        default=None,
        help="Detach recurrent state every N PPO timesteps. Use 0 to disable chunking.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Override logging frequency in updates.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override checkpoint save frequency in updates.",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Override checkpoint output directory."
    )
    parser.add_argument(
        "--opponent-groups",
        type=int,
        default=None,
        help="Number of independently sampled league checkpoints across envs. Defaults to 2 when possible.",
    )
    parser.add_argument(
        "--team-pool-format",
        default=None,
        help="Load teams from data/team_pools/<format>/manifest.json.",
    )
    parser.add_argument(
        "--team-pool-manifest",
        default=None,
        help="Path to a specific team pool manifest.json.",
    )
    parser.add_argument(
        "--team-pool-subset",
        choices=["curated", "generated", "pool"],
        help="Which manifest subset to sample teams from.",
    )
    parser.add_argument(
        "--team-pool-seed",
        type=int,
        default=None,
        help="Override the team-sampling seed derived from --seed.",
    )
    parser.add_argument(
        "--league-update-interval",
        type=int,
        default=None,
        help="Override how often the current policy is snapshotted into the league.",
    )
    parser.add_argument(
        "--league-refresh-interval",
        type=int,
        default=None,
        help=(
            "Resample each opponent group every N updates. Defaults to every update, "
            "independently of league checkpoint creation."
        ),
    )
    parser.add_argument(
        "--league-size",
        type=int,
        default=None,
        help=(
            "Maximum frozen checkpoint models resident in each inference cache. "
            "Every league snapshot remains in the permanent on-disk archive."
        ),
    )
    parser.add_argument(
        "--league-payoff-half-life-updates",
        type=float,
        default=None,
        help="Half-life for current-policy opponent payoff evidence.",
    )
    parser.add_argument(
        "--league-pfsp-exponent",
        type=float,
        default=None,
        help="Exponent applied to PFSP hard-opponent priorities.",
    )
    parser.add_argument(
        "--league-pfsp-uniform-fraction",
        type=float,
        default=None,
        help="Uniform archive-coverage fraction mixed into PFSP sampling.",
    )
    parser.add_argument(
        "--league-cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use bounded CUDA graphs for frozen battle-opponent inference. "
            "Disable with --no-league-cuda-graphs for eager A/B runs."
        ),
    )
    parser.add_argument(
        "--league-cuda-graph-batch-size",
        type=int,
        default=None,
        help="Fixed row capacity of each frozen-opponent CUDA graph replay.",
    )
    parser.add_argument(
        "--league-cuda-graph-cache-size",
        type=int,
        default=None,
        help="Maximum frozen-checkpoint CUDA graphs resident at once.",
    )
    parser.add_argument(
        "--league-preview-device",
        choices=("action", "cpu"),
        default=None,
        help=(
            "Run frozen team-preview inference on the shared action service/device "
            "or on a separate CPU service."
        ),
    )
    parser.add_argument(
        "--tera-exploration-start",
        type=float,
        default=None,
        help="Initial epsilon for factorized Tera gate/slot exploration.",
    )
    parser.add_argument(
        "--tera-exploration-end",
        type=float,
        default=None,
        help="Final tera-exploration epsilon after the decay window.",
    )
    parser.add_argument(
        "--tera-exploration-decay-updates",
        type=int,
        default=None,
        help=(
            "Absolute number of learner updates used to anneal Tera exploration. "
            "Required for a fresh learner and restored from its checkpoint on resume."
        ),
    )
    parser.add_argument(
        "--tera-gate-entropy-coef",
        type=float,
        default=None,
        help="Entropy coefficient for the base Tera use-now gate.",
    )
    parser.add_argument(
        "--tera-slot-entropy-coef",
        type=float,
        default=None,
        help="Entropy coefficient for the conditional eligible Tera slot.",
    )
    parser.add_argument(
        "--aux-hidden-info-coef",
        type=float,
        default=None,
        help="Coefficient on the opponent hidden-info prediction loss (item/ability/tera/moves).",
    )
    parser.add_argument(
        "--aux-opponent-intent-coef",
        type=float,
        default=None,
        help=(
            "Coefficient on masked semantic opponent-intent prediction "
            "(kind/move/target/switch/Tera)."
        ),
    )
    parser.add_argument(
        "--aux-public-consequence-coef",
        type=float,
        default=None,
        help=(
            "Positive coefficient on masked public action-consequence prediction "
            "(resolution, opponent CANT, and next public field)."
        ),
    )
    parser.add_argument(
        "--diagnose-memory-gradients",
        action="store_true",
        help=(
            "Measure weighted actor/value/auxiliary gradient norms and cosine "
            "conflicts on the shared memory for one PPO minibatch per update."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a trainer checkpoint to resume training from (e.g. checkpoints/step_204800.pt).",
    )
    parser.add_argument(
        "--init-from", help="Start a fresh learner from these model weights."
    )
    parser.add_argument(
        "--opponent-checkpoint",
        help="Train against this fixed policy instead of self-play.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Write a machine-readable JSON run summary to this path.",
    )
    return parser.parse_args()


def training_config(args: argparse.Namespace) -> TrainingConfig:
    if args.total_timesteps is not None and args.num_updates is not None:
        raise ValueError("Use either --total-timesteps or --num-updates, not both.")

    overrides = {
        "num_envs": args.num_envs,
        "seed": args.seed,
        "optimizer_name": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "rollout_steps": args.rollout_steps,
        "total_timesteps": args.total_timesteps,
        "num_epochs": args.num_epochs,
        "num_minibatches": args.num_minibatches,
        "bptt_chunk_steps": args.bptt_chunk_steps,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "checkpoint_dir": args.checkpoint_dir,
        "league_size": args.league_size,
        "league_update_interval": args.league_update_interval,
        "league_refresh_interval": args.league_refresh_interval,
        "league_payoff_half_life_updates": args.league_payoff_half_life_updates,
        "league_pfsp_exponent": args.league_pfsp_exponent,
        "league_pfsp_uniform_fraction": args.league_pfsp_uniform_fraction,
        "league_cuda_graphs": args.league_cuda_graphs,
        "league_cuda_graph_batch_size": args.league_cuda_graph_batch_size,
        "league_cuda_graph_cache_size": args.league_cuda_graph_cache_size,
        "league_preview_device": args.league_preview_device,
        "tera_exploration_start": args.tera_exploration_start,
        "tera_exploration_end": args.tera_exploration_end,
        "tera_exploration_decay_updates": args.tera_exploration_decay_updates,
        "tera_gate_entropy_coef": args.tera_gate_entropy_coef,
        "tera_slot_entropy_coef": args.tera_slot_entropy_coef,
        "aux_hidden_info_coef": args.aux_hidden_info_coef,
        "aux_opponent_intent_coef": args.aux_opponent_intent_coef,
        "aux_public_consequence_coef": args.aux_public_consequence_coef,
        "diagnose_memory_gradients": args.diagnose_memory_gradients,
    }
    settings = {name: value for name, value in overrides.items() if value is not None}
    if settings.get("bptt_chunk_steps") == 0:
        settings["bptt_chunk_steps"] = None
    train_config = TrainingConfig(**settings)

    if args.num_updates is not None:
        train_config.total_timesteps = (
            args.num_updates * train_config.rollout_steps * train_config.num_envs
        )

    return train_config


def apply_torch_runtime_config(args: argparse.Namespace) -> dict[str, object]:
    if args.torch_num_threads is not None and args.torch_num_threads <= 0:
        raise ValueError("--torch-num-threads must be positive.")
    if (
        args.torch_num_interop_threads is not None
        and args.torch_num_interop_threads <= 0
    ):
        raise ValueError("--torch-num-interop-threads must be positive.")

    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

    runtime_config: dict[str, object] = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    if torch.cuda.is_available():
        runtime_config["cuda_matmul_allow_tf32"] = bool(
            torch.backends.cuda.matmul.allow_tf32
        )
        runtime_config["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    return runtime_config


def resolve_update_bounds(
    *,
    total_timesteps: int,
    steps_per_update: int,
    global_step: int,
    completed_updates: int,
    additional_updates: bool,
) -> tuple[int, int]:
    """Return completed and target update counts for this invocation."""
    if steps_per_update <= 0:
        raise ValueError("steps_per_update must be positive.")
    if type(completed_updates) is not int or completed_updates < 0:
        raise ValueError("completed_updates must be a non-negative integer.")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be a non-negative integer.")
    if additional_updates:
        requested_updates = total_timesteps // steps_per_update
    else:
        remaining_steps = max(0, total_timesteps - global_step)
        requested_updates = math.ceil(remaining_steps / steps_per_update)
    target_update = completed_updates + requested_updates
    return completed_updates, target_update


def tera_exploration_probability(
    *,
    start: float,
    end: float,
    update: int,
    decay_updates: int,
) -> float:
    """Linearly anneal epsilon, reaching ``end`` within the decay window."""
    if update <= 0:
        raise ValueError("update must be positive.")
    if decay_updates < 0:
        raise ValueError("decay_updates must be non-negative.")
    if decay_updates <= 1:
        return float(end)
    progress = min(1.0, max(0.0, (update - 1) / (decay_updates - 1)))
    return float(start + (end - start) * progress)


def resolve_opponent_groups(raw_value: int | None, num_envs: int) -> int:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if raw_value is None:
        return min(2, num_envs)
    if raw_value <= 0:
        raise ValueError("--opponent-groups must be positive.")
    return min(raw_value, num_envs)


def make_envs(
    num_envs: int,
    game_data: GameData,
    league: League,
    learner: PokeAgent,
    opponent_groups: int,
    policy_version: int,
    team_pair_provider,
    collect_timing: bool = False,
    initial_checkpoint_by_group: dict[int, str] | None = None,
) -> PikaproVectorEnv:
    """Create parallel battle environments backed by league opponents."""
    if initial_checkpoint_by_group is None:
        checkpoint_by_group = dict(
            enumerate(league.sample_checkpoint_paths(opponent_groups))
        )
    else:
        if set(initial_checkpoint_by_group) != set(range(opponent_groups)):
            raise ValueError(
                "Saved opponent assignments do not match --opponent-groups."
            )
        checkpoint_by_group = initial_checkpoint_by_group
    seed = np.random.SeedSequence(
        [league.config.seed, policy_version, 5]
    ).generate_state(1)[0]
    return PikaproVectorEnv(
        num_envs,
        game_data,
        league,
        learner,
        policy_version,
        team_pair_provider,
        checkpoint_by_group,
        seed=int(seed),
        collect_timing=collect_timing,
    )


def refresh_opponents(
    parallel_env: PikaproVectorEnv,
    league: League,
) -> dict[int, str]:
    """Sample opponents for newly starting battles in each group."""
    parallel_env.assignments = dict(
        enumerate(league.sample_checkpoint_paths(len(parallel_env.assignments)))
    )
    return dict(parallel_env.assignments)


def _maintain_league_after_update(
    *,
    update: int,
    num_updates: int,
    global_step: int,
    agent: PokeAgent,
    parallel_env: PikaproVectorEnv,
    league: League,
    config: TrainingConfig,
) -> LeagueMaintenance:
    """Snapshot first, then sample next-battle opponents from the new pool."""
    checkpoint_added = update % config.league_update_interval == 0
    if checkpoint_added:
        league.add_checkpoint(agent, global_step, update=update)

    assignments: dict[int, str] = {}
    if update < num_updates and update % config.league_refresh_interval == 0:
        assignments = refresh_opponents(parallel_env, league)
    return LeagueMaintenance(
        checkpoint_added=checkpoint_added,
        assignments_by_group=assignments,
    )


def clone_hidden_state(
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if hidden is None:
        return None
    return tuple(state.detach().clone() for state in hidden)


def collect_synchronous_rollout(
    *,
    parallel_env: PikaproVectorEnv,
    agent: PokeAgent,
    buffer: RolloutBuffer,
    obs: dict[str, np.ndarray],
    action_masks: dict[str, np.ndarray],
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
    device: torch.device,
    num_steps: int,
    preview_policy_version: int,
    tera_exploration_prob: float,
    collect_timing: bool,
    capture_stream: torch.cuda.Stream | None = None,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    tuple[torch.Tensor, torch.Tensor] | None,
    list[dict],
    dict[str, float],
]:
    """Collect one fixed-length, fully batched rollout.

    Every policy forward covers all environments. This keeps player inference
    efficient on GPUs whose forward latency is nearly batch-size invariant and
    preserves the standard synchronous PPO trajectory layout.
    """
    buffer.set_tera_behavior_policy(
        exploration_prob=tera_exploration_prob,
    )
    initial_preview_starts = parallel_env.drain_team_preview_starts()
    buffer.register_team_preview_starts(
        initial_preview_starts,
        policy_version=preview_policy_version,
    )
    profile = {
        "obs_to_device_sec": 0.0,
        "mask_to_device_sec": 0.0,
        "policy_forward_sec": 0.0,
        "policy_postprocess_sec": 0.0,
        "env_step_sec": 0.0,
        "feedback_to_device_sec": 0.0,
        "buffer_add_sec": 0.0,
        "hidden_reset_sec": 0.0,
        "team_preview_episode_starts": float(len(initial_preview_starts)),
    }
    rollout_infos: list[dict] = []
    count = parallel_env.num_envs
    consequences = empty_consequence_batch(count)
    slot_shape = (count, 2, NUM_SLOT_INTENT_COMPONENTS)
    tera_shape = (count, NUM_TERA_INTENT_COMPONENTS)
    feedback = InputTransfer(
        {
            "rewards": np.empty(count, np.float32),
            "dones": np.empty(count, bool),
            "opponent_intent_slot_labels": np.empty(slot_shape, np.int64),
            "opponent_intent_slot_masks": np.empty(slot_shape, bool),
            "opponent_intent_tera_labels": np.empty(tera_shape, np.int64),
            "opponent_intent_tera_masks": np.empty(tera_shape, bool),
            **{
                f"public_consequence_{name}": values
                for name, values in consequences.items()
            },
        },
        device,
    )
    host = feedback.host.arrays
    public_consequence_np = {
        name: host[f"public_consequence_{name}"] for name in consequences
    }
    policy = RolloutPolicy(
        agent, obs, action_masks, device, tera_exploration_prob, capture_stream
    )
    for _ in range(num_steps):
        with torch.no_grad(), cuda_model_execution(device):
            started = time.perf_counter() if collect_timing else 0.0
            obs_t = policy.observations.stage(obs)
            if collect_timing:
                profile["obs_to_device_sec"] += time.perf_counter() - started

            started = time.perf_counter() if collect_timing else 0.0
            masks_t = policy.masks.stage(action_masks)
            if collect_timing:
                profile["mask_to_device_sec"] += time.perf_counter() - started

            started = time.perf_counter() if collect_timing else 0.0
            out = policy.sample(hidden)
            if collect_timing:
                profile["policy_forward_sec"] += time.perf_counter() - started

            started = time.perf_counter() if collect_timing else 0.0
            actions_t = out["actions"]
            log_probs_t = out["log_prob"]
            values_t = out["value"]
            actions = actions_t.detach().cpu().numpy()
            hidden = out["hidden"]
            if collect_timing:
                profile["policy_postprocess_sec"] += time.perf_counter() - started

            if device.type == "cpu":
                obs_for_buffer = {
                    key: value.detach().clone() for key, value in obs_t.items()
                }
                masks_for_buffer = {
                    key: value.detach().clone() for key, value in masks_t.items()
                }
            else:
                obs_for_buffer = obs_t
                masks_for_buffer = masks_t

        started = time.perf_counter() if collect_timing else 0.0
        next_obs, next_masks, rewards, dones, infos = parallel_env.step(actions)
        if collect_timing:
            profile["env_step_sec"] += time.perf_counter() - started

        started = time.perf_counter() if collect_timing else 0.0
        host["rewards"][:] = rewards
        host["dones"][:] = dones
        host["opponent_intent_slot_labels"].fill(-1)
        host["opponent_intent_slot_masks"].fill(False)
        host["opponent_intent_tera_labels"].fill(-1)
        host["opponent_intent_tera_masks"].fill(False)
        consequence_obs = next_obs
        terminal_lanes = np.flatnonzero(dones)
        if len(terminal_lanes):
            final = [
                infos[lane].pop("final_observation", None) for lane in terminal_lanes
            ]
            if any(row is None for row in final):
                raise RuntimeError(
                    "Auto-reset transition omitted final_observation for terminal auxiliary labels."
                )
            # The learner's next inputs describe new battles in these lanes.
            # Splice their final public events into separate training columns.
            consequence_obs = {}
            for key in PUBLIC_CONSEQUENCE_OBSERVATION_KEYS:
                if key in next_obs:
                    column = next_obs[key].copy()
                    column[terminal_lanes] = [row[key] for row in final]
                    consequence_obs[key] = column
        encode_consequence_batch(consequence_obs, public_consequence_np)
        # A terminal action resolves its moves without advancing the public field a turn.
        public_consequence_np["field_masks"][terminal_lanes] = False
        for env_index, info in enumerate(infos):
            slot_labels = info.get("opponent_intent_slot_labels")
            if slot_labels is None:
                continue
            host["opponent_intent_slot_labels"][env_index] = slot_labels
            host["opponent_intent_slot_masks"][env_index] = info[
                "opponent_intent_slot_masks"
            ]
            host["opponent_intent_tera_labels"][env_index] = info[
                "opponent_intent_tera_labels"
            ]
            host["opponent_intent_tera_masks"][env_index] = info[
                "opponent_intent_tera_masks"
            ]

        feedback_t = feedback.upload()

        if collect_timing:
            profile["feedback_to_device_sec"] += time.perf_counter() - started

        started = time.perf_counter() if collect_timing else 0.0
        buffer.add(
            obs_for_buffer,
            actions_t,
            masks_for_buffer,
            log_probs_t,
            values_t,
            **feedback_t,
        )
        preview_starts = parallel_env.drain_team_preview_starts()
        buffer.register_team_preview_starts(
            preview_starts,
            policy_version=preview_policy_version,
        )
        profile["team_preview_episode_starts"] += float(len(preview_starts))
        if collect_timing:
            profile["buffer_add_sec"] += time.perf_counter() - started

        if hidden is not None:
            started = time.perf_counter() if collect_timing else 0.0
            h, c = hidden
            dones_t = feedback_t["dones"]
            not_done = (~dones_t).to(h.dtype).view(1, dones_t.shape[0], 1)
            hidden = (h * not_done, c * not_done)
            if collect_timing:
                profile["hidden_reset_sec"] += time.perf_counter() - started

        rollout_infos.extend(infos)
        obs, action_masks = next_obs, next_masks

    return obs, action_masks, hidden, rollout_infos, profile


def _json_ready_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_observation_history(
    obs: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Report whether the causal event and prior-intent inputs are populated."""
    required = {"recent_event_meta", "recent_event_mask", "prev_own_valid"}
    if not required.issubset(obs):
        return {}
    with torch.no_grad():
        meta = obs["recent_event_meta"]
        valid = meta[..., 0] > 0.5
        partial = (meta[..., 1] > 0.5) & valid
        overflow = (meta[..., 2] > 0.0) & valid
        event_counts = (~obs["recent_event_mask"]).sum(dim=-1).to(torch.float32)
        valid_count = valid.sum().item()
        return {
            "history_event_segment_coverage": valid.to(torch.float32).mean().item(),
            "history_partial_segment_fraction": _safe_div(
                partial.sum().item(), valid_count
            ),
            "history_overflow_segment_fraction": _safe_div(
                overflow.sum().item(), valid_count
            ),
            "history_events_per_valid_segment": _safe_div(
                event_counts[valid].sum().item(), valid_count
            ),
            "history_previous_intent_slot_coverage": obs["prev_own_valid"]
            .to(torch.float32)
            .mean()
            .item(),
        }


def summarize_first_tera_turns(
    first_turns: list[int],
    completed_battles: int,
) -> dict[str, object]:
    counts: dict[int, int] = {}
    for turn in first_turns:
        counts[turn] = counts.get(turn, 0) + 1
    return {
        "completed_battles": completed_battles,
        "tera_used_battles": len(first_turns),
        "tera_used_fraction": _safe_div(len(first_turns), completed_battles),
        "turn_one_fraction": _safe_div(counts.get(1, 0), completed_battles),
        "avg_first_tera_turn": (float(np.mean(first_turns)) if first_turns else 0.0),
        "first_tera_turn_distribution": [
            {"turn": turn, "count": count} for turn, count in sorted(counts.items())
        ],
    }


def _write_run_summary(path: str, summary: dict) -> None:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)


def _maybe_write_run_summary(path: str | None, summary: dict) -> None:
    if path:
        _write_run_summary(path, summary)


def train():
    args = parse_args()
    if args.resume is not None and any(
        value is not None
        for value in (
            args.tera_exploration_start,
            args.tera_exploration_end,
            args.tera_exploration_decay_updates,
        )
    ):
        raise ValueError(
            "Tera exploration schedule options cannot be used with --resume; "
            "the complete schedule is restored from the trainer checkpoint."
        )
    if args.resume and args.init_from:
        raise ValueError("Choose either --resume or --init-from.")
    torch_runtime_config = apply_torch_runtime_config(args)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    run_started = time.perf_counter()
    collect_timing = args.collect_timing

    checkpoint_path = args.resume or args.init_from
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path
        else None
    )
    state = checkpoint["run_state"] if args.resume else None
    train_config = training_config(args)
    model_config = (
        ModelConfig(**checkpoint["model_config"])
        if checkpoint is not None
        else ModelConfig()
    )
    if state is not None:
        train_config.seed = state["seed"]
        if args.opponent_groups is None:
            args.opponent_groups = len(state["opponent_assignments"])
    if args.team_pool_manifest:
        manifest = REPO_ROOT / args.team_pool_manifest
    elif args.team_pool_format or state is None:
        manifest = default_team_pool_manifest_path(
            REPO_ROOT, args.team_pool_format or REGULATION_F
        )
    else:
        manifest = Path(state["team_manifest"])
    subset = args.team_pool_subset or (
        state["team_subset"] if state is not None else "pool"
    )
    if state is not None and (
        manifest.resolve() != Path(state["team_manifest"]).resolve()
        or subset != state["team_subset"]
    ):
        raise ValueError(
            "The saved run uses a different team pool. Use --init-from to start a new run on changed teams."
        )

    if args.resume is None and train_config.tera_exploration_decay_updates is None:
        raise ValueError(
            "--tera-exploration-decay-updates is required when starting a fresh "
            "learner. The value is restored automatically with --resume."
        )
    seed_streams = configure_training_randomness(
        train_config.seed,
        team_pool_seed=args.team_pool_seed,
    )
    print(f"Device: {device}")
    print(
        "Random streams: "
        f"master={seed_streams['master']} "
        f"torch={seed_streams['torch']} "
        f"league={seed_streams['league']} "
        f"team_pool={seed_streams['team_pool']}"
    )
    print(
        "Torch runtime: "
        f"float32_matmul_precision={torch_runtime_config['float32_matmul_precision']} "
        f"threads={torch_runtime_config['torch_num_threads']} "
        f"interop={torch_runtime_config['torch_num_interop_threads']}"
    )
    team_pool = load_team_pool(manifest, subset=subset, seed=seed_streams["team_pool"])
    if team_pool.format_id != REGULATION_F:
        raise ValueError(f"Expected Regulation F teams, got {team_pool.format_id}.")
    opponent_groups = resolve_opponent_groups(
        args.opponent_groups, train_config.num_envs
    )
    steps_per_update = train_config.rollout_steps * train_config.num_envs
    if train_config.total_timesteps < steps_per_update:
        raise ValueError(
            f"total_timesteps ({train_config.total_timesteps}) must be at least one full update "
            f"({steps_per_update} = rollout_steps * num_envs)."
        )

    # Game data and model
    game_data = GameData(gen=9)
    agent = PokeAgent(model_config, game_data)
    agent.to(device)
    print(f"Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print(
        f"Team pool: {team_pool.format_id} [{team_pool.subset}] with {team_pool.size} teams"
    )

    # Training
    trainer = PPOTrainer(agent, train_config, device)
    fixed_opponent = (
        str(Path(args.opponent_checkpoint).resolve())
        if args.opponent_checkpoint
        else None
    )
    restored_opponent_assignments = None
    if args.resume:
        trainer.load_state_dict(checkpoint)
        if fixed_opponent is not None and fixed_opponent != state["fixed_opponent"]:
            raise ValueError("--opponent-checkpoint differs from the saved run.")
        fixed_opponent = state["fixed_opponent"]
        team_pool.restore_rng_state(state["team_pool_rng"])
        restored_opponent_assignments = state["opponent_assignments"]
        print(f"Resumed update {trainer.completed_updates} from {args.resume}")
    else:
        if args.init_from:
            agent.load_state_dict(checkpoint["agent"])
        if fixed_opponent is not None:
            restored_opponent_assignments = dict.fromkeys(
                range(opponent_groups), fixed_opponent
            )
    del checkpoint
    optimizer_settings = trainer.optimizer_settings()
    if args.resume:
        for option, key in (
            ("optimizer", "name"),
            ("lr", "lr"),
            ("weight_decay", "weight_decay"),
        ):
            requested = getattr(args, option)
            if requested is not None and requested != optimizer_settings[key]:
                raise ValueError(
                    f"--{option.replace('_', '-')}={requested} disagrees with the saved optimizer ({optimizer_settings[key]})."
                )
    print(
        "Optimizer: "
        f"{optimizer_settings['name']} "
        f"lr={optimizer_settings['lr']:g} "
        f"betas={optimizer_settings['betas']} "
        f"eps={optimizer_settings['eps']:g} "
        f"weight_decay={optimizer_settings['weight_decay']:g} "
        f"grouping={optimizer_settings['decay_grouping']}"
    )
    opponent_template = agent
    if fixed_opponent is not None:
        with torch.random.fork_rng(devices=[]):
            opponent_template = load_policy(
                fixed_opponent, game_data, torch.device("cpu")
            )
    with ExitStack() as resources:
        league = League(
            train_config,
            opponent_template,
            device,
            rng=random.Random(seed_streams["league"]),
        )
        resources.callback(league.close)
        del opponent_template
        if state is not None:
            league.load_state_dict(state["league"])
        if fixed_opponent is None and not league.checkpoints:
            with cuda_model_execution(device):
                league.add_checkpoint(agent, step=trainer.global_step)
            print(f"Seeded league with agent at step {trainer.global_step}")

        agent.eval()
        parallel_env = make_envs(
            train_config.num_envs,
            game_data,
            league,
            agent,
            opponent_groups,
            policy_version=trainer.completed_updates,
            team_pair_provider=team_pool.sample_packed_pair,
            collect_timing=collect_timing,
            initial_checkpoint_by_group=restored_opponent_assignments,
        )
        resources.callback(parallel_env.close)

        def save_checkpoint(path: str) -> None:
            trainer.save(
                path,
                run_state={
                    "league": league.state_dict(),
                    "team_manifest": str(manifest),
                    "team_subset": team_pool.subset,
                    "team_pool_rng": team_pool.rng_state(),
                    "opponent_assignments": dict(parallel_env.assignments),
                    "fixed_opponent": fixed_opponent,
                    "seed": train_config.seed,
                },
            )

        current_opponent_assignments = dict(parallel_env.assignments)
        run_assigned_opponents = set(current_opponent_assignments.values())
        run_observed_opponents: set[str] = set()
        run_opponent_outcomes: OpponentOutcomeCounts = {}
        run_summary = {
            "status": "starting",
            "config": {
                "fixed_opponent": fixed_opponent,
                "initializer": args.init_from,
                "device": str(device),
                "seed": train_config.seed,
                "seed_streams": seed_streams,
                "torch_runtime": torch_runtime_config,
                "battle_format": REGULATION_F,
                "num_envs": train_config.num_envs,
                "team_size": model_config.team_size,
                "pick_size": model_config.pick_size,
                "rollout_steps": train_config.rollout_steps,
                "num_epochs": train_config.num_epochs,
                "num_minibatches": train_config.num_minibatches,
                "bptt_chunk_steps": train_config.bptt_chunk_steps,
                "optimizer": trainer.optimizer_settings(),
                "gamma": train_config.gamma,
                "gae_lambda": train_config.gae_lambda,
                "clip_eps": train_config.clip_eps,
                "entropy_coef": train_config.entropy_coef,
                "value_coef": train_config.value_coef,
                "max_grad_norm": train_config.max_grad_norm,
                "team_preview_value_coef": train_config.team_preview_value_coef,
                "team_preview_bring_coef": train_config.team_preview_bring_coef,
                "team_preview_lead_coef": train_config.team_preview_lead_coef,
                "tera_exploration_start": train_config.tera_exploration_start,
                "tera_exploration_end": train_config.tera_exploration_end,
                "tera_exploration_decay_updates": (
                    train_config.tera_exploration_decay_updates
                ),
                "tera_gate_entropy_coef": train_config.tera_gate_entropy_coef,
                "tera_slot_entropy_coef": train_config.tera_slot_entropy_coef,
                "aux_hidden_info_coef": train_config.aux_hidden_info_coef,
                "aux_opponent_intent_coef": train_config.aux_opponent_intent_coef,
                "aux_public_consequence_coef": (
                    train_config.aux_public_consequence_coef
                ),
                "diagnose_memory_gradients": train_config.diagnose_memory_gradients,
                "league_size": train_config.league_size,
                "league_update_interval": train_config.league_update_interval,
                "league_refresh_interval": train_config.league_refresh_interval,
                "league_cuda_graphs": train_config.league_cuda_graphs,
                "league_cuda_graph_batch_size": (
                    train_config.league_cuda_graph_batch_size
                ),
                "league_cuda_graph_cache_size": (
                    train_config.league_cuda_graph_cache_size
                ),
                "league_preview_device": train_config.league_preview_device,
                "league_payoff_half_life_updates": (
                    train_config.league_payoff_half_life_updates
                ),
                "league_pfsp_exponent": train_config.league_pfsp_exponent,
                "league_pfsp_uniform_fraction": (
                    train_config.league_pfsp_uniform_fraction
                ),
                "league_inference_device": str(league.inference_device),
                "league_preview_inference_device": str(league.preview_inference_device),
                "collect_timing": collect_timing,
                "total_timesteps": train_config.total_timesteps,
                "backend": "pikapro",
                "opponent_groups": opponent_groups,
                "vector_threads": parallel_env.threads,
                "team_pool": team_pool.describe(),
            },
            "opponents": {
                "distinct_assigned_opponents": len(run_assigned_opponents),
                "assigned_checkpoint_paths": sorted(run_assigned_opponents),
                "distinct_observed_opponents": 0,
                "observed_checkpoint_paths": [],
                "outcomes": _summarize_opponent_outcomes(run_opponent_outcomes),
            },
            "league": league.describe(include_members=True),
            "updates": [],
        }

        try:
            # Rollout buffer
            reset_started = time.perf_counter()
            obs, masks = parallel_env.reset()
            initial_reset_sec = time.perf_counter() - reset_started
            initial_env_profile = (
                parallel_env.snapshot_timing() if collect_timing else {}
            )
            if collect_timing:
                parallel_env.reset_timing()
            with cuda_model_execution(device):
                buffer = RolloutBuffer(
                    train_config.rollout_steps,
                    train_config.num_envs,
                    obs,
                    masks,
                    device,
                )
                hidden = agent.initial_hidden(train_config.num_envs)
                recurrent_history = RecurrentObservationHistory.empty(
                    train_config.num_envs
                )
            updates_done, num_updates = resolve_update_bounds(
                total_timesteps=train_config.total_timesteps,
                steps_per_update=steps_per_update,
                global_step=trainer.global_step,
                completed_updates=trainer.completed_updates,
                additional_updates=args.resume is not None
                and args.num_updates is not None,
            )
            tera_exploration_decay_updates = train_config.tera_exploration_decay_updates
            if tera_exploration_decay_updates is None:
                raise RuntimeError(
                    "Tera exploration decay was neither configured for fresh training "
                    "nor restored from the trainer checkpoint."
                )

            print(f"Total updates: {num_updates}")
            print(f"Steps per update: {steps_per_update}")
            print(
                "Tera exploration: "
                f"{train_config.tera_exploration_start:g} -> "
                f"{train_config.tera_exploration_end:g} over "
                f"{tera_exploration_decay_updates} update(s)"
            )
            print(f"Opponent groups: {opponent_groups}")
            if fixed_opponent is None:
                print(
                    f"League: permanent archive, resident cache {train_config.league_size}; "
                    f"snapshot every {train_config.league_update_interval} update(s), "
                    f"refresh every {train_config.league_refresh_interval} update(s); "
                    f"payoff half-life {train_config.league_payoff_half_life_updates:g}; "
                    f"PFSP exponent {train_config.league_pfsp_exponent:g} with "
                    f"{train_config.league_pfsp_uniform_fraction:.0%} uniform coverage"
                )
            else:
                print(f"Fixed opponent: {fixed_opponent}")
            print(
                "Frozen inference: "
                f"action={league.inference_device}, preview={league.preview_inference_device}; "
                f"CUDA graphs={'on' if train_config.league_cuda_graphs else 'off'} "
                f"batch={train_config.league_cuda_graph_batch_size} "
                f"cache={train_config.league_cuda_graph_cache_size}"
            )
            print(
                f"Pikapro VecEnv | {parallel_env.num_envs} battles | {parallel_env.threads} Rust threads"
            )
            print(f"Initial reset: {initial_reset_sec:.2f}s")
            if updates_done:
                print(f"Resuming from update {updates_done + 1} / {num_updates}")
            print("Starting training...")

            run_summary["initial_reset_sec"] = initial_reset_sec
            if collect_timing:
                run_summary["initial_env_profile"] = initial_env_profile
            run_summary["status"] = "running"
            _maybe_write_run_summary(args.summary_path, run_summary)

            episode_rewards: list[float] = []
            run_first_tera_turns: list[int] = []
            start_time = time.time()
            invocation_start_global_step = trainer.global_step

            # cuBLAS keeps a workspace for each stream used by the learner.
            capture_stream = (
                torch.cuda.Stream(device=device) if device.type == "cuda" else None
            )
            for update in range(updates_done + 1, num_updates + 1):
                rollout_opponent_assignments = dict(current_opponent_assignments)
                rollout_policy_version = parallel_env.policy_version
                update_wall_started = time.perf_counter()
                with cuda_model_execution(device):
                    agent.eval()
                    buffer.reset(initial_hidden=clone_hidden_state(hidden))
                if collect_timing:
                    parallel_env.reset_timing()
                rollout_started = time.perf_counter()

                tera_eps = tera_exploration_probability(
                    start=train_config.tera_exploration_start,
                    end=train_config.tera_exploration_end,
                    update=update,
                    decay_updates=tera_exploration_decay_updates,
                )
                update_episode_rewards: list[float] = []
                update_episode_secs: list[float] = []
                update_episode_turns: list[float] = []
                update_first_tera_turns: list[int] = []
                update_episode_opponent_choice_calls: list[int] = []
                update_episode_opponent_repeated_turn_requests: list[int] = []
                update_episode_opponent_max_calls_same_turn: list[int] = []
                update_pathological_episodes: list[dict[str, object]] = []
                update_opponent_outcomes: OpponentOutcomeCounts = {}
                update_observed_opponents: set[str] = set()

                # --- Collect rollout ---
                obs, masks, hidden, rollout_infos, rollout_profile = (
                    collect_synchronous_rollout(
                        parallel_env=parallel_env,
                        agent=agent,
                        buffer=buffer,
                        obs=obs,
                        action_masks=masks,
                        hidden=hidden,
                        device=device,
                        num_steps=train_config.rollout_steps,
                        preview_policy_version=rollout_policy_version,
                        tera_exploration_prob=tera_eps,
                        collect_timing=collect_timing,
                        capture_stream=capture_stream,
                    )
                )

                # Track completed episodes.
                for info in rollout_infos:
                    if "episode_reward" in info:
                        reward = float(info["episode_reward"])
                        episode_rewards.append(reward)
                        update_episode_rewards.append(reward)
                        opponent_path = _record_opponent_outcome(
                            update_opponent_outcomes,
                            info.get("episode_opponent_checkpoint_path"),
                            reward,
                        )
                        _record_opponent_outcome(
                            run_opponent_outcomes,
                            opponent_path,
                            reward,
                        )
                        if opponent_path is not None:
                            update_observed_opponents.add(opponent_path)
                            run_observed_opponents.add(opponent_path)
                    if "episode_sec" in info:
                        update_episode_secs.append(info["episode_sec"])
                    if "battle_turn" in info and info.get("episode_reward") is not None:
                        update_episode_turns.append(info["battle_turn"])
                        first_tera_turn = info.get("episode_first_tera_turn")
                        if first_tera_turn is not None:
                            first_tera_turn = int(first_tera_turn)
                            update_first_tera_turns.append(first_tera_turn)
                            run_first_tera_turns.append(first_tera_turn)
                    if "episode_opponent_choice_calls" in info:
                        update_episode_opponent_choice_calls.append(
                            int(info["episode_opponent_choice_calls"])
                        )
                    if "episode_opponent_repeated_turn_requests" in info:
                        repeated_turn_requests = int(
                            info["episode_opponent_repeated_turn_requests"]
                        )
                        update_episode_opponent_repeated_turn_requests.append(
                            repeated_turn_requests
                        )
                        update_episode_opponent_max_calls_same_turn.append(
                            int(info.get("episode_opponent_max_calls_same_turn", 0))
                        )
                        if repeated_turn_requests > 0:
                            update_pathological_episodes.append(
                                {
                                    "battle_tag": info.get("battle_tag", ""),
                                    "battle_turn": int(info.get("battle_turn", 0)),
                                    "episode_sec": float(info.get("episode_sec", 0.0)),
                                    "opponent_choice_calls": int(
                                        info.get("episode_opponent_choice_calls", 0)
                                    ),
                                    "opponent_repeated_turn_requests": repeated_turn_requests,
                                    "opponent_max_calls_same_turn": int(
                                        info.get(
                                            "episode_opponent_max_calls_same_turn", 0
                                        )
                                    ),
                                    "opponent_checkpoint_path": info.get(
                                        "episode_opponent_checkpoint_path"
                                    ),
                                }
                            )

                rollout_sec = time.perf_counter() - rollout_started
                if collect_timing:
                    rollout_profile["rollout_sec"] = rollout_sec
                env_profile = parallel_env.snapshot_timing() if collect_timing else {}

                # --- Compute advantages ---
                advantage_started = time.perf_counter()
                with torch.no_grad(), cuda_model_execution(device):
                    obs_t = {
                        key: torch.from_numpy(value).to(device)
                        for key, value in obs.items()
                    }
                    last_value = agent.get_value(obs_t, hidden)
                    batch = buffer.compute_gae(
                        last_value,
                        train_config.gamma,
                        train_config.gae_lambda,
                    )
                    observation_history_metrics = summarize_observation_history(
                        batch.obs
                    )
                advantage_sec = time.perf_counter() - advantage_started

                # --- PPO update ---
                ppo_started = time.perf_counter()
                with cuda_model_execution(device):
                    agent.train()
                    metrics = trainer.update(batch)
                    metrics["tera_exploration_prob"] = tera_eps
                    metrics.update(observation_history_metrics)
                    metrics["team_preview_policy_version"] = float(
                        rollout_policy_version
                    )

                    agent.eval()
                    parallel_env.advance_policy_version()
                hidden_refresh_started = time.perf_counter()
                with cuda_model_execution(device):
                    if hidden is not None:
                        recurrent_history = recurrent_history.extend(
                            batch.obs,
                            batch.dones,
                        )
                        hidden = recurrent_history.recompute_hidden(agent)
                        metrics["recurrent_history_mean_steps"] = (
                            recurrent_history.mean_length
                        )
                        metrics["recurrent_history_max_steps"] = float(
                            recurrent_history.max_length
                        )
                hidden_refresh_sec = time.perf_counter() - hidden_refresh_started
                ppo_sec = time.perf_counter() - ppo_started

                league_maintenance_started = time.perf_counter()
                if fixed_opponent is None:
                    # The updated main policy enters the archive before opponent
                    # groups are resampled, making the new snapshot immediately
                    # eligible under the updated PFSP evidence.
                    league.record_outcomes(update_opponent_outcomes, update=update)
                    with cuda_model_execution(device):
                        league_maintenance = _maintain_league_after_update(
                            update=update,
                            num_updates=num_updates,
                            global_step=trainer.global_step,
                            agent=agent,
                            parallel_env=parallel_env,
                            league=league,
                            config=train_config,
                        )
                else:
                    league_maintenance = LeagueMaintenance(False, {})
                league_update_sec = time.perf_counter() - league_maintenance_started
                if league_maintenance.assignments_by_group:
                    current_opponent_assignments = dict(
                        league_maintenance.assignments_by_group
                    )
                    run_assigned_opponents.update(current_opponent_assignments.values())
                if league_maintenance.checkpoint_added:
                    refresh_note = (
                        ", opponents refreshed"
                        if league_maintenance.assignments_by_group
                        else ""
                    )
                    print(
                        f"League updated (pool: {len(league.checkpoints)}){refresh_note}"
                    )

                trainer.completed_updates = update

                checkpoint_sec = 0.0
                if update % train_config.save_interval == 0:
                    checkpoint_started = time.perf_counter()
                    path = (
                        f"{train_config.checkpoint_dir}/step_{trainer.global_step}.pt"
                    )
                    with cuda_model_execution(device):
                        save_checkpoint(path)
                    checkpoint_sec = time.perf_counter() - checkpoint_started
                    print(f"Saved checkpoint: {path}")

                avg_episode_sec = (
                    float(np.mean(update_episode_secs)) if update_episode_secs else 0.0
                )
                avg_episode_turns = (
                    float(np.mean(update_episode_turns))
                    if update_episode_turns
                    else 0.0
                )
                avg_episode_opponent_choice_calls = (
                    float(np.mean(update_episode_opponent_choice_calls))
                    if update_episode_opponent_choice_calls
                    else 0.0
                )
                avg_episode_opponent_repeated_turn_requests = (
                    float(np.mean(update_episode_opponent_repeated_turn_requests))
                    if update_episode_opponent_repeated_turn_requests
                    else 0.0
                )
                max_episode_opponent_choice_calls = (
                    max(update_episode_opponent_choice_calls)
                    if update_episode_opponent_choice_calls
                    else 0
                )
                max_episode_opponent_repeated_turn_requests = (
                    max(update_episode_opponent_repeated_turn_requests)
                    if update_episode_opponent_repeated_turn_requests
                    else 0
                )
                max_episode_opponent_max_calls_same_turn = (
                    max(update_episode_opponent_max_calls_same_turn)
                    if update_episode_opponent_max_calls_same_turn
                    else 0
                )
                metrics_json = _json_ready_metrics(metrics)
                update_steps_per_second = (
                    steps_per_update / (rollout_sec + ppo_sec)
                    if (rollout_sec + ppo_sec) > 0
                    else 0.0
                )
                update_accounted_sec = rollout_sec + advantage_sec + ppo_sec
                update_wall_sec = time.perf_counter() - update_wall_started
                update_outcome_summary = _summarize_opponent_outcomes(
                    update_opponent_outcomes
                )
                rollout_assignment_paths = _sorted_assignment_paths(
                    rollout_opponent_assignments
                )
                next_assignment_paths = (
                    _sorted_assignment_paths(current_opponent_assignments)
                    if update < num_updates
                    else []
                )
                team_preview_episode_starts = int(
                    rollout_profile["team_preview_episode_starts"]
                )
                update_tera_summary = summarize_first_tera_turns(
                    update_first_tera_turns,
                    len(update_episode_rewards),
                )
                update_summary = {
                    "update": update,
                    "global_step": trainer.global_step,
                    "completed_battles": len(update_episode_rewards),
                    "team_preview_episode_starts": team_preview_episode_starts,
                    "rollout_sec": rollout_sec,
                    "advantage_sec": advantage_sec,
                    "ppo_sec": ppo_sec,
                    "hidden_refresh_sec": hidden_refresh_sec,
                    "recurrent_history_mean_steps": metrics_json.get(
                        "recurrent_history_mean_steps",
                        0.0,
                    ),
                    "recurrent_history_max_steps": metrics_json.get(
                        "recurrent_history_max_steps",
                        0.0,
                    ),
                    "update_accounted_sec": update_accounted_sec,
                    "update_steps_per_second": update_steps_per_second,
                    "avg_episode_sec": avg_episode_sec,
                    "avg_episode_turns": avg_episode_turns,
                    "tera": update_tera_summary,
                    "avg_episode_opponent_choice_calls": avg_episode_opponent_choice_calls,
                    "avg_episode_opponent_repeated_turn_requests": avg_episode_opponent_repeated_turn_requests,
                    "max_episode_opponent_choice_calls": max_episode_opponent_choice_calls,
                    "max_episode_opponent_repeated_turn_requests": max_episode_opponent_repeated_turn_requests,
                    "max_episode_opponent_max_calls_same_turn": max_episode_opponent_max_calls_same_turn,
                    "opponents": {
                        "assignments_by_group": {
                            str(group): path
                            for group, path in sorted(
                                rollout_opponent_assignments.items()
                            )
                        },
                        "assigned_checkpoint_paths": rollout_assignment_paths,
                        "distinct_assigned_opponents": len(rollout_assignment_paths),
                        "observed_checkpoint_paths": sorted(update_observed_opponents),
                        "distinct_observed_opponents": len(update_observed_opponents),
                        "outcomes": update_outcome_summary,
                        "refresh_performed": bool(
                            league_maintenance.assignments_by_group
                        ),
                        "next_assignments_by_group": {
                            str(group): path
                            for group, path in sorted(
                                current_opponent_assignments.items()
                            )
                        }
                        if update < num_updates
                        else {},
                        "next_assigned_checkpoint_paths": next_assignment_paths,
                        "next_distinct_assigned_opponents": len(next_assignment_paths),
                    },
                    "checkpoint_sec": checkpoint_sec,
                    "league_update_sec": league_update_sec,
                    "league_checkpoint_added": league_maintenance.checkpoint_added,
                    "league": league.describe(update=update, include_members=False),
                    "update_wall_sec": update_wall_sec,
                    "update_wall_steps_per_second": (
                        steps_per_update / update_wall_sec
                        if update_wall_sec > 0
                        else 0.0
                    ),
                    "update_overhead_sec": max(
                        0.0,
                        update_wall_sec
                        - update_accounted_sec
                        - checkpoint_sec
                        - league_update_sec,
                    ),
                    "metrics": metrics_json,
                }
                if device.type == "cuda":
                    update_summary["cuda_memory_allocated_bytes"] = (
                        torch.cuda.memory_allocated(device)
                    )
                    update_summary["cuda_memory_reserved_bytes"] = (
                        torch.cuda.memory_reserved(device)
                    )
                if update_pathological_episodes:
                    update_summary["pathological_episodes"] = sorted(
                        update_pathological_episodes,
                        key=lambda item: (
                            int(item["opponent_repeated_turn_requests"]),
                            int(item["opponent_choice_calls"]),
                        ),
                        reverse=True,
                    )[:5]
                if collect_timing:
                    update_summary["rollout_profile"] = rollout_profile
                    update_summary["env_profile"] = env_profile
                run_summary["opponents"] = {
                    "distinct_assigned_opponents": len(run_assigned_opponents),
                    "assigned_checkpoint_paths": sorted(run_assigned_opponents),
                    "distinct_observed_opponents": len(run_observed_opponents),
                    "observed_checkpoint_paths": sorted(run_observed_opponents),
                    "outcomes": _summarize_opponent_outcomes(run_opponent_outcomes),
                }
                run_summary["tera"] = summarize_first_tera_turns(
                    run_first_tera_turns,
                    len(episode_rewards),
                )
                # Keep the complete latest archive distribution once at top level;
                # per-update rows retain only compact PFSP/cache diagnostics.
                run_summary["league"] = league.describe(
                    update=update,
                    include_members=True,
                )
                run_summary["updates"].append(update_summary)
                run_summary["status"] = "running"
                run_summary["update_count"] = update
                run_summary["global_step"] = trainer.global_step
                _maybe_write_run_summary(args.summary_path, run_summary)

                # --- Logging ---
                if update % train_config.log_interval == 0:
                    elapsed = time.time() - start_time
                    invocation_steps = (
                        trainer.global_step - invocation_start_global_step
                    )
                    sps = invocation_steps / elapsed if elapsed > 0 else 0.0
                    win_rate = (
                        np.mean([r > 0 for r in episode_rewards[-100:]])
                        if episode_rewards
                        else 0
                    )
                    diagnostics_text = ""
                    if collect_timing:
                        diagnostics_text = (
                            f" | Player views {env_profile['native_observation_sec']:.2f}s"
                            f" | Rust steps {env_profile['native_step_and_checks_sec']:.2f}s"
                            f" | Opponent actions {env_profile.get('native_opponent_battle_sec', 0):.2f}s"
                            f" | Previews {env_profile.get('native_preview_sec', 0):.2f}s"
                        )
                    population_label = (
                        f"League pool {len(league.checkpoints)}"
                        if fixed_opponent is None
                        else "Fixed opponent"
                    )
                    print(
                        f"Update {update}/{num_updates} | "
                        f"Step {trainer.global_step:,} | "
                        f"Envs {train_config.num_envs} | "
                        f"Opp groups {opponent_groups} | "
                        f"Avg SPS {sps:.0f} | "
                        f"Upd SPS {update_steps_per_second:.0f} | "
                        f"Rollout {rollout_sec:.2f}s | "
                        f"PPO {ppo_sec:.2f}s | "
                        f"Completed battles {len(update_episode_rewards)} | "
                        f"Avg battle {avg_episode_sec:.2f}s/{avg_episode_turns:.1f} turns | "
                        f"Opp calls/ep {avg_episode_opponent_choice_calls:.1f} | "
                        f"Max repeat-turn req {max_episode_opponent_repeated_turn_requests} | "
                        f"Win rate (100) {win_rate:.2%} | "
                        f"W/L/D {update_outcome_summary['wins']}/"
                        f"{update_outcome_summary['losses']}/"
                        f"{update_outcome_summary['draws']} | "
                        f"Opp seen {len(update_observed_opponents)} | "
                        f"Policy loss {metrics['policy_loss']:.4f} | "
                        f"Value loss {metrics['value_loss']:.4f} | "
                        f"Entropy {metrics['entropy']:.4f} | "
                        f"Tera H(gate/slot) {metrics['base_tera_gate_entropy']:.3f}/"
                        f"{metrics['base_tera_slot_entropy']:.3f} | "
                        f"Tera t1 {update_tera_summary['turn_one_fraction']:.1%} | "
                        f"Pre-PPO KL {metrics['pre_ppo_approx_kl']:.5f}/"
                        f"clip {metrics['pre_ppo_clip_fraction']:.1%} | "
                        f"Preview loss {metrics.get('team_preview_loss', float('nan')):.4f} | "
                        f"Preview value {metrics.get('team_preview_value_loss', float('nan')):.4f} | "
                        f"{population_label}"
                        f"{diagnostics_text}"
                    )

            print("Training complete.")
            final_path = f"{train_config.checkpoint_dir}/final.pt"
            with cuda_model_execution(device):
                save_checkpoint(final_path)
            print(f"Saved final checkpoint: {final_path}")
            total_wall_sec = time.perf_counter() - run_started
            steps_this_run = trainer.global_step - invocation_start_global_step
            run_summary["final"] = {
                "global_step": trainer.global_step,
                "steps_this_run": steps_this_run,
                "wall_time_sec": total_wall_sec,
                "steps_per_second": steps_this_run / total_wall_sec
                if total_wall_sec > 0
                else 0.0,
                "final_checkpoint": final_path,
            }
            run_summary["status"] = "completed"
            _maybe_write_run_summary(args.summary_path, run_summary)
            return run_summary
        except Exception as exc:
            run_summary["status"] = "failed"
            run_summary["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _maybe_write_run_summary(args.summary_path, run_summary)
            raise


if __name__ == "__main__":
    train()
