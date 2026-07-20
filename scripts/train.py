"""Training entry point.

Usage:
    python -m scripts.train
    python -m scripts.train --device cuda
    python -m scripts.train --num-envs 1 --rollout-steps 8 --total-timesteps 24
    python -m scripts.train --resume checkpoints/step_204800.pt
"""

import argparse
import asyncio
import json
import math
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from pokerl.config import ModelConfig, EnvConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.env.actions import (
    NUM_SLOT_INTENT_COMPONENTS,
    NUM_TERA_INTENT_COMPONENTS,
)
from pokerl.env.observation import NUM_PUBLIC_TARGET_POSITIONS
from pokerl.model.agent import PokeAgent
from pokerl.public_consequence import (
    PUBLIC_CONSEQUENCE_FIELD_DIM,
    PUBLIC_CONSEQUENCE_OBSERVATION_KEYS,
    encode_public_consequence,
)
from pokerl.env.showdown_env import (
    BattleEnv,
    LoopThread,
    ParallelBattleEnv,
    battlestream_summary,
    default_showdown_dir,
    resolve_battle_loop_count,
    resolve_reset_concurrency,
)
from pokerl.team_pool import TeamPool, default_team_pool_manifest_path, load_team_pool
from pokerl.training.rollout import RolloutBuffer
from pokerl.training.ppo import PPOTrainer
from pokerl.training.league import League
from pokerl.training.preview import (
    TeamPreviewBatcher,
    TeamPreviewResult,
)
from pokerl.training.recurrent_history import RecurrentObservationHistory
from pokerl.training.cuda_execution import cuda_model_execution
from pokerl.training.phase import (
    FrozenMixtureSampler,
    PhaseKind,
    PhaseRunState,
    PhaseSpec,
)
from pokerl.training.exploiter import validate_exploiter_phase_semantics
from pokerl.training.population import PopulationRevision


REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_MODULUS = 2**32
_MAIN_RUN_IDENTITY_VERSION = 1


def _derived_seed(seed: int, offset: int) -> int:
    return (seed + offset) % _SEED_MODULUS


def configure_training_randomness(
    seed: int,
    *,
    team_pool_seed: int | None,
) -> dict[str, int]:
    """Seed independent host streams and the shared torch learner stream.

    Battle execution is asynchronous, so this makes configured sampling streams
    reproducible without claiming bitwise-identical end-to-end training runs.
    """
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


CheckpointSampler = Callable[[int], list[str]]


@dataclass(frozen=True)
class FrozenPhaseOpponentSource:
    """Map one pinned policy-id mixture to verified checkpoint paths."""

    sampler: FrozenMixtureSampler
    checkpoint_by_policy_id: dict[str, str]

    def __post_init__(self) -> None:
        target_ids = {item.policy_id for item in self.sampler.distribution}
        if set(self.checkpoint_by_policy_id) != target_ids:
            raise ValueError(
                "Frozen checkpoint mapping must exactly match mixture support."
            )
        if len(set(self.checkpoint_by_policy_id.values())) != len(target_ids):
            raise ValueError("Frozen mixture policies must use distinct checkpoints.")

    def sample_checkpoint_paths(self, count: int) -> list[str]:
        return [
            self.checkpoint_by_policy_id[policy_id]
            for policy_id in self.sampler.sample_policy_ids(count)
        ]


@dataclass(frozen=True)
class BestResponsePhaseBundle:
    spec_path: Path
    spec: PhaseSpec
    population_path: Path
    population: PopulationRevision
    checkpoint_by_policy_id: dict[str, str]


def load_best_response_phase(path: str) -> BestResponsePhaseBundle:
    spec_path = Path(path).expanduser().resolve()
    spec = PhaseSpec.load(spec_path, verify_initializer=True)
    if spec.kind is not PhaseKind.BEST_RESPONSE:
        raise ValueError("--phase-spec currently supports only kind='best_response'.")
    population_path = spec.resolve_target_population_path(spec_path.parent)
    population = PopulationRevision.load(
        population_path,
        verify_checkpoints=True,
    )
    spec.validate_against_population(population)
    validate_exploiter_phase_semantics(spec, population)
    checkpoint_by_policy_id = {
        member.policy_id: str(member.checkpoint.resolve(population_path.parent))
        for member in population.policies
    }
    return BestResponsePhaseBundle(
        spec_path=spec_path,
        spec=spec,
        population_path=population_path,
        population=population,
        checkpoint_by_policy_id=checkpoint_by_policy_id,
    )


def _close_training_resources(*resources: tuple[str, object]) -> None:
    """Attempt every close even when an earlier resource reports an error."""
    errors: list[Exception] = []
    for name, resource in resources:
        try:
            resource.close()
        except Exception as exc:
            exc.add_note(f"Raised while closing training resource {name!r}.")
            errors.append(exc)
    if errors:
        raise ExceptionGroup("One or more training resources failed to close.", errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pokeRL with PPO.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Master training seed. Separate deterministic streams are derived for "
            "league and team sampling; asynchronous battles are not bitwise deterministic."
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
        "--battle-format", default=None, help="Override the Showdown battle format."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override the number of parallel battle environments.",
    )
    parser.add_argument(
        "--team-size",
        type=int,
        default=None,
        help="Override the full team size exposed in observations.",
    )
    parser.add_argument(
        "--pick-size",
        type=int,
        default=None,
        help="Override how many Pokemon are selected at team preview.",
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
        "--showdown-dir",
        default=None,
        help="Pokemon Showdown checkout used by local BattleStream workers.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Local BattleStream worker process count. Defaults to one worker per three envs.",
    )
    parser.add_argument(
        "--battles-per-worker",
        type=int,
        default=None,
        help="Maximum concurrent local battles per worker. Defaults to the capacity needed by all envs.",
    )
    parser.add_argument(
        "--runtime-loop-count",
        type=int,
        default=None,
        help="Number of runtime loops used by local BattleStream workers. Defaults to at most two.",
    )
    parser.add_argument(
        "--battle-loop-count",
        type=int,
        default=None,
        help="Event-loop threads hosting battle sessions. Defaults to min(num-envs, 4).",
    )
    parser.add_argument(
        "--battlestream-startup-timeout",
        type=float,
        default=None,
        help="Seconds to wait for a local BattleStream battle to produce its first protocol output.",
    )
    parser.add_argument(
        "--reset-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent BattleEnv resets. Defaults to all envs.",
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
        default="pool",
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
        "--tera-exploration-decay-fraction",
        type=float,
        default=None,
        help="Fraction of planned updates used to anneal Tera exploration.",
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
        "--league-coalesce-ms",
        type=float,
        default=None,
        help="Override the league-opponent inference coalescing window in milliseconds.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a trainer checkpoint to resume training from (e.g. checkpoints/step_204800.pt).",
    )
    parser.add_argument(
        "--phase-spec",
        default=None,
        help=(
            "Immutable best-response PhaseSpec JSON. Without --resume it starts "
            "a fresh weight-only learner; with --resume it must match the paired "
            "phase state exactly."
        ),
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Write a machine-readable JSON run summary to this path.",
    )
    return parser.parse_args()


def build_configs(
    args: argparse.Namespace,
) -> tuple[ModelConfig, EnvConfig, TrainingConfig]:
    model_config = ModelConfig()
    env_config = EnvConfig()
    train_config = TrainingConfig()

    if args.total_timesteps is not None and args.num_updates is not None:
        raise ValueError("Use either --total-timesteps or --num-updates, not both.")

    if args.battle_format is not None:
        env_config.battle_format = args.battle_format

    if args.num_envs is not None:
        env_config.num_envs = args.num_envs

    if args.team_size is not None:
        env_config.team_size = args.team_size

    if args.pick_size is not None:
        env_config.pick_size = args.pick_size
    model_config.team_size = env_config.team_size
    model_config.pick_size = env_config.pick_size

    if args.showdown_dir is not None:
        env_config.showdown_dir = args.showdown_dir
    if args.worker_count is not None:
        env_config.worker_count = args.worker_count
    if args.battles_per_worker is not None:
        env_config.battles_per_worker = args.battles_per_worker
    if args.runtime_loop_count is not None:
        env_config.runtime_loop_count = args.runtime_loop_count
    if args.battle_loop_count is not None:
        resolve_battle_loop_count(args.battle_loop_count, env_config.num_envs)
        env_config.battle_loop_count = args.battle_loop_count
    if args.battlestream_startup_timeout is not None:
        env_config.battlestream_startup_timeout = args.battlestream_startup_timeout
    if args.reset_concurrency is not None:
        resolve_reset_concurrency(args.reset_concurrency, env_config.num_envs)
        env_config.reset_concurrency = args.reset_concurrency
    overrides = {
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
        "league_coalesce_ms": args.league_coalesce_ms,
        "league_payoff_half_life_updates": args.league_payoff_half_life_updates,
        "league_pfsp_exponent": args.league_pfsp_exponent,
        "league_pfsp_uniform_fraction": args.league_pfsp_uniform_fraction,
        "league_cuda_graphs": args.league_cuda_graphs,
        "league_cuda_graph_batch_size": args.league_cuda_graph_batch_size,
        "league_cuda_graph_cache_size": args.league_cuda_graph_cache_size,
        "league_preview_device": args.league_preview_device,
        "tera_exploration_start": args.tera_exploration_start,
        "tera_exploration_end": args.tera_exploration_end,
        "tera_exploration_decay_fraction": args.tera_exploration_decay_fraction,
        "tera_gate_entropy_coef": args.tera_gate_entropy_coef,
        "tera_slot_entropy_coef": args.tera_slot_entropy_coef,
        "aux_hidden_info_coef": args.aux_hidden_info_coef,
        "aux_opponent_intent_coef": args.aux_opponent_intent_coef,
        "aux_public_consequence_coef": args.aux_public_consequence_coef,
        "diagnose_memory_gradients": args.diagnose_memory_gradients,
    }
    for field_name, value in overrides.items():
        if value is not None:
            if field_name == "bptt_chunk_steps" and value == 0:
                value = None
            setattr(train_config, field_name, value)

    if args.num_updates is not None:
        train_config.total_timesteps = (
            args.num_updates * train_config.rollout_steps * env_config.num_envs
        )

    train_config.__post_init__()
    return model_config, env_config, train_config


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


def validate_resume_update_state(
    *,
    trainer_completed_updates: int,
    league_current_update: int,
) -> None:
    """Reject a trainer/league pair that describes different update boundaries."""
    if trainer_completed_updates != league_current_update:
        raise ValueError(
            "Trainer checkpoint and paired league state disagree on completed "
            f"updates: trainer={trainer_completed_updates}, "
            f"league={league_current_update}."
        )


def resolve_tera_exploration_decay_updates(
    total_updates: int,
    decay_fraction: float,
) -> int:
    """Convert a horizon fraction into an absolute exploration window."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive.")
    if not 0.0 <= decay_fraction <= 1.0:
        raise ValueError("decay_fraction must be within [0, 1].")
    if decay_fraction == 0.0:
        return 0
    return max(1, math.ceil(total_updates * decay_fraction))


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


def make_league_opponent_factory(
    league: League,
    checkpoint_path: str | None = None,
):
    def league_factory(loop: asyncio.AbstractEventLoop):
        opponent = league.make_opponent(
            loop,
            checkpoint_path=checkpoint_path,
        )
        if opponent is None:
            raise RuntimeError(
                "League opponent requested, but the league is empty. "
                "This should not happen if the league was seeded before training."
            )
        return opponent

    return league_factory


def resolve_opponent_groups(raw_value: int | None, num_envs: int) -> int:
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if raw_value is None:
        return min(2, num_envs)
    if raw_value <= 0:
        raise ValueError("--opponent-groups must be positive.")
    return min(raw_value, num_envs)


def make_team_preview_fn(batcher: TeamPreviewBatcher):
    """Create a team preview callback that uses the current agent policy.

    The closure captures the agent object (not a copy), so it always reflects
    the latest weights during rollout collection.
    """

    def team_preview_fn(
        obs: dict[str, np.ndarray],
    ) -> tuple[TeamPreviewResult, str]:
        result = batcher.choose(obs)
        picks = result.picks
        team_str = "/team " + "".join(str(int(p) + 1) for p in picks)
        return result, team_str

    return team_preview_fn


def load_requested_team_pool(
    args: argparse.Namespace,
    *,
    seed: int | None = None,
) -> TeamPool | None:
    manifest_path: Path | None = None

    if args.team_pool_manifest:
        manifest_path = Path(args.team_pool_manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
    elif args.team_pool_format:
        manifest_path = default_team_pool_manifest_path(
            REPO_ROOT, args.team_pool_format
        )

    if manifest_path is None:
        return None

    return load_team_pool(
        manifest_path,
        subset=args.team_pool_subset,
        seed=args.team_pool_seed if args.team_pool_seed is not None else seed,
    )


def make_team_pair_provider(team_pool: TeamPool | None):
    if team_pool is None:
        return None

    def provider() -> tuple[str, str]:
        return team_pool.sample_packed_pair(distinct=True)

    return provider


def _sample_checkpoints_by_group(
    league: League,
    opponent_groups: int,
    checkpoint_sampler: CheckpointSampler | None = None,
) -> dict[int, str]:
    sampler = checkpoint_sampler or league.sample_checkpoint_paths
    paths = sampler(opponent_groups)
    if len(paths) != opponent_groups:
        raise RuntimeError(
            "League returned the wrong number of opponent assignments: "
            f"requested={opponent_groups}, returned={len(paths)}."
        )
    return dict(enumerate(paths))


def make_envs(
    env_config: EnvConfig,
    game_data: GameData,
    league: League,
    opponent_groups: int,
    team_preview_fn,
    team_preview_batcher: TeamPreviewBatcher | None = None,
    team_pair_provider=None,
    collect_timing: bool = False,
    initial_checkpoint_by_group: dict[int, str] | None = None,
    checkpoint_sampler: CheckpointSampler | None = None,
) -> ParallelBattleEnv:
    """Create parallel battle environments backed by league opponents."""
    if initial_checkpoint_by_group is None:
        checkpoint_by_group = _sample_checkpoints_by_group(
            league,
            opponent_groups,
            checkpoint_sampler,
        )
    else:
        expected_groups = set(range(opponent_groups))
        if set(initial_checkpoint_by_group) != expected_groups:
            raise ValueError(
                "Initial opponent assignments do not match configured groups: "
                f"expected={sorted(expected_groups)}, "
                f"saved={sorted(initial_checkpoint_by_group)}."
            )
        archive_paths = set(league.checkpoints)
        if any(
            not isinstance(path, str) or path not in archive_paths
            for path in initial_checkpoint_by_group.values()
        ):
            raise ValueError(
                "Initial opponent assignments must reference archived checkpoints."
            )
        checkpoint_by_group = dict(initial_checkpoint_by_group)
        # Loading strict resume assignments must not consume another PFSP RNG
        # draw. ``make_opponent`` prefetches each restored checkpoint below.
    loop_count = resolve_battle_loop_count(
        env_config.battle_loop_count,
        env_config.num_envs,
    )
    loop_threads: list[LoopThread] = []
    envs: list[BattleEnv] = []
    try:
        for _ in range(loop_count):
            loop_threads.append(LoopThread.start())

        for i in range(env_config.num_envs):
            env_instance_config = replace(
                env_config,
                opponent_group=i % opponent_groups,
            )
            opponent_factory = make_league_opponent_factory(
                league,
                checkpoint_path=checkpoint_by_group[env_instance_config.opponent_group],
            )
            env = BattleEnv(
                env_instance_config,
                game_data,
                opponent_factory,
                team_preview_fn=team_preview_fn,
                team_preview_batcher=team_preview_batcher,
                team_pair_provider=team_pair_provider,
                loop_thread=loop_threads[i % loop_count],
                collect_timing=collect_timing,
            )
            envs.append(env)
        return ParallelBattleEnv(
            envs,
            collect_timing=collect_timing,
            loop_threads=loop_threads,
        )
    except BaseException as construction_error:
        cleanup_errors: list[BaseException] = []
        for env in reversed(envs):
            try:
                env.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for loop_thread in loop_threads:
            try:
                loop_thread.close()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        for cleanup_error in cleanup_errors:
            construction_error.add_note(
                f"Parallel environment construction cleanup failed: {cleanup_error!r}"
            )
        raise


def _sample_refresh_assignments(
    parallel_env: ParallelBattleEnv,
    league: League,
    checkpoint_sampler: CheckpointSampler | None = None,
) -> list[tuple[BattleEnv, str]]:
    assignments: list[tuple[BattleEnv, str]] = []
    groups = sorted({int(env.config.opponent_group) for env in parallel_env.envs})
    sampler = checkpoint_sampler or league.sample_checkpoint_paths
    sampled_paths = sampler(len(groups))
    checkpoint_by_group = dict(zip(groups, sampled_paths, strict=True))
    for env in parallel_env.envs:
        opponent_group = int(env.config.opponent_group)
        checkpoint_path = checkpoint_by_group[opponent_group]
        assignments.append((env, checkpoint_path))
    return assignments


def refresh_opponents(
    parallel_env: ParallelBattleEnv,
    league: League,
    checkpoint_sampler: CheckpointSampler | None = None,
) -> dict[int, str]:
    """Swap out opponents in existing environments.

    Called periodically so the agent faces a freshly sampled set of past
    selves. Envs in the same opponent group share one sampled checkpoint, which
    keeps opponent inference batches large on local setups.
    """
    assignments = _sample_refresh_assignments(
        parallel_env,
        league,
        checkpoint_sampler,
    )
    checkpoint_by_group: dict[int, str] = {}
    for env, checkpoint_path in assignments:
        checkpoint_by_group[int(env.config.opponent_group)] = checkpoint_path
        if env.stage_next_opponent_checkpoint(checkpoint_path):
            continue
        env.set_next_opponent(
            make_league_opponent_factory(league, checkpoint_path=checkpoint_path)
        )
    return checkpoint_by_group


def _current_opponent_assignments(
    parallel_env: ParallelBattleEnv,
) -> dict[int, str]:
    assignments: dict[int, str] = {}
    for env in parallel_env.envs:
        checkpoint_path = getattr(env.opponent, "checkpoint_path", None)
        if isinstance(checkpoint_path, str) and checkpoint_path:
            assignments[int(env.config.opponent_group)] = checkpoint_path
    return assignments


def _maintain_league_after_update(
    *,
    update: int,
    num_updates: int,
    global_step: int,
    agent: PokeAgent,
    parallel_env: ParallelBattleEnv,
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


def _maintain_best_response_after_update(
    *,
    update: int,
    num_updates: int,
    parallel_env: ParallelBattleEnv,
    league: League,
    config: TrainingConfig,
    checkpoint_sampler: CheckpointSampler,
) -> LeagueMaintenance:
    """Refresh a frozen target without updating payoffs or adding the learner."""
    assignments: dict[int, str] = {}
    if update < num_updates and update % config.league_refresh_interval == 0:
        assignments = refresh_opponents(
            parallel_env,
            league,
            checkpoint_sampler,
        )
    return LeagueMaintenance(
        checkpoint_added=False,
        assignments_by_group=assignments,
    )


def obs_to_device(
    obs: dict[str, np.ndarray],
    device: torch.device,
    buffers: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if buffers is None:
        buffers = {}

    for key, value in obs.items():
        source = torch.from_numpy(value)
        buffer = buffers.get(key)
        if (
            buffer is None
            or buffer.shape != source.shape
            or buffer.dtype != source.dtype
            or buffer.device != device
        ):
            buffer = torch.empty(source.shape, dtype=source.dtype, device=device)
            buffers[key] = buffer
        buffer.copy_(source)

    stale_keys = [key for key in buffers if key not in obs]
    for key in stale_keys:
        del buffers[key]

    return buffers


def action_masks_to_device(
    masks: dict[str, np.ndarray],
    device: torch.device,
    buffers: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    if buffers is None:
        buffers = {}

    for key, value in masks.items():
        buffer = buffers.get(key)
        expected_shape = value.shape
        if (
            buffer is None
            or buffer.shape != expected_shape
            or buffer.dtype != torch.bool
            or buffer.device != device
        ):
            buffer = torch.empty(expected_shape, dtype=torch.bool, device=device)
            buffers[key] = buffer
        buffer.copy_(torch.from_numpy(value))

    stale_keys = [key for key in buffers if key not in masks]
    for key in stale_keys:
        del buffers[key]

    return buffers


def array_to_device(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
    buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Copy a NumPy array into a reusable tensor on ``device``."""
    source = torch.from_numpy(array)
    target_dtype = source.dtype if dtype is None else dtype
    if (
        buffer is None
        or buffer.shape != source.shape
        or buffer.dtype != target_dtype
        or buffer.device != device
    ):
        buffer = torch.empty(source.shape, dtype=target_dtype, device=device)
    buffer.copy_(source)
    return buffer


def clone_hidden_state(
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if hidden is None:
        return None
    return tuple(state.detach().clone() for state in hidden)


def collect_synchronous_rollout(
    *,
    parallel_env: ParallelBattleEnv,
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
    obs_device_buffers: dict[str, torch.Tensor] | None = None
    mask_device_buffers: dict[str, torch.Tensor] | None = None
    rewards_device_buffer: torch.Tensor | None = None
    dones_device_buffer: torch.Tensor | None = None
    opponent_intent_slot_labels_np = np.empty(
        (parallel_env.num_envs, 2, NUM_SLOT_INTENT_COMPONENTS),
        dtype=np.int64,
    )
    opponent_intent_slot_masks_np = np.empty_like(
        opponent_intent_slot_labels_np,
        dtype=bool,
    )
    opponent_intent_tera_labels_np = np.empty(
        (parallel_env.num_envs, NUM_TERA_INTENT_COMPONENTS),
        dtype=np.int64,
    )
    opponent_intent_tera_masks_np = np.empty_like(
        opponent_intent_tera_labels_np,
        dtype=bool,
    )
    consequence_slot_shape = (parallel_env.num_envs, 2)
    consequence_target_shape = (
        parallel_env.num_envs,
        2,
        NUM_PUBLIC_TARGET_POSITIONS,
    )
    public_consequence_np = {
        "move_emitted_targets": np.empty(consequence_slot_shape, dtype=np.float32),
        "move_emitted_masks": np.empty(consequence_slot_shape, dtype=bool),
        "nonmove_reason_labels": np.empty(consequence_slot_shape, dtype=np.int64),
        "nonmove_reason_masks": np.empty(consequence_slot_shape, dtype=bool),
        "move_branch_labels": np.empty(consequence_slot_shape, dtype=np.int64),
        "move_branch_masks": np.empty(consequence_slot_shape, dtype=bool),
        "route_targets": np.empty(consequence_target_shape, dtype=np.float32),
        "route_masks": np.empty(consequence_target_shape, dtype=bool),
        "interaction_targets": np.empty(consequence_target_shape, dtype=np.float32),
        "interaction_masks": np.empty(consequence_target_shape, dtype=bool),
        "interaction_reason_labels": np.empty(consequence_target_shape, dtype=np.int64),
        "interaction_reason_masks": np.empty(consequence_target_shape, dtype=bool),
        "hit_count_labels": np.empty(consequence_target_shape, dtype=np.int64),
        "hit_count_masks": np.empty(consequence_target_shape, dtype=bool),
    }
    public_consequence_cant_targets_np = np.empty(
        (parallel_env.num_envs, 2, 2),
        dtype=np.float32,
    )
    public_consequence_cant_masks_np = np.empty_like(
        public_consequence_cant_targets_np,
        dtype=bool,
    )
    public_consequence_field_targets_np = np.empty(
        (parallel_env.num_envs, 2, PUBLIC_CONSEQUENCE_FIELD_DIM),
        dtype=np.float32,
    )
    public_consequence_field_masks_np = np.empty(
        (parallel_env.num_envs, 2),
        dtype=bool,
    )
    opponent_intent_slot_labels_device: torch.Tensor | None = None
    opponent_intent_slot_masks_device: torch.Tensor | None = None
    opponent_intent_tera_labels_device: torch.Tensor | None = None
    opponent_intent_tera_masks_device: torch.Tensor | None = None
    public_consequence_device_buffers: dict[str, torch.Tensor | None] = {
        key: None for key in public_consequence_np
    }
    public_consequence_cant_targets_device: torch.Tensor | None = None
    public_consequence_cant_masks_device: torch.Tensor | None = None
    public_consequence_field_targets_device: torch.Tensor | None = None
    public_consequence_field_masks_device: torch.Tensor | None = None
    for _ in range(num_steps):
        with torch.no_grad(), cuda_model_execution(device):
            started = time.perf_counter() if collect_timing else 0.0
            obs_t = obs_to_device(obs, device, buffers=obs_device_buffers)
            obs_device_buffers = obs_t
            if collect_timing:
                profile["obs_to_device_sec"] += time.perf_counter() - started

            started = time.perf_counter() if collect_timing else 0.0
            masks_t = action_masks_to_device(
                action_masks,
                device,
                buffers=mask_device_buffers,
            )
            mask_device_buffers = masks_t
            if collect_timing:
                profile["mask_to_device_sec"] += time.perf_counter() - started

            started = time.perf_counter() if collect_timing else 0.0
            out = agent(
                obs_t,
                hidden,
                action_masks=masks_t,
                tera_exploration_prob=tera_exploration_prob,
            )
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
        rewards_t = array_to_device(
            rewards,
            device,
            dtype=torch.float32,
            buffer=rewards_device_buffer,
        )
        dones_t = array_to_device(
            dones,
            device,
            dtype=torch.bool,
            buffer=dones_device_buffer,
        )
        rewards_device_buffer = rewards_t
        dones_device_buffer = dones_t

        opponent_intent_slot_labels_np.fill(-1)
        opponent_intent_slot_masks_np.fill(False)
        opponent_intent_tera_labels_np.fill(-1)
        opponent_intent_tera_masks_np.fill(False)
        for consequence_name, consequence_array in public_consequence_np.items():
            if consequence_name.endswith("labels"):
                consequence_array.fill(-1)
            elif consequence_name.endswith("masks"):
                consequence_array.fill(False)
            else:
                consequence_array.fill(0.0)
        public_consequence_cant_targets_np.fill(0.0)
        public_consequence_cant_masks_np.fill(False)
        public_consequence_field_targets_np.fill(0.0)
        public_consequence_field_masks_np.fill(False)
        for env_index, info in enumerate(infos):
            if dones[env_index]:
                consequence_observation = info.pop("final_observation", None)
                if consequence_observation is None:
                    raise RuntimeError(
                        "Auto-reset transition omitted final_observation for "
                        "terminal auxiliary labels."
                    )
            else:
                consequence_observation = {
                    key: next_obs[key][env_index]
                    for key in PUBLIC_CONSEQUENCE_OBSERVATION_KEYS
                    if key in next_obs
                }
            consequence = encode_public_consequence(consequence_observation)
            for consequence_name, consequence_array in public_consequence_np.items():
                consequence_array[env_index] = getattr(consequence, consequence_name)
            public_consequence_cant_targets_np[env_index] = consequence.cant_targets
            public_consequence_cant_masks_np[env_index] = consequence.cant_masks
            public_consequence_field_targets_np[env_index] = consequence.field_targets
            # A terminal observation resolves the selected moves, but Showdown
            # does not advance battle.turn after the battle-ending action. The
            # field head's persistence baseline is explicitly a one-turn
            # transition, so supervising that non-advanced terminal field would
            # teach a systematic false deviation. Other exact consequences stay
            # valid on the terminal segment.
            if not dones[env_index]:
                public_consequence_field_masks_np[env_index] = consequence.field_masks

            slot_labels = info.get("opponent_intent_slot_labels")
            if slot_labels is None:
                continue
            required_keys = {
                "opponent_intent_slot_masks",
                "opponent_intent_tera_labels",
                "opponent_intent_tera_masks",
            }
            missing_keys = required_keys.difference(info)
            if missing_keys:
                raise RuntimeError(
                    "Semantic opponent intent omitted keys: "
                    + ", ".join(sorted(missing_keys))
                )
            opponent_intent_slot_labels_np[env_index] = slot_labels
            opponent_intent_slot_masks_np[env_index] = info[
                "opponent_intent_slot_masks"
            ]
            opponent_intent_tera_labels_np[env_index] = info[
                "opponent_intent_tera_labels"
            ]
            opponent_intent_tera_masks_np[env_index] = info[
                "opponent_intent_tera_masks"
            ]

        opponent_intent_slot_labels_t = array_to_device(
            opponent_intent_slot_labels_np,
            device,
            dtype=torch.long,
            buffer=opponent_intent_slot_labels_device,
        )
        opponent_intent_slot_masks_t = array_to_device(
            opponent_intent_slot_masks_np,
            device,
            dtype=torch.bool,
            buffer=opponent_intent_slot_masks_device,
        )
        opponent_intent_tera_labels_t = array_to_device(
            opponent_intent_tera_labels_np,
            device,
            dtype=torch.long,
            buffer=opponent_intent_tera_labels_device,
        )
        opponent_intent_tera_masks_t = array_to_device(
            opponent_intent_tera_masks_np,
            device,
            dtype=torch.bool,
            buffer=opponent_intent_tera_masks_device,
        )
        opponent_intent_slot_labels_device = opponent_intent_slot_labels_t
        opponent_intent_slot_masks_device = opponent_intent_slot_masks_t
        opponent_intent_tera_labels_device = opponent_intent_tera_labels_t
        opponent_intent_tera_masks_device = opponent_intent_tera_masks_t

        public_consequence_t: dict[str, torch.Tensor] = {}
        for consequence_name, consequence_array in public_consequence_np.items():
            consequence_dtype = (
                torch.long
                if consequence_name.endswith("labels")
                else (
                    torch.bool if consequence_name.endswith("masks") else torch.float32
                )
            )
            public_consequence_t[consequence_name] = array_to_device(
                consequence_array,
                device,
                dtype=consequence_dtype,
                buffer=public_consequence_device_buffers[consequence_name],
            )
            public_consequence_device_buffers[consequence_name] = public_consequence_t[
                consequence_name
            ]
        public_consequence_cant_targets_t = array_to_device(
            public_consequence_cant_targets_np,
            device,
            dtype=torch.float32,
            buffer=public_consequence_cant_targets_device,
        )
        public_consequence_cant_masks_t = array_to_device(
            public_consequence_cant_masks_np,
            device,
            dtype=torch.bool,
            buffer=public_consequence_cant_masks_device,
        )
        public_consequence_field_targets_t = array_to_device(
            public_consequence_field_targets_np,
            device,
            dtype=torch.float32,
            buffer=public_consequence_field_targets_device,
        )
        public_consequence_field_masks_t = array_to_device(
            public_consequence_field_masks_np,
            device,
            dtype=torch.bool,
            buffer=public_consequence_field_masks_device,
        )
        public_consequence_cant_targets_device = public_consequence_cant_targets_t
        public_consequence_cant_masks_device = public_consequence_cant_masks_t
        public_consequence_field_targets_device = public_consequence_field_targets_t
        public_consequence_field_masks_device = public_consequence_field_masks_t

        if collect_timing:
            profile["feedback_to_device_sec"] += time.perf_counter() - started

        started = time.perf_counter() if collect_timing else 0.0
        buffer.add(
            obs_for_buffer,
            actions_t,
            masks_for_buffer,
            log_probs_t,
            values_t,
            rewards_t,
            dones_t,
            opponent_intent_slot_labels=opponent_intent_slot_labels_t,
            opponent_intent_slot_masks=opponent_intent_slot_masks_t,
            opponent_intent_tera_labels=opponent_intent_tera_labels_t,
            opponent_intent_tera_masks=opponent_intent_tera_masks_t,
            public_consequence_cant_targets=public_consequence_cant_targets_t,
            public_consequence_cant_masks=public_consequence_cant_masks_t,
            public_consequence_field_targets=public_consequence_field_targets_t,
            public_consequence_field_masks=public_consequence_field_masks_t,
            **{
                f"public_consequence_{name}": value
                for name, value in public_consequence_t.items()
            },
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
            not_done = (~dones_t).to(h.dtype).view(1, dones_t.shape[0], 1)
            hidden = (h * not_done, c * not_done)
            if collect_timing:
                profile["hidden_reset_sec"] += time.perf_counter() - started

        rollout_infos.extend(infos)
        obs, action_masks = next_obs, next_masks

    return obs, action_masks, hidden, rollout_infos, profile


def _league_state_path_for_resume(resume_checkpoint: str) -> Path:
    return Path(resume_checkpoint).resolve().with_suffix(".league_state.json")


def _phase_state_path_for_resume(resume_checkpoint: str) -> Path:
    return Path(resume_checkpoint).resolve().with_suffix(".phase_state.json")


def _main_run_identity(
    *,
    battle_format: str,
    team_pool_signature: str | None,
    model_signature: str,
    population_revision_id: str | None,
) -> dict[str, object]:
    """Return the immutable scientific identity carried by main checkpoints."""

    if (
        not isinstance(model_signature, str)
        or len(model_signature) != 64
        or any(character not in "0123456789abcdef" for character in model_signature)
    ):
        raise ValueError("Main-run model_signature must be a lowercase SHA-256 digest.")
    if population_revision_id is not None and (
        not isinstance(population_revision_id, str)
        or len(population_revision_id) != 64
        or any(
            character not in "0123456789abcdef" for character in population_revision_id
        )
    ):
        raise ValueError(
            "Main-run population_revision_id must be a lowercase SHA-256 digest."
        )
    if team_pool_signature is not None and (
        not isinstance(team_pool_signature, str)
        or len(team_pool_signature) != 64
        or any(character not in "0123456789abcdef" for character in team_pool_signature)
    ):
        raise ValueError(
            "Main-run team_pool_signature must be a lowercase SHA-256 digest."
        )
    if not isinstance(battle_format, str) or not battle_format:
        raise ValueError("Main-run battle_format must be a non-empty string.")
    return {
        "format_version": _MAIN_RUN_IDENTITY_VERSION,
        "battle_format": battle_format,
        "team_pool_signature": team_pool_signature,
        "model_signature": model_signature,
        "population_revision_id": population_revision_id,
    }


def _validate_main_run_identity(
    value: object,
    *,
    battle_format: str,
    team_pool_signature: str | None,
    model_signature: str,
) -> str | None:
    if not isinstance(value, dict):
        raise ValueError("Paired league state omitted the main-run identity.")
    expected_keys = {
        "format_version",
        "battle_format",
        "team_pool_signature",
        "model_signature",
        "population_revision_id",
    }
    if set(value) != expected_keys:
        raise ValueError(
            "Paired main-run identity has invalid fields: "
            f"missing={sorted(expected_keys - set(value))}, "
            f"unexpected={sorted(set(value) - expected_keys)}."
        )
    if (
        type(value["format_version"]) is not int
        or value["format_version"] != _MAIN_RUN_IDENTITY_VERSION
    ):
        raise ValueError("Unsupported main-run identity format version.")
    declared = _main_run_identity(
        battle_format=value["battle_format"],
        team_pool_signature=value["team_pool_signature"],
        model_signature=value["model_signature"],
        population_revision_id=value["population_revision_id"],
    )
    for name, runtime in (
        ("battle_format", battle_format),
        ("team_pool_signature", team_pool_signature),
        ("model_signature", model_signature),
    ):
        if declared[name] != runtime:
            raise ValueError(
                f"Resume main-run {name} mismatch: "
                f"checkpoint={declared[name]!r}, runtime={runtime!r}."
            )
    population_revision_id = declared["population_revision_id"]
    return None if population_revision_id is None else str(population_revision_id)


def _serialize_opponent_assignments(
    assignments: dict[int, str],
    league: League,
) -> dict[str, str]:
    expected_groups = set(range(len(assignments)))
    if not assignments or set(assignments) != expected_groups:
        raise ValueError(
            "Opponent assignments must use contiguous group indices starting at zero."
        )
    archive_paths = set(league.checkpoints)
    if any(
        not isinstance(path, str) or path not in archive_paths
        for path in assignments.values()
    ):
        raise ValueError("Opponent assignments must reference archived checkpoints.")
    return {str(group): assignments[group] for group in sorted(assignments)}


def _restore_opponent_assignments(
    value: object,
    *,
    opponent_groups: int,
    league: League,
) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ValueError("Paired league state omitted opponent assignments.")
    expected_keys = {str(group) for group in range(opponent_groups)}
    if set(value) != expected_keys:
        raise ValueError(
            "Saved opponent assignments do not match configured opponent groups."
        )
    archive_paths = set(league.checkpoints)
    restored: dict[int, str] = {}
    for group in range(opponent_groups):
        checkpoint_path = value[str(group)]
        if not isinstance(checkpoint_path, str) or checkpoint_path not in archive_paths:
            raise ValueError(
                "Saved opponent assignment references a checkpoint outside the archive."
            )
        restored[group] = checkpoint_path
    return restored


def _save_training_state(
    league: League,
    checkpoint_path: str,
    team_pool: TeamPool | None,
    training_state_id: str,
    opponent_assignments: dict[int, str],
    run_identity: dict[str, object],
) -> None:
    team_pool_state = (
        None
        if team_pool is None
        else {
            "signature": team_pool.signature(),
            "rng_state": team_pool.rng_state(),
        }
    )
    league.save_state(
        str(_league_state_path_for_resume(checkpoint_path)),
        extra_state={
            "training_state_id": training_state_id,
            "run_identity": run_identity,
            "team_pool": team_pool_state,
            "opponent_assignments": _serialize_opponent_assignments(
                opponent_assignments,
                league,
            ),
        },
    )


def _save_best_response_training_state(
    *,
    checkpoint_path: str,
    phase: PhaseSpec,
    sampler: FrozenMixtureSampler,
    team_pool: TeamPool,
    training_state_id: str,
    completed_updates: int,
    opponent_assignments: dict[int, str],
    league: League,
) -> None:
    serialized_assignments = _serialize_opponent_assignments(
        opponent_assignments,
        league,
    )
    state = PhaseRunState(
        phase_hash=phase.phase_hash,
        target_population_revision_id=phase.target_population_revision_id,
        training_state_id=training_state_id,
        completed_updates=completed_updates,
        sampler_state=sampler.state_dict(),
        team_pool_signature=team_pool.signature(),
        team_pool_rng_state=team_pool.rng_state(),
        opponent_assignments=tuple(
            (int(group), checkpoint_path)
            for group, checkpoint_path in serialized_assignments.items()
        ),
    )
    state.save(_phase_state_path_for_resume(checkpoint_path))


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


def build_rollout_diagnostics(
    rollout_profile: dict[str, float],
    env_profile: dict[str, float],
) -> dict[str, float]:
    """Summarize timing counters into stable throughput diagnostics."""
    env_step_total_sec = float(env_profile.get("env_step_total_sec", 0.0))
    env_step_calls = float(env_profile.get("env_step_calls", 0.0))
    env_queue_wait_sec = float(env_profile.get("env_queue_wait_sec", 0.0))
    parallel_step_wall_sec = float(env_profile.get("parallel_step_wall_sec", 0.0))
    parallel_step_calls = float(env_profile.get("parallel_step_calls", 0.0))
    parallel_step_completion_span_sec = float(
        env_profile.get("parallel_step_completion_span_sec", 0.0)
    )
    parallel_step_first_completion_sec = float(
        env_profile.get("parallel_step_first_completion_sec", 0.0)
    )
    parallel_step_median_completion_sec = float(
        env_profile.get("parallel_step_median_completion_sec", 0.0)
    )
    parallel_step_p90_completion_sec = float(
        env_profile.get("parallel_step_p90_completion_sec", 0.0)
    )
    opponent_choice_sec = float(env_profile.get("opponent_choice_sec", 0.0))
    opponent_choice_calls = float(env_profile.get("opponent_choice_calls", 0.0))
    agent_action_latency_sec = float(
        env_profile.get("env_agent_action_latency_sec", 0.0)
    )
    agent_action_latency_calls = float(
        env_profile.get("env_agent_action_latency_calls", 0.0)
    )
    action_queue_get_wait_sec = float(
        env_profile.get("player_action_queue_get_wait_sec", 0.0)
    )
    action_queue_get_calls = float(
        env_profile.get("player_action_queue_get_calls", 0.0)
    )
    action_queue_item_age_sec = float(
        env_profile.get("player_action_queue_item_age_sec", 0.0)
    )
    action_queue_loop_put_delay_sec = float(
        env_profile.get("player_action_queue_loop_put_delay_sec", 0.0)
    )
    action_queue_put_completed_calls = float(
        env_profile.get("player_action_queue_threadsafe_put_completed_calls", 0.0)
    )
    obs_queue_get_wait_sec = float(
        env_profile.get("player_obs_queue_get_wait_sec", 0.0)
    )
    obs_queue_get_calls = float(env_profile.get("player_obs_queue_get_calls", 0.0))
    obs_queue_item_age_sec = float(
        env_profile.get("player_obs_queue_item_age_sec", 0.0)
    )
    opponent_infer_action_wait_sec = float(
        env_profile.get("opponent_infer_action_wait_sec", 0.0)
    )
    opponent_infer_action_queue_wait_sec = float(
        env_profile.get("opponent_infer_action_queue_wait_sec", 0.0)
    )
    opponent_infer_action_queue_wait_calls = float(
        env_profile.get("opponent_infer_action_queue_wait_calls", 0.0)
    )
    opponent_infer_preview_queue_wait_sec = float(
        env_profile.get("opponent_infer_preview_queue_wait_sec", 0.0)
    )
    opponent_infer_preview_queue_wait_calls = float(
        env_profile.get("opponent_infer_preview_queue_wait_calls", 0.0)
    )
    rollout_sec = float(rollout_profile.get("rollout_sec", 0.0))
    device_transfer_sec = (
        float(rollout_profile.get("obs_to_device_sec", 0.0))
        + float(rollout_profile.get("mask_to_device_sec", 0.0))
        + float(rollout_profile.get("feedback_to_device_sec", 0.0))
    )

    return {
        "env_queue_wait_share": _safe_div(env_queue_wait_sec, env_step_total_sec),
        "avg_env_step_ms": _safe_div(env_step_total_sec, env_step_calls) * 1000.0,
        "avg_parallel_step_wall_ms": _safe_div(
            parallel_step_wall_sec, parallel_step_calls
        )
        * 1000.0,
        "avg_parallel_step_first_completion_ms": _safe_div(
            parallel_step_first_completion_sec,
            parallel_step_calls,
        )
        * 1000.0,
        "avg_parallel_step_median_completion_ms": _safe_div(
            parallel_step_median_completion_sec,
            parallel_step_calls,
        )
        * 1000.0,
        "avg_parallel_step_p90_completion_ms": _safe_div(
            parallel_step_p90_completion_sec,
            parallel_step_calls,
        )
        * 1000.0,
        "avg_parallel_step_completion_span_ms": _safe_div(
            parallel_step_completion_span_sec,
            parallel_step_calls,
        )
        * 1000.0,
        "parallel_step_completion_span_share": _safe_div(
            parallel_step_completion_span_sec,
            parallel_step_wall_sec,
        ),
        "avg_opponent_choice_ms": _safe_div(opponent_choice_sec, opponent_choice_calls)
        * 1000.0,
        "avg_agent_action_latency_ms": _safe_div(
            agent_action_latency_sec,
            agent_action_latency_calls,
        )
        * 1000.0,
        "avg_player_action_queue_get_wait_ms": _safe_div(
            action_queue_get_wait_sec,
            action_queue_get_calls,
        )
        * 1000.0,
        "avg_player_action_queue_item_age_ms": _safe_div(
            action_queue_item_age_sec,
            action_queue_get_calls,
        )
        * 1000.0,
        "avg_player_action_queue_loop_put_delay_ms": _safe_div(
            action_queue_loop_put_delay_sec,
            action_queue_put_completed_calls,
        )
        * 1000.0,
        "avg_player_obs_queue_get_wait_ms": _safe_div(
            obs_queue_get_wait_sec,
            obs_queue_get_calls,
        )
        * 1000.0,
        "avg_player_obs_queue_item_age_ms": _safe_div(
            obs_queue_item_age_sec,
            obs_queue_get_calls,
        )
        * 1000.0,
        "avg_opponent_infer_action_queue_wait_ms": _safe_div(
            opponent_infer_action_queue_wait_sec,
            opponent_infer_action_queue_wait_calls,
        )
        * 1000.0,
        "opponent_infer_action_queue_wait_share": _safe_div(
            opponent_infer_action_queue_wait_sec,
            opponent_infer_action_wait_sec,
        ),
        "avg_opponent_infer_preview_queue_wait_ms": _safe_div(
            opponent_infer_preview_queue_wait_sec,
            opponent_infer_preview_queue_wait_calls,
        )
        * 1000.0,
        "policy_forward_share": _safe_div(
            float(rollout_profile.get("policy_forward_sec", 0.0)), rollout_sec
        ),
        "device_transfer_share": _safe_div(device_transfer_sec, rollout_sec),
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
    phase_bundle = (
        load_best_response_phase(args.phase_spec) if args.phase_spec else None
    )
    phase_spec = phase_bundle.spec if phase_bundle is not None else None
    if phase_spec is not None:
        if (
            args.battle_format is not None
            and args.battle_format != phase_spec.battle_format
        ):
            raise ValueError(
                "--battle-format cannot override the immutable phase format: "
                f"cli={args.battle_format!r}, phase={phase_spec.battle_format!r}."
            )
        args.battle_format = phase_spec.battle_format
        if args.team_pool_manifest is None and args.team_pool_format is None:
            args.team_pool_format = phase_spec.battle_format
    torch_runtime_config = apply_torch_runtime_config(args)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    run_started = time.perf_counter()
    collect_timing = args.collect_timing

    model_config, env_config, train_config = build_configs(args)
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
    if env_config.showdown_dir is None:
        env_config.showdown_dir = default_showdown_dir()
    if env_config.showdown_dir is None:
        raise SystemExit(
            "--showdown-dir is required. Set it to a Pokemon Showdown checkout, "
            "or set POKERL_SHOWDOWN_DIR."
        )
    if not Path(env_config.showdown_dir).exists():
        raise SystemExit(
            f"Pokemon Showdown checkout not found: {env_config.showdown_dir}"
        )
    opponent_groups = resolve_opponent_groups(args.opponent_groups, env_config.num_envs)
    team_pool = load_requested_team_pool(args, seed=seed_streams["team_pool"])
    if phase_spec is not None:
        if team_pool is None:
            raise ValueError(
                "A best-response phase requires the team pool identified by its "
                "team_pool_signature."
            )
        if team_pool.signature() != phase_spec.team_pool_signature:
            raise ValueError(
                "Configured team pool does not match the immutable phase signature."
            )

    steps_per_update = train_config.rollout_steps * env_config.num_envs
    if train_config.total_timesteps < steps_per_update:
        raise ValueError(
            f"total_timesteps ({train_config.total_timesteps}) must be at least one full update "
            f"({steps_per_update} = rollout_steps * num_envs)."
        )

    # Game data and model
    game_data = GameData(gen=9)
    agent = PokeAgent(model_config, game_data)
    agent.to(device)
    if phase_spec is not None and agent.model_signature != phase_spec.model_signature:
        raise ValueError(
            "Runtime architecture/vocabulary does not match the phase model signature."
        )
    main_run_identity = (
        None
        if phase_spec is not None
        else _main_run_identity(
            battle_format=env_config.battle_format,
            team_pool_signature=(None if team_pool is None else team_pool.signature()),
            model_signature=agent.model_signature,
            population_revision_id=None,
        )
    )

    print(f"Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
    if team_pool is not None:
        print(
            f"Team pool: {team_pool.format_id} [{team_pool.subset}] with {team_pool.size} teams"
        )

    # Training
    trainer = PPOTrainer(agent, train_config, device)
    optimizer_settings = trainer.optimizer_settings()
    print(
        "Optimizer: "
        f"{optimizer_settings['name']} "
        f"lr={optimizer_settings['lr']:g} "
        f"betas={optimizer_settings['betas']} "
        f"eps={optimizer_settings['eps']:g} "
        f"weight_decay={optimizer_settings['weight_decay']:g} "
        f"grouping={optimizer_settings['decay_grouping']}"
    )
    league = League(
        train_config,
        env_config,
        game_data,
        agent,
        device,
        rng=random.Random(seed_streams["league"]),
    )

    frozen_sampler: FrozenMixtureSampler | None = None
    phase_opponent_source: FrozenPhaseOpponentSource | None = None
    checkpoint_sampler: CheckpointSampler | None = None
    if phase_bundle is not None and phase_spec is not None:
        frozen_sampler = FrozenMixtureSampler(
            phase_spec.target_distribution,
            seed=seed_streams["league"],
        )
        target_policy_ids = {item.policy_id for item in phase_spec.target_distribution}
        target_checkpoints = {
            policy_id: checkpoint_path
            for policy_id, checkpoint_path in phase_bundle.checkpoint_by_policy_id.items()
            if policy_id in target_policy_ids
        }
        phase_opponent_source = FrozenPhaseOpponentSource(
            sampler=frozen_sampler,
            checkpoint_by_policy_id=target_checkpoints,
        )
        checkpoint_sampler = phase_opponent_source.sample_checkpoint_paths
        league.install_frozen_population(list(target_checkpoints.values()))

    # --- Resume from checkpoint ---
    restored_opponent_assignments: dict[int, str] | None = None
    if phase_spec is not None and phase_bundle is not None:
        if team_pool is None or frozen_sampler is None:
            raise RuntimeError("Best-response phase initialization was incomplete.")
        if args.resume:
            trainer.load(args.resume)
            print(
                f"Resumed best-response trainer from {args.resume!r} "
                f"(phase update {trainer.completed_updates}, "
                f"global step {trainer.global_step:,})"
            )
            phase_state_path = _phase_state_path_for_resume(args.resume)
            if not phase_state_path.exists():
                raise FileNotFoundError(
                    "Strict best-response resume requires the checkpoint's paired "
                    f"phase state: {phase_state_path!s}"
                )
            phase_state = PhaseRunState.load(phase_state_path)
            phase_state.validate_resume(
                phase=phase_spec,
                training_state_id=trainer.loaded_training_state_id,
                trainer_completed_updates=trainer.completed_updates,
                sampler=frozen_sampler,
            )
            if phase_state.team_pool_signature != team_pool.signature():
                raise ValueError(
                    "Configured team pool does not match the paired phase state."
                )
            if phase_state.team_pool_rng_state is None:
                raise ValueError("Paired phase state omitted the team-pool RNG state.")
            team_pool.restore_rng_state(phase_state.team_pool_rng_state)
            restored_opponent_assignments = _restore_opponent_assignments(
                {
                    str(group): checkpoint_path
                    for group, checkpoint_path in phase_state.opponent_assignments
                },
                opponent_groups=opponent_groups,
                league=league,
            )
            print(
                "Restored frozen mixture sampler at "
                f"draw {frozen_sampler.draw_count} from {phase_state_path!s}"
            )
        else:
            initializer_path = Path(
                phase_bundle.checkpoint_by_policy_id[
                    phase_spec.initialization.source_policy_id
                ]
            )
            trainer.initialize_from_trainer_weights(str(initializer_path))
            print(
                "Initialized fresh best-response learner from weights "
                f"{initializer_path!s}; optimizer, counters, and RNG are fresh"
            )
    elif args.resume:
        trainer.load(args.resume)
        print(
            f"Resumed trainer from {args.resume!r} (global step {trainer.global_step:,})"
        )

        league_state = _league_state_path_for_resume(args.resume)
        if not league_state.exists():
            raise FileNotFoundError(
                "Strict resume requires the checkpoint's paired league state: "
                f"{league_state!s}"
            )
        extra_state = league.load_state(str(league_state))
        if extra_state.get("training_state_id") != trainer.loaded_training_state_id:
            raise ValueError(
                "Trainer checkpoint and paired league state have different IDs."
            )
        validate_resume_update_state(
            trainer_completed_updates=trainer.completed_updates,
            league_current_update=league.current_update,
        )
        saved_team_pool = extra_state.get("team_pool")
        if team_pool is None:
            if saved_team_pool is not None:
                raise ValueError(
                    "Checkpoint used a team pool, but this resume disabled it."
                )
        else:
            if not isinstance(saved_team_pool, dict):
                raise ValueError("Paired league state omitted the team-pool state.")
            if saved_team_pool.get("signature") != team_pool.signature():
                raise ValueError(
                    "Configured team pool does not match the resume checkpoint."
                )
            team_pool_rng_state = saved_team_pool.get("rng_state")
            if not isinstance(team_pool_rng_state, tuple):
                raise ValueError("Paired league state has invalid team-pool RNG state.")
            team_pool.restore_rng_state(team_pool_rng_state)
        population_revision_id = _validate_main_run_identity(
            extra_state.get("run_identity"),
            battle_format=env_config.battle_format,
            team_pool_signature=(None if team_pool is None else team_pool.signature()),
            model_signature=agent.model_signature,
        )
        main_run_identity = _main_run_identity(
            battle_format=env_config.battle_format,
            team_pool_signature=(None if team_pool is None else team_pool.signature()),
            model_signature=agent.model_signature,
            population_revision_id=population_revision_id,
        )
        restored_opponent_assignments = _restore_opponent_assignments(
            extra_state.get("opponent_assignments"),
            opponent_groups=opponent_groups,
            league=league,
        )
        print(
            f"Restored league from {league_state!s}: {len(league.checkpoints)} checkpoint(s)"
        )
    # --- Seed league so self-play is available from update 1 ---
    # On a fresh start this ensures the agent never falls back to bots.
    if phase_spec is None and not league.checkpoints:
        with cuda_model_execution(device):
            league.add_checkpoint(agent, step=trainer.global_step)
        print(f"Seeded league with agent at step {trainer.global_step}")

    # --- Environments ---
    # Initial team previews are submitted during environment construction/reset.
    # Put the shared live policy in inference mode before the batcher thread can
    # observe it.
    agent.eval()
    team_preview_batcher = TeamPreviewBatcher(
        agent,
        device,
        policy_version=trainer.completed_updates,
    )
    team_preview_fn = make_team_preview_fn(team_preview_batcher)
    team_pair_provider = make_team_pair_provider(team_pool)
    parallel_env = make_envs(
        env_config,
        game_data,
        league,
        opponent_groups,
        team_preview_fn=team_preview_fn,
        team_preview_batcher=team_preview_batcher,
        team_pair_provider=team_pair_provider,
        collect_timing=collect_timing,
        initial_checkpoint_by_group=restored_opponent_assignments,
        checkpoint_sampler=checkpoint_sampler,
    )
    current_opponent_assignments = _current_opponent_assignments(parallel_env)
    run_assigned_opponents = set(current_opponent_assignments.values())
    run_observed_opponents: set[str] = set()
    run_opponent_outcomes: OpponentOutcomeCounts = {}
    phase_summary = None
    if (
        phase_spec is not None
        and phase_bundle is not None
        and phase_opponent_source is not None
        and frozen_sampler is not None
    ):
        phase_summary = {
            "phase_id": phase_spec.phase_id,
            "phase_hash": phase_spec.phase_hash,
            "kind": phase_spec.kind.value,
            "learner_policy_id": phase_spec.learner_policy_id,
            "learner_role": phase_spec.learner_role.value,
            "initialization_mode": phase_spec.initialization.mode.value,
            "initializer_checkpoint": phase_bundle.checkpoint_by_policy_id[
                phase_spec.initialization.source_policy_id
            ],
            "initialized_from_training_state_id": (
                trainer.initialized_from_training_state_id
            ),
            "target_population_path": str(phase_bundle.population_path),
            "target_population_revision_id": (phase_spec.target_population_revision_id),
            "opponent_sampler": phase_spec.opponent_sampler.value,
            "mixture_id": frozen_sampler.mixture_id,
            "sampler_draw_count_at_start": frozen_sampler.draw_count,
            "archive_policy": phase_spec.archive_policy.value,
            "target_distribution": [
                {
                    "policy_id": item.policy_id,
                    "probability": item.probability,
                    "checkpoint_path": (
                        phase_opponent_source.checkpoint_by_policy_id[item.policy_id]
                    ),
                }
                for item in phase_spec.target_distribution
            ],
        }
    run_summary = {
        "status": "starting",
        "phase": phase_summary,
        "run_identity": main_run_identity,
        "config": {
            "training_mode": (
                "best_response" if phase_spec is not None else "main_self_play"
            ),
            "device": str(device),
            "seed": train_config.seed,
            "seed_streams": seed_streams,
            "torch_runtime": torch_runtime_config,
            "battle_format": env_config.battle_format,
            "num_envs": env_config.num_envs,
            "team_size": env_config.team_size,
            "pick_size": env_config.pick_size,
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
            "tera_exploration_decay_fraction": (
                train_config.tera_exploration_decay_fraction
            ),
            "tera_gate_entropy_coef": train_config.tera_gate_entropy_coef,
            "tera_slot_entropy_coef": train_config.tera_slot_entropy_coef,
            "aux_hidden_info_coef": train_config.aux_hidden_info_coef,
            "aux_opponent_intent_coef": train_config.aux_opponent_intent_coef,
            "aux_public_consequence_coef": (train_config.aux_public_consequence_coef),
            "diagnose_memory_gradients": train_config.diagnose_memory_gradients,
            "league_size": train_config.league_size,
            "league_update_interval": train_config.league_update_interval,
            "league_refresh_interval": train_config.league_refresh_interval,
            "league_coalesce_ms": train_config.league_coalesce_ms,
            "league_cuda_graphs": train_config.league_cuda_graphs,
            "league_cuda_graph_batch_size": (train_config.league_cuda_graph_batch_size),
            "league_cuda_graph_cache_size": (train_config.league_cuda_graph_cache_size),
            "league_preview_device": train_config.league_preview_device,
            "league_payoff_half_life_updates": (
                train_config.league_payoff_half_life_updates
            ),
            "league_pfsp_exponent": train_config.league_pfsp_exponent,
            "league_pfsp_uniform_fraction": (train_config.league_pfsp_uniform_fraction),
            "league_inference_device": str(league.inference_device),
            "league_preview_inference_device": str(league.preview_inference_device),
            "collect_timing": collect_timing,
            "total_timesteps": train_config.total_timesteps,
            "backend": "local_battlestream",
            "opponent_groups": opponent_groups,
            "battlestream": battlestream_summary(env_config),
            "team_pool": team_pool.describe() if team_pool is not None else None,
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
        initial_env_profile = parallel_env.snapshot_timing() if collect_timing else {}
        if collect_timing:
            parallel_env.reset_timing()
        with cuda_model_execution(device):
            buffer = RolloutBuffer(
                train_config.rollout_steps,
                env_config.num_envs,
                obs,
                masks,
                device,
            )
            hidden = agent.initial_hidden(env_config.num_envs)
            recurrent_history = RecurrentObservationHistory.empty(env_config.num_envs)
        updates_done, num_updates = resolve_update_bounds(
            total_timesteps=train_config.total_timesteps,
            steps_per_update=steps_per_update,
            global_step=trainer.global_step,
            completed_updates=trainer.completed_updates,
            additional_updates=args.resume is not None and args.num_updates is not None,
        )
        tera_exploration_decay_updates = resolve_tera_exploration_decay_updates(
            num_updates,
            train_config.tera_exploration_decay_fraction,
        )
        run_summary["config"]["tera_exploration_decay_updates"] = (
            tera_exploration_decay_updates
        )

        print(f"Total updates: {num_updates}")
        print(f"Steps per update: {steps_per_update}")
        print(
            "Tera exploration: "
            f"{train_config.tera_exploration_start:g} -> "
            f"{train_config.tera_exploration_end:g} over "
            f"{tera_exploration_decay_updates} update(s)"
        )
        print("Backend: local BattleStream")
        print(f"Opponent groups: {opponent_groups}")
        if phase_spec is None:
            print(
                f"League: permanent archive, resident cache {train_config.league_size}; "
                f"snapshot every {train_config.league_update_interval} update(s), "
                f"refresh every {train_config.league_refresh_interval} update(s); "
                f"payoff half-life {train_config.league_payoff_half_life_updates:g}; "
                f"PFSP exponent {train_config.league_pfsp_exponent:g} with "
                f"{train_config.league_pfsp_uniform_fraction:.0%} uniform coverage"
            )
        else:
            assert frozen_sampler is not None
            print(
                "Best response: immutable target population "
                f"{phase_spec.target_population_revision_id[:12]} with "
                f"{len(phase_spec.target_distribution)} mixture member(s); "
                f"refresh every {train_config.league_refresh_interval} update(s); "
                "learner snapshots and payoff adaptation disabled"
            )
        print(
            "Frozen inference: "
            f"action={league.inference_device}, preview={league.preview_inference_device}; "
            f"CUDA graphs={'on' if train_config.league_cuda_graphs else 'off'} "
            f"batch={train_config.league_cuda_graph_batch_size} "
            f"cache={train_config.league_cuda_graph_cache_size}"
        )
        local_summary = battlestream_summary(env_config)
        print(
            "Local BattleStream: "
            f"workers={local_summary['worker_count']} "
            f"battles/worker={local_summary['max_battles_per_worker']} "
            f"runtime_loops={local_summary['runtime_loop_count']} "
            f"battle_loops={local_summary['battle_loop_count']}"
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

        for update in range(updates_done + 1, num_updates + 1):
            rollout_opponent_assignments = dict(current_opponent_assignments)
            rollout_policy_version = team_preview_batcher.policy_version
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
                                    info.get("episode_opponent_max_calls_same_turn", 0)
                                ),
                                "opponent_checkpoint_path": info.get(
                                    "episode_opponent_checkpoint_path"
                                ),
                            }
                        )

            # All behavior-policy preview work must be complete before the
            # shared parameters change. Include any residual wait in rollout
            # timing rather than leaving it as unaccounted update overhead.
            parallel_env.wait_for_prepared_team_previews()
            rollout_sec = time.perf_counter() - rollout_started
            if collect_timing:
                rollout_profile["rollout_sec"] = rollout_sec
            env_profile = parallel_env.snapshot_timing() if collect_timing else {}

            # --- Compute advantages ---
            advantage_started = time.perf_counter()
            with torch.no_grad(), cuda_model_execution(device):
                obs_t = obs_to_device(obs, device)
                last_value = agent.get_value(obs_t, hidden)
                batch = buffer.compute_gae(
                    last_value,
                    train_config.gamma,
                    train_config.gae_lambda,
                )
                observation_history_metrics = summarize_observation_history(batch.obs)
            advantage_sec = time.perf_counter() - advantage_started

            # --- PPO update ---
            ppo_started = time.perf_counter()
            with cuda_model_execution(device):
                agent.train()
                metrics = trainer.update(batch)
                metrics["tera_exploration_prob"] = tera_eps
                metrics.update(observation_history_metrics)
                metrics["team_preview_policy_version"] = float(rollout_policy_version)

                # TeamPreviewBatcher shares the live policy object.  Prepared
                # previews are submitted immediately below and may execute on
                # its background thread before the next rollout starts, so the
                # policy must leave this serialized training section in eval
                # mode.  Otherwise a preview can observe train-mode layers or
                # race the next update's agent.eval() transition.
                agent.eval()
                next_preview_policy_version = (
                    team_preview_batcher.advance_policy_version()
                )
                if next_preview_policy_version != rollout_policy_version + 1:
                    raise RuntimeError(
                        "Team-preview policy version did not advance exactly once."
                    )
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
            if phase_spec is None:
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
                if checkpoint_sampler is None:
                    raise RuntimeError(
                        "Best-response phase omitted its frozen checkpoint sampler."
                    )
                # Best responses report outcomes for diagnostics only.  Their
                # target weights, population, and learner archive stay frozen.
                with cuda_model_execution(device):
                    league_maintenance = _maintain_best_response_after_update(
                        update=update,
                        num_updates=num_updates,
                        parallel_env=parallel_env,
                        league=league,
                        config=train_config,
                        checkpoint_sampler=checkpoint_sampler,
                    )
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
                print(f"League updated (pool: {len(league.checkpoints)}){refresh_note}")

            if phase_spec is None and league.current_update != update:
                raise RuntimeError(
                    "League maintenance did not finish at the current update: "
                    f"expected={update}, actual={league.current_update}."
                )
            trainer.completed_updates = update

            # Pair the trainer with the mutable state for its training regime:
            # adaptive league state for main self-play, or exact frozen-sampler
            # and team RNG state for a best response.
            checkpoint_sec = 0.0
            if update % train_config.save_interval == 0:
                checkpoint_started = time.perf_counter()
                path = f"{train_config.checkpoint_dir}/step_{trainer.global_step}.pt"
                training_state_id = uuid.uuid4().hex
                if phase_spec is None:
                    if main_run_identity is None:
                        raise RuntimeError("Main training omitted its run identity.")
                    _save_training_state(
                        league,
                        path,
                        team_pool,
                        training_state_id,
                        current_opponent_assignments,
                        main_run_identity,
                    )
                else:
                    if team_pool is None or frozen_sampler is None:
                        raise RuntimeError(
                            "Best-response checkpoint state was not initialized."
                        )
                    _save_best_response_training_state(
                        checkpoint_path=path,
                        phase=phase_spec,
                        sampler=frozen_sampler,
                        team_pool=team_pool,
                        training_state_id=training_state_id,
                        completed_updates=trainer.completed_updates,
                        opponent_assignments=current_opponent_assignments,
                        league=league,
                    )
                with cuda_model_execution(device):
                    trainer.save(path, training_state_id=training_state_id)
                checkpoint_sec = time.perf_counter() - checkpoint_started
                print(f"Saved checkpoint: {path}")

            preview_refresh_sec = 0.0
            if update < num_updates:
                preview_refresh_started = time.perf_counter()
                parallel_env.refresh_prepared_team_previews()
                preview_refresh_sec = time.perf_counter() - preview_refresh_started

            avg_episode_sec = (
                float(np.mean(update_episode_secs)) if update_episode_secs else 0.0
            )
            avg_episode_turns = (
                float(np.mean(update_episode_turns)) if update_episode_turns else 0.0
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
            retained_player_battles = sum(
                len(env.player.battles) for env in parallel_env.envs
            )
            retained_opponent_battles = sum(
                len(env.opponent.battles) for env in parallel_env.envs
            )
            update_steps_per_second = (
                steps_per_update / (rollout_sec + ppo_sec)
                if (rollout_sec + ppo_sec) > 0
                else 0.0
            )
            update_accounted_sec = (
                rollout_sec + advantage_sec + ppo_sec + preview_refresh_sec
            )
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
                "phase": (
                    None
                    if phase_spec is None or frozen_sampler is None
                    else {
                        "phase_hash": phase_spec.phase_hash,
                        "target_population_revision_id": (
                            phase_spec.target_population_revision_id
                        ),
                        "mixture_id": frozen_sampler.mixture_id,
                        "sampler_draw_count": frozen_sampler.draw_count,
                        "target_adapted": False,
                        "learner_archived": False,
                    }
                ),
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
                "preview_refresh_sec": preview_refresh_sec,
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
                "retained_player_battles": retained_player_battles,
                "retained_opponent_battles": retained_opponent_battles,
                "opponents": {
                    "assignments_by_group": {
                        str(group): path
                        for group, path in sorted(rollout_opponent_assignments.items())
                    },
                    "assigned_checkpoint_paths": rollout_assignment_paths,
                    "distinct_assigned_opponents": len(rollout_assignment_paths),
                    "observed_checkpoint_paths": sorted(update_observed_opponents),
                    "distinct_observed_opponents": len(update_observed_opponents),
                    "outcomes": update_outcome_summary,
                    "refresh_performed": bool(league_maintenance.assignments_by_group),
                    "next_assignments_by_group": {
                        str(group): path
                        for group, path in sorted(current_opponent_assignments.items())
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
                    steps_per_update / update_wall_sec if update_wall_sec > 0 else 0.0
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
                update_summary["rollout_diagnostics"] = build_rollout_diagnostics(
                    rollout_profile,
                    env_profile,
                )
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
            if phase_summary is not None and frozen_sampler is not None:
                phase_summary["sampler_draw_count"] = frozen_sampler.draw_count
            run_summary["updates"].append(update_summary)
            run_summary["status"] = "running"
            run_summary["update_count"] = update
            run_summary["global_step"] = trainer.global_step
            _maybe_write_run_summary(args.summary_path, run_summary)

            # --- Logging ---
            if update % train_config.log_interval == 0:
                elapsed = time.time() - start_time
                invocation_steps = trainer.global_step - invocation_start_global_step
                sps = invocation_steps / elapsed if elapsed > 0 else 0.0
                win_rate = (
                    np.mean([r > 0 for r in episode_rewards[-100:]])
                    if episode_rewards
                    else 0
                )
                diagnostics = update_summary.get("rollout_diagnostics", {})
                diagnostics_text = ""
                if diagnostics:
                    diagnostics_text = (
                        f" | Env wait {diagnostics['env_queue_wait_share']:.0%}"
                        f" | Vec step {diagnostics['avg_parallel_step_wall_ms']:.0f}ms"
                        f" | Step p90 {diagnostics.get('avg_parallel_step_p90_completion_ms', 0.0):.0f}ms"
                        f" | Strag span {diagnostics.get('avg_parallel_step_completion_span_ms', 0.0):.0f}ms"
                        f" | Opp choice {diagnostics['avg_opponent_choice_ms']:.1f}ms"
                        f" | Act wait {diagnostics.get('avg_agent_action_latency_ms', 0.0):.1f}ms"
                        f" | ActQ age {diagnostics.get('avg_player_action_queue_item_age_ms', 0.0):.1f}ms"
                        f" | ObsQ wait {diagnostics.get('avg_player_obs_queue_get_wait_ms', 0.0):.1f}ms"
                        f" | OppQ {diagnostics.get('avg_opponent_infer_action_queue_wait_ms', 0.0):.1f}ms"
                    )
                population_label = (
                    f"League pool {len(league.checkpoints)}"
                    if phase_spec is None
                    else f"Frozen targets {len(league.checkpoints)}"
                )
                print(
                    f"Update {update}/{num_updates} | "
                    f"Step {trainer.global_step:,} | "
                    f"Envs {env_config.num_envs} | "
                    f"Opp groups {opponent_groups} | "
                    f"Avg SPS {sps:.0f} | "
                    f"Upd SPS {update_steps_per_second:.0f} | "
                    f"Rollout {rollout_sec:.2f}s | "
                    f"PPO {ppo_sec:.2f}s | "
                    f"Completed battles {len(update_episode_rewards)} | "
                    f"Avg battle {avg_episode_sec:.2f}s/{avg_episode_turns:.1f} turns | "
                    f"Opp calls/ep {avg_episode_opponent_choice_calls:.1f} | "
                    f"Max repeat-turn req {max_episode_opponent_repeated_turn_requests} | "
                    f"Retained battles {retained_player_battles + retained_opponent_battles} | "
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
        training_state_id = uuid.uuid4().hex
        if phase_spec is None:
            if main_run_identity is None:
                raise RuntimeError("Main training omitted its run identity.")
            _save_training_state(
                league,
                final_path,
                team_pool,
                training_state_id,
                current_opponent_assignments,
                main_run_identity,
            )
        else:
            if team_pool is None or frozen_sampler is None:
                raise RuntimeError("Best-response final state was not initialized.")
            _save_best_response_training_state(
                checkpoint_path=final_path,
                phase=phase_spec,
                sampler=frozen_sampler,
                team_pool=team_pool,
                training_state_id=training_state_id,
                completed_updates=trainer.completed_updates,
                opponent_assignments=current_opponent_assignments,
                league=league,
            )
        with cuda_model_execution(device):
            trainer.save(final_path, training_state_id=training_state_id)
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
            "phase_state": (
                str(_phase_state_path_for_resume(final_path))
                if phase_spec is not None
                else None
            ),
        }
        if phase_summary is not None and frozen_sampler is not None:
            phase_summary["sampler_draw_count"] = frozen_sampler.draw_count
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
    finally:
        _close_training_resources(
            ("parallel_env", parallel_env),
            ("team_preview_batcher", team_preview_batcher),
            ("league", league),
        )


if __name__ == "__main__":
    train()
