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
import random
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from pokerl.config import ModelConfig, EnvConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.model.agent import PokeAgent
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
    VersionedTeamPreviewTrainingSample,
    filter_current_team_preview_training_samples,
    update_preview_then_battle,
)
from pokerl.training.recurrent_history import RecurrentObservationHistory
from pokerl.training.cuda_execution import cuda_model_execution


REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_MODULUS = 2**32


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
        help="Override how many frozen checkpoints are retained in the self-play league.",
    )
    parser.add_argument(
        "--team-preview-max-samples",
        type=int,
        default=None,
        help="Uniform current-rollout preview sample cap.",
    )
    parser.add_argument(
        "--tera-exploration-start",
        type=float,
        default=None,
        help="Initial probability of uniform-over-valid tera sampling during rollout.",
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
        help="PPO updates over which to linearly decay tera-exploration epsilon.",
    )
    parser.add_argument(
        "--tera-force-delay-start",
        type=float,
        default=None,
        help="Initial probability of forcing early sampled tera actions to no-tera during rollout.",
    )
    parser.add_argument(
        "--tera-force-delay-end",
        type=float,
        default=None,
        help="Final early-turn tera-delay probability after the decay window.",
    )
    parser.add_argument(
        "--tera-force-delay-decay-updates",
        type=int,
        default=None,
        help="PPO updates over which to linearly decay early-turn tera-delay exploration.",
    )
    parser.add_argument(
        "--tera-force-delay-max-turn",
        type=int,
        default=None,
        help="Apply forced tera delay only through this battle turn. Use 0 to disable.",
    )
    parser.add_argument(
        "--aux-hidden-info-coef",
        type=float,
        default=None,
        help="Coefficient on the opponent hidden-info prediction loss (item/ability/tera/moves).",
    )
    parser.add_argument(
        "--aux-opp-action-coef",
        type=float,
        default=None,
        help="Coefficient on the opponent-action prediction loss (a1/a2/tera).",
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
        model_config.pick_size = args.pick_size

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
        "team_preview_max_samples": args.team_preview_max_samples,
        "tera_exploration_start": args.tera_exploration_start,
        "tera_exploration_end": args.tera_exploration_end,
        "tera_exploration_decay_updates": args.tera_exploration_decay_updates,
        "tera_force_delay_start": args.tera_force_delay_start,
        "tera_force_delay_end": args.tera_force_delay_end,
        "tera_force_delay_decay_updates": args.tera_force_delay_decay_updates,
        "tera_force_delay_max_turn": args.tera_force_delay_max_turn,
        "aux_hidden_info_coef": args.aux_hidden_info_coef,
        "aux_opp_action_coef": args.aux_opp_action_coef,
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
    additional_updates: bool,
) -> tuple[int, int]:
    """Return completed and target update counts for this invocation."""
    if steps_per_update <= 0:
        raise ValueError("steps_per_update must be positive.")
    requested_updates = total_timesteps // steps_per_update
    updates_done = global_step // steps_per_update
    target_update = (
        updates_done + requested_updates if additional_updates else requested_updates
    )
    return updates_done, target_update


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


def _sample_league_checkpoint(league: League, *, action: str) -> str:
    checkpoint_path = league.sample_checkpoint_path()
    if checkpoint_path is None:
        raise RuntimeError(
            f"League opponent {action} requested, but the league is empty. "
            "This should not happen if the league was seeded before training."
        )
    return checkpoint_path


def _sample_checkpoints_by_group(
    league: League, opponent_groups: int
) -> dict[int, str]:
    return {
        opponent_group: _sample_league_checkpoint(league, action="assignment")
        for opponent_group in range(opponent_groups)
    }


def make_envs(
    env_config: EnvConfig,
    game_data: GameData,
    league: League,
    opponent_groups: int,
    team_preview_fn,
    team_preview_batcher: TeamPreviewBatcher | None = None,
    team_pair_provider=None,
    collect_timing: bool = False,
) -> ParallelBattleEnv:
    """Create parallel battle environments backed by league opponents."""
    checkpoint_by_group = _sample_checkpoints_by_group(league, opponent_groups)
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
) -> list[tuple[BattleEnv, str]]:
    assignments: list[tuple[BattleEnv, str]] = []
    checkpoint_by_group: dict[int, str] = {}
    for env in parallel_env.envs:
        opponent_group = int(env.config.opponent_group)
        checkpoint_path = checkpoint_by_group.get(opponent_group)
        if checkpoint_path is None:
            checkpoint_path = _sample_league_checkpoint(league, action="refresh")
            checkpoint_by_group[opponent_group] = checkpoint_path
        assignments.append((env, checkpoint_path))
    return assignments


def refresh_opponents(
    parallel_env: ParallelBattleEnv,
    league: League,
) -> dict[int, str]:
    """Swap out opponents in existing environments.

    Called periodically so the agent faces a freshly sampled set of past
    selves. Envs in the same opponent group share one sampled checkpoint, which
    keeps opponent inference batches large on local setups.
    """
    assignments = _sample_refresh_assignments(parallel_env, league)
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
        league.add_checkpoint(agent, global_step)

    assignments: dict[int, str] = {}
    if update < num_updates and update % config.league_refresh_interval == 0:
        assignments = refresh_opponents(parallel_env, league)
    return LeagueMaintenance(
        checkpoint_added=checkpoint_added,
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
    tera_exploration_prob: float,
    tera_force_delay_prob: float,
    tera_force_delay_max_turn: int,
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
        force_delay_prob=tera_force_delay_prob,
        force_delay_max_turn=tera_force_delay_max_turn,
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
        "exact_aux_label_episodes": 0.0,
    }
    rollout_infos: list[dict] = []
    obs_device_buffers: dict[str, torch.Tensor] | None = None
    mask_device_buffers: dict[str, torch.Tensor] | None = None
    rewards_device_buffer: torch.Tensor | None = None
    dones_device_buffer: torch.Tensor | None = None
    opp_actions_np_buffer = np.empty((parallel_env.num_envs, 3), dtype=np.int64)
    opp_actions_valid_np_buffer = np.empty(parallel_env.num_envs, dtype=bool)
    opp_actions_device_buffer: torch.Tensor | None = None
    opp_actions_valid_device_buffer: torch.Tensor | None = None
    aux_label_obs_device_buffers: dict[str, torch.Tensor] | None = None

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
                tera_force_delay_prob=tera_force_delay_prob,
                tera_force_delay_max_turn=tera_force_delay_max_turn,
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

        opp_actions_np_buffer.fill(-1)
        opp_actions_valid_np_buffer.fill(False)
        for env_index, info in enumerate(infos):
            opponent_actions = info.get("opp_actions")
            if opponent_actions is not None:
                opp_actions_np_buffer[env_index] = opponent_actions
                opp_actions_valid_np_buffer[env_index] = True

        opp_actions_t = array_to_device(
            opp_actions_np_buffer,
            device,
            dtype=torch.long,
            buffer=opp_actions_device_buffer,
        )
        opp_actions_valid_t = array_to_device(
            opp_actions_valid_np_buffer,
            device,
            dtype=torch.bool,
            buffer=opp_actions_valid_device_buffer,
        )
        opp_actions_device_buffer = opp_actions_t
        opp_actions_valid_device_buffer = opp_actions_valid_t

        done_env_indices = np.flatnonzero(dones)
        aux_label_obs_t = None
        if done_env_indices.size:
            label_observations: dict[int, dict[str, np.ndarray]] = {}
            for env_index in done_env_indices:
                info = infos[int(env_index)]
                final_observation = info.pop(
                    "final_observation",
                    None,
                )
                if final_observation is None:
                    raise RuntimeError(
                        "Auto-reset transition omitted final_observation for "
                        "terminal auxiliary labels."
                    )
                private_labels = info.pop("opponent_aux_labels", None)
                if private_labels is not None:
                    missing_keys = set(RolloutBuffer.AUX_OBS_KEYS).difference(
                        private_labels
                    )
                    if missing_keys:
                        raise RuntimeError(
                            "Trainer-private opponent labels omitted keys: "
                            + ", ".join(sorted(missing_keys))
                        )
                    label_observations[int(env_index)] = private_labels
                    profile["exact_aux_label_episodes"] += 1.0
                else:
                    label_observations[int(env_index)] = final_observation

            if all(key in next_obs for key in RolloutBuffer.AUX_OBS_KEYS):
                aux_label_obs_np = {
                    key: np.empty_like(next_obs[key])
                    for key in RolloutBuffer.AUX_OBS_KEYS
                }
                for env_index, label_observation in label_observations.items():
                    for key in RolloutBuffer.AUX_OBS_KEYS:
                        aux_label_obs_np[key][env_index] = label_observation[key]
                aux_label_obs_t = obs_to_device(
                    aux_label_obs_np,
                    device,
                    buffers=aux_label_obs_device_buffers,
                )
                aux_label_obs_device_buffers = aux_label_obs_t
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
            opp_actions=opp_actions_t,
            opp_actions_valid=opp_actions_valid_t,
            aux_label_obs=aux_label_obs_t,
            done_env_indices=done_env_indices,
        )
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


def _save_training_state(
    league: League,
    checkpoint_path: str,
    team_pool: TeamPool | None,
    training_state_id: str,
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
            "team_pool": team_pool_state,
        },
    )


def _json_ready_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


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

    print(f"Model parameters: {sum(p.numel() for p in agent.parameters()):,}")
    if team_pool is not None:
        print(
            f"Team pool: {team_pool.format_id} [{team_pool.subset}] with {team_pool.size} teams"
        )

    # Training
    trainer = PPOTrainer(agent, train_config, device)
    league = League(
        train_config,
        env_config,
        game_data,
        agent,
        device,
        rng=random.Random(seed_streams["league"]),
    )

    # --- Resume from checkpoint ---
    if args.resume:
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
        print(
            f"Restored league from {league_state!s}: {len(league.checkpoints)} checkpoint(s)"
        )

    # --- Seed league so self-play is available from update 1 ---
    # On a fresh start this ensures the agent never falls back to bots.
    if not league.checkpoints:
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
        policy_version=trainer.global_step // steps_per_update,
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
    )
    current_opponent_assignments = _current_opponent_assignments(parallel_env)
    run_assigned_opponents = set(current_opponent_assignments.values())
    run_observed_opponents: set[str] = set()
    run_opponent_outcomes: OpponentOutcomeCounts = {}
    run_summary = {
        "status": "starting",
        "config": {
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
            "team_preview_value_coef": train_config.team_preview_value_coef,
            "team_preview_bring_coef": train_config.team_preview_bring_coef,
            "team_preview_lead_coef": train_config.team_preview_lead_coef,
            "team_preview_max_samples": train_config.team_preview_max_samples,
            "tera_exploration_start": train_config.tera_exploration_start,
            "tera_exploration_end": train_config.tera_exploration_end,
            "tera_exploration_decay_updates": train_config.tera_exploration_decay_updates,
            "tera_force_delay_start": train_config.tera_force_delay_start,
            "tera_force_delay_end": train_config.tera_force_delay_end,
            "tera_force_delay_decay_updates": train_config.tera_force_delay_decay_updates,
            "tera_force_delay_max_turn": train_config.tera_force_delay_max_turn,
            "aux_hidden_info_coef": train_config.aux_hidden_info_coef,
            "aux_opp_action_coef": train_config.aux_opp_action_coef,
            "diagnose_memory_gradients": train_config.diagnose_memory_gradients,
            "league_size": train_config.league_size,
            "league_update_interval": train_config.league_update_interval,
            "league_refresh_interval": train_config.league_refresh_interval,
            "league_coalesce_ms": train_config.league_coalesce_ms,
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
            additional_updates=args.resume is not None and args.num_updates is not None,
        )

        print(f"Total updates: {num_updates}")
        print(f"Steps per update: {steps_per_update}")
        print("Backend: local BattleStream")
        print(f"Opponent groups: {opponent_groups}")
        print(
            "League cadence: "
            f"snapshot every {train_config.league_update_interval} update(s), "
            f"refresh every {train_config.league_refresh_interval} update(s)"
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

            # Linear decay of tera exploration. A zero-length decay window means
            # "skip decay, apply end immediately". Negative values are rejected
            # by TrainingConfig validation.
            decay_updates = train_config.tera_exploration_decay_updates
            if decay_updates == 0:
                tera_eps = train_config.tera_exploration_end
            else:
                progress = min(1.0, max(0.0, (update - 1) / decay_updates))
                tera_eps = (
                    train_config.tera_exploration_start
                    + (
                        train_config.tera_exploration_end
                        - train_config.tera_exploration_start
                    )
                    * progress
                )
            delay_decay_updates = train_config.tera_force_delay_decay_updates
            if delay_decay_updates == 0:
                tera_force_delay_prob = train_config.tera_force_delay_end
            else:
                delay_progress = min(1.0, max(0.0, (update - 1) / delay_decay_updates))
                tera_force_delay_prob = (
                    train_config.tera_force_delay_start
                    + (
                        train_config.tera_force_delay_end
                        - train_config.tera_force_delay_start
                    )
                    * delay_progress
                )
            update_episode_rewards: list[float] = []
            update_episode_secs: list[float] = []
            update_episode_turns: list[float] = []
            update_episode_opponent_choice_calls: list[int] = []
            update_episode_opponent_repeated_turn_requests: list[int] = []
            update_episode_opponent_max_calls_same_turn: list[int] = []
            update_pathological_episodes: list[dict[str, object]] = []
            versioned_preview_data: list[VersionedTeamPreviewTrainingSample] = []
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
                    tera_exploration_prob=tera_eps,
                    tera_force_delay_prob=tera_force_delay_prob,
                    tera_force_delay_max_turn=(train_config.tera_force_delay_max_turn),
                    collect_timing=collect_timing,
                )
            )

            # Track episode rewards and team preview decisions
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
                    if "team_preview_obs" in info:
                        if "opponent_team_preview_picks" not in info:
                            raise RuntimeError(
                                "Episode recorded a team preview decision without opponent preview labels."
                            )
                        preview_policy_version = info.get("team_preview_policy_version")
                        if type(preview_policy_version) is not int:
                            raise RuntimeError(
                                "Training episode omitted its integer team-preview "
                                "policy version."
                            )
                        versioned_preview_data.append(
                            VersionedTeamPreviewTrainingSample(
                                sample=(
                                    info["team_preview_obs"],
                                    info["team_preview_picks"],
                                    reward,
                                    info["opponent_team_preview_picks"],
                                ),
                                policy_version=preview_policy_version,
                            )
                        )
                if "episode_sec" in info:
                    update_episode_secs.append(info["episode_sec"])
                if "battle_turn" in info and info.get("episode_reward") is not None:
                    update_episode_turns.append(info["battle_turn"])
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

            try:
                preview_training_data = filter_current_team_preview_training_samples(
                    versioned_preview_data,
                    rollout_policy_version,
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Invalid team-preview policy version in rollout data."
                ) from exc
            update_preview_data = list(preview_training_data.samples)
            stale_preview_samples = preview_training_data.stale_excluded_count

            rollout_sec = time.perf_counter() - rollout_started
            if collect_timing:
                rollout_profile["rollout_sec"] = rollout_sec
            env_profile = parallel_env.snapshot_timing() if collect_timing else {}
            parallel_env.wait_for_prepared_team_previews()

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
            advantage_sec = time.perf_counter() - advantage_started

            # --- PPO update ---
            ppo_started = time.perf_counter()
            with cuda_model_execution(device):
                agent.train()
                metrics = update_preview_then_battle(
                    trainer,
                    batch,
                    update_preview_data,
                    train_config.team_preview_max_samples,
                    seed=train_config.seed,
                    update=update,
                )
                metrics["tera_exploration_prob"] = tera_eps
                metrics["tera_force_delay_prob"] = tera_force_delay_prob
                metrics["team_preview_samples_stale_excluded"] = float(
                    stale_preview_samples
                )
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

            # The updated policy enters the league before opponent groups are
            # resampled, making a newly created snapshot eligible immediately.
            league_maintenance_started = time.perf_counter()
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

            # Save league state after adding the current snapshot so a trainer
            # checkpoint and its paired .league_state.json describe one point
            # in training.
            checkpoint_sec = 0.0
            if update % train_config.save_interval == 0:
                checkpoint_started = time.perf_counter()
                path = f"{train_config.checkpoint_dir}/step_{trainer.global_step}.pt"
                training_state_id = uuid.uuid4().hex
                _save_training_state(
                    league,
                    path,
                    team_pool,
                    training_state_id,
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
            exact_aux_label_episodes = int(rollout_profile["exact_aux_label_episodes"])
            update_summary = {
                "update": update,
                "global_step": trainer.global_step,
                "completed_battles": len(update_episode_rewards),
                "exact_aux_label_episodes": exact_aux_label_episodes,
                "exact_aux_label_episode_coverage": _safe_div(
                    exact_aux_label_episodes,
                    len(update_episode_rewards),
                ),
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
                    f"Base tera H(choice) {metrics['base_tera_entropy']:.3f} | "
                    f"Pre-PPO KL {metrics['pre_ppo_approx_kl']:.5f}/"
                    f"clip {metrics['pre_ppo_clip_fraction']:.1%} | "
                    f"Preview loss {metrics.get('team_preview_loss', float('nan')):.4f} | "
                    f"Preview value {metrics.get('team_preview_value_loss', float('nan')):.4f} | "
                    f"League pool {len(league.checkpoints)}"
                    f"{diagnostics_text}"
                )

        print("Training complete.")
        final_path = f"{train_config.checkpoint_dir}/final.pt"
        training_state_id = uuid.uuid4().hex
        _save_training_state(
            league,
            final_path,
            team_pool,
            training_state_id,
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
    finally:
        _close_training_resources(
            ("parallel_env", parallel_env),
            ("team_preview_batcher", team_preview_batcher),
            ("league", league),
        )


if __name__ == "__main__":
    train()
