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
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from pokerl.config import ModelConfig, EnvConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.model.agent import PokeAgent
from pokerl.env.showdown_env import (
    BattleEnv,
    ParallelBattleEnv,
    battlestream_summary,
    default_showdown_dir,
)
from pokerl.team_pool import TeamPool, default_team_pool_manifest_path, load_team_pool
from pokerl.training.rollout import RolloutBuffer
from pokerl.training.ppo import PPOTrainer
from pokerl.training.league import League
from pokerl.training.preview import TeamPreviewBatcher


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pokeRL with PPO.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument("--battle-format", default=None, help="Override the Showdown battle format.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the number of parallel battle environments.")
    parser.add_argument("--team-size", type=int, default=None, help="Override the full team size exposed in observations.")
    parser.add_argument("--pick-size", type=int, default=None, help="Override how many Pokemon are selected at team preview.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override rollout steps collected per update.")
    parser.add_argument("--num-updates", type=int, default=None, help="Override how many rollout/PPO updates to run.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override the total environment steps to train for.")
    parser.add_argument(
        "--collect-timing",
        action="store_true",
        help="Collect detailed rollout/env micro-profiles even without --summary-path.",
    )
    parser.add_argument("--num-epochs", type=int, default=None, help="Override PPO epochs per update.")
    parser.add_argument("--num-minibatches", type=int, default=None, help="Override PPO minibatches per update.")
    parser.add_argument(
        "--bptt-chunk-steps",
        type=int,
        default=None,
        help="Detach recurrent state every N PPO timesteps. Use 0 to disable chunking.",
    )
    parser.add_argument("--log-interval", type=int, default=None, help="Override logging frequency in updates.")
    parser.add_argument("--save-interval", type=int, default=None, help="Override checkpoint save frequency in updates.")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint output directory.")
    parser.add_argument(
        "--showdown-dir",
        default=None,
        help="Pokemon Showdown checkout used by local BattleStream workers.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Local BattleStream worker process count. Defaults from local worker mode.",
    )
    parser.add_argument(
        "--battles-per-worker",
        type=int,
        default=None,
        help="Maximum concurrent local battles per worker. Defaults from local worker mode.",
    )
    parser.add_argument(
        "--pool-mode",
        choices=("single", "shared"),
        default=None,
        help="Local BattleStream worker mode. Defaults to bounded shared.",
    )
    parser.add_argument(
        "--runtime-loop-count",
        type=int,
        default=None,
        help="Number of runtime loops used by local BattleStream workers. Defaults from local worker mode.",
    )
    parser.add_argument(
        "--battlestream-startup-timeout",
        type=float,
        default=None,
        help="Seconds to wait for a local BattleStream battle to produce its first protocol output.",
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
        help="Seed for fixed-team sampling. Useful for reproducible benchmarks.",
    )
    parser.add_argument(
        "--league-update-interval",
        type=int,
        default=None,
        help="Override league checkpoint and opponent refresh frequency in updates.",
    )
    parser.add_argument(
        "--league-size",
        type=int,
        default=None,
        help="Override how many frozen checkpoints are retained in the self-play league.",
    )
    parser.add_argument(
        "--team-preview-update-interval",
        type=int,
        default=None,
        help="Flush the preview accumulator and run a gradient step every N PPO updates.",
    )
    parser.add_argument(
        "--team-preview-max-samples",
        type=int,
        default=None,
        help="Cap on preview accumulator size; oldest samples are dropped first.",
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


def build_configs(args: argparse.Namespace) -> tuple[ModelConfig, EnvConfig, TrainingConfig]:
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
    if args.pool_mode is not None:
        env_config.pool_mode = args.pool_mode
    if args.runtime_loop_count is not None:
        env_config.runtime_loop_count = args.runtime_loop_count
    if args.battlestream_startup_timeout is not None:
        env_config.battlestream_startup_timeout = args.battlestream_startup_timeout

    overrides = {
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
        "league_coalesce_ms": args.league_coalesce_ms,
        "team_preview_update_interval": args.team_preview_update_interval,
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
    }
    for field_name, value in overrides.items():
        if value is not None:
            if field_name == "bptt_chunk_steps" and value == 0:
                value = None
            setattr(train_config, field_name, value)

    if args.num_updates is not None:
        train_config.total_timesteps = args.num_updates * train_config.rollout_steps * env_config.num_envs

    train_config.__post_init__()
    return model_config, env_config, train_config


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
    def team_preview_fn(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
        picks = batcher.choose(obs)
        team_str = "/team " + "".join(str(int(p) + 1) for p in picks)
        return picks, team_str
    return team_preview_fn


def load_requested_team_pool(args: argparse.Namespace) -> TeamPool | None:
    manifest_path: Path | None = None

    if args.team_pool_manifest:
        manifest_path = Path(args.team_pool_manifest)
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
    elif args.team_pool_format:
        manifest_path = default_team_pool_manifest_path(REPO_ROOT, args.team_pool_format)

    if manifest_path is None:
        return None

    return load_team_pool(
        manifest_path,
        subset=args.team_pool_subset,
        seed=args.team_pool_seed,
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


def _sample_checkpoints_by_group(league: League, opponent_groups: int) -> dict[int, str]:
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
    envs = []
    checkpoint_by_group = _sample_checkpoints_by_group(league, opponent_groups)
    for i in range(env_config.num_envs):
        env_instance_config = replace(env_config, opponent_group=i % opponent_groups)
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
            collect_timing=collect_timing,
        )
        envs.append(env)
    return ParallelBattleEnv(envs, collect_timing=collect_timing)


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


def refresh_opponents(parallel_env: ParallelBattleEnv, league: League) -> None:
    """Swap out opponents in existing environments.

    Called periodically so the agent faces a freshly sampled set of past
    selves. Envs in the same opponent group share one sampled checkpoint, which
    keeps opponent inference batches large on local setups.
    """
    assignments = _sample_refresh_assignments(parallel_env, league)
    for env, checkpoint_path in assignments:
        if env.stage_next_opponent_checkpoint(checkpoint_path):
            continue
        env.set_next_opponent(
            make_league_opponent_factory(league, checkpoint_path=checkpoint_path)
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


def _league_state_path(train_config: TrainingConfig) -> str:
    return f"{train_config.checkpoint_dir}/league_state.json"


def _league_state_path_for_resume(resume_checkpoint: str) -> Path:
    return Path(resume_checkpoint).resolve().with_name("league_state.json")


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
    opponent_choice_sec = float(env_profile.get("opponent_choice_sec", 0.0))
    opponent_choice_calls = float(env_profile.get("opponent_choice_calls", 0.0))
    rollout_sec = float(rollout_profile.get("rollout_sec", 0.0))
    device_transfer_sec = float(rollout_profile.get("obs_to_device_sec", 0.0)) + float(
        rollout_profile.get("mask_to_device_sec", 0.0)
    ) + float(rollout_profile.get("feedback_to_device_sec", 0.0))

    return {
        "env_queue_wait_share": _safe_div(env_queue_wait_sec, env_step_total_sec),
        "avg_env_step_ms": _safe_div(env_step_total_sec, env_step_calls) * 1000.0,
        "avg_parallel_step_wall_ms": _safe_div(parallel_step_wall_sec, parallel_step_calls) * 1000.0,
        "avg_opponent_choice_ms": _safe_div(opponent_choice_sec, opponent_choice_calls) * 1000.0,
        "policy_forward_share": _safe_div(float(rollout_profile.get("policy_forward_sec", 0.0)), rollout_sec),
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
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    run_started = time.perf_counter()
    collect_timing = args.collect_timing

    model_config, env_config, train_config = build_configs(args)
    print(f"Device: {device}")
    if env_config.showdown_dir is None:
        env_config.showdown_dir = default_showdown_dir()
    if env_config.showdown_dir is None:
        raise SystemExit(
            "--showdown-dir is required. Set it to a Pokemon Showdown checkout, "
            "or set POKERL_SHOWDOWN_DIR."
        )
    if not Path(env_config.showdown_dir).exists():
        raise SystemExit(f"Pokemon Showdown checkout not found: {env_config.showdown_dir}")
    opponent_groups = resolve_opponent_groups(args.opponent_groups, env_config.num_envs)
    team_pool = load_requested_team_pool(args)

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
    league = League(train_config, env_config, game_data, agent, device)

    # --- Resume from checkpoint ---
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed trainer from {args.resume!r} (global step {trainer.global_step:,})")

        league_state = _league_state_path_for_resume(args.resume)
        if league_state.exists():
            league.load_state(league_state)
            print(f"Restored league from {league_state!s}: {len(league.checkpoints)} checkpoint(s)")
        else:
            print("Warning: no league state file found alongside resumed checkpoint — league starts empty.")

    # --- Seed league so self-play is available from update 1 ---
    # If resuming, this only runs when the league file was missing (edge case).
    # On a fresh start, this ensures the agent never falls back to bots from the beginning.
    if not league.checkpoints:
        league.add_checkpoint(agent, step=trainer.global_step)
        print(f"Seeded league with agent at step {trainer.global_step}")

    # --- Environments ---
    team_preview_batcher = TeamPreviewBatcher(agent, device)
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
    run_summary = {
        "status": "starting",
        "config": {
            "device": str(device),
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
            "team_preview_update_interval": train_config.team_preview_update_interval,
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
            "league_coalesce_ms": train_config.league_coalesce_ms,
            "collect_timing": collect_timing,
            "total_timesteps": train_config.total_timesteps,
            "backend": "local_battlestream",
            "opponent_groups": opponent_groups,
            "battlestream": battlestream_summary(env_config),
            "team_pool": team_pool.describe() if team_pool is not None else None,
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
        buffer = RolloutBuffer(
            train_config.rollout_steps,
            env_config.num_envs,
            obs,
            masks,
            device,
        )

        hidden = agent.initial_hidden(env_config.num_envs)
        obs_device_buffers: dict[str, torch.Tensor] | None = None
        mask_device_buffers: dict[str, torch.Tensor] | None = None
        rewards_device_buffer: torch.Tensor | None = None
        dones_device_buffer: torch.Tensor | None = None
        opp_actions_np_buffer = np.empty((env_config.num_envs, 3), dtype=np.int64)
        opp_actions_valid_np_buffer = np.empty(env_config.num_envs, dtype=bool)
        opp_actions_device_buffer: torch.Tensor | None = None
        opp_actions_valid_device_buffer: torch.Tensor | None = None
        aux_label_obs_device_buffers: dict[str, torch.Tensor] | None = None
        num_updates = train_config.total_timesteps // steps_per_update
        updates_done = trainer.global_step // steps_per_update

        print(f"Total updates: {num_updates}")
        print(f"Steps per update: {steps_per_update}")
        print("Backend: local BattleStream")
        print(f"Opponent groups: {opponent_groups}")
        local_summary = battlestream_summary(env_config)
        print(
            "Local BattleStream: "
            f"workers={local_summary['worker_count']} "
            f"battles/worker={local_summary['max_battles_per_worker']} "
            f"pool={local_summary['pool_mode']} "
            f"runtime_loops={local_summary['runtime_loop_count']}"
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

        # Preview samples are tiny per update (~20-50 completed episodes), so batch
        # across multiple updates before running a gradient step. Oldest samples are
        # dropped first if the accumulator overflows.
        accumulated_preview_data: list[tuple[dict, np.ndarray, float, np.ndarray]] = []

        for update in range(updates_done + 1, num_updates + 1):
            update_wall_started = time.perf_counter()
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
                    + (train_config.tera_exploration_end - train_config.tera_exploration_start)
                    * progress
                )
            delay_decay_updates = train_config.tera_force_delay_decay_updates
            if delay_decay_updates == 0:
                tera_force_delay_prob = train_config.tera_force_delay_end
            else:
                delay_progress = min(1.0, max(0.0, (update - 1) / delay_decay_updates))
                tera_force_delay_prob = (
                    train_config.tera_force_delay_start
                    + (train_config.tera_force_delay_end - train_config.tera_force_delay_start)
                    * delay_progress
                )
            update_episode_rewards: list[float] = []
            update_episode_secs: list[float] = []
            update_episode_turns: list[float] = []
            update_episode_opponent_choice_calls: list[int] = []
            update_episode_opponent_repeated_turn_requests: list[int] = []
            update_episode_opponent_max_calls_same_turn: list[int] = []
            update_pathological_episodes: list[dict[str, object]] = []
            update_preview_data: list[tuple[dict, np.ndarray, float, np.ndarray]] = []
            rollout_profile = {}
            if collect_timing:
                rollout_profile = {
                    "obs_to_device_sec": 0.0,
                    "mask_to_device_sec": 0.0,
                    "policy_forward_sec": 0.0,
                    "policy_postprocess_sec": 0.0,
                    "env_step_sec": 0.0,
                    "feedback_to_device_sec": 0.0,
                    "buffer_add_sec": 0.0,
                    "hidden_reset_sec": 0.0,
                }

            # --- Collect rollout ---
            for step in range(train_config.rollout_steps):
                with torch.no_grad():
                    if collect_timing:
                        obs_device_started = time.perf_counter()
                    obs_t = obs_to_device(obs, device, buffers=obs_device_buffers)
                    obs_device_buffers = obs_t
                    if collect_timing:
                        rollout_profile["obs_to_device_sec"] += time.perf_counter() - obs_device_started

                    if collect_timing:
                        mask_device_started = time.perf_counter()
                    masks_t = action_masks_to_device(masks, device, buffers=mask_device_buffers)
                    mask_device_buffers = masks_t
                    if collect_timing:
                        rollout_profile["mask_to_device_sec"] += time.perf_counter() - mask_device_started

                    if collect_timing:
                        policy_started = time.perf_counter()
                    out = agent(
                        obs_t,
                        hidden,
                        action_masks=masks_t,
                        tera_exploration_prob=tera_eps,
                        tera_force_delay_prob=tera_force_delay_prob,
                        tera_force_delay_max_turn=train_config.tera_force_delay_max_turn,
                    )
                    if collect_timing:
                        rollout_profile["policy_forward_sec"] += time.perf_counter() - policy_started

                    if collect_timing:
                        policy_post_started = time.perf_counter()
                    actions_t = out["actions"]
                    log_probs_t = out["log_prob"]
                    values_t = out["value"]
                    actions = actions_t.cpu().numpy()
                    hidden = out["hidden"]
                    if collect_timing:
                        rollout_profile["policy_postprocess_sec"] += time.perf_counter() - policy_post_started
                    if device.type == "cpu":
                        obs_for_buffer = {
                            key: value.detach().clone()
                            for key, value in obs_t.items()
                        }
                        masks_for_buffer = {
                            key: value.detach().clone()
                            for key, value in masks_t.items()
                        }
                    else:
                        obs_for_buffer = obs_t
                        masks_for_buffer = masks_t

                if collect_timing:
                    env_step_started = time.perf_counter()
                next_obs, next_masks, rewards, dones, infos = parallel_env.step(actions)
                if collect_timing:
                    rollout_profile["env_step_sec"] += time.perf_counter() - env_step_started

                if collect_timing:
                    feedback_device_started = time.perf_counter()
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
                if collect_timing:
                    rollout_profile["feedback_to_device_sec"] += time.perf_counter() - feedback_device_started

                # Stack opponent actions for the aux head. Envs that didn't expose
                # them (non-league opponents) leave opp_actions_valid=False and the
                # PPO update masks those samples.
                opp_actions_np_buffer.fill(-1)
                opp_actions_valid_np_buffer.fill(False)
                for idx, info in enumerate(infos):
                    if "opp_actions" in info:
                        opp_actions_np_buffer[idx] = info["opp_actions"]
                        opp_actions_valid_np_buffer[idx] = True
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

                aux_label_obs_t = None
                done_env_indices = np.flatnonzero(dones)
                if done_env_indices.size and all(key in next_obs for key in RolloutBuffer.AUX_LABEL_KEYS):
                    aux_label_obs_t = obs_to_device(
                        {key: next_obs[key] for key in RolloutBuffer.AUX_LABEL_KEYS},
                        device,
                        buffers=aux_label_obs_device_buffers,
                    )
                    aux_label_obs_device_buffers = aux_label_obs_t

                if collect_timing:
                    buffer_started = time.perf_counter()
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
                    rollout_profile["buffer_add_sec"] += time.perf_counter() - buffer_started

                # Reset hidden state for done environments
                if hidden is not None:
                    if collect_timing:
                        hidden_reset_started = time.perf_counter()
                    h, c = hidden
                    not_done = (~dones_t).to(h.dtype).view(1, dones_t.shape[0], 1)
                    hidden = (h * not_done, c * not_done)
                    if collect_timing:
                        rollout_profile["hidden_reset_sec"] += time.perf_counter() - hidden_reset_started

                # Track episode rewards and team preview decisions
                for info in infos:
                    if "episode_reward" in info:
                        reward = info["episode_reward"]
                        episode_rewards.append(reward)
                        update_episode_rewards.append(reward)
                        if "team_preview_obs" in info:
                            if "opponent_team_preview_picks" not in info:
                                raise RuntimeError(
                                    "Episode recorded a team preview decision without opponent preview labels."
                                )
                            update_preview_data.append(
                                (
                                    info["team_preview_obs"],
                                    info["team_preview_picks"],
                                    reward,
                                    info["opponent_team_preview_picks"],
                                )
                            )
                    if "episode_sec" in info:
                        update_episode_secs.append(info["episode_sec"])
                    if "battle_turn" in info and info.get("episode_reward") is not None:
                        update_episode_turns.append(info["battle_turn"])
                    if "episode_opponent_choice_calls" in info:
                        update_episode_opponent_choice_calls.append(int(info["episode_opponent_choice_calls"]))
                    if "episode_opponent_repeated_turn_requests" in info:
                        repeated_turn_requests = int(info["episode_opponent_repeated_turn_requests"])
                        update_episode_opponent_repeated_turn_requests.append(repeated_turn_requests)
                        update_episode_opponent_max_calls_same_turn.append(
                            int(info.get("episode_opponent_max_calls_same_turn", 0))
                        )
                        if repeated_turn_requests > 0:
                            update_pathological_episodes.append(
                                {
                                    "battle_tag": info.get("battle_tag", ""),
                                    "battle_turn": int(info.get("battle_turn", 0)),
                                    "episode_sec": float(info.get("episode_sec", 0.0)),
                                    "opponent_choice_calls": int(info.get("episode_opponent_choice_calls", 0)),
                                    "opponent_repeated_turn_requests": repeated_turn_requests,
                                    "opponent_max_calls_same_turn": int(
                                        info.get("episode_opponent_max_calls_same_turn", 0)
                                    ),
                                    "opponent_checkpoint_path": info.get("episode_opponent_checkpoint_path"),
                                }
                            )

                obs, masks = next_obs, next_masks

            rollout_sec = time.perf_counter() - rollout_started
            if collect_timing:
                rollout_profile["rollout_sec"] = rollout_sec
            env_profile = parallel_env.snapshot_timing() if collect_timing else {}
            parallel_env.wait_for_prepared_team_previews()
            refresh_after_update = (
                update % train_config.league_update_interval == 0 and update < num_updates
            )
            if refresh_after_update:
                refresh_opponents(parallel_env, league)

            # --- Compute advantages ---
            advantage_started = time.perf_counter()
            with torch.no_grad():
                obs_t = obs_to_device(obs, device)
                last_value = agent.get_value(obs_t, hidden)

            batch = buffer.compute_gae(last_value, train_config.gamma, train_config.gae_lambda)
            advantage_sec = time.perf_counter() - advantage_started

            # --- PPO update ---
            agent.train()
            ppo_started = time.perf_counter()
            metrics = trainer.update(batch)
            metrics["tera_exploration_prob"] = tera_eps

            # Extend the preview accumulator and flush on the configured interval.
            # FIFO drop keeps the buffer bounded without silently losing the freshest data.
            if update_preview_data:
                accumulated_preview_data.extend(update_preview_data)
                overflow = len(accumulated_preview_data) - train_config.team_preview_max_samples
                if overflow > 0:
                    del accumulated_preview_data[:overflow]
            preview_interval = max(1, train_config.team_preview_update_interval)
            if accumulated_preview_data and update % preview_interval == 0:
                metrics.update(trainer.update_team_preview(accumulated_preview_data))
                metrics["team_preview_batch_size"] = float(len(accumulated_preview_data))
                # Clear unconditionally: non-finite preview losses are usually caused by
                # something specific to this batch, so retaining the samples would risk
                # re-triggering on the next flush.
                accumulated_preview_data = []
            ppo_sec = time.perf_counter() - ppo_started
            preview_refresh_started = time.perf_counter()
            parallel_env.refresh_prepared_team_previews()
            preview_refresh_sec = time.perf_counter() - preview_refresh_started

            avg_episode_sec = float(np.mean(update_episode_secs)) if update_episode_secs else 0.0
            avg_episode_turns = float(np.mean(update_episode_turns)) if update_episode_turns else 0.0
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
                steps_per_update / (rollout_sec + ppo_sec) if (rollout_sec + ppo_sec) > 0 else 0.0
            )
            update_accounted_sec = rollout_sec + advantage_sec + ppo_sec + preview_refresh_sec
            update_summary = {
                "update": update,
                "global_step": trainer.global_step,
                "completed_battles": len(update_episode_rewards),
                "rollout_sec": rollout_sec,
                "advantage_sec": advantage_sec,
                "ppo_sec": ppo_sec,
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
                "metrics": metrics_json,
            }
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
            run_summary["updates"].append(update_summary)
            run_summary["status"] = "running"
            run_summary["update_count"] = update
            run_summary["global_step"] = trainer.global_step
            _maybe_write_run_summary(args.summary_path, run_summary)

            # --- Logging ---
            if update % train_config.log_interval == 0:
                elapsed = time.time() - start_time
                sps = trainer.global_step / elapsed
                win_rate = np.mean([r > 0 for r in episode_rewards[-100:]]) if episode_rewards else 0
                diagnostics = update_summary.get("rollout_diagnostics", {})
                diagnostics_text = ""
                if diagnostics:
                    diagnostics_text = (
                        f" | Env wait {diagnostics['env_queue_wait_share']:.0%}"
                        f" | Vec step {diagnostics['avg_parallel_step_wall_ms']:.0f}ms"
                        f" | Opp choice {diagnostics['avg_opponent_choice_ms']:.1f}ms"
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
                    f"Win rate (100) {win_rate:.2%} | "
                    f"Policy loss {metrics['policy_loss']:.4f} | "
                    f"Value loss {metrics['value_loss']:.4f} | "
                    f"Entropy {metrics['entropy']:.4f} | "
                    f"Preview loss {metrics.get('team_preview_loss', float('nan')):.4f} | "
                    f"Preview value {metrics.get('team_preview_value_loss', float('nan')):.4f} | "
                    f"League pool {len(league.checkpoints)}"
                    f"{diagnostics_text}"
                )

            # --- Checkpointing ---
            checkpoint_sec = 0.0
            if update % train_config.save_interval == 0:
                checkpoint_started = time.perf_counter()
                path = f"{train_config.checkpoint_dir}/step_{trainer.global_step}.pt"
                trainer.save(path)
                league.save_state(_league_state_path(train_config))
                checkpoint_sec = time.perf_counter() - checkpoint_started
                print(f"Saved checkpoint: {path}")

            # --- League: add checkpoint and refresh opponents ---
            league_update_sec = 0.0
            if update % train_config.league_update_interval == 0:
                league_update_started = time.perf_counter()
                league.add_checkpoint(agent, trainer.global_step)
                league_update_sec = time.perf_counter() - league_update_started
                if refresh_after_update:
                    print(f"League updated (pool: {len(league.checkpoints)}), opponents refreshed")
                else:
                    print(f"League updated (pool: {len(league.checkpoints)})")

            update_wall_sec = time.perf_counter() - update_wall_started
            update_summary["checkpoint_sec"] = checkpoint_sec
            update_summary["league_update_sec"] = league_update_sec
            update_summary["update_wall_sec"] = update_wall_sec
            update_summary["update_wall_steps_per_second"] = (
                steps_per_update / update_wall_sec if update_wall_sec > 0 else 0.0
            )
            update_summary["update_overhead_sec"] = max(
                0.0,
                update_wall_sec
                - update_accounted_sec
                - checkpoint_sec
                - league_update_sec,
            )
            _maybe_write_run_summary(args.summary_path, run_summary)

        # Flush any leftover preview samples before saving the final checkpoint.
        # num_updates is rarely a multiple of team_preview_update_interval, so up to
        # one interval's worth of samples would otherwise be collected but never used.
        if accumulated_preview_data:
            trainer.update_team_preview(accumulated_preview_data)
            accumulated_preview_data = []

        print("Training complete.")
        final_path = f"{train_config.checkpoint_dir}/final.pt"
        trainer.save(final_path)
        league.save_state(_league_state_path(train_config))
        print(f"Saved final checkpoint: {final_path}")
        total_wall_sec = time.perf_counter() - run_started
        run_summary["final"] = {
            "global_step": trainer.global_step,
            "wall_time_sec": total_wall_sec,
            "steps_per_second": trainer.global_step / total_wall_sec if total_wall_sec > 0 else 0.0,
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
        parallel_env.close()
        team_preview_batcher.close()
        league.close()


if __name__ == "__main__":
    train()
