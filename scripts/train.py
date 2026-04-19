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
import socket
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
    parser.add_argument("--num-epochs", type=int, default=None, help="Override PPO epochs per update.")
    parser.add_argument("--num-minibatches", type=int, default=None, help="Override PPO minibatches per update.")
    parser.add_argument("--log-interval", type=int, default=None, help="Override logging frequency in updates.")
    parser.add_argument("--save-interval", type=int, default=None, help="Override checkpoint save frequency in updates.")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint output directory.")
    parser.add_argument(
        "--server-urls",
        default=None,
        help="Comma-separated server URLs to shard envs across, for example localhost:8000,localhost:8001.",
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

    overrides = {
        "rollout_steps": args.rollout_steps,
        "total_timesteps": args.total_timesteps,
        "num_epochs": args.num_epochs,
        "num_minibatches": args.num_minibatches,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "checkpoint_dir": args.checkpoint_dir,
        "league_size": args.league_size,
        "league_update_interval": args.league_update_interval,
        "league_coalesce_ms": args.league_coalesce_ms,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(train_config, field_name, value)

    if args.num_updates is not None:
        train_config.total_timesteps = args.num_updates * train_config.rollout_steps * env_config.num_envs

    return model_config, env_config, train_config


def make_league_opponent_factory(
    env_config: EnvConfig,
    league: League,
    checkpoint_path: str | None = None,
):
    """Create a league opponent factory bound to a specific env's server URL."""
    server_url = env_config.server_url

    def league_factory(loop: asyncio.AbstractEventLoop):
        opponent = league.make_opponent(
            loop,
            server_url=server_url,
            checkpoint_path=checkpoint_path,
        )
        if opponent is None:
            raise RuntimeError(
                "League opponent requested, but the league is empty. "
                "This should not happen if the league was seeded before training."
            )
        return opponent

    return league_factory


def parse_server_urls(raw_value: str | None, default_url: str) -> list[str]:
    if not raw_value:
        return [default_url]
    server_urls = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not server_urls:
        raise ValueError("--server-urls did not contain any valid host:port entries.")
    return server_urls


def _port_open(server_url: str) -> bool:
    host, port_raw = server_url.split(":", maxsplit=1)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, int(port_raw))) == 0


def ensure_server_urls_ready(server_urls: list[str]) -> None:
    missing = [url for url in server_urls if not _port_open(url)]
    if missing:
        missing_ports = ", ".join(missing)
        raise SystemExit(
            f"The following Showdown servers are not reachable: {missing_ports}. "
            "Start them first with scripts/start_showdown_server.ps1 or choose active ports."
        )


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


def make_envs(
    env_config: EnvConfig,
    game_data: GameData,
    league: League,
    server_urls: list[str],
    team_preview_fn,
    team_preview_batcher: TeamPreviewBatcher | None = None,
    team_pair_provider=None,
) -> ParallelBattleEnv:
    """Create parallel battle environments backed by league opponents."""
    envs = []
    for i in range(env_config.num_envs):
        env_server_url = server_urls[i % len(server_urls)]
        env_instance_config = replace(env_config, server_url=env_server_url)
        opponent_factory = make_league_opponent_factory(env_instance_config, league)
        env = BattleEnv(
            env_instance_config,
            game_data,
            opponent_factory,
            team_preview_fn=team_preview_fn,
            team_preview_batcher=team_preview_batcher,
            team_pair_provider=team_pair_provider,
        )
        envs.append(env)
    return ParallelBattleEnv(envs)


def _sample_refresh_assignments(
    parallel_env: ParallelBattleEnv,
    league: League,
) -> list[tuple[BattleEnv, str]]:
    assignments: list[tuple[BattleEnv, str]] = []
    checkpoint_by_server: dict[str, str] = {}
    for env in parallel_env.envs:
        server_url = env.config.server_url
        checkpoint_path = checkpoint_by_server.get(server_url)
        if checkpoint_path is None:
            checkpoint_path = league.sample_checkpoint_path()
            if checkpoint_path is None:
                raise RuntimeError(
                    "League opponent refresh requested, but the league is empty. "
                    "This should not happen if the league was seeded before training."
                )
            checkpoint_by_server[server_url] = checkpoint_path
        assignments.append((env, checkpoint_path))
    return assignments


def refresh_opponents(parallel_env: ParallelBattleEnv, league: League) -> None:
    """Swap out opponents in existing environments.

    Called periodically so the agent faces a freshly sampled set of past
    selves. All envs on the same Showdown server share one sampled checkpoint,
    which keeps opponent inference batches large on small local setups.
    """
    assignments = _sample_refresh_assignments(parallel_env, league)
    for env, checkpoint_path in assignments:
        if env.stage_next_opponent_checkpoint(checkpoint_path):
            continue
        env.set_next_opponent(
            make_league_opponent_factory(env.config, league, checkpoint_path=checkpoint_path)
        )


def obs_to_device(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in obs.items()}


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


def _write_run_summary(path: str, summary: dict) -> None:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def train():
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    run_started = time.perf_counter()
    collect_timing = args.summary_path is not None

    print(f"Device: {device}")

    model_config, env_config, train_config = build_configs(args)
    server_urls = parse_server_urls(args.server_urls, env_config.server_url)
    ensure_server_urls_ready(server_urls)
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
        server_urls,
        team_preview_fn=team_preview_fn,
        team_preview_batcher=team_preview_batcher,
        team_pair_provider=team_pair_provider,
    )
    run_summary = {
        "config": {
            "device": str(device),
            "battle_format": env_config.battle_format,
            "num_envs": env_config.num_envs,
            "team_size": env_config.team_size,
            "pick_size": env_config.pick_size,
            "rollout_steps": train_config.rollout_steps,
            "num_epochs": train_config.num_epochs,
            "num_minibatches": train_config.num_minibatches,
            "total_timesteps": train_config.total_timesteps,
            "server_urls": server_urls,
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
        num_updates = train_config.total_timesteps // steps_per_update
        updates_done = trainer.global_step // steps_per_update

        print(f"Total updates: {num_updates}")
        print(f"Steps per update: {steps_per_update}")
        print(f"Server URLs: {', '.join(server_urls)}")
        print(f"Initial reset: {initial_reset_sec:.2f}s")
        if updates_done:
            print(f"Resuming from update {updates_done + 1} / {num_updates}")
        print("Starting training...")

        run_summary["initial_reset_sec"] = initial_reset_sec
        if collect_timing:
            run_summary["initial_env_profile"] = initial_env_profile

        episode_rewards: list[float] = []
        start_time = time.time()

        for update in range(updates_done + 1, num_updates + 1):
            agent.eval()
            buffer.reset(initial_hidden=clone_hidden_state(hidden))
            if collect_timing:
                parallel_env.reset_timing()
            rollout_started = time.perf_counter()
            update_episode_rewards: list[float] = []
            update_episode_secs: list[float] = []
            update_episode_turns: list[float] = []
            update_episode_opponent_choice_calls: list[int] = []
            update_episode_opponent_repeated_turn_requests: list[int] = []
            update_episode_opponent_max_calls_same_turn: list[int] = []
            update_pathological_episodes: list[dict[str, object]] = []
            team_preview_data: list[tuple[dict, np.ndarray, float]] = []
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
                    obs_device_started = time.perf_counter()
                    obs_t = obs_to_device(obs, device)
                    rollout_profile["obs_to_device_sec"] += time.perf_counter() - obs_device_started

                    mask_device_started = time.perf_counter()
                    masks_t = torch.from_numpy(masks).to(device=device, dtype=torch.bool)
                    rollout_profile["mask_to_device_sec"] += time.perf_counter() - mask_device_started

                    policy_started = time.perf_counter()
                    out = agent(obs_t, hidden, action_masks=masks_t)
                    rollout_profile["policy_forward_sec"] += time.perf_counter() - policy_started

                    policy_post_started = time.perf_counter()
                    actions_t = out["actions"]
                    log_probs_t = out["log_prob"]
                    values_t = out["value"]
                    actions = actions_t.cpu().numpy()
                    hidden = out["hidden"]
                    rollout_profile["policy_postprocess_sec"] += time.perf_counter() - policy_post_started
                    if device.type == "cpu":
                        obs_for_buffer = {
                            key: value.detach().clone()
                            for key, value in obs_t.items()
                        }
                        masks_for_buffer = masks_t.detach().clone()
                    else:
                        obs_for_buffer = obs_t
                        masks_for_buffer = masks_t

                env_step_started = time.perf_counter()
                next_obs, next_masks, rewards, dones, infos = parallel_env.step(actions)
                rollout_profile["env_step_sec"] += time.perf_counter() - env_step_started

                feedback_device_started = time.perf_counter()
                rewards_t = torch.from_numpy(rewards).to(device)
                dones_t = torch.from_numpy(dones).to(device)
                rollout_profile["feedback_to_device_sec"] += time.perf_counter() - feedback_device_started

                buffer_started = time.perf_counter()
                buffer.add(
                    obs_for_buffer,
                    actions_t,
                    masks_for_buffer,
                    log_probs_t,
                    values_t,
                    rewards_t,
                    dones_t,
                )
                rollout_profile["buffer_add_sec"] += time.perf_counter() - buffer_started

                # Reset hidden state for done environments
                if hidden is not None:
                    hidden_reset_started = time.perf_counter()
                    h, c = hidden
                    not_done = (~dones_t).to(h.dtype).view(1, dones_t.shape[0], 1)
                    hidden = (h * not_done, c * not_done)
                    rollout_profile["hidden_reset_sec"] += time.perf_counter() - hidden_reset_started

                # Track episode rewards and team preview decisions
                for info in infos:
                    if "episode_reward" in info:
                        reward = info["episode_reward"]
                        episode_rewards.append(reward)
                        update_episode_rewards.append(reward)
                        if "team_preview_obs" in info:
                            team_preview_data.append(
                                (info["team_preview_obs"], info["team_preview_picks"], reward)
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
            rollout_profile["rollout_sec"] = rollout_sec
            env_profile = parallel_env.snapshot_timing() if collect_timing else {}
            parallel_env.wait_for_prepared_team_previews()
            refresh_after_update = (
                update % train_config.league_update_interval == 0 and update < num_updates
            )
            if refresh_after_update:
                refresh_opponents(parallel_env, league)

            # --- Compute advantages ---
            with torch.no_grad():
                obs_t = obs_to_device(obs, device)
                last_value = agent.get_value(obs_t, hidden)

            batch = buffer.compute_gae(last_value, train_config.gamma, train_config.gae_lambda)

            # --- PPO update ---
            agent.train()
            ppo_started = time.perf_counter()
            metrics = trainer.update(batch)
            if team_preview_data:
                metrics["team_preview_loss"] = trainer.update_team_preview(team_preview_data)
            ppo_sec = time.perf_counter() - ppo_started
            parallel_env.refresh_prepared_team_previews()

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
            update_summary = {
                "update": update,
                "global_step": trainer.global_step,
                "completed_battles": len(update_episode_rewards),
                "rollout_sec": rollout_sec,
                "ppo_sec": ppo_sec,
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
            run_summary["updates"].append(update_summary)

            # --- Logging ---
            if update % train_config.log_interval == 0:
                elapsed = time.time() - start_time
                sps = trainer.global_step / elapsed
                win_rate = np.mean([r > 0 for r in episode_rewards[-100:]]) if episode_rewards else 0
                print(
                    f"Update {update}/{num_updates} | "
                    f"Step {trainer.global_step:,} | "
                    f"Envs {env_config.num_envs} | "
                    f"Servers {len(server_urls)} | "
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
                    f"League pool {len(league.checkpoints)}"
                )

            # --- Checkpointing ---
            if update % train_config.save_interval == 0:
                path = f"{train_config.checkpoint_dir}/step_{trainer.global_step}.pt"
                trainer.save(path)
                league.save_state(_league_state_path(train_config))
                print(f"Saved checkpoint: {path}")

            # --- League: add checkpoint and refresh opponents ---
            if update % train_config.league_update_interval == 0:
                league.add_checkpoint(agent, trainer.global_step)
                if refresh_after_update:
                    print(f"League updated (pool: {len(league.checkpoints)}), opponents refreshed")
                else:
                    print(f"League updated (pool: {len(league.checkpoints)})")

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
        if args.summary_path:
            _write_run_summary(args.summary_path, run_summary)
        return run_summary
    finally:
        parallel_env.close()
        team_preview_batcher.close()
        league.close()


if __name__ == "__main__":
    train()
