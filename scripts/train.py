"""Training entry point.

Usage:
    python -m scripts.train
    python -m scripts.train --device cuda
    python -m scripts.train --num-envs 1 --rollout-steps 8 --total-timesteps 24
"""

import argparse
import asyncio
import time

import numpy as np
import torch

from poke_env.player import MaxBasePowerPlayer

from pokerl.config import ModelConfig, EnvConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.model.agent import PokeAgent
from pokerl.env.showdown_env import (
    BattleEnv,
    ParallelBattleEnv,
    make_local_account_configuration,
    make_server_configuration,
)
from pokerl.training.rollout import RolloutBuffer
from pokerl.training.ppo import PPOTrainer
from pokerl.training.league import League


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pokeRL with PPO.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the number of parallel battle environments.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override rollout steps collected per update.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override the total environment steps to train for.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override PPO epochs per update.")
    parser.add_argument("--num-minibatches", type=int, default=None, help="Override PPO minibatches per update.")
    parser.add_argument("--log-interval", type=int, default=None, help="Override logging frequency in updates.")
    parser.add_argument("--save-interval", type=int, default=None, help="Override checkpoint save frequency in updates.")
    parser.add_argument(
        "--league-update-interval",
        type=int,
        default=None,
        help="Override league checkpoint and opponent refresh frequency in updates.",
    )
    parser.add_argument(
        "--profile-rollout",
        action="store_true",
        help="Log a detailed per-update rollout timing breakdown.",
    )
    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> tuple[ModelConfig, EnvConfig, TrainingConfig]:
    model_config = ModelConfig()
    env_config = EnvConfig()
    train_config = TrainingConfig()

    if args.num_envs is not None:
        env_config.num_envs = args.num_envs

    overrides = {
        "rollout_steps": args.rollout_steps,
        "total_timesteps": args.total_timesteps,
        "num_epochs": args.num_epochs,
        "num_minibatches": args.num_minibatches,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "league_update_interval": args.league_update_interval,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(train_config, field_name, value)

    return model_config, env_config, train_config


def make_opponent_factory(env_config, game_data, league, device):
    """Create an opponent factory for one environment."""
    if league.should_use_league():
        def league_factory(loop: asyncio.AbstractEventLoop):
            opponent = league.make_opponent(loop)
            if opponent is None:
                raise RuntimeError("League opponent requested, but the league is empty.")
            return opponent

        return league_factory, "league"

    def baseline_factory(loop: asyncio.AbstractEventLoop):
        return MaxBasePowerPlayer(
            account_configuration=make_local_account_configuration("maxbase"),
            battle_format=env_config.battle_format,
            server_configuration=make_server_configuration(env_config),
            loop=loop,
        )

    return baseline_factory, "max-base-power"


def make_envs(env_config, game_data, league, device):
    """Create parallel battle environments with mixed opponents."""
    envs = []
    for i in range(env_config.num_envs):
        opponent_factory, _ = make_opponent_factory(env_config, game_data, league, device)
        env = BattleEnv(env_config, game_data, opponent_factory)
        envs.append(env)
    return ParallelBattleEnv(envs)


def refresh_opponents(parallel_env, env_config, game_data, league, device):
    """Swap out opponents in existing environments.

    Called periodically so the agent faces a mix of league
    and random opponents as training progresses.
    """
    for env in parallel_env.envs:
        opponent_factory, _ = make_opponent_factory(env_config, game_data, league, device)
        env.set_next_opponent(opponent_factory)


def obs_to_device(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in obs.items()}


def _share(part: float, whole: float) -> float:
    return (part / whole * 100.0) if whole > 0 else 0.0


def log_rollout_profile(profile: dict[str, float], env_profile: dict[str, float]) -> None:
    rollout_sec = profile.get("rollout_sec", 0.0)
    env_step_sec = profile.get("env_step_sec", 0.0)
    print(
        "Rollout profile | "
        f"obs->device {profile.get('obs_to_device_sec', 0.0):.3f}s ({_share(profile.get('obs_to_device_sec', 0.0), rollout_sec):.1f}%) | "
        f"mask->device {profile.get('mask_to_device_sec', 0.0):.3f}s ({_share(profile.get('mask_to_device_sec', 0.0), rollout_sec):.1f}%) | "
        f"policy fwd {profile.get('policy_forward_sec', 0.0):.3f}s ({_share(profile.get('policy_forward_sec', 0.0), rollout_sec):.1f}%) | "
        f"policy post {profile.get('policy_postprocess_sec', 0.0):.3f}s ({_share(profile.get('policy_postprocess_sec', 0.0), rollout_sec):.1f}%) | "
        f"env wall {env_step_sec:.3f}s ({_share(env_step_sec, rollout_sec):.1f}%) | "
        f"buffer {profile.get('buffer_add_sec', 0.0):.3f}s ({_share(profile.get('buffer_add_sec', 0.0), rollout_sec):.1f}%) | "
        f"hidden reset {profile.get('hidden_reset_sec', 0.0):.3f}s ({_share(profile.get('hidden_reset_sec', 0.0), rollout_sec):.1f}%)"
    )

    env_steps = int(env_profile.get("env_step_calls", 0.0))
    env_step_total = env_profile.get("env_step_total_sec", 0.0)
    env_queue_wait = env_profile.get("env_queue_wait_sec", 0.0)
    emit_total = env_profile.get("player_emit_total_sec", 0.0)
    make_order_total = env_profile.get("player_make_order_sec", 0.0)
    opponent_choice_total = env_profile.get("opponent_choice_sec", 0.0)
    approx_other_wait = max(env_queue_wait - emit_total - make_order_total - opponent_choice_total, 0.0)

    def avg_ms(total: float, count: int) -> float:
        return (total / count * 1000.0) if count else 0.0

    print(
        "Env profile (avg per env step) | "
        f"step total {avg_ms(env_step_total, env_steps):.2f}ms | "
        f"queue wait {avg_ms(env_queue_wait, env_steps):.2f}ms ({_share(env_queue_wait, env_step_total):.1f}% of env step) | "
        f"battle wait other {avg_ms(approx_other_wait, env_steps):.2f}ms | "
        f"emit total {avg_ms(emit_total, env_steps):.2f}ms | "
        f"obs build {avg_ms(env_profile.get('player_emit_obs_sec', 0.0), env_steps):.2f}ms | "
        f"mask build {avg_ms(env_profile.get('player_emit_mask_sec', 0.0), env_steps):.2f}ms | "
        f"order build {avg_ms(make_order_total, env_steps):.2f}ms | "
        f"opponent choice {avg_ms(opponent_choice_total, env_steps):.2f}ms"
    )

    reset_calls = int(env_profile.get("env_reset_calls", 0.0))
    parallel_step_calls = int(env_profile.get("parallel_step_calls", 0.0))
    parallel_step_wall = env_profile.get("parallel_step_wall_sec", 0.0)
    parallel_wait = env_profile.get("parallel_step_wait_sec", 0.0)
    parallel_collate = env_profile.get("parallel_collate_sec", 0.0)
    print(
        "Parallel env wall | "
        f"parallel step {avg_ms(parallel_step_wall, parallel_step_calls):.2f}ms | "
        f"executor wait {avg_ms(parallel_wait, parallel_step_calls):.2f}ms | "
        f"collate {avg_ms(parallel_collate, parallel_step_calls):.2f}ms | "
        f"battle reset {avg_ms(env_profile.get('env_reset_total_sec', 0.0), reset_calls):.2f}ms over {reset_calls} resets"
    )


def train():
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(f"Device: {device}")

    model_config, env_config, train_config = build_configs(args)

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

    # Training
    trainer = PPOTrainer(agent, train_config, device)
    league = League(train_config, env_config, game_data, agent, device)

    # Environments (start with max-base-power opponents, league is empty)
    parallel_env = make_envs(env_config, game_data, league, device)

    try:
        # Rollout buffer
        obs, masks = parallel_env.reset()
        parallel_env.reset_timing()
        obs_keys = list(obs.keys())
        buffer = RolloutBuffer(train_config.rollout_steps, env_config.num_envs, obs_keys)

        hidden = agent.initial_hidden(env_config.num_envs)
        num_updates = train_config.total_timesteps // steps_per_update

        print(f"Total updates: {num_updates}")
        print(f"Steps per update: {steps_per_update}")
        print("Starting training...")

        episode_rewards = []
        start_time = time.time()

        for update in range(1, num_updates + 1):
            agent.eval()
            buffer.reset()
            if args.profile_rollout:
                parallel_env.reset_timing()
            rollout_started = time.perf_counter()
            update_episode_rewards = []
            update_episode_secs = []
            update_episode_turns = []
            rollout_profile = {
                "obs_to_device_sec": 0.0,
                "mask_to_device_sec": 0.0,
                "policy_forward_sec": 0.0,
                "policy_postprocess_sec": 0.0,
                "env_step_sec": 0.0,
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
                    masks_t = torch.from_numpy(masks).to(device).bool()
                    rollout_profile["mask_to_device_sec"] += time.perf_counter() - mask_device_started

                    # Store the hidden state BEFORE the forward pass so PPO
                    # can reconstruct the same computation during updates.
                    pre_hidden = hidden

                    policy_started = time.perf_counter()
                    out = agent(obs_t, hidden, action_masks=masks_t)
                    rollout_profile["policy_forward_sec"] += time.perf_counter() - policy_started

                    policy_post_started = time.perf_counter()
                    actions = out["actions"].cpu().numpy()
                    log_probs = out["log_prob"].cpu().numpy()
                    values = out["value"].cpu().numpy()
                    hidden = out["hidden"]
                    rollout_profile["policy_postprocess_sec"] += time.perf_counter() - policy_post_started

                # Step environments
                env_step_started = time.perf_counter()
                next_obs, next_masks, rewards, dones, infos = parallel_env.step(actions)
                rollout_profile["env_step_sec"] += time.perf_counter() - env_step_started

                buffer_started = time.perf_counter()
                buffer.add(obs, actions, masks, log_probs, values, rewards, dones, pre_hidden)
                rollout_profile["buffer_add_sec"] += time.perf_counter() - buffer_started

                # Reset hidden state for done environments
                if hidden is not None and dones.any():
                    hidden_reset_started = time.perf_counter()
                    h, c = hidden
                    done_idx = torch.from_numpy(dones).to(device)
                    h[:, done_idx] = 0
                    c[:, done_idx] = 0
                    hidden = (h, c)
                    rollout_profile["hidden_reset_sec"] += time.perf_counter() - hidden_reset_started

                # Track episode rewards
                for info in infos:
                    if "episode_reward" in info:
                        reward = info["episode_reward"]
                        episode_rewards.append(reward)
                        update_episode_rewards.append(reward)
                    if "episode_sec" in info:
                        update_episode_secs.append(info["episode_sec"])
                    if "battle_turn" in info and info.get("episode_reward") is not None:
                        update_episode_turns.append(info["battle_turn"])

                obs, masks = next_obs, next_masks

            rollout_sec = time.perf_counter() - rollout_started
            rollout_profile["rollout_sec"] = rollout_sec
            env_profile = parallel_env.snapshot_timing() if args.profile_rollout else {}

            # --- Compute advantages ---
            with torch.no_grad():
                obs_t = obs_to_device(obs, device)
                last_value = agent.get_value(obs_t, hidden).cpu().numpy()

            batch = buffer.compute_gae(last_value, train_config.gamma, train_config.gae_lambda)

            # --- PPO update ---
            agent.train()
            ppo_started = time.perf_counter()
            metrics = trainer.update(batch)
            ppo_sec = time.perf_counter() - ppo_started

            # --- Logging ---
            if update % train_config.log_interval == 0:
                elapsed = time.time() - start_time
                sps = trainer.global_step / elapsed
                win_rate = np.mean([r > 0 for r in episode_rewards[-100:]]) if episode_rewards else 0
                avg_episode_sec = float(np.mean(update_episode_secs)) if update_episode_secs else 0.0
                avg_episode_turns = float(np.mean(update_episode_turns)) if update_episode_turns else 0.0
                print(
                    f"Update {update}/{num_updates} | "
                    f"Step {trainer.global_step:,} | "
                    f"Envs {env_config.num_envs} | "
                    f"SPS {sps:.0f} | "
                    f"Rollout {rollout_sec:.2f}s | "
                    f"PPO {ppo_sec:.2f}s | "
                    f"Completed battles {len(update_episode_rewards)} | "
                    f"Avg battle {avg_episode_sec:.2f}s/{avg_episode_turns:.1f} turns | "
                    f"Win rate (100) {win_rate:.2%} | "
                    f"Policy loss {metrics['policy_loss']:.4f} | "
                    f"Value loss {metrics['value_loss']:.4f} | "
                    f"Entropy {metrics['entropy']:.4f} | "
                    f"League pool {len(league.checkpoints)}"
                )
                if args.profile_rollout:
                    log_rollout_profile(rollout_profile, env_profile)

            # --- Checkpointing ---
            if update % train_config.save_interval == 0:
                path = f"{train_config.checkpoint_dir}/step_{trainer.global_step}.pt"
                trainer.save(path)
                print(f"Saved checkpoint: {path}")

            # --- League: add checkpoint and refresh opponents ---
            if update % train_config.league_update_interval == 0:
                league.add_checkpoint(agent, trainer.global_step)
                refresh_opponents(parallel_env, env_config, game_data, league, device)
                print(f"League updated (pool: {len(league.checkpoints)}), opponents refreshed")

        print("Training complete.")
        trainer.save(f"{train_config.checkpoint_dir}/final.pt")
    finally:
        parallel_env.close()
if __name__ == "__main__":
    train()
