"""Training entry point.

Usage:
    python -m scripts.train
    python -m scripts.train --device cuda
"""

import sys
import time
import numpy as np
import torch

from pokerl.config import ModelConfig, EnvConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.model.agent import PokeAgent
from pokerl.env.showdown_env import BattleEnv, ParallelBattleEnv, ShowdownPlayer
from pokerl.env.observation import ObservationBuilder
from pokerl.env.actions import ACTIONS_PER_SLOT
from pokerl.training.rollout import RolloutBuffer
from pokerl.training.ppo import PPOTrainer
from pokerl.training.league import League


def make_envs(env_config: EnvConfig, game_data: GameData) -> ParallelBattleEnv:
    """Create parallel battle environments against random opponents."""
    from poke_env.player import RandomPlayer

    envs = []
    for i in range(env_config.num_envs):
        opponent = RandomPlayer(
            battle_format=env_config.battle_format,
            server_configuration=_server_config(env_config),
        )
        env = BattleEnv(env_config, game_data, opponent)
        envs.append(env)
    return ParallelBattleEnv(envs)


def obs_to_device(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in obs.items()}


def train():
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if len(sys.argv) > 1 and sys.argv[1] == "--device":
        device_name = sys.argv[2]
    device = torch.device(device_name)

    print(f"Device: {device}")

    model_config = ModelConfig()
    env_config = EnvConfig()
    train_config = TrainingConfig()

    # Game data and model
    game_data = GameData(gen=9)
    agent = PokeAgent(model_config, game_data)
    agent.to(device)

    print(f"Model parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Training
    trainer = PPOTrainer(agent, train_config, device)
    league = League(train_config, device)

    # Environments
    parallel_env = make_envs(env_config, game_data)

    # Rollout buffer
    obs, masks = parallel_env.reset()
    obs_keys = list(obs.keys())
    buffer = RolloutBuffer(train_config.rollout_steps, env_config.num_envs, obs_keys)

    hidden = agent.initial_hidden(env_config.num_envs)
    num_updates = train_config.total_timesteps // (train_config.rollout_steps * env_config.num_envs)

    print(f"Total updates: {num_updates}")
    print(f"Steps per update: {train_config.rollout_steps * env_config.num_envs}")
    print("Starting training...")

    episode_rewards = []
    start_time = time.time()

    for update in range(1, num_updates + 1):
        agent.eval()
        buffer.reset()

        # --- Collect rollout ---
        for step in range(train_config.rollout_steps):
            with torch.no_grad():
                obs_t = obs_to_device(obs, device)
                masks_t = torch.from_numpy(masks).to(device).bool()

                out = agent(obs_t, hidden, action_masks=masks_t)

                actions = out["actions"].cpu().numpy()
                log_probs = out["log_prob"].cpu().numpy()
                values = out["value"].cpu().numpy()
                hidden = out["hidden"]

            # Step environments
            next_obs, next_masks, rewards, dones, infos = parallel_env.step(actions)

            buffer.add(obs, actions, masks, log_probs, values, rewards, dones, hidden)

            # Reset hidden state for done environments
            if hidden is not None and dones.any():
                h, c = hidden
                done_idx = torch.from_numpy(dones).to(device)
                h[:, done_idx] = 0
                c[:, done_idx] = 0
                hidden = (h, c)

            # Track episode rewards
            for info in infos:
                if "episode_reward" in info:
                    episode_rewards.append(info["episode_reward"])

            obs, masks = next_obs, next_masks

        # --- Compute advantages ---
        with torch.no_grad():
            obs_t = obs_to_device(obs, device)
            last_value = agent.get_value(obs_t, hidden).cpu().numpy()

        batch = buffer.compute_gae(last_value, train_config.gamma, train_config.gae_lambda)

        # --- PPO update ---
        agent.train()
        metrics = trainer.update(batch)

        # --- Logging ---
        if update % train_config.log_interval == 0:
            elapsed = time.time() - start_time
            sps = trainer.global_step / elapsed
            win_rate = np.mean([r > 0 for r in episode_rewards[-100:]]) if episode_rewards else 0
            print(
                f"Update {update}/{num_updates} | "
                f"Step {trainer.global_step:,} | "
                f"SPS {sps:.0f} | "
                f"Win rate (100) {win_rate:.2%} | "
                f"Policy loss {metrics['policy_loss']:.4f} | "
                f"Value loss {metrics['value_loss']:.4f} | "
                f"Entropy {metrics['entropy']:.4f} | "
                f"KL {metrics['approx_kl']:.4f}"
            )

        # --- Checkpointing ---
        if update % train_config.save_interval == 0:
            path = f"{train_config.checkpoint_dir}/step_{trainer.global_step}.pt"
            trainer.save(path)
            print(f"Saved checkpoint: {path}")

        # --- League update ---
        if update % train_config.league_update_interval == 0:
            league.add_checkpoint(agent, trainer.global_step)
            print(f"Added league checkpoint (pool size: {len(league.checkpoints)})")

    print("Training complete.")
    trainer.save(f"{train_config.checkpoint_dir}/final.pt")


def _server_config(env_config: EnvConfig):
    from poke_env import ServerConfiguration
    host, port = env_config.server_url.split(":")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php",
    )


if __name__ == "__main__":
    train()
