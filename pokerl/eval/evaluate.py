"""Evaluate a trained agent against bots or league opponents."""

import numpy as np
import torch

from ..config import ModelConfig, EnvConfig
from ..data import GameData
from ..model.agent import PokeAgent
from ..env.showdown_env import BattleEnv, make_local_account_configuration, make_server_configuration


def evaluate(
    agent: PokeAgent,
    env: BattleEnv,
    num_games: int,
    device: torch.device,
) -> dict:
    """Play games and return statistics."""
    agent.eval()
    wins, losses = 0, 0
    game_lengths = []

    for game in range(num_games):
        obs, masks = env.reset()
        hidden = agent.initial_hidden(1)
        done = False
        turn = 0

        while not done:
            with torch.no_grad():
                obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs.items()}
                masks_t = torch.from_numpy(masks).unsqueeze(0).to(device).bool()

                out = agent(obs_t, hidden, action_masks=masks_t)
                action = out["actions"].cpu().numpy()[0]
                hidden = out["hidden"]

            obs, masks, reward, done, info = env.step(action)
            turn += 1

        if reward > 0:
            wins += 1
        else:
            losses += 1
        game_lengths.append(turn)

    return {
        "wins": wins,
        "losses": losses,
        "win_rate": wins / num_games,
        "avg_game_length": np.mean(game_lengths),
        "num_games": num_games,
    }


def load_and_evaluate(
    checkpoint_path: str,
    num_games: int = 100,
    device_name: str = "cpu",
):
    """Load a checkpoint and evaluate against a max-base-power bot."""
    from poke_env.player import MaxBasePowerPlayer

    device = torch.device(device_name)
    game_data = GameData(gen=9)
    model_config = ModelConfig()
    env_config = EnvConfig()

    agent = PokeAgent(model_config, game_data)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.load_state_dict(checkpoint["agent"])
    agent.to(device)

    def opponent_factory(loop):
        return MaxBasePowerPlayer(
            account_configuration=make_local_account_configuration("eval"),
            battle_format=env_config.battle_format,
            server_configuration=make_server_configuration(env_config),
            loop=loop,
        )

    env = BattleEnv(env_config, game_data, opponent_factory)

    try:
        results = evaluate(agent, env, num_games, device)
        print(f"Results over {num_games} games:")
        print(f"  Win rate: {results['win_rate']:.1%}")
        print(f"  Avg game length: {results['avg_game_length']:.1f} turns")
        return results
    finally:
        env.close()
