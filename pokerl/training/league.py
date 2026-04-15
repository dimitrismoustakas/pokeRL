"""Self-play league: a pool of past agent checkpoints as opponents."""

import asyncio
import random
import copy
from pathlib import Path
from collections import deque

import torch
import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.player import DoubleBattleOrder, PassBattleOrder, Player

from ..model.agent import PokeAgent
from ..data import GameData
from ..config import TrainingConfig, EnvConfig
from ..env.observation import ObservationBuilder
from ..env.actions import build_action_mask, action_to_order
from ..env.showdown_env import make_local_account_configuration


class LeaguePlayer(Player):
    """A poke-env player that uses a frozen agent checkpoint to choose moves.

    This is the opponent in self-play: it loads a past version of the agent
    and plays battles autonomously (no queue bridging needed).
    """

    def __init__(
        self,
        agent: PokeAgent,
        obs_builder: ObservationBuilder,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent = agent
        self.agent.eval()
        self.obs_builder = obs_builder
        self.device = device
        self._hidden = None

    def choose_move(self, battle: DoubleBattle):
        obs = self.obs_builder.build(battle)
        masks = np.stack([build_action_mask(battle, s) for s in range(2)])

        with torch.no_grad():
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
            masks_t = torch.from_numpy(masks).unsqueeze(0).to(self.device).bool()

            if self._hidden is None:
                self._hidden = self.agent.initial_hidden(1)

            out = self.agent(obs_t, self._hidden, action_masks=masks_t)
            self._hidden = out["hidden"]
            actions = out["actions"].cpu().numpy()[0]

        orders = []
        for slot in range(2):
            orders.append(action_to_order(int(actions[slot]), battle, slot))

        if all(isinstance(order, PassBattleOrder) for order in orders):
            return self.choose_default_move()
        return DoubleBattleOrder(orders[0], orders[1])

    def teampreview(self, battle: DoubleBattle) -> str:
        # Use the agent's team preview head
        obs = self.obs_builder.build(battle)
        with torch.no_grad():
            obs_t = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
            out = self.agent.team_preview(obs_t)
            picks = out["picks"].cpu().numpy()[0]
        return "/team " + "".join(str(int(p) + 1) for p in picks)

    def reset_hidden(self):
        """Call between battles to clear LSTM state."""
        self._hidden = None


class League:
    """Manages a pool of opponent checkpoints for self-play training.

    Opponents are sampled with a mix of:
    - Recent checkpoints (to track current skill)
    - Older checkpoints (to prevent forgetting)
    - Built-in bots (for diversity)
    """

    def __init__(
        self,
        config: TrainingConfig,
        env_config: EnvConfig,
        game_data: GameData,
        agent_template: PokeAgent,
        device: torch.device,
    ):
        self.config = config
        self.env_config = env_config
        self.game_data = game_data
        self.agent_template = agent_template
        self.device = device
        self.checkpoints: deque[str] = deque(maxlen=config.league_size)
        self.win_rates: dict[str, float] = {}

    def add_checkpoint(self, agent: PokeAgent, step: int):
        """Snapshot the current agent as a new league member."""
        path = f"{self.config.checkpoint_dir}/league/step_{step}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": agent.state_dict()}, path)
        self.checkpoints.append(path)

    def make_opponent(self, loop: asyncio.AbstractEventLoop) -> LeaguePlayer | None:
        """Create a LeaguePlayer from a sampled checkpoint.

        Returns None if the league is empty.
        """
        path = self._sample_checkpoint()
        if path is None:
            return None

        agent_copy = copy.deepcopy(self.agent_template)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        agent_copy.load_state_dict(checkpoint["agent"])
        agent_copy.to(self.device)

        from poke_env import ServerConfiguration
        host, port = self.env_config.server_url.split(":")
        server_config = ServerConfiguration(
            f"ws://{host}:{port}/showdown/websocket",
            f"http://{host}:{port}/action.php",
        )

        obs_builder = ObservationBuilder(self.game_data, self.env_config.team_size)
        return LeaguePlayer(
            agent_copy,
            obs_builder,
            self.device,
            account_configuration=make_local_account_configuration("league"),
            battle_format=self.env_config.battle_format,
            server_configuration=server_config,
            loop=loop,
        )

    def should_use_league(self) -> bool:
        """Whether to play against the league vs a built-in bot."""
        if not self.checkpoints:
            return False
        return random.random() < self.config.self_play_prob

    def record_result(self, opponent_name: str, won: bool):
        """Track win rate against each opponent (for monitoring)."""
        if opponent_name not in self.win_rates:
            self.win_rates[opponent_name] = 0.5
        alpha = 0.1
        self.win_rates[opponent_name] = (
            alpha * float(won) + (1 - alpha) * self.win_rates[opponent_name]
        )

    def _sample_checkpoint(self) -> str | None:
        """Sample a checkpoint path. 70% recent, 30% older."""
        if not self.checkpoints:
            return None

        checkpoints = list(self.checkpoints)
        mid = len(checkpoints) // 2

        if len(checkpoints) < 2 or random.random() < 0.7:
            return random.choice(checkpoints[mid:]) if mid < len(checkpoints) else random.choice(checkpoints)
        else:
            return random.choice(checkpoints[:mid])
