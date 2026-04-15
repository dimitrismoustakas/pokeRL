"""Self-play league: a pool of past agent checkpoints as opponents."""

import random
import copy
from pathlib import Path
from collections import deque

import torch
from poke_env.player import Player, RandomPlayer

from ..model.agent import PokeAgent
from ..config import TrainingConfig


class LeagueOpponent:
    """Wraps a frozen agent checkpoint as a poke-env Player for self-play."""

    def __init__(self, agent: PokeAgent, name: str):
        self.agent = agent
        self.agent.eval()
        self.name = name

    @classmethod
    def from_checkpoint(cls, path: str, agent_template: PokeAgent, device: torch.device):
        agent_copy = copy.deepcopy(agent_template)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        agent_copy.load_state_dict(checkpoint["agent"])
        agent_copy.to(device)
        return cls(agent_copy, name=Path(path).stem)


class League:
    """Manages a pool of opponent checkpoints for self-play training.

    Opponents are sampled with a mix of:
    - Recent checkpoints (to track current skill)
    - Older checkpoints (to prevent forgetting)
    - Built-in bots (for diversity)
    """

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.checkpoints: deque[str] = deque(maxlen=config.league_size)
        self.win_rates: dict[str, float] = {}

    def add_checkpoint(self, agent: PokeAgent, step: int):
        """Snapshot the current agent as a new league member."""
        path = f"{self.config.checkpoint_dir}/league/step_{step}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": agent.state_dict()}, path)
        self.checkpoints.append(path)

    def sample_opponent_path(self) -> str | None:
        """Sample a checkpoint path from the league.

        Sampling strategy: 70% from recent half, 30% from older half.
        Returns None if the league is empty (use a built-in bot instead).
        """
        if not self.checkpoints:
            return None

        checkpoints = list(self.checkpoints)
        mid = len(checkpoints) // 2

        if len(checkpoints) < 2 or random.random() < 0.7:
            # Recent
            return random.choice(checkpoints[mid:]) if mid < len(checkpoints) else random.choice(checkpoints)
        else:
            # Older
            return random.choice(checkpoints[:mid])

    def should_use_league(self) -> bool:
        """Whether to play against the league vs a built-in bot."""
        if not self.checkpoints:
            return False
        return random.random() < self.config.self_play_prob

    def record_result(self, opponent_name: str, won: bool):
        """Track win rate against each opponent (for monitoring)."""
        if opponent_name not in self.win_rates:
            self.win_rates[opponent_name] = 0.5
        # Exponential moving average
        alpha = 0.1
        self.win_rates[opponent_name] = (
            alpha * float(won) + (1 - alpha) * self.win_rates[opponent_name]
        )
