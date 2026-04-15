"""Rollout buffer for collecting experience from parallel environments."""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class RolloutBatch:
    """A batch of experience for PPO updates."""
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor          # (batch, 2)
    action_masks: torch.Tensor     # (batch, 2, num_actions)
    log_probs: torch.Tensor        # (batch,)
    values: torch.Tensor           # (batch,)
    rewards: torch.Tensor          # (batch,)
    dones: torch.Tensor            # (batch,)
    advantages: torch.Tensor       # (batch,)
    returns: torch.Tensor          # (batch,)

    # LSTM hidden states at the start of each step (for sequence training)
    hidden_h: torch.Tensor | None = None  # (batch, hidden_dim)
    hidden_c: torch.Tensor | None = None  # (batch, hidden_dim)


class RolloutBuffer:
    """Stores rollout data and computes GAE advantages.

    Designed for parallel environments with LSTM policies:
    stores data in (num_steps, num_envs) layout to preserve
    temporal ordering per environment.
    """

    def __init__(self, num_steps: int, num_envs: int, obs_keys: list[str]):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_keys = obs_keys
        self.step = 0
        self._obs: dict[str, list] = {k: [] for k in obs_keys}
        self._actions: list = []
        self._action_masks: list = []
        self._log_probs: list = []
        self._values: list = []
        self._rewards: list = []
        self._dones: list = []
        self._hidden_h: list = []
        self._hidden_c: list = []

    def add(
        self,
        obs: dict[str, np.ndarray],
        actions: np.ndarray,
        action_masks: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        hidden: tuple | None = None,
    ):
        for k in self.obs_keys:
            self._obs[k].append(obs[k])
        self._actions.append(actions)
        self._action_masks.append(action_masks)
        self._log_probs.append(log_probs)
        self._values.append(values)
        self._rewards.append(rewards)
        self._dones.append(dones)

        if hidden is not None:
            h, c = hidden
            self._hidden_h.append(h.squeeze(0).cpu().numpy())
            self._hidden_c.append(c.squeeze(0).cpu().numpy())

        self.step += 1

    def compute_gae(
        self, last_value: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> RolloutBatch:
        """Compute GAE advantages and returns, then flatten into a batch."""
        values = np.array(self._values)       # (steps, envs)
        rewards = np.array(self._rewards)     # (steps, envs)
        dones = np.array(self._dones)         # (steps, envs)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t].astype(np.float32)

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        returns = advantages + values

        # Flatten (steps, envs, ...) -> (steps * envs, ...)
        def to_tensor(arr):
            flat = arr.reshape(-1, *arr.shape[2:])
            return torch.from_numpy(flat)

        obs_tensors = {}
        for k in self.obs_keys:
            arr = np.array(self._obs[k])  # (steps, envs, ...)
            obs_tensors[k] = to_tensor(arr)

        batch = RolloutBatch(
            obs=obs_tensors,
            actions=to_tensor(np.array(self._actions)),
            action_masks=to_tensor(np.array(self._action_masks)),
            log_probs=to_tensor(np.array(self._log_probs)),
            values=to_tensor(np.array(self._values)),
            rewards=to_tensor(np.array(rewards)),
            dones=to_tensor(np.array(dones)),
            advantages=to_tensor(advantages),
            returns=to_tensor(returns),
        )

        if self._hidden_h:
            batch.hidden_h = to_tensor(np.array(self._hidden_h))
            batch.hidden_c = to_tensor(np.array(self._hidden_c))

        return batch

    def reset(self):
        self.step = 0
        self._obs = {k: [] for k in self.obs_keys}
        self._actions = []
        self._action_masks = []
        self._log_probs = []
        self._values = []
        self._rewards = []
        self._dones = []
        self._hidden_h = []
        self._hidden_c = []
