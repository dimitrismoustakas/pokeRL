"""Rollout buffer for collecting experience from parallel environments."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    """A device-resident PPO batch stored in sequence layout.

    All tensors keep their original ``(num_steps, num_envs, ...)`` layout so the
    PPO trainer can operate on full environment sequences without a flatten /
    reshape round-trip.
    """

    obs: dict[str, torch.Tensor]
    actions: torch.Tensor          # (num_steps, num_envs, 2)
    action_masks: torch.Tensor     # (num_steps, num_envs, 2, num_actions)
    log_probs: torch.Tensor        # (num_steps, num_envs)
    values: torch.Tensor           # (num_steps, num_envs)
    rewards: torch.Tensor          # (num_steps, num_envs)
    dones: torch.Tensor            # (num_steps, num_envs)
    advantages: torch.Tensor       # (num_steps, num_envs)
    returns: torch.Tensor          # (num_steps, num_envs)
    num_steps: int
    num_envs: int
    initial_hidden_h: torch.Tensor | None = None  # (num_envs, hidden_dim)
    initial_hidden_c: torch.Tensor | None = None  # (num_envs, hidden_dim)


class RolloutBuffer:
    """Stores rollout data and computes GAE advantages.

    The buffer preallocates tensors on the trainer device so rollout collection
    can reuse the same storage across updates and PPO can train directly from
    the collected sequences without copying the full batch back to the device.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_example: dict[str, np.ndarray],
        action_mask_example: np.ndarray,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.obs_keys = tuple(obs_example.keys())
        self.step = 0

        self._obs = {
            key: torch.empty(
                (num_steps, num_envs, *value.shape[1:]),
                dtype=torch.from_numpy(value).dtype,
                device=device,
            )
            for key, value in obs_example.items()
        }
        self._actions = torch.empty((num_steps, num_envs, 2), dtype=torch.long, device=device)
        self._action_masks = torch.empty(
            (num_steps, num_envs, *action_mask_example.shape[1:]),
            dtype=torch.bool,
            device=device,
        )
        self._log_probs = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._values = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._rewards = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._dones = torch.empty((num_steps, num_envs), dtype=torch.bool, device=device)
        self._advantages = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._returns = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._initial_hidden_h: torch.Tensor | None = None
        self._initial_hidden_c: torch.Tensor | None = None

    def add(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise IndexError(
                f"RolloutBuffer is full ({self.num_steps} steps); call reset() before adding more data."
            )

        slot = self.step
        for key in self.obs_keys:
            self._obs[key][slot].copy_(obs[key].detach())
        self._actions[slot].copy_(actions.detach())
        self._action_masks[slot].copy_(action_masks.detach())
        self._log_probs[slot].copy_(log_probs.detach())
        self._values[slot].copy_(values.detach())
        self._rewards[slot].copy_(rewards.detach())
        self._dones[slot].copy_(dones.detach())

        self.step += 1

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> RolloutBatch:
        """Compute GAE advantages and returns for the collected rollout."""
        if self.step != self.num_steps:
            raise RuntimeError(
                f"Cannot compute GAE with an incomplete rollout: collected {self.step} / {self.num_steps} steps."
            )

        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_value = last_value
            else:
                next_value = self._values[t + 1]
            next_non_terminal = (~self._dones[t]).float()

            delta = self._rewards[t] + gamma * next_value * next_non_terminal - self._values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self._advantages[t].copy_(last_gae)

        self._returns.copy_(self._advantages + self._values)

        return RolloutBatch(
            obs=self._obs,
            actions=self._actions,
            action_masks=self._action_masks,
            log_probs=self._log_probs,
            values=self._values,
            rewards=self._rewards,
            dones=self._dones,
            advantages=self._advantages,
            returns=self._returns,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            initial_hidden_h=self._initial_hidden_h,
            initial_hidden_c=self._initial_hidden_c,
        )

    def reset(
        self,
        initial_hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.step = 0
        self._initial_hidden_h = None
        self._initial_hidden_c = None

        if initial_hidden is None:
            return

        h, c = initial_hidden
        if h.shape[:2] != (1, self.num_envs) or c.shape[:2] != (1, self.num_envs):
            raise ValueError(
                "Initial hidden state must have shape (1, num_envs, hidden_dim) for both h and c."
            )

        self._initial_hidden_h = h.squeeze(0).detach().to(self.device).clone()
        self._initial_hidden_c = c.squeeze(0).detach().to(self.device).clone()
