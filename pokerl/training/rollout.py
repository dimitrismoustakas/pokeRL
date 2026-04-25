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
    actions: torch.Tensor          # (num_steps, num_envs, 3)
    action_masks: dict[str, torch.Tensor]
    log_probs: torch.Tensor        # (num_steps, num_envs)
    values: torch.Tensor           # (num_steps, num_envs)
    rewards: torch.Tensor          # (num_steps, num_envs)
    dones: torch.Tensor            # (num_steps, num_envs)
    advantages: torch.Tensor       # (num_steps, num_envs)
    returns: torch.Tensor          # (num_steps, num_envs)
    # Aux labels for hidden-info prediction. Each tensor holds the end-of-episode
    # opponent observation copied onto every timestep of that episode. Unfinished
    # episodes at rollout boundary leave aux_label_valid=False for that slice.
    aux_labels: dict[str, torch.Tensor]
    aux_label_valid: torch.Tensor  # (num_steps, num_envs) bool
    # Observed opponent actions for the aux opponent-action head. Entries where
    # the env did not expose them carry opp_actions_valid=False.
    opp_actions: torch.Tensor      # (num_steps, num_envs, 3) long; -1 when invalid
    opp_actions_valid: torch.Tensor  # (num_steps, num_envs) bool
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

    # Obs keys whose end-of-episode value is stored as an aux label.
    AUX_LABEL_KEYS: tuple[str, ...] = (
        "opp_item",
        "opp_ability",
        "opp_types",
        "opp_moves",
        "opp_continuous",
        "opp_mask",
    )

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_example: dict[str, np.ndarray],
        action_mask_example: dict[str, np.ndarray],
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
        self._actions = torch.empty((num_steps, num_envs, 3), dtype=torch.long, device=device)
        self._action_masks = {
            key: torch.empty(
                (num_steps, num_envs, *value.shape[1:]),
                dtype=torch.from_numpy(value).dtype,
                device=device,
            )
            for key, value in action_mask_example.items()
        }
        self._log_probs = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._values = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._rewards = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._dones = torch.empty((num_steps, num_envs), dtype=torch.bool, device=device)
        self._advantages = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._returns = torch.empty((num_steps, num_envs), dtype=torch.float32, device=device)
        self._initial_hidden_h: torch.Tensor | None = None
        self._initial_hidden_c: torch.Tensor | None = None

        # Aux labels require a Pokemon-shaped observation. Tests that pass a
        # minimal obs dict without these keys simply get empty aux buffers and
        # aux_label_valid=False everywhere; no label-tracking work is done.
        self._aux_enabled = all(key in obs_example for key in self.AUX_LABEL_KEYS)
        if self._aux_enabled:
            self._aux_labels = {
                key: torch.empty(
                    (num_steps, num_envs, *obs_example[key].shape[1:]),
                    dtype=torch.from_numpy(obs_example[key]).dtype,
                    device=device,
                )
                for key in self.AUX_LABEL_KEYS
            }
        else:
            self._aux_labels = {}
        self._aux_label_valid = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self._opp_actions = torch.full(
            (num_steps, num_envs, 3),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        self._opp_actions_valid = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        # Tracks the timestep within the current rollout at which each env's
        # active episode began. Used to retroactively fill aux labels on done.
        self._episode_start = torch.zeros(num_envs, dtype=torch.long, device=device)

    def add(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_masks: dict[str, torch.Tensor],
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        opp_actions: torch.Tensor | None = None,
        opp_actions_valid: torch.Tensor | None = None,
        aux_label_obs: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if self.step >= self.num_steps:
            raise IndexError(
                f"RolloutBuffer is full ({self.num_steps} steps); call reset() before adding more data."
            )

        slot = self.step
        for key in self.obs_keys:
            self._obs[key][slot].copy_(obs[key].detach())
        self._actions[slot].copy_(actions.detach())
        for key in self._action_masks:
            self._action_masks[key][slot].copy_(action_masks[key].detach())
        self._log_probs[slot].copy_(log_probs.detach())
        self._values[slot].copy_(values.detach())
        self._rewards[slot].copy_(rewards.detach())
        self._dones[slot].copy_(dones.detach())

        if opp_actions is not None:
            self._opp_actions[slot].copy_(opp_actions.detach())
        else:
            self._opp_actions[slot].fill_(-1)
        if opp_actions_valid is not None:
            self._opp_actions_valid[slot].copy_(opp_actions_valid.detach())
        else:
            self._opp_actions_valid[slot].fill_(False)

        # Retroactively fill aux labels: every env whose episode ended at this
        # step gets the terminal opponent state copied into every timestep from
        # episode start through the done step.
        if self._aux_enabled:
            done_envs = dones.detach().nonzero(as_tuple=False).squeeze(-1)
            if done_envs.numel() > 0:
                label_obs = aux_label_obs if aux_label_obs is not None else obs
                for env_idx in done_envs.tolist():
                    start = int(self._episode_start[env_idx].item())
                    for key in self.AUX_LABEL_KEYS:
                        label_value = label_obs[key][env_idx].detach()
                        self._aux_labels[key][start : slot + 1, env_idx].copy_(
                            label_value.unsqueeze(0).expand(slot + 1 - start, *label_value.shape)
                        )
                    self._aux_label_valid[start : slot + 1, env_idx] = True
                    self._episode_start[env_idx] = slot + 1

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
            aux_labels=self._aux_labels,
            aux_label_valid=self._aux_label_valid,
            opp_actions=self._opp_actions,
            opp_actions_valid=self._opp_actions_valid,
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
        self._aux_label_valid.fill_(False)
        self._opp_actions.fill_(-1)
        self._opp_actions_valid.fill_(False)
        # New rollout starts a fresh episode-start tracker for every env.
        self._episode_start.zero_()

        if initial_hidden is None:
            return

        h, c = initial_hidden
        if h.shape[:2] != (1, self.num_envs) or c.shape[:2] != (1, self.num_envs):
            raise ValueError(
                "Initial hidden state must have shape (1, num_envs, hidden_dim) for both h and c."
            )

        self._initial_hidden_h = h.squeeze(0).detach().to(self.device).clone()
        self._initial_hidden_c = c.squeeze(0).detach().to(self.device).clone()
