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
    actions: torch.Tensor  # (num_steps, num_envs, 3)
    action_masks: dict[str, torch.Tensor]
    log_probs: torch.Tensor  # (num_steps, num_envs)
    values: torch.Tensor  # (num_steps, num_envs)
    rewards: torch.Tensor  # (num_steps, num_envs)
    dones: torch.Tensor  # (num_steps, num_envs)
    advantages: torch.Tensor  # (num_steps, num_envs)
    returns: torch.Tensor  # (num_steps, num_envs)
    # Trainer-private exact labels for the opponent's selected team are aligned
    # to the same Pokemon at each earlier timestep. Unfinished episodes,
    # unselected Pokemon, and ambiguous identities leave aux_label_valid=False.
    aux_labels: dict[str, torch.Tensor]
    aux_label_valid: torch.Tensor  # (num_steps, num_envs, team_size) bool
    # Observed opponent actions for the aux opponent-action head. Entries where
    # the env did not expose them carry opp_actions_valid=False.
    opp_actions: torch.Tensor  # (num_steps, num_envs, 3) long; -1 when invalid
    opp_actions_valid: torch.Tensor  # (num_steps, num_envs) bool
    num_steps: int
    num_envs: int
    initial_hidden_h: torch.Tensor | None = None  # (num_envs, hidden_dim)
    initial_hidden_c: torch.Tensor | None = None  # (num_envs, hidden_dim)
    # Fixed behavior-policy parameters used for every transition in this
    # rollout. PPO must evaluate actions under the same transformed tera
    # distribution that sampled them.
    tera_exploration_prob: float = 0.0
    tera_force_delay_prob: float = 0.0
    tera_force_delay_max_turn: int = 0


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
    AUX_IDENTITY_KEY = "opp_source_index"
    AUX_OBS_KEYS: tuple[str, ...] = (*AUX_LABEL_KEYS, AUX_IDENTITY_KEY)

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
        self._actions = torch.empty(
            (num_steps, num_envs, 3), dtype=torch.long, device=device
        )
        self._action_masks = {
            key: torch.empty(
                (num_steps, num_envs, *value.shape[1:]),
                dtype=torch.from_numpy(value).dtype,
                device=device,
            )
            for key, value in action_mask_example.items()
        }
        self._log_probs = torch.empty(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self._values = torch.empty(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self._rewards = torch.empty(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self._dones = torch.empty(
            (num_steps, num_envs), dtype=torch.bool, device=device
        )
        self._advantages = torch.empty(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self._returns = torch.empty(
            (num_steps, num_envs), dtype=torch.float32, device=device
        )
        self._initial_hidden_h: torch.Tensor | None = None
        self._initial_hidden_c: torch.Tensor | None = None
        self._tera_exploration_prob = 0.0
        self._tera_force_delay_prob = 0.0
        self._tera_force_delay_max_turn = 0

        # Aux labels require a Pokemon-shaped observation. Tests that pass a
        # minimal obs dict without these keys simply get empty aux buffers and
        # aux_label_valid=False everywhere; no label-tracking work is done.
        self._aux_enabled = all(key in obs_example for key in self.AUX_OBS_KEYS)
        if self._aux_enabled:
            self._aux_labels = {
                key: torch.zeros(
                    (num_steps, num_envs, *obs_example[key].shape[1:]),
                    dtype=torch.from_numpy(obs_example[key]).dtype,
                    device=device,
                )
                for key in self.AUX_LABEL_KEYS
            }
            team_size = obs_example[self.AUX_IDENTITY_KEY].shape[-1]
        else:
            self._aux_labels = {}
            team_size = 0
        self._aux_label_valid = torch.zeros(
            (num_steps, num_envs, team_size),
            dtype=torch.bool,
            device=device,
        )
        self._opp_actions = torch.full(
            (num_steps, num_envs, 3),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        self._opp_actions_valid = torch.zeros(
            (num_steps, num_envs), dtype=torch.bool, device=device
        )
        # Tracks the timestep within the current rollout at which each env's
        # active episode began. Used to retroactively fill aux labels on done.
        self._episode_start = [0] * num_envs

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
        done_env_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
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

        # Retroactively fill aux labels for completed episodes. Terminal slots
        # are matched by their episode-local source index rather than output
        # position because observation slots move when the active Pokemon
        # switch. A unique match is required; untracked identities are skipped.
        if self._aux_enabled:
            if done_env_indices is None:
                done_envs = dones.detach().nonzero(as_tuple=False).squeeze(-1).tolist()
            else:
                done_envs = done_env_indices
            if len(done_envs) > 0:
                label_obs = aux_label_obs if aux_label_obs is not None else obs
                env_indices = [int(env_idx) for env_idx in done_envs]
                self._fill_aligned_aux_labels(
                    starts=[self._episode_start[env_idx] for env_idx in env_indices],
                    end=slot + 1,
                    env_indices=env_indices,
                    label_obs=label_obs,
                )
                for env_idx in env_indices:
                    self._episode_start[env_idx] = slot + 1

        self.step += 1

    def _fill_aligned_aux_labels(
        self,
        *,
        starts: list[int],
        end: int,
        env_indices: list[int],
        label_obs: dict[str, torch.Tensor],
    ) -> None:
        lengths = [end - start for start in starts]
        flat_times_cpu = torch.cat(
            [torch.arange(start, end, dtype=torch.int64) for start in starts]
        )
        env_indices_cpu = torch.tensor(env_indices, dtype=torch.int64)
        lengths_cpu = torch.tensor(lengths, dtype=torch.int64)
        flat_envs_cpu = env_indices_cpu.repeat_interleave(lengths_cpu)
        terminal_rows_cpu = torch.arange(
            len(env_indices),
            dtype=torch.int64,
        ).repeat_interleave(lengths_cpu)

        flat_times = flat_times_cpu.to(device=self.device)
        flat_envs = flat_envs_cpu.to(device=self.device)
        terminal_rows = terminal_rows_cpu.to(device=self.device)
        done_envs = env_indices_cpu.to(device=self.device)

        current_identity = self._obs[self.AUX_IDENTITY_KEY][flat_times, flat_envs]
        terminal_identity = (
            label_obs[self.AUX_IDENTITY_KEY]
            .index_select(
                0,
                done_envs,
            )
            .index_select(0, terminal_rows)
        )
        current_present = ~self._obs["opp_mask"][flat_times, flat_envs]
        terminal_present = ~label_obs["opp_mask"].index_select(
            0,
            done_envs,
        ).index_select(0, terminal_rows)

        matches = current_identity.unsqueeze(-1).eq(terminal_identity.unsqueeze(-2))
        matches &= current_present.unsqueeze(-1)
        matches &= terminal_present.unsqueeze(-2)
        matches &= current_identity.unsqueeze(-1).ge(0)
        unique_match = matches.sum(dim=-1).eq(1)
        terminal_indices = matches.to(torch.int64).argmax(dim=-1)
        pair_indices = torch.arange(
            terminal_indices.shape[0],
            device=self.device,
        ).unsqueeze(-1)

        for key in self.AUX_LABEL_KEYS:
            terminal_value = (
                label_obs[key]
                .index_select(
                    0,
                    done_envs,
                )
                .index_select(0, terminal_rows)
            )
            aligned_value = terminal_value[pair_indices, terminal_indices]
            trailing_dims = aligned_value.ndim - unique_match.ndim
            valid_value = unique_match.reshape(
                *unique_match.shape,
                *((1,) * trailing_dims),
            ).expand_as(aligned_value)
            self._aux_labels[key][flat_times, flat_envs] = aligned_value.masked_fill(
                ~valid_value,
                0,
            )
        self._aux_label_valid[flat_times, flat_envs] = unique_match

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

            delta = (
                self._rewards[t]
                + gamma * next_value * next_non_terminal
                - self._values[t]
            )
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
            tera_exploration_prob=self._tera_exploration_prob,
            tera_force_delay_prob=self._tera_force_delay_prob,
            tera_force_delay_max_turn=self._tera_force_delay_max_turn,
        )

    def set_tera_behavior_policy(
        self,
        *,
        exploration_prob: float,
        force_delay_prob: float,
        force_delay_max_turn: int,
    ) -> None:
        """Record the fixed tera behavior policy before collecting a rollout."""
        if self.step != 0:
            raise RuntimeError(
                "Tera behavior policy cannot change after rollout collection starts."
            )
        for name, value in (
            ("exploration_prob", exploration_prob),
            ("force_delay_prob", force_delay_prob),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be finite and within [0, 1], got {value}."
                )
        if force_delay_max_turn < 0:
            raise ValueError("force_delay_max_turn must be non-negative.")
        self._tera_exploration_prob = float(exploration_prob)
        self._tera_force_delay_prob = float(force_delay_prob)
        self._tera_force_delay_max_turn = int(force_delay_max_turn)

    def reset(
        self,
        initial_hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.step = 0
        self._initial_hidden_h = None
        self._initial_hidden_c = None
        self._tera_exploration_prob = 0.0
        self._tera_force_delay_prob = 0.0
        self._tera_force_delay_max_turn = 0
        self._aux_label_valid.fill_(False)
        self._opp_actions.fill_(-1)
        self._opp_actions_valid.fill_(False)
        # New rollout starts a fresh episode-start tracker for every env.
        self._episode_start = [0] * self.num_envs

        if initial_hidden is None:
            return

        h, c = initial_hidden
        if h.shape[:2] != (1, self.num_envs) or c.shape[:2] != (1, self.num_envs):
            raise ValueError(
                "Initial hidden state must have shape (1, num_envs, hidden_dim) for both h and c."
            )

        self._initial_hidden_h = h.squeeze(0).detach().to(self.device).clone()
        self._initial_hidden_c = c.squeeze(0).detach().to(self.device).clone()
