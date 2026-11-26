"""Rollout buffer for collecting experience from parallel environments."""

from dataclasses import dataclass

import numpy as np
import torch

from ..env.actions import (
    NUM_SLOT_INTENT_COMPONENTS,
    NUM_TERA_INTENT_COMPONENTS,
)
from ..env.observation import NUM_PUBLIC_TARGET_POSITIONS
from ..public_consequence import PUBLIC_CONSEQUENCE_FIELD_DIM
from .preview import TeamPreviewEpisodeStart


@dataclass
class TeamPreviewRolloutBatch:
    """Episode-start preview decisions aligned to one fixed battle rollout."""

    obs: dict[str, torch.Tensor]
    picks: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    opponent_picks: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminal: torch.Tensor
    horizons: torch.Tensor


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
    # online to the same Pokemon at each timestep. Unselected Pokemon,
    # unavailable packed teams, and ambiguous identities leave
    # aux_label_valid=False.
    aux_labels: dict[str, torch.Tensor]
    aux_label_valid: torch.Tensor  # (num_steps, num_envs, team_size) bool
    # Trainer-only semantic opponent-intent targets. Component-specific masks
    # exclude forced/default choices and unavailable conditional decisions.
    opponent_intent_slot_labels: torch.Tensor  # (steps, envs, 2, 4) long
    opponent_intent_slot_masks: torch.Tensor  # (steps, envs, 2, 4) bool
    opponent_intent_tera_labels: torch.Tensor  # (steps, envs, 2) long
    opponent_intent_tera_masks: torch.Tensor  # (steps, envs, 2) bool
    public_consequence_move_emitted_targets: torch.Tensor
    public_consequence_move_emitted_masks: torch.Tensor
    public_consequence_nonmove_reason_labels: torch.Tensor
    public_consequence_nonmove_reason_masks: torch.Tensor
    public_consequence_move_branch_labels: torch.Tensor
    public_consequence_move_branch_masks: torch.Tensor
    public_consequence_route_targets: torch.Tensor
    public_consequence_route_masks: torch.Tensor
    public_consequence_interaction_targets: torch.Tensor
    public_consequence_interaction_masks: torch.Tensor
    public_consequence_interaction_reason_labels: torch.Tensor
    public_consequence_interaction_reason_masks: torch.Tensor
    public_consequence_hit_count_labels: torch.Tensor
    public_consequence_hit_count_masks: torch.Tensor
    public_consequence_damage_targets: torch.Tensor
    public_consequence_damage_masks: torch.Tensor
    public_consequence_cant_targets: torch.Tensor  # (steps, envs, 2, 2) float
    public_consequence_cant_masks: torch.Tensor  # (steps, envs, 2, 2) bool
    public_consequence_field_targets: torch.Tensor  # (steps, envs, 2, 33) float
    public_consequence_field_masks: torch.Tensor  # (steps, envs, 2) bool
    team_preview: TeamPreviewRolloutBatch | None
    num_steps: int
    num_envs: int
    initial_hidden_h: torch.Tensor | None = None  # (num_envs, hidden_dim)
    initial_hidden_c: torch.Tensor | None = None  # (num_envs, hidden_dim)
    # Fixed behavior-policy parameters used for every transition in this
    # rollout. PPO must evaluate actions under the same transformed tera
    # distribution that sampled them.
    tera_exploration_prob: float = 0.0


class RolloutBuffer:
    """Stores rollout data and computes GAE advantages.

    The buffer preallocates tensors on the trainer device so rollout collection
    can reuse the same storage across updates and PPO can train directly from
    the collected sequences without copying the full batch back to the device.
    """

    # Trainer-only selected-team fields supplied at episode start.
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
            self._active_aux_labels = {
                key: torch.zeros(
                    (num_envs, *obs_example[key].shape[1:]),
                    dtype=torch.from_numpy(obs_example[key]).dtype,
                    device=device,
                )
                for key in self.AUX_OBS_KEYS
            }
            team_size = obs_example[self.AUX_IDENTITY_KEY].shape[-1]
        else:
            self._aux_labels = {}
            self._active_aux_labels = {}
            team_size = 0
        self._active_aux_episode = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )
        self._aux_label_valid = torch.zeros(
            (num_steps, num_envs, team_size),
            dtype=torch.bool,
            device=device,
        )
        self._opponent_intent_slot_labels = torch.full(
            (num_steps, num_envs, 2, NUM_SLOT_INTENT_COMPONENTS),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        self._opponent_intent_slot_masks = torch.zeros(
            (num_steps, num_envs, 2, NUM_SLOT_INTENT_COMPONENTS),
            dtype=torch.bool,
            device=device,
        )
        self._opponent_intent_tera_labels = torch.full(
            (num_steps, num_envs, NUM_TERA_INTENT_COMPONENTS),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )
        self._opponent_intent_tera_masks = torch.zeros(
            (num_steps, num_envs, NUM_TERA_INTENT_COMPONENTS),
            dtype=torch.bool,
            device=device,
        )
        self._public_consequence_move_emitted_targets = torch.zeros(
            (num_steps, num_envs, 2),
            dtype=torch.float32,
            device=device,
        )
        self._public_consequence_move_emitted_masks = torch.zeros(
            (num_steps, num_envs, 2),
            dtype=torch.bool,
            device=device,
        )
        for label_name in (
            "nonmove_reason",
            "move_branch",
        ):
            setattr(
                self,
                f"_public_consequence_{label_name}_labels",
                torch.full(
                    (num_steps, num_envs, 2),
                    fill_value=-1,
                    dtype=torch.long,
                    device=device,
                ),
            )
            setattr(
                self,
                f"_public_consequence_{label_name}_masks",
                torch.zeros(
                    (num_steps, num_envs, 2),
                    dtype=torch.bool,
                    device=device,
                ),
            )
        target_shape = (num_steps, num_envs, 2, NUM_PUBLIC_TARGET_POSITIONS)
        for target_name in ("route", "interaction", "damage"):
            setattr(
                self,
                f"_public_consequence_{target_name}_targets",
                torch.zeros(target_shape, dtype=torch.float32, device=device),
            )
            setattr(
                self,
                f"_public_consequence_{target_name}_masks",
                torch.zeros(target_shape, dtype=torch.bool, device=device),
            )
        for label_name in ("interaction_reason", "hit_count"):
            setattr(
                self,
                f"_public_consequence_{label_name}_labels",
                torch.full(
                    target_shape,
                    fill_value=-1,
                    dtype=torch.long,
                    device=device,
                ),
            )
            setattr(
                self,
                f"_public_consequence_{label_name}_masks",
                torch.zeros(target_shape, dtype=torch.bool, device=device),
            )
        self._public_consequence_cant_targets = torch.zeros(
            (num_steps, num_envs, 2, 2),
            dtype=torch.float32,
            device=device,
        )
        self._public_consequence_cant_masks = torch.zeros(
            (num_steps, num_envs, 2, 2),
            dtype=torch.bool,
            device=device,
        )
        self._public_consequence_field_targets = torch.zeros(
            (
                num_steps,
                num_envs,
                2,
                PUBLIC_CONSEQUENCE_FIELD_DIM,
            ),
            dtype=torch.float32,
            device=device,
        )
        self._public_consequence_field_masks = torch.zeros(
            (num_steps, num_envs, 2),
            dtype=torch.bool,
            device=device,
        )
        self._team_preview_starts: list[tuple[int, int, TeamPreviewEpisodeStart]] = []
        self._team_preview_start_keys: set[tuple[int, int]] = set()

    def register_team_preview_starts(
        self,
        starts: list[tuple[int, TeamPreviewEpisodeStart]],
        *,
        policy_version: int,
    ) -> None:
        """Register episode starts whose first battle state is ``self.step``."""
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError(
                f"policy_version must be a non-negative integer, got {policy_version!r}."
            )
        if self.step > self.num_steps:
            raise RuntimeError(
                "Team-preview start cannot be registered beyond the rollout boundary."
            )

        accepted: list[tuple[int, TeamPreviewEpisodeStart]] = []
        for env_index, start in starts:
            if (
                isinstance(env_index, bool)
                or not isinstance(env_index, int)
                or not 0 <= env_index < self.num_envs
            ):
                raise IndexError(
                    f"Team-preview env index {env_index!r} is outside "
                    f"[0, {self.num_envs})."
                )
            if start.policy_version != policy_version:
                raise RuntimeError(
                    "Team-preview episode start was produced by the wrong policy "
                    f"version: expected={policy_version}, "
                    f"actual={start.policy_version}."
                )
            start_key = (env_index, self.step)
            if start_key in self._team_preview_start_keys:
                raise RuntimeError(
                    "An environment registered more than one team-preview start "
                    f"at rollout step {self.step}: env={env_index}."
                )
            self._team_preview_start_keys.add(start_key)
            self._team_preview_starts.append((env_index, self.step, start))
            accepted.append((env_index, start))

        if not accepted:
            return
        accepted_envs = torch.tensor(
            [env_index for env_index, _ in accepted],
            dtype=torch.long,
            device=self.device,
        )
        self._active_aux_episode.index_fill_(0, accepted_envs, False)
        for labels in self._active_aux_labels.values():
            labels.index_fill_(0, accepted_envs, 0)
        if not self._aux_enabled:
            return

        labeled = [
            (env_index, start)
            for env_index, start in accepted
            if start.opponent_aux_labels is not None
        ]
        if not labeled:
            return
        labeled_envs = torch.tensor(
            [env_index for env_index, _ in labeled],
            dtype=torch.long,
            device=self.device,
        )
        for _, start in labeled:
            assert start.opponent_aux_labels is not None
            missing = set(self.AUX_OBS_KEYS).difference(start.opponent_aux_labels)
            if missing:
                raise ValueError(
                    "Team-preview episode start omitted trainer-only opponent "
                    "label key(s): " + ", ".join(sorted(missing))
                )
        for key in self.AUX_OBS_KEYS:
            destination = self._active_aux_labels[key]
            source = torch.from_numpy(
                np.stack(
                    [
                        start.opponent_aux_labels[key]
                        for _, start in labeled
                        if start.opponent_aux_labels is not None
                    ]
                )
            ).to(device=self.device, dtype=destination.dtype)
            expected_shape = (len(labeled), *destination.shape[1:])
            if source.shape != expected_shape:
                raise ValueError(
                    f"Trainer-only opponent label {key!r} has batched shape "
                    f"{tuple(source.shape)}; expected {expected_shape}."
                )
            destination.index_copy_(0, labeled_envs, source)
        self._active_aux_episode.index_fill_(0, labeled_envs, True)

    def add(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_masks: dict[str, torch.Tensor],
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        opponent_intent_slot_labels: torch.Tensor | None = None,
        opponent_intent_slot_masks: torch.Tensor | None = None,
        opponent_intent_tera_labels: torch.Tensor | None = None,
        opponent_intent_tera_masks: torch.Tensor | None = None,
        public_consequence_move_emitted_targets: torch.Tensor | None = None,
        public_consequence_move_emitted_masks: torch.Tensor | None = None,
        public_consequence_nonmove_reason_labels: torch.Tensor | None = None,
        public_consequence_nonmove_reason_masks: torch.Tensor | None = None,
        public_consequence_move_branch_labels: torch.Tensor | None = None,
        public_consequence_move_branch_masks: torch.Tensor | None = None,
        public_consequence_route_targets: torch.Tensor | None = None,
        public_consequence_route_masks: torch.Tensor | None = None,
        public_consequence_interaction_targets: torch.Tensor | None = None,
        public_consequence_interaction_masks: torch.Tensor | None = None,
        public_consequence_interaction_reason_labels: torch.Tensor | None = None,
        public_consequence_interaction_reason_masks: torch.Tensor | None = None,
        public_consequence_hit_count_labels: torch.Tensor | None = None,
        public_consequence_hit_count_masks: torch.Tensor | None = None,
        public_consequence_damage_targets: torch.Tensor | None = None,
        public_consequence_damage_masks: torch.Tensor | None = None,
        public_consequence_cant_targets: torch.Tensor | None = None,
        public_consequence_cant_masks: torch.Tensor | None = None,
        public_consequence_field_targets: torch.Tensor | None = None,
        public_consequence_field_masks: torch.Tensor | None = None,
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

        if opponent_intent_slot_labels is not None:
            self._opponent_intent_slot_labels[slot].copy_(
                opponent_intent_slot_labels.detach()
            )
        else:
            self._opponent_intent_slot_labels[slot].fill_(-1)
        if opponent_intent_slot_masks is not None:
            self._opponent_intent_slot_masks[slot].copy_(
                opponent_intent_slot_masks.detach()
            )
        else:
            self._opponent_intent_slot_masks[slot].fill_(False)
        if opponent_intent_tera_labels is not None:
            self._opponent_intent_tera_labels[slot].copy_(
                opponent_intent_tera_labels.detach()
            )
        else:
            self._opponent_intent_tera_labels[slot].fill_(-1)
        if opponent_intent_tera_masks is not None:
            self._opponent_intent_tera_masks[slot].copy_(
                opponent_intent_tera_masks.detach()
            )
        else:
            self._opponent_intent_tera_masks[slot].fill_(False)

        consequence_values = {
            "move_emitted_targets": public_consequence_move_emitted_targets,
            "move_emitted_masks": public_consequence_move_emitted_masks,
            "nonmove_reason_labels": public_consequence_nonmove_reason_labels,
            "nonmove_reason_masks": public_consequence_nonmove_reason_masks,
            "move_branch_labels": public_consequence_move_branch_labels,
            "move_branch_masks": public_consequence_move_branch_masks,
            "route_targets": public_consequence_route_targets,
            "route_masks": public_consequence_route_masks,
            "interaction_targets": public_consequence_interaction_targets,
            "interaction_masks": public_consequence_interaction_masks,
            "interaction_reason_labels": public_consequence_interaction_reason_labels,
            "interaction_reason_masks": public_consequence_interaction_reason_masks,
            "hit_count_labels": public_consequence_hit_count_labels,
            "hit_count_masks": public_consequence_hit_count_masks,
            "damage_targets": public_consequence_damage_targets,
            "damage_masks": public_consequence_damage_masks,
        }
        for consequence_name, consequence_value in consequence_values.items():
            destination = getattr(self, f"_public_consequence_{consequence_name}")[slot]
            if consequence_value is not None:
                destination.copy_(consequence_value.detach())
            elif consequence_name.endswith("labels"):
                destination.fill_(-1)
            elif consequence_name.endswith("masks"):
                destination.fill_(False)
            else:
                destination.zero_()
        if public_consequence_cant_targets is not None:
            self._public_consequence_cant_targets[slot].copy_(
                public_consequence_cant_targets.detach()
            )
        else:
            self._public_consequence_cant_targets[slot].zero_()
        if public_consequence_cant_masks is not None:
            self._public_consequence_cant_masks[slot].copy_(
                public_consequence_cant_masks.detach()
            )
        else:
            self._public_consequence_cant_masks[slot].fill_(False)
        if public_consequence_field_targets is not None:
            self._public_consequence_field_targets[slot].copy_(
                public_consequence_field_targets.detach()
            )
        else:
            self._public_consequence_field_targets[slot].zero_()
        if public_consequence_field_masks is not None:
            self._public_consequence_field_masks[slot].copy_(
                public_consequence_field_masks.detach()
            )
        else:
            self._public_consequence_field_masks[slot].fill_(False)

        # Exact selected-team labels are known to the trainer at episode start.
        # Align them to the current public slot order immediately so episodes
        # spanning rollout boundaries receive the same supervision as shorter
        # episodes. The labels never enter the policy observation.
        if self._aux_enabled:
            self._fill_current_aux_labels(slot)

        # The terminal action still belongs to the old episode, so clear its
        # trainer-only state only after filling this slot. Auto-reset starts are
        # registered by the collector after add() and before the next action.
        done_mask = dones.detach().to(device=self.device, dtype=torch.bool)
        self._active_aux_episode.logical_and_(~done_mask)
        for labels in self._active_aux_labels.values():
            trailing_dims = labels.ndim - 1
            labels.masked_fill_(
                done_mask.reshape(
                    self.num_envs,
                    *((1,) * trailing_dims),
                ),
                0,
            )

        self.step += 1

    def _fill_current_aux_labels(self, slot: int) -> None:
        current_identity = self._obs[self.AUX_IDENTITY_KEY][slot]
        label_identity = self._active_aux_labels[self.AUX_IDENTITY_KEY]
        current_present = ~self._obs["opp_mask"][slot]
        label_present = ~self._active_aux_labels["opp_mask"]

        matches = current_identity.unsqueeze(-1).eq(label_identity.unsqueeze(-2))
        matches &= current_present.unsqueeze(-1)
        matches &= label_present.unsqueeze(-2)
        matches &= current_identity.unsqueeze(-1).ge(0)
        unique_match = matches.sum(dim=-1).eq(1) & self._active_aux_episode.unsqueeze(
            -1
        )
        label_indices = matches.to(torch.int64).argmax(dim=-1)
        env_indices = torch.arange(
            self.num_envs,
            device=self.device,
        ).unsqueeze(-1)

        for key in self.AUX_LABEL_KEYS:
            aligned_value = self._active_aux_labels[key][env_indices, label_indices]
            trailing_dims = aligned_value.ndim - unique_match.ndim
            valid_value = unique_match.reshape(
                *unique_match.shape,
                *((1,) * trailing_dims),
            ).expand_as(aligned_value)
            self._aux_labels[key][slot].copy_(
                aligned_value.masked_fill(
                    ~valid_value,
                    0,
                )
            )
        self._aux_label_valid[slot].copy_(unique_match)

    def compute_gae(
        self,
        last_value: torch.Tensor,
        gamma: float = 1.0,
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
        team_preview = self._build_team_preview_batch(
            last_value=last_value,
            gamma=gamma,
        )

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
            opponent_intent_slot_labels=self._opponent_intent_slot_labels,
            opponent_intent_slot_masks=self._opponent_intent_slot_masks,
            opponent_intent_tera_labels=self._opponent_intent_tera_labels,
            opponent_intent_tera_masks=self._opponent_intent_tera_masks,
            public_consequence_move_emitted_targets=(
                self._public_consequence_move_emitted_targets
            ),
            public_consequence_move_emitted_masks=(
                self._public_consequence_move_emitted_masks
            ),
            public_consequence_nonmove_reason_labels=(
                self._public_consequence_nonmove_reason_labels
            ),
            public_consequence_nonmove_reason_masks=(
                self._public_consequence_nonmove_reason_masks
            ),
            public_consequence_move_branch_labels=(
                self._public_consequence_move_branch_labels
            ),
            public_consequence_move_branch_masks=(
                self._public_consequence_move_branch_masks
            ),
            public_consequence_route_targets=self._public_consequence_route_targets,
            public_consequence_route_masks=self._public_consequence_route_masks,
            public_consequence_interaction_targets=(
                self._public_consequence_interaction_targets
            ),
            public_consequence_interaction_masks=(
                self._public_consequence_interaction_masks
            ),
            public_consequence_interaction_reason_labels=(
                self._public_consequence_interaction_reason_labels
            ),
            public_consequence_interaction_reason_masks=(
                self._public_consequence_interaction_reason_masks
            ),
            public_consequence_hit_count_labels=(
                self._public_consequence_hit_count_labels
            ),
            public_consequence_hit_count_masks=(
                self._public_consequence_hit_count_masks
            ),
            public_consequence_damage_targets=self._public_consequence_damage_targets,
            public_consequence_damage_masks=self._public_consequence_damage_masks,
            public_consequence_cant_targets=self._public_consequence_cant_targets,
            public_consequence_cant_masks=self._public_consequence_cant_masks,
            public_consequence_field_targets=self._public_consequence_field_targets,
            public_consequence_field_masks=self._public_consequence_field_masks,
            team_preview=team_preview,
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            initial_hidden_h=self._initial_hidden_h,
            initial_hidden_c=self._initial_hidden_c,
            tera_exploration_prob=self._tera_exploration_prob,
        )

    def _build_team_preview_batch(
        self,
        *,
        last_value: torch.Tensor,
        gamma: float,
    ) -> TeamPreviewRolloutBatch | None:
        if not self._team_preview_starts:
            return None

        records = [record for _, _, record in self._team_preview_starts]
        env_indices = torch.tensor(
            [env_index for env_index, _, _ in self._team_preview_starts],
            dtype=torch.long,
            device=self.device,
        )
        first_battle_steps = torch.tensor(
            [first_step for _, first_step, _ in self._team_preview_starts],
            dtype=torch.long,
            device=self.device,
        )

        # Return from every battle state to its own episode's first terminal or
        # the rollout boundary. This is a direct lambda=1 truncated n-step
        # target and therefore does not inherit battle GAE's lambda.
        continuation_returns = torch.empty(
            (self.num_steps + 1, self.num_envs),
            dtype=torch.float32,
            device=self.device,
        )
        terminal = torch.empty(
            (self.num_steps + 1, self.num_envs),
            dtype=torch.bool,
            device=self.device,
        )
        horizons = torch.empty(
            (self.num_steps + 1, self.num_envs),
            dtype=torch.long,
            device=self.device,
        )
        continuation_returns[-1].copy_(last_value)
        terminal[-1].fill_(False)
        horizons[-1].zero_()
        for step in reversed(range(self.num_steps)):
            non_terminal = ~self._dones[step]
            continuation_returns[step].copy_(
                self._rewards[step]
                + gamma
                * non_terminal.to(torch.float32)
                * continuation_returns[step + 1]
            )
            terminal[step].copy_(
                self._dones[step] | (non_terminal & terminal[step + 1])
            )
            horizons[step].copy_(1 + non_terminal.to(torch.long) * horizons[step + 1])

        # Preview itself is a zero-reward transition into the first battle
        # state, so it contributes one additional factor of gamma. Production
        # uses gamma=1, while retaining the factor keeps discounted ablations
        # correctly indexed.
        returns = gamma * continuation_returns[first_battle_steps, env_indices]
        values = torch.tensor(
            [record.value for record in records],
            dtype=torch.float32,
            device=self.device,
        )

        return TeamPreviewRolloutBatch(
            obs={
                key: torch.from_numpy(
                    np.stack([record.obs[key] for record in records])
                ).to(self.device)
                for key in records[0].obs
            },
            picks=torch.from_numpy(np.stack([record.picks for record in records])).to(
                device=self.device, dtype=torch.long
            ),
            log_probs=torch.tensor(
                [record.log_prob for record in records],
                dtype=torch.float32,
                device=self.device,
            ),
            values=values,
            opponent_picks=torch.from_numpy(
                np.stack([record.opponent_picks for record in records])
            ).to(device=self.device, dtype=torch.long),
            advantages=returns - values,
            returns=returns,
            terminal=terminal[first_battle_steps, env_indices],
            horizons=horizons[first_battle_steps, env_indices],
        )

    def set_tera_behavior_policy(
        self,
        *,
        exploration_prob: float,
    ) -> None:
        """Record the fixed tera behavior policy before collecting a rollout."""
        if self.step != 0:
            raise RuntimeError(
                "Tera behavior policy cannot change after rollout collection starts."
            )
        if not np.isfinite(exploration_prob) or not 0.0 <= exploration_prob <= 1.0:
            raise ValueError(
                "exploration_prob must be finite and within [0, 1], "
                f"got {exploration_prob}."
            )
        self._tera_exploration_prob = float(exploration_prob)

    def reset(
        self,
        initial_hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.step = 0
        self._initial_hidden_h = None
        self._initial_hidden_c = None
        self._tera_exploration_prob = 0.0
        self._aux_label_valid.fill_(False)
        self._opponent_intent_slot_labels.fill_(-1)
        self._opponent_intent_slot_masks.fill_(False)
        self._opponent_intent_tera_labels.fill_(-1)
        self._opponent_intent_tera_masks.fill_(False)
        self._public_consequence_move_emitted_targets.zero_()
        self._public_consequence_move_emitted_masks.fill_(False)
        for label_name in (
            "nonmove_reason",
            "move_branch",
            "interaction_reason",
            "hit_count",
        ):
            getattr(self, f"_public_consequence_{label_name}_labels").fill_(-1)
            getattr(self, f"_public_consequence_{label_name}_masks").fill_(False)
        for target_name in ("route", "interaction", "damage"):
            getattr(self, f"_public_consequence_{target_name}_targets").zero_()
            getattr(self, f"_public_consequence_{target_name}_masks").fill_(False)
        self._public_consequence_cant_targets.zero_()
        self._public_consequence_cant_masks.fill_(False)
        self._public_consequence_field_targets.zero_()
        self._public_consequence_field_masks.fill_(False)
        self._team_preview_starts.clear()
        self._team_preview_start_keys.clear()

        if initial_hidden is None:
            return

        h, c = initial_hidden
        if h.shape[:2] != (1, self.num_envs) or c.shape[:2] != (1, self.num_envs):
            raise ValueError(
                "Initial hidden state must have shape (1, num_envs, hidden_dim) for both h and c."
            )

        self._initial_hidden_h = h.squeeze(0).detach().to(self.device).clone()
        self._initial_hidden_c = c.squeeze(0).detach().to(self.device).clone()
