"""PPO trainer with support for action masking and LSTM policies."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ..config import TrainingConfig
from ..env.observation import (
    ABILITY_KNOWN_OFFSET,
    ITEM_KNOWN_OFFSET,
    ROLE_BENCH,
    TERA_KNOWN_OFFSET,
)
from ..model.agent import PokeAgent, load_state_dict_compatible
from .rollout import RolloutBatch


class PPOTrainer:
    """Proximal Policy Optimization with clipped surrogate objective."""

    def __init__(self, agent: PokeAgent, config: TrainingConfig, device: torch.device):
        self.agent = agent.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)
        self.global_step = 0

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        """Run PPO update epochs on a rollout batch.

        Minibatches are slices of environments, not random timesteps, so that
        the LSTM can be unrolled through the full rollout sequence with correct
        gradient flow across episode boundaries (hidden state is zeroed on done).

        Returns a dict of loss metrics for logging.
        """
        T = batch.num_steps
        E = batch.num_envs
        if self.config.num_minibatches > E:
            raise ValueError(
                f"num_minibatches ({self.config.num_minibatches}) cannot exceed num_envs ({E}); "
                "reduce num_minibatches."
            )
        batch_device = next(iter(batch.obs.values())).device
        if batch_device.type != self.device.type:
            raise ValueError(
                f"RolloutBatch is on {batch_device}, but PPOTrainer is on {self.device}. "
                "Collect rollouts on the trainer device."
            )
        if self.device.index is not None and batch_device.index != self.device.index:
            raise ValueError(
                f"RolloutBatch is on {batch_device}, but PPOTrainer is on {self.device}. "
                "Collect rollouts on the trainer device."
            )

        # Normalize advantages over the full batch once, before the update loop.
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_seq = batch.obs
        actions_seq = batch.actions              # (T, E, 3)
        masks_seq = batch.action_masks
        old_lp_seq = batch.log_probs             # (T, E)
        adv_seq = adv                            # (T, E)
        returns_seq = batch.returns              # (T, E)
        values_old_seq = batch.values            # (T, E)
        dones_seq = batch.dones                  # (T, E)

        # Per-env initial hidden states at the start of the rollout.
        if batch.initial_hidden_h is not None:
            init_h_all = batch.initial_hidden_h
            init_c_all = batch.initial_hidden_c
        else:
            init_h_all = None
            init_c_all = None

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        num_updates = 0

        aux_labels_all = batch.aux_labels
        aux_label_valid_all = batch.aux_label_valid
        opp_actions_all = batch.opp_actions
        opp_actions_valid_all = batch.opp_actions_valid
        # Aux losses only run when both the buffer recorded labels AND the
        # agent has the aux heads attached. The test doubles ship a minimal
        # agent without aux heads.
        aux_enabled = bool(aux_labels_all) and hasattr(self.agent, "hidden_info_head") and hasattr(self.agent, "opponent_action_head")
        if aux_enabled:
            metrics["aux_hidden_info_loss"] = 0.0
            metrics["aux_opp_action_loss"] = 0.0

        for epoch in range(self.config.num_epochs):
            env_order = torch.randperm(E, device=self.device)

            for mb_envs in torch.tensor_split(env_order, self.config.num_minibatches):
                if mb_envs.numel() == 0:
                    continue
                E_mb = mb_envs.numel()

                mb_obs_seq = {k: v[:, mb_envs] for k, v in obs_seq.items()}
                mb_actions_seq = actions_seq[:, mb_envs]      # (T, E_mb, 3)
                mb_masks_seq = {key: value[:, mb_envs] for key, value in masks_seq.items()}
                mb_dones_seq = dones_seq[:, mb_envs]          # (T, E_mb)
                mb_old_lp = old_lp_seq[:, mb_envs].reshape(-1)
                mb_adv = adv_seq[:, mb_envs].reshape(-1)
                mb_returns = returns_seq[:, mb_envs].reshape(-1)
                mb_values_old = values_old_seq[:, mb_envs].reshape(-1)

                # Initial hidden for this env subset.
                if init_h_all is not None:
                    hidden: tuple | None = (
                        init_h_all[mb_envs].unsqueeze(0),
                        init_c_all[mb_envs].unsqueeze(0),
                    )
                else:
                    hidden = None

                # Spatial encoder: process all T*E_mb pokemon in one batched pass
                # (heavy attention layers run once; only the lightweight LSTM loops).
                flat_obs = {
                    k: v.reshape(T * E_mb, *v.shape[2:])
                    for k, v in mb_obs_seq.items()
                }
                context_flat, active_reps_flat, own_reps_flat, opp_reps_flat, opp_active_reps_flat = self.agent.spatial(flat_obs)
                context_seq_mb = context_flat.reshape(T, E_mb, -1)
                active_reps_seq_mb = active_reps_flat.reshape(T, E_mb, 2, -1)
                own_reps_seq_mb = own_reps_flat.reshape(T, E_mb, own_reps_flat.shape[1], -1)
                opp_reps_seq_mb = opp_reps_flat.reshape(T, E_mb, opp_reps_flat.shape[1], -1)
                opp_active_reps_seq_mb = opp_active_reps_flat.reshape(T, E_mb, 2, -1)
                own_bench_mask_seq_mb = (mb_obs_seq["own_roles"] != ROLE_BENCH) | mb_obs_seq["own_mask"]
                opp_bench_mask_seq_mb = (mb_obs_seq["opp_roles"] != ROLE_BENCH) | mb_obs_seq["opp_mask"]

                # LSTM + heads: step through the sequence one turn at a time.
                # Gradients flow back through all turns within each non-terminal
                # episode segment; the hidden is zeroed at episode boundaries.
                all_log_probs: list[torch.Tensor] = []
                all_entropies: list[torch.Tensor] = []
                all_values: list[torch.Tensor] = []
                all_ctx: list[torch.Tensor] = []

                for t in range(T):
                    ctx_t, hidden = self.agent.memory(context_seq_mb[t], hidden)
                    all_ctx.append(ctx_t)

                    action_out = self.agent.action_head(
                        ctx_t,
                        active_reps_seq_mb[t],
                        own_reps_seq_mb[t],
                        opp_reps_seq_mb[t],
                        opp_bench_mask_seq_mb[t],
                        opp_active_reps_seq_mb[t],
                        {key: value[t] for key, value in mb_masks_seq.items()},
                        mb_actions_seq[t],
                        battle_obs={key: value[t] for key, value in mb_obs_seq.items()},
                    )
                    value_t = self.agent.value_head(
                        ctx_t,
                        own_reps_seq_mb[t],
                        opp_reps_seq_mb[t],
                        own_bench_mask_seq_mb[t],
                        opp_bench_mask_seq_mb[t],
                    )

                    all_log_probs.append(action_out["log_prob"])
                    all_entropies.append(action_out["entropy"])
                    all_values.append(value_t)

                    # Zero hidden state at episode boundaries without host-side syncs.
                    if hidden is not None:
                        not_done = (~mb_dones_seq[t]).to(ctx_t.dtype).view(1, E_mb, 1)
                        h, c = hidden
                        hidden = (h * not_done, c * not_done)
                        if (
                            self.config.bptt_chunk_steps is not None
                            and self.config.bptt_chunk_steps > 0
                            and (t + 1) % self.config.bptt_chunk_steps == 0
                            and t + 1 < T
                        ):
                            hidden = tuple(state.detach() for state in hidden)

                # Flatten (T, E_mb) → (T*E_mb,) for loss computation.
                log_probs_flat = torch.stack(all_log_probs).reshape(-1)
                entropies_flat = torch.stack(all_entropies).reshape(-1)
                values_flat = torch.stack(all_values).reshape(-1)

                # Policy loss (clipped surrogate objective).
                log_ratio = log_probs_flat - mb_old_lp
                ratio = log_ratio.exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with clipping (prevents large value updates).
                values_clipped = mb_values_old + torch.clamp(
                    values_flat - mb_values_old,
                    -self.config.clip_eps,
                    self.config.clip_eps,
                )
                value_loss = 0.5 * torch.max(
                    (values_flat - mb_returns).pow(2),
                    (values_clipped - mb_returns).pow(2),
                ).mean()

                entropy_loss = -entropies_flat.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # --- Aux losses ---
                # Hidden info: predict end-of-episode opponent item/ability/
                # tera-type/move-set. Opp action: predict the opp's chosen
                # (a1, a2, tera) using memory output plus both-side actives.
                # Labels come from the buffer's retroactive fill on done.
                hidden_info_loss = None
                opp_action_loss = None
                if aux_enabled:
                    aux_labels_mb = {key: value[:, mb_envs] for key, value in aux_labels_all.items()}
                    aux_label_valid_mb = aux_label_valid_all[:, mb_envs]
                    hidden_info_out = self.agent.hidden_info_head(opp_reps_seq_mb)
                    hidden_info_loss = self._hidden_info_loss(
                        logits=hidden_info_out,
                        labels=aux_labels_mb,
                        label_valid=aux_label_valid_mb,
                    )

                    ctx_seq_mb = torch.stack(all_ctx, dim=0)  # (T, E_mb, mem_dim)
                    opp_action_out = self.agent.opponent_action_head(
                        ctx_seq_mb.reshape(T * E_mb, -1),
                        opp_active_reps_seq_mb.reshape(T * E_mb, 2, -1),
                        active_reps_seq_mb.reshape(T * E_mb, 2, -1),
                    )
                    opp_actions_mb = opp_actions_all[:, mb_envs]
                    opp_actions_valid_mb = opp_actions_valid_all[:, mb_envs]
                    opp_action_loss = self._opp_action_loss(
                        logits=opp_action_out,
                        labels=opp_actions_mb,
                        valid=opp_actions_valid_mb,
                    )

                    loss = (
                        loss
                        + self.config.aux_hidden_info_coef * hidden_info_loss
                        + self.config.aux_opp_action_coef * opp_action_loss
                    )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                if aux_enabled:
                    metrics["aux_hidden_info_loss"] += hidden_info_loss.item()
                    metrics["aux_opp_action_loss"] += opp_action_loss.item()
                num_updates += 1

        self.global_step += T * E
        return {k: v / num_updates for k, v in metrics.items()}

    @staticmethod
    def _hidden_info_loss(
        logits: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        label_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Masked CE (item/ability/tera) + BCE (moves) against end-of-episode labels."""
        opp_pad = labels["opp_mask"]  # (T, E_mb, 6) True if padding slot
        opp_cont = labels["opp_continuous"]  # (T, E_mb, 6, cont_dim)
        item_known = opp_cont[..., ITEM_KNOWN_OFFSET] > 0.5
        ability_known = opp_cont[..., ABILITY_KNOWN_OFFSET] > 0.5
        tera_known = opp_cont[..., TERA_KNOWN_OFFSET] > 0.5

        valid_step = label_valid.unsqueeze(-1).expand_as(opp_pad)  # (T, E_mb, 6)
        base_mask = valid_step & ~opp_pad

        item_mask = (base_mask & item_known).reshape(-1)
        ability_mask = (base_mask & ability_known).reshape(-1)
        # Tera-type label is types[..., 2]; treat 0 as "no tera known" rather than a valid class.
        tera_types = labels["opp_types"][..., 2].reshape(-1)
        tera_mask = (base_mask & tera_known).reshape(-1) & (tera_types > 0)

        item_logits = logits["item_logits"].reshape(-1, logits["item_logits"].shape[-1])
        ability_logits = logits["ability_logits"].reshape(-1, logits["ability_logits"].shape[-1])
        tera_logits = logits["tera_type_logits"].reshape(-1, logits["tera_type_logits"].shape[-1])
        move_logits = logits["move_logits"]  # (T, E_mb, 6, num_moves)

        item_labels = labels["opp_item"].reshape(-1)
        ability_labels = labels["opp_ability"].reshape(-1)

        zero = torch.zeros((), device=item_logits.device)
        item_loss = (
            F.cross_entropy(item_logits[item_mask], item_labels[item_mask])
            if item_mask.any()
            else zero
        )
        ability_loss = (
            F.cross_entropy(ability_logits[ability_mask], ability_labels[ability_mask])
            if ability_mask.any()
            else zero
        )
        tera_loss = (
            F.cross_entropy(tera_logits[tera_mask], tera_types[tera_mask])
            if tera_mask.any()
            else zero
        )

        # Multi-hot move labels from slot indices. Slot value 0 is "no move" and
        # is not a valid move id, so we ignore it. A pokemon's label is valid if
        # at least one move slot is > 0.
        move_slots = labels["opp_moves"]  # (T, E_mb, 6, 4)
        num_moves = move_logits.shape[-1]
        multihot = torch.zeros_like(move_logits)
        valid_slot = move_slots > 0
        safe_slots = move_slots.clamp(min=0, max=num_moves - 1)
        multihot.scatter_(-1, safe_slots, valid_slot.float())
        # Zero out slot 0 contribution (it was set to 1.0 anywhere safe_slots==0).
        multihot[..., 0] = 0.0

        move_slot_mask = base_mask & valid_slot.any(dim=-1)  # (T, E_mb, 6)
        move_loss_raw = F.binary_cross_entropy_with_logits(move_logits, multihot, reduction="none")
        # Mean across the move vocab for each pokemon, then mask-average.
        per_pokemon_bce = move_loss_raw.mean(dim=-1)
        denom = move_slot_mask.float().sum().clamp(min=1.0)
        move_loss = (per_pokemon_bce * move_slot_mask.float()).sum() / denom

        return item_loss + ability_loss + tera_loss + move_loss

    @staticmethod
    def _opp_action_loss(
        logits: dict[str, torch.Tensor],
        labels: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Masked CE on opp a1 / a2 / tera using observed opponent actions."""
        labels_flat = labels.reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        a1_labels = labels_flat[:, 0]
        a2_labels = labels_flat[:, 1]
        tera_labels = labels_flat[:, 2]

        a1_mask = valid_flat & (a1_labels >= 0)
        a2_mask = valid_flat & (a2_labels >= 0)
        tera_mask = valid_flat & (tera_labels >= 0)

        zero = torch.zeros((), device=logits["opp_a1_logits"].device)
        a1_loss = (
            F.cross_entropy(logits["opp_a1_logits"][a1_mask], a1_labels[a1_mask])
            if a1_mask.any()
            else zero
        )
        a2_loss = (
            F.cross_entropy(logits["opp_a2_logits"][a2_mask], a2_labels[a2_mask])
            if a2_mask.any()
            else zero
        )
        tera_loss = (
            F.cross_entropy(logits["opp_tera_logits"][tera_mask], tera_labels[tera_mask])
            if tera_mask.any()
            else zero
        )
        return a1_loss + a2_loss + tera_loss

    def update_team_preview(
        self,
        data: list[tuple[dict, np.ndarray, float, np.ndarray]],
    ) -> dict[str, float]:
        """REINFORCE update for team preview decisions.

        Called once per PPO update cycle using all team preview decisions made
        during the rollout. Each entry is ``(obs_dict, picks_np, episode_reward,
        opponent_picks_np)`` where episode_reward is +1 for a win and -1 for a loss.

        Uses the value function as a baseline to reduce gradient variance.

        Returns a dict of preview-loss metrics for logging.
        """
        obs_batch = {
            k: torch.from_numpy(np.stack([d[0][k] for d in data])).to(self.device)
            for k in data[0][0]
        }
        picks_batch = torch.from_numpy(np.stack([d[1] for d in data])).long().to(self.device)
        rewards_batch = torch.tensor([d[2] for d in data], dtype=torch.float32, device=self.device)
        opponent_picks_batch = torch.from_numpy(np.stack([d[3] for d in data])).long().to(self.device)

        out = self.agent.team_preview(obs_batch, picks=picks_batch)
        log_prob = out["log_prob"]
        entropy = out["entropy"]
        baselines = out["preview_value"]
        bring_logits = out["opponent_bring_logits"]
        lead_pair_logits = out["opponent_lead_pair_logits"]

        # Preview actor-critic: use a dedicated preview value head as the baseline.
        advantages = rewards_batch - baselines.detach()
        policy_loss = -(log_prob * advantages).mean()
        entropy_bonus = self.config.entropy_coef * entropy.mean()
        value_loss = 0.5 * (baselines - rewards_batch).pow(2).mean()

        team_size = bring_logits.shape[1]
        bring_targets = torch.zeros_like(bring_logits)
        bring_targets.scatter_(1, opponent_picks_batch, 1.0)
        valid_slots = (~obs_batch["opp_mask"]).to(bring_logits.dtype)
        bring_loss_raw = F.binary_cross_entropy_with_logits(bring_logits, bring_targets, reduction="none")
        bring_loss = (bring_loss_raw * valid_slots).sum() / valid_slots.sum().clamp(min=1.0)

        lead_targets = opponent_picks_batch[:, 0] * team_size + opponent_picks_batch[:, 1]
        lead_loss = F.cross_entropy(lead_pair_logits, lead_targets)

        loss = (
            policy_loss
            - entropy_bonus
            + self.config.team_preview_value_coef * value_loss
            + self.config.team_preview_bring_coef * bring_loss
            + self.config.team_preview_lead_coef * lead_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "team_preview_loss": loss.item(),
            "team_preview_policy_loss": policy_loss.item(),
            "team_preview_value_loss": value_loss.item(),
            "team_preview_entropy": entropy.mean().item(),
            "team_preview_bring_loss": bring_loss.item(),
            "team_preview_lead_loss": lead_loss.item(),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        load_report = load_state_dict_compatible(self.agent, checkpoint["agent"])
        optimizer_compatible = not any(load_report.values())
        if optimizer_compatible:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.config.lr, eps=1e-5)
            print(
                "Checkpoint architecture changed; loaded compatible agent weights only. "
                f"Skipped={len(load_report['skipped_keys'])}, "
                f"Missing={len(load_report['missing_keys'])}, "
                f"Unexpected={len(load_report['unexpected_keys'])}. "
                "Optimizer state was reinitialized."
            )
        self.global_step = checkpoint["global_step"]
