"""PPO trainer with support for action masking and LSTM policies."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from ..config import TrainingConfig
from ..env.observation import ROLE_BENCH
from ..model.agent import PokeAgent, load_state_dict_compatible
from .rollout import RolloutBatch


class PPOTrainer:
    """Proximal Policy Optimization with clipped surrogate objective."""

    _LOG_RATIO_CLAMP = 20.0

    def __init__(self, agent: PokeAgent, config: TrainingConfig, device: torch.device):
        self.agent = agent.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)
        self.global_step = 0

    def _has_finite_gradients(self) -> bool:
        for param in self.agent.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
        return True

    def _apply_update_step(self, loss: torch.Tensor) -> bool:
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return False

        self.optimizer.zero_grad()
        loss.backward()
        if not self._has_finite_gradients():
            self.optimizer.zero_grad()
            return False

        total_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
        if not torch.isfinite(total_norm):
            self.optimizer.zero_grad()
            return False

        self.optimizer.step()
        return True

    @staticmethod
    def _skipped_preview_metrics() -> dict[str, float]:
        return {
            "team_preview_loss": 0.0,
            "team_preview_policy_loss": 0.0,
            "team_preview_value_loss": 0.0,
            "team_preview_entropy": 0.0,
            "team_preview_bring_loss": 0.0,
            "team_preview_lead_loss": 0.0,
            "team_preview_skipped_nonfinite": 1.0,
        }

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

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        num_updates = 0
        skipped_nonfinite = 0

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

                for t in range(T):
                    ctx_t, hidden = self.agent.memory(context_seq_mb[t], hidden)

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
                # Rare extreme policy shifts can overflow exp(log_ratio) in float32 and
                # poison Adam with non-finite gradients. Clamping here only affects
                # already-pathological ratios far outside PPO's trust region.
                ratio = torch.exp(log_ratio.clamp(min=-self._LOG_RATIO_CLAMP, max=self._LOG_RATIO_CLAMP))
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

                if not self._apply_update_step(loss):
                    skipped_nonfinite += 1
                    continue

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio.clamp(
                        min=-self._LOG_RATIO_CLAMP,
                        max=self._LOG_RATIO_CLAMP,
                    )).mean().item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                num_updates += 1

        if num_updates == 0:
            raise RuntimeError(
                "PPO update skipped every minibatch because the loss or gradients became non-finite."
            )

        self.global_step += T * E
        averaged = {k: v / num_updates for k, v in metrics.items()}
        if skipped_nonfinite > 0:
            averaged["skipped_nonfinite_minibatches"] = float(skipped_nonfinite)
        return averaged

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

        invalid_tensor_names = [
            name
            for name, tensor in {
                "log_prob": log_prob,
                "entropy": entropy,
                "preview_value": baselines,
                "opponent_bring_logits": bring_logits,
                "opponent_lead_pair_logits": lead_pair_logits,
            }.items()
            if not torch.isfinite(tensor).all()
        ]
        if invalid_tensor_names:
            print(
                "Warning: skipped team preview update due to non-finite tensors: "
                + ", ".join(invalid_tensor_names)
            )
            return self._skipped_preview_metrics()

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

        invalid_loss_names = [
            name
            for name, tensor in {
                "team_preview_loss": loss,
                "team_preview_policy_loss": policy_loss,
                "team_preview_value_loss": value_loss,
                "team_preview_entropy": entropy.mean(),
                "team_preview_bring_loss": bring_loss,
                "team_preview_lead_loss": lead_loss,
            }.items()
            if not torch.isfinite(tensor).all()
        ]
        if invalid_loss_names:
            print(
                "Warning: skipped team preview update due to non-finite losses: "
                + ", ".join(invalid_loss_names)
            )
            return self._skipped_preview_metrics()

        if not self._apply_update_step(loss):
            print("Warning: skipped team preview update due to non-finite loss or gradients.")
            return self._skipped_preview_metrics()

        return {
            "team_preview_loss": loss.item(),
            "team_preview_policy_loss": policy_loss.item(),
            "team_preview_value_loss": value_loss.item(),
            "team_preview_entropy": entropy.mean().item(),
            "team_preview_bring_loss": bring_loss.item(),
            "team_preview_lead_loss": lead_loss.item(),
            "team_preview_skipped_nonfinite": 0.0,
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
