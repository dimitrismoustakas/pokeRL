"""PPO trainer with support for action masking and LSTM policies."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from ..config import TrainingConfig
from ..model.agent import PokeAgent
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

        Returns a dict of loss metrics for logging.
        """
        batch_size = batch.actions.shape[0]
        minibatch_size = batch_size // self.config.num_minibatches

        # Normalize advantages
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "approx_kl": 0}
        num_updates = 0

        for epoch in range(self.config.num_epochs):
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                mb_obs = {k: v[mb_idx].to(self.device) for k, v in batch.obs.items()}
                mb_actions = batch.actions[mb_idx].to(self.device)
                mb_masks = batch.action_masks[mb_idx].to(self.device).bool()
                mb_old_log_probs = batch.log_probs[mb_idx].to(self.device)
                mb_advantages = adv[mb_idx].to(self.device)
                mb_returns = batch.returns[mb_idx].to(self.device)

                # Reconstruct hidden state if using LSTM
                hidden = None
                if batch.hidden_h is not None:
                    h = batch.hidden_h[mb_idx].unsqueeze(0).to(self.device)
                    c = batch.hidden_c[mb_idx].unsqueeze(0).to(self.device)
                    hidden = (h, c)

                # Forward pass: evaluate the stored actions
                out = self.agent(mb_obs, hidden, actions=mb_actions, action_masks=mb_masks)

                # Policy loss (clipped surrogate)
                log_ratio = out["log_prob"] - mb_old_log_probs
                ratio = log_ratio.exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (simple MSE, could clip too)
                value_loss = 0.5 * (out["value"] - mb_returns).pow(2).mean()

                # Entropy bonus
                entropy_loss = -out["entropy"].mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["approx_kl"] += approx_kl
                num_updates += 1

        self.global_step += batch_size

        return {k: v / num_updates for k, v in metrics.items()}

    def update_team_preview(self, data: list[tuple[dict, np.ndarray, float]]) -> float:
        """REINFORCE update for team preview decisions.

        Called once per PPO update cycle using all team preview decisions made
        during the rollout. Each entry is (obs_dict, picks_np, episode_reward)
        where episode_reward is +1 for a win and -1 for a loss.

        Returns the loss value for logging.
        """
        obs_batch = {
            k: torch.from_numpy(np.stack([d[0][k] for d in data])).to(self.device)
            for k in data[0][0]
        }
        picks_batch = torch.from_numpy(np.stack([d[1] for d in data])).long().to(self.device)
        rewards_batch = torch.tensor([d[2] for d in data], dtype=torch.float32, device=self.device)

        out = self.agent.team_preview(obs_batch, picks=picks_batch)
        log_prob = out["log_prob"]
        entropy = out["entropy"]

        # REINFORCE: minimize -E[log_prob * reward]; add entropy bonus to encourage exploration.
        loss = -(log_prob * rewards_batch).mean() - self.config.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "agent": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(checkpoint["agent"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
