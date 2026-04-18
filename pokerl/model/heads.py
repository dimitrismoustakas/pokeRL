"""Action head (autoregressive for doubles) and value head."""

import torch
import torch.nn as nn

from ..env.actions import ACTIONS_PER_SLOT, FIRST_SLOT_MASK_ROW


class ActionHead(nn.Module):
    """Autoregressive action selection for doubles.

    Both heads see BOTH active pokemon representations so they can
    coordinate. Head 1 sees [context, rep1, rep2], head 2 additionally
    sees the embedding of pokemon 1's chosen action.
    """

    def __init__(self, context_dim: int, pokemon_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = ACTIONS_PER_SLOT

        # Embeds a chosen action so the second head can condition on it
        self.action_emb = nn.Embedding(self.num_actions, hidden_dim)

        # First pokemon: sees context + own rep + partner rep
        self.head1 = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

        # Second pokemon: sees context + own rep + partner rep + partner's action
        self.head2 = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(
        self,
        context: torch.Tensor,        # (batch, context_dim)
        active_reps: torch.Tensor,     # (batch, 2, pokemon_dim)
        masks: torch.Tensor,           # (batch, num_actions + 1, num_actions)
        actions: torch.Tensor | None = None,  # (batch, 2) if provided, evaluate these
    ) -> dict[str, torch.Tensor]:
        """
        When actions is None: sample actions (for rollout).
        When actions is provided: evaluate log-probs and entropy (for PPO update).
        """
        rep1 = active_reps[:, 0]
        rep2 = active_reps[:, 1]
        mask1 = masks[:, FIRST_SLOT_MASK_ROW]

        # --- Pokemon 1 (sees both active reps for coordination) ---
        logits1 = self.head1(torch.cat([context, rep1, rep2], dim=-1))
        logits1 = logits1.masked_fill(~mask1, -1e8)
        dist1 = torch.distributions.Categorical(logits=logits1)

        if actions is None:
            a1 = dist1.sample()
        else:
            a1 = actions[:, 0]

        if not mask1.gather(1, a1.unsqueeze(1)).squeeze(1).all():
            raise RuntimeError(
                "ActionHead received an illegal first-slot action for the current joint mask."
            )

        # --- Pokemon 2 (sees both reps + pokemon 1's action) ---
        batch_indices = torch.arange(context.shape[0], device=context.device)
        mask2 = masks[batch_indices, a1 + 1]
        if not mask2.any(dim=1).all():
            raise RuntimeError(
                "ActionHead received an empty conditional joint action mask for pokemon 2. "
                "This means build_joint_action_mask() produced an inconsistent legality table."
            )

        a1_emb = self.action_emb(a1)
        logits2 = self.head2(torch.cat([context, rep2, rep1, a1_emb], dim=-1))
        logits2 = logits2.masked_fill(~mask2, -1e8)
        dist2 = torch.distributions.Categorical(logits=logits2)

        if actions is None:
            a2 = dist2.sample()
        else:
            a2 = actions[:, 1]

        if not mask2.gather(1, a2.unsqueeze(1)).squeeze(1).all():
            raise RuntimeError(
                "ActionHead received an illegal second-slot action for the current joint mask."
            )

        # Joint log-prob: log P(a1, a2) = log P(a1) + log P(a2 | a1)
        log_prob = dist1.log_prob(a1) + dist2.log_prob(a2)
        entropy = dist1.entropy() + dist2.entropy()

        return {
            "actions": torch.stack([a1, a2], dim=1),  # (batch, 2)
            "log_prob": log_prob,                       # (batch,)
            "entropy": entropy,                         # (batch,)
        }


class TeamPreviewHead(nn.Module):
    """Autoregressive team selection for team preview (pick 4 from 6)."""

    def __init__(self, pokemon_dim: int, context_dim: int, pick_size: int = 4):
        super().__init__()
        self.pick_size = pick_size
        # Scores each pokemon for selection, conditioned on context + already picked
        self.score = nn.Sequential(
            nn.Linear(pokemon_dim + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Encodes the "already picked" set -- simple: sum of picked pokemon reps
        self.context_proj = nn.Linear(context_dim + pokemon_dim, context_dim)

    def forward(
        self,
        pokemon_reps: torch.Tensor,    # (batch, 6, pokemon_dim)
        context: torch.Tensor,         # (batch, context_dim)
        picks: torch.Tensor | None = None,  # (batch, pick_size) if evaluating
    ) -> dict[str, torch.Tensor]:
        batch, team_size, pdim = pokemon_reps.shape

        all_log_probs = []
        all_entropies = []
        all_picks = []
        available = torch.ones(batch, team_size, dtype=torch.bool, device=context.device)
        picked_sum = torch.zeros(batch, pdim, device=context.device)

        for step in range(self.pick_size):
            # Context incorporates what we've already picked
            ctx = self.context_proj(torch.cat([context, picked_sum], dim=-1))

            # Score each available pokemon
            ctx_expanded = ctx.unsqueeze(1).expand(-1, team_size, -1)
            inputs = torch.cat([pokemon_reps, ctx_expanded], dim=-1)
            scores = self.score(inputs).squeeze(-1)  # (batch, team_size)
            scores = scores.masked_fill(~available, -1e8)

            dist = torch.distributions.Categorical(logits=scores)

            if picks is None:
                chosen = dist.sample()
            else:
                chosen = picks[:, step]

            all_picks.append(chosen)
            all_log_probs.append(dist.log_prob(chosen))
            all_entropies.append(dist.entropy())

            # Update: mark chosen as unavailable, add to picked sum
            available.scatter_(1, chosen.unsqueeze(1), False)
            chosen_rep = pokemon_reps.gather(
                1, chosen.unsqueeze(1).unsqueeze(2).expand(-1, -1, pdim)
            ).squeeze(1)
            picked_sum = picked_sum + chosen_rep

        return {
            "picks": torch.stack(all_picks, dim=1),            # (batch, pick_size)
            "log_prob": torch.stack(all_log_probs, dim=1).sum(1),  # (batch,)
            "entropy": torch.stack(all_entropies, dim=1).sum(1),   # (batch,)
        }


class ValueHead(nn.Module):
    """Estimates V(s) -- the probability of winning from the current state."""

    def __init__(self, context_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.mlp(context).squeeze(-1)
