"""Action head (autoregressive for doubles) and value head."""

import math

import torch
import torch.nn as nn

from ..env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    MAX_TEAM_SLOTS,
    NUM_MOVES,
    NUM_TARGETS,
    TARGET_NONE,
    decode_action,
)


class MaskedSetAttention(nn.Module):
    """Query-conditioned pooling over a masked set of pokemon representations."""

    def __init__(self, query_dim: int, item_dim: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(item_dim, hidden_dim)
        self.value_proj = nn.Linear(item_dim, item_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        items: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_items, _ = items.shape
        if query.shape[0] != batch_size:
            raise ValueError(
                f"MaskedSetAttention expected batch-aligned query/items, got "
                f"{query.shape[0]} queries for {batch_size} item sets."
            )
        if mask is not None and mask.shape != (batch_size, num_items):
            raise ValueError(
                f"MaskedSetAttention expected mask shape {(batch_size, num_items)}, got {tuple(mask.shape)}."
            )

        query_proj = self.query_proj(query).unsqueeze(1)
        key_proj = self.key_proj(items)
        scores = (query_proj * key_proj).sum(dim=-1) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, -1e8)

        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights.masked_fill(mask, 0.0)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(weights_sum > 0, weights / weights_sum.clamp(min=1e-8), weights)

        values = self.value_proj(items)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1)


class ActionHead(nn.Module):
    """Autoregressive action selection for doubles.

    Each slot scores every fixed action against the live battle entities:
    the acting pokemon, its partner, both opponent actives, and the action's
    explicit target or switch destination. The second slot additionally sees
    the first slot's chosen action embedding so the joint decision remains
    coordinated.
    """

    def __init__(self, context_dim: int, pokemon_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.num_actions = ACTIONS_PER_SLOT

        self.action_id_emb = nn.Embedding(self.num_actions, hidden_dim)
        self.kind_emb = nn.Embedding(4, hidden_dim)
        self.move_slot_emb = nn.Embedding(NUM_MOVES + 1, hidden_dim)
        self.target_emb = nn.Embedding(NUM_TARGETS + 1, hidden_dim)
        self.switch_slot_emb = nn.Embedding(MAX_TEAM_SLOTS + 1, hidden_dim)

        self.query1 = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 4, hidden_dim),
            nn.ReLU(),
        )
        self.query2 = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 4 + hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.reserve_attention = MaskedSetAttention(hidden_dim, pokemon_dim, hidden_dim)
        self.query1_fuse = nn.Sequential(
            nn.Linear(hidden_dim + pokemon_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query2_fuse = nn.Sequential(
            nn.Linear(hidden_dim + pokemon_dim, hidden_dim),
            nn.ReLU(),
        )
        self.entity_proj = nn.Sequential(
            nn.Linear(pokemon_dim, hidden_dim),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        metadata = self._build_action_metadata()
        self.register_buffer("_action_ids", metadata["action_ids"], persistent=False)
        self.register_buffer("_action_kind", metadata["kind"], persistent=False)
        self.register_buffer("_action_move_slot", metadata["move_slot"], persistent=False)
        self.register_buffer("_action_target_slot", metadata["target_slot"], persistent=False)
        self.register_buffer("_action_switch_slot", metadata["switch_slot"], persistent=False)
        self.register_buffer("_action_is_switch", metadata["is_switch"], persistent=False)

    def forward(
        self,
        context: torch.Tensor,
        active_reps: torch.Tensor,
        own_team_reps: torch.Tensor,
        opp_team_reps: torch.Tensor,
        opp_reserve_mask: torch.Tensor,
        opp_active_reps: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        rep1 = active_reps[:, 0]
        rep2 = active_reps[:, 1]
        opp1 = opp_active_reps[:, 0]
        opp2 = opp_active_reps[:, 1]
        mask1 = masks[:, FIRST_SLOT_MASK_ROW]

        base_query1 = self.query1(torch.cat([context, rep1, rep2, opp1, opp2], dim=-1))
        reserve_summary1 = self.reserve_attention(base_query1, opp_team_reps, opp_reserve_mask)
        query1 = self.query1_fuse(torch.cat([base_query1, reserve_summary1], dim=-1))
        logits1 = self._score_actions(query1, rep1, rep2, own_team_reps, opp_active_reps)
        logits1 = logits1.masked_fill(~mask1, -1e8)
        dist1 = torch.distributions.Categorical(logits=logits1)

        if actions is None:
            a1 = logits1.argmax(dim=1) if greedy else dist1.sample()
        else:
            a1 = actions[:, 0]

        if not mask1.gather(1, a1.unsqueeze(1)).squeeze(1).all():
            raise RuntimeError(
                "ActionHead received an illegal first-slot action for the current joint mask."
            )

        batch_indices = torch.arange(context.shape[0], device=context.device)
        mask2 = masks[batch_indices, a1 + 1]
        if not mask2.any(dim=1).all():
            raise RuntimeError(
                "ActionHead received an empty conditional joint action mask for pokemon 2. "
                "This means build_joint_action_mask() produced an inconsistent legality table."
            )

        a1_emb = self.action_id_emb(a1)
        base_query2 = self.query2(torch.cat([context, rep2, rep1, opp1, opp2, a1_emb], dim=-1))
        reserve_summary2 = self.reserve_attention(base_query2, opp_team_reps, opp_reserve_mask)
        query2 = self.query2_fuse(torch.cat([base_query2, reserve_summary2], dim=-1))
        logits2 = self._score_actions(query2, rep2, rep1, own_team_reps, opp_active_reps)
        logits2 = logits2.masked_fill(~mask2, -1e8)
        dist2 = torch.distributions.Categorical(logits=logits2)

        if actions is None:
            a2 = logits2.argmax(dim=1) if greedy else dist2.sample()
        else:
            a2 = actions[:, 1]

        if not mask2.gather(1, a2.unsqueeze(1)).squeeze(1).all():
            raise RuntimeError(
                "ActionHead received an illegal second-slot action for the current joint mask."
            )

        log_prob = dist1.log_prob(a1) + dist2.log_prob(a2)
        entropy = dist1.entropy() + dist2.entropy()

        return {
            "actions": torch.stack([a1, a2], dim=1),
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def _score_actions(
        self,
        query: torch.Tensor,
        actor_rep: torch.Tensor,
        partner_rep: torch.Tensor,
        own_team_reps: torch.Tensor,
        opp_active_reps: torch.Tensor,
    ) -> torch.Tensor:
        query_features = query.unsqueeze(1).expand(-1, self.num_actions, -1)
        action_features = self._static_action_features(query.device).unsqueeze(0).expand(query.shape[0], -1, -1)
        entity_features = self.entity_proj(
            self._dynamic_action_entities(actor_rep, partner_rep, own_team_reps, opp_active_reps)
        )
        return self.score(torch.cat([query_features, action_features, entity_features], dim=-1)).squeeze(-1)

    def _static_action_features(self, device: torch.device) -> torch.Tensor:
        action_ids = self._action_ids.to(device)
        kind = self._action_kind.to(device)
        move_slot = self._action_move_slot.to(device)
        target_slot = self._action_target_slot.to(device)
        switch_slot = self._action_switch_slot.to(device)
        return (
            self.action_id_emb(action_ids)
            + self.kind_emb(kind)
            + self.move_slot_emb(move_slot)
            + self.target_emb(target_slot)
            + self.switch_slot_emb(switch_slot)
        )

    def _dynamic_action_entities(
        self,
        actor_rep: torch.Tensor,
        partner_rep: torch.Tensor,
        own_team_reps: torch.Tensor,
        opp_active_reps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, pokemon_dim = actor_rep.shape
        zero_rep = torch.zeros_like(actor_rep)

        target_bank = torch.stack(
            [
                opp_active_reps[:, 0],
                opp_active_reps[:, 1],
                partner_rep,
                actor_rep,
                zero_rep,
                zero_rep,
            ],
            dim=1,
        )
        target_indices = self._action_target_slot.view(1, self.num_actions, 1).expand(batch_size, -1, pokemon_dim)
        target_entities = target_bank.gather(1, target_indices)

        switch_bank = torch.cat([own_team_reps, zero_rep.unsqueeze(1)], dim=1)
        switch_indices = self._action_switch_slot.view(1, self.num_actions, 1).expand(batch_size, -1, pokemon_dim)
        switch_entities = switch_bank.gather(1, switch_indices)

        is_switch = self._action_is_switch.view(1, self.num_actions, 1).to(device=actor_rep.device)
        return torch.where(is_switch, switch_entities, target_entities)

    @staticmethod
    def _build_action_metadata() -> dict[str, torch.Tensor]:
        action_ids = []
        kinds = []
        move_slots = []
        target_slots = []
        switch_slots = []
        is_switch = []

        move_none_slot = NUM_MOVES
        target_none_slot = NUM_TARGETS
        switch_none_slot = MAX_TEAM_SLOTS
        kind_lookup = {
            ("move", False): 0,
            ("move", True): 1,
            ("switch", False): 2,
            ("pass", False): 3,
        }

        for action_idx in range(ACTIONS_PER_SLOT):
            kind, slot_idx, target_idx, tera = decode_action(action_idx)
            action_ids.append(action_idx)
            kinds.append(kind_lookup[(kind, tera if kind == "move" else False)])
            move_slots.append(slot_idx if kind == "move" else move_none_slot)
            target_slots.append(target_idx if kind == "move" else (TARGET_NONE if kind == "pass" else target_none_slot))
            switch_slots.append(slot_idx if kind == "switch" else switch_none_slot)
            is_switch.append(kind == "switch")

        return {
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "kind": torch.tensor(kinds, dtype=torch.long),
            "move_slot": torch.tensor(move_slots, dtype=torch.long),
            "target_slot": torch.tensor(target_slots, dtype=torch.long),
            "switch_slot": torch.tensor(switch_slots, dtype=torch.long),
            "is_switch": torch.tensor(is_switch, dtype=torch.bool),
        }


class TeamPreviewHead(nn.Module):
    """Autoregressive team selection for team preview (pick 4 from 6)."""

    def __init__(self, pokemon_dim: int, context_dim: int, pick_size: int = 4):
        super().__init__()
        self.pick_size = pick_size
        self.opponent_attention = MaskedSetAttention(context_dim, pokemon_dim, 128)
        self.score = nn.Sequential(
            nn.Linear(pokemon_dim * 2 + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.context_proj = nn.Linear(context_dim + pokemon_dim, context_dim)

    def forward(
        self,
        pokemon_reps: torch.Tensor,
        opponent_reps: torch.Tensor,
        context: torch.Tensor,
        team_mask: torch.Tensor | None = None,
        opponent_mask: torch.Tensor | None = None,
        picks: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch, team_size, pdim = pokemon_reps.shape

        all_log_probs = []
        all_entropies = []
        all_picks = []
        if team_mask is None:
            available = torch.ones(batch, team_size, dtype=torch.bool, device=context.device)
        else:
            available = ~team_mask.clone()
        if (available.sum(dim=1) < self.pick_size).any():
            raise ValueError(
                f"TeamPreviewHead needs at least {self.pick_size} available pokemon per batch item."
            )
        picked_sum = torch.zeros(batch, pdim, device=context.device)

        for step in range(self.pick_size):
            ctx = self.context_proj(torch.cat([context, picked_sum], dim=-1))
            opponent_summary = self.opponent_attention(ctx, opponent_reps, opponent_mask)

            ctx_expanded = ctx.unsqueeze(1).expand(-1, team_size, -1)
            opponent_expanded = opponent_summary.unsqueeze(1).expand(-1, team_size, -1)
            inputs = torch.cat([pokemon_reps, ctx_expanded, opponent_expanded], dim=-1)
            scores = self.score(inputs).squeeze(-1)
            scores = scores.masked_fill(~available, -1e8)

            dist = torch.distributions.Categorical(logits=scores)

            if picks is None:
                chosen = scores.argmax(dim=1) if greedy else dist.sample()
            else:
                chosen = picks[:, step]

            all_picks.append(chosen)
            all_log_probs.append(dist.log_prob(chosen))
            all_entropies.append(dist.entropy())

            available.scatter_(1, chosen.unsqueeze(1), False)
            chosen_rep = pokemon_reps.gather(
                1, chosen.unsqueeze(1).unsqueeze(2).expand(-1, -1, pdim)
            ).squeeze(1)
            picked_sum = picked_sum + chosen_rep

        return {
            "picks": torch.stack(all_picks, dim=1),
            "log_prob": torch.stack(all_log_probs, dim=1).sum(1),
            "entropy": torch.stack(all_entropies, dim=1).sum(1),
        }


class OpponentPreviewHead(nn.Module):
    """Auxiliary opponent-model head used to shape preview representations."""

    def __init__(self, pokemon_dim: int, context_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.bring_score = nn.Sequential(
            nn.Linear(pokemon_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.lead_pair_score = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        pokemon_reps: torch.Tensor,
        context: torch.Tensor,
        team_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, team_size, pokemon_dim = pokemon_reps.shape
        context_expanded = context.unsqueeze(1).expand(-1, team_size, -1)
        bring_logits = self.bring_score(torch.cat([pokemon_reps, context_expanded], dim=-1)).squeeze(-1)

        left = pokemon_reps.unsqueeze(2).expand(-1, team_size, team_size, -1)
        right = pokemon_reps.unsqueeze(1).expand(-1, team_size, team_size, -1)
        pair_context = context.unsqueeze(1).unsqueeze(2).expand(-1, team_size, team_size, -1)
        lead_pair_logits = self.lead_pair_score(torch.cat([left, right, pair_context], dim=-1)).squeeze(-1)

        invalid_pairs = torch.eye(team_size, dtype=torch.bool, device=context.device).unsqueeze(0)
        if team_mask is not None:
            invalid_pairs = invalid_pairs | team_mask.unsqueeze(1) | team_mask.unsqueeze(2)
        lead_pair_logits = lead_pair_logits.masked_fill(invalid_pairs, -1e8).reshape(batch_size, team_size * team_size)

        return {
            "opponent_bring_logits": bring_logits,
            "opponent_lead_pair_logits": lead_pair_logits,
        }


class ValueHead(nn.Module):
    """Estimates V(s) -- the probability of winning from the current state."""

    def __init__(self, context_dim: int, pokemon_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.own_bench_attention = MaskedSetAttention(context_dim, pokemon_dim, hidden_dim)
        self.opp_bench_attention = MaskedSetAttention(context_dim, pokemon_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + 2 * pokemon_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        own_team_reps: torch.Tensor,
        opp_team_reps: torch.Tensor,
        own_bench_mask: torch.Tensor,
        opp_bench_mask: torch.Tensor,
    ) -> torch.Tensor:
        own_bench = self.own_bench_attention(context, own_team_reps, own_bench_mask)
        opp_bench = self.opp_bench_attention(context, opp_team_reps, opp_bench_mask)
        return self.mlp(torch.cat([context, own_bench, opp_bench], dim=-1)).squeeze(-1)
