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
    NUM_TERA_CHOICES,
    TARGET_NONE,
    decode_action,
)
from ..env.observation import TERASTALLIZED_OFFSET


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
    coordinated. Terastallization is modeled as a third conditional decision
    over {no tera, tera slot 0, tera slot 1}.
    """

    _MATCHUP_FEATURE_DIM = 22
    _TARGET_FLAG_DIM = 6

    def __init__(
        self,
        context_dim: int,
        pokemon_dim: int,
        hidden_dim: int = 256,
        *,
        move_type_lookup: torch.Tensor | None = None,
        move_base_power_lookup: torch.Tensor | None = None,
        move_accuracy_lookup: torch.Tensor | None = None,
        move_priority_lookup: torch.Tensor | None = None,
        move_category_lookup: torch.Tensor | None = None,
        move_target_flags: torch.Tensor | None = None,
        type_effectiveness: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_actions = ACTIONS_PER_SLOT

        self.action_id_emb = nn.Embedding(self.num_actions, hidden_dim)
        self.kind_emb = nn.Embedding(3, hidden_dim)
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
        self.tera_query = nn.Sequential(
            nn.Linear(
                context_dim + pokemon_dim * 4 + hidden_dim * 2 + self._MATCHUP_FEATURE_DIM * 2,
                hidden_dim,
            ),
            nn.ReLU(),
        )
        self.entity_proj = nn.Sequential(
            nn.Linear(pokemon_dim, hidden_dim),
            nn.ReLU(),
        )
        self.matchup_proj = nn.Sequential(
            nn.Linear(self._MATCHUP_FEATURE_DIM, hidden_dim),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.tera_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_TERA_CHOICES),
        )

        metadata = self._build_action_metadata()
        self.register_buffer("_action_ids", metadata["action_ids"], persistent=False)
        self.register_buffer("_action_kind", metadata["kind"], persistent=False)
        self.register_buffer("_action_move_slot", metadata["move_slot"], persistent=False)
        self.register_buffer("_action_target_slot", metadata["target_slot"], persistent=False)
        self.register_buffer("_action_switch_slot", metadata["switch_slot"], persistent=False)
        self.register_buffer("_action_is_switch", metadata["is_switch"], persistent=False)
        self.register_buffer("_action_is_move", metadata["is_move"], persistent=False)

        self._has_explicit_metadata = all(
            tensor is not None
            for tensor in (
                move_type_lookup,
                move_base_power_lookup,
                move_accuracy_lookup,
                move_priority_lookup,
                move_category_lookup,
                move_target_flags,
                type_effectiveness,
            )
        )
        self.register_buffer(
            "_move_type_lookup",
            self._ensure_long_buffer(move_type_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_base_power_lookup",
            self._ensure_float_buffer(move_base_power_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_accuracy_lookup",
            self._ensure_float_buffer(move_accuracy_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_priority_lookup",
            self._ensure_float_buffer(move_priority_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_category_lookup",
            self._ensure_long_buffer(move_category_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_target_flags",
            self._ensure_float_buffer(move_target_flags, dims=2, width=self._TARGET_FLAG_DIM),
            persistent=False,
        )
        self.register_buffer(
            "_type_effectiveness_lookup",
            self._ensure_float_buffer(type_effectiveness, dims=2, width=1),
            persistent=False,
        )

    def forward(
        self,
        context: torch.Tensor,
        active_reps: torch.Tensor,
        own_team_reps: torch.Tensor,
        opp_team_reps: torch.Tensor,
        opp_reserve_mask: torch.Tensor,
        opp_active_reps: torch.Tensor,
        masks: dict[str, torch.Tensor],
        actions: torch.Tensor | None = None,
        greedy: bool = False,
        battle_obs: dict[str, torch.Tensor] | None = None,
        tera_exploration_prob: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if "joint" not in masks or "can_tera" not in masks:
            raise ValueError("ActionHead expected mask keys {'joint', 'can_tera'}.")
        if actions is not None and actions.shape[1] != 3:
            raise ValueError(
                f"ActionHead expected actions with shape (batch, 3), got {tuple(actions.shape)}."
            )

        rep1 = active_reps[:, 0]
        rep2 = active_reps[:, 1]
        opp1 = opp_active_reps[:, 0]
        opp2 = opp_active_reps[:, 1]
        joint_masks = masks["joint"]
        can_tera = masks["can_tera"]
        mask1 = joint_masks[:, FIRST_SLOT_MASK_ROW]
        matchup_features_1 = self._candidate_action_features(
            battle_obs=battle_obs,
            actor_slot=0,
            can_tera_slot=can_tera[:, 0],
            batch_size=context.shape[0],
            device=context.device,
        )
        matchup_features_2 = self._candidate_action_features(
            battle_obs=battle_obs,
            actor_slot=1,
            can_tera_slot=can_tera[:, 1],
            batch_size=context.shape[0],
            device=context.device,
        )

        base_query1 = self.query1(torch.cat([context, rep1, rep2, opp1, opp2], dim=-1))
        reserve_summary1 = self.reserve_attention(base_query1, opp_team_reps, opp_reserve_mask)
        query1 = self.query1_fuse(torch.cat([base_query1, reserve_summary1], dim=-1))
        logits1 = self._score_actions(
            query1,
            rep1,
            rep2,
            own_team_reps,
            opp_active_reps,
            matchup_features_1,
        )
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
        mask2 = joint_masks[batch_indices, a1 + 1]
        if not mask2.any(dim=1).all():
            raise RuntimeError(
                "ActionHead received an empty conditional joint action mask for pokemon 2. "
                "This means build_joint_action_mask() produced an inconsistent legality table."
            )

        a1_emb = self.action_id_emb(a1)
        chosen_features_1 = matchup_features_1[batch_indices, a1]
        base_query2 = self.query2(torch.cat([context, rep2, rep1, opp1, opp2, a1_emb], dim=-1))
        reserve_summary2 = self.reserve_attention(base_query2, opp_team_reps, opp_reserve_mask)
        query2 = self.query2_fuse(torch.cat([base_query2, reserve_summary2], dim=-1))
        logits2 = self._score_actions(
            query2,
            rep2,
            rep1,
            own_team_reps,
            opp_active_reps,
            matchup_features_2,
        )
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

        a2_emb = self.action_id_emb(a2)
        chosen_features_2 = matchup_features_2[batch_indices, a2]
        a1_is_move = self._action_is_move[a1]
        a2_is_move = self._action_is_move[a2]
        tera_mask = torch.stack(
            [
                torch.ones_like(a1_is_move, dtype=torch.bool),
                can_tera[:, 0] & a1_is_move,
                can_tera[:, 1] & a2_is_move,
            ],
            dim=1,
        )
        tera_query = self.tera_query(
            torch.cat(
                [context, rep1, rep2, opp1, opp2, a1_emb, a2_emb, chosen_features_1, chosen_features_2],
                dim=-1,
            )
        )
        tera_logits = self.tera_score(tera_query).masked_fill(~tera_mask, -1e8)
        tera_dist = torch.distributions.Categorical(logits=tera_logits)

        if actions is None:
            tera_choice = tera_logits.argmax(dim=1) if greedy else tera_dist.sample()
            if tera_exploration_prob > 0.0 and not greedy:
                # Only explore when there is a real tera choice (>= 2 valid options).
                # Otherwise tera_mask is [True, False, False] and uniform == policy.
                has_choice = tera_mask.sum(dim=1) >= 2
                explore = (
                    torch.rand(context.shape[0], device=context.device) < tera_exploration_prob
                ) & has_choice
                if explore.any():
                    uniform_logits = torch.zeros_like(tera_logits).masked_fill(~tera_mask, -1e8)
                    uniform_sample = torch.distributions.Categorical(logits=uniform_logits).sample()
                    tera_choice = torch.where(explore, uniform_sample, tera_choice)
        else:
            tera_choice = actions[:, 2]

        if not tera_mask.gather(1, tera_choice.unsqueeze(1)).squeeze(1).all():
            raise RuntimeError(
                "ActionHead received an illegal tera choice for the current joint action pair."
            )

        log_prob = dist1.log_prob(a1) + dist2.log_prob(a2) + tera_dist.log_prob(tera_choice)
        entropy = dist1.entropy() + dist2.entropy() + tera_dist.entropy()

        return {
            "actions": torch.stack([a1, a2, tera_choice], dim=1),
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
        matchup_features: torch.Tensor,
    ) -> torch.Tensor:
        query_features = query.unsqueeze(1).expand(-1, self.num_actions, -1)
        action_features = self._static_action_features(query.device).unsqueeze(0).expand(query.shape[0], -1, -1)
        entity_features = self.entity_proj(
            self._dynamic_action_entities(actor_rep, partner_rep, own_team_reps, opp_active_reps)
        )
        matchup_emb = self.matchup_proj(matchup_features)
        return self.score(
            torch.cat([query_features, action_features, entity_features, matchup_emb], dim=-1)
        ).squeeze(-1)

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

    def _candidate_action_features(
        self,
        *,
        battle_obs: dict[str, torch.Tensor] | None,
        actor_slot: int,
        can_tera_slot: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            not self._has_explicit_metadata
            or battle_obs is None
            or any(
                key not in battle_obs
                for key in (
                    "own_moves",
                    "own_types",
                    "own_continuous",
                    "own_active_indices",
                    "opp_types",
                    "opp_continuous",
                    "opp_active_indices",
                    "opp_active_present",
                )
            )
        ):
            return torch.zeros(
                (batch_size, self.num_actions, self._MATCHUP_FEATURE_DIM),
                dtype=torch.float32,
                device=device,
            )

        own_active_indices = battle_obs["own_active_indices"]
        opp_active_indices = battle_obs["opp_active_indices"]
        own_types_active = self._gather_team_slots(battle_obs["own_types"], own_active_indices)
        own_cont_active = self._gather_team_slots(battle_obs["own_continuous"], own_active_indices)
        opp_types_active = self._gather_team_slots(battle_obs["opp_types"], opp_active_indices)
        opp_cont_active = self._gather_team_slots(battle_obs["opp_continuous"], opp_active_indices)

        actor_indices = own_active_indices[:, actor_slot]
        actor_moves = self._gather_team_rows(battle_obs["own_moves"], actor_indices)
        actor_types = own_types_active[:, actor_slot]
        actor_continuous = own_cont_active[:, actor_slot]
        actor_present = self._active_slot_present(battle_obs, "own", actor_slot)

        partner_slot = 1 - actor_slot
        partner_types = own_types_active[:, partner_slot]
        partner_continuous = own_cont_active[:, partner_slot]
        partner_present = self._active_slot_present(battle_obs, "own", partner_slot)

        actor_def_type_1, actor_def_type_2 = self._defensive_type_pair(actor_types, actor_continuous)
        partner_def_type_1, partner_def_type_2 = self._defensive_type_pair(partner_types, partner_continuous)
        opp_def_type_1, opp_def_type_2 = self._defensive_type_pair(opp_types_active, opp_cont_active)
        opp_present = battle_obs["opp_active_present"].to(dtype=torch.bool)

        move_slots = self._action_move_slot.clamp(max=NUM_MOVES - 1).view(1, self.num_actions).expand(batch_size, -1)
        action_move_ids = actor_moves.gather(1, move_slots)
        action_move_ids = torch.where(
            self._action_is_move.view(1, self.num_actions),
            action_move_ids,
            torch.zeros_like(action_move_ids),
        )
        action_move_ids = action_move_ids.clamp(max=self._move_type_lookup.shape[0] - 1)
        move_types = self._move_type_lookup[action_move_ids]
        base_power = self._move_base_power_lookup[action_move_ids]
        accuracy = self._move_accuracy_lookup[action_move_ids]
        priority = self._move_priority_lookup[action_move_ids]
        categories = self._move_category_lookup[action_move_ids]
        target_flags = self._move_target_flags[action_move_ids]
        is_move = self._action_is_move.view(1, self.num_actions).to(device=device, dtype=torch.float32).expand(batch_size, -1)
        is_switch = self._action_is_switch.view(1, self.num_actions).to(device=device, dtype=torch.float32).expand(batch_size, -1)
        is_pass = 1.0 - is_move - is_switch

        current_stab, post_tera_stab = self._stab_multipliers(
            move_types=move_types,
            actor_types=actor_types,
            actor_continuous=actor_continuous,
            can_tera_slot=can_tera_slot,
        )

        opp1_eff = self._type_effectiveness(move_types, opp_def_type_1[:, 0:1], opp_def_type_2[:, 0:1])
        opp2_eff = self._type_effectiveness(move_types, opp_def_type_1[:, 1:2], opp_def_type_2[:, 1:2])
        self_eff = self._type_effectiveness(move_types, actor_def_type_1.unsqueeze(1), actor_def_type_2.unsqueeze(1))
        partner_eff = self._type_effectiveness(
            move_types,
            partner_def_type_1.unsqueeze(1),
            partner_def_type_2.unsqueeze(1),
        )

        live_foe_count = opp_present.to(dtype=torch.float32).sum(dim=1, keepdim=True)
        foe_mean_eff = (
            opp1_eff * opp_present[:, 0:1].to(dtype=opp1_eff.dtype)
            + opp2_eff * opp_present[:, 1:2].to(dtype=opp2_eff.dtype)
        ) / live_foe_count.clamp(min=1.0)
        foe_mean_eff = torch.where(live_foe_count > 0, foe_mean_eff, torch.ones_like(foe_mean_eff))
        foe_best_eff = torch.maximum(
            torch.where(opp_present[:, 0:1], opp1_eff, torch.zeros_like(opp1_eff)),
            torch.where(opp_present[:, 1:2], opp2_eff, torch.zeros_like(opp2_eff)),
        )
        foe_best_eff = torch.where(live_foe_count > 0, foe_best_eff, torch.ones_like(foe_best_eff))

        target_type_1_bank = torch.stack(
            [
                opp_def_type_1[:, 0],
                opp_def_type_1[:, 1],
                partner_def_type_1,
                actor_def_type_1,
                torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.zeros(batch_size, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        target_type_2_bank = torch.stack(
            [
                opp_def_type_2[:, 0],
                opp_def_type_2[:, 1],
                partner_def_type_2,
                actor_def_type_2,
                torch.zeros(batch_size, dtype=torch.long, device=device),
                torch.zeros(batch_size, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        target_present_bank = torch.stack(
            [
                opp_present[:, 0],
                opp_present[:, 1],
                partner_present,
                actor_present,
                torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.zeros(batch_size, dtype=torch.bool, device=device),
            ],
            dim=1,
        )
        target_indices = self._action_target_slot.view(1, self.num_actions).expand(batch_size, -1)
        target_type_1 = target_type_1_bank.gather(1, target_indices)
        target_type_2 = target_type_2_bank.gather(1, target_indices)
        target_present = target_present_bank.gather(1, target_indices)
        targeted_eff = self._type_effectiveness(move_types, target_type_1, target_type_2)
        targeted_eff = torch.where(target_present, targeted_eff, torch.ones_like(targeted_eff))

        no_target_move = self._action_is_move.view(1, self.num_actions) & (
            self._action_target_slot.view(1, self.num_actions) == TARGET_NONE
        )
        no_target_eff = torch.ones_like(targeted_eff)
        no_target_eff = torch.where(
            (target_flags[..., 0] > 0) | (target_flags[..., 1] > 0),
            foe_mean_eff,
            no_target_eff,
        )
        no_target_eff = torch.where(target_flags[..., 2] > 0, partner_eff, no_target_eff)
        no_target_eff = torch.where(target_flags[..., 3] > 0, self_eff, no_target_eff)
        current_target_eff = torch.where(no_target_move, no_target_eff, targeted_eff)

        current_power = base_power * current_stab * current_target_eff / 4.0
        tera_power_gain = base_power * current_target_eff * (post_tera_stab - current_stab) / 4.0
        target_eff_scaled = current_target_eff / 4.0
        foe_best_scaled = foe_best_eff / 4.0
        foe_mean_scaled = foe_mean_eff / 4.0

        features = torch.stack(
            [
                is_move,
                is_switch,
                is_pass,
                base_power * is_move,
                accuracy * is_move,
                priority * is_move,
                (categories == 0).to(dtype=torch.float32) * is_move,
                (categories == 1).to(dtype=torch.float32) * is_move,
                (categories == 2).to(dtype=torch.float32) * is_move,
                (current_stab - 1.0) * is_move,
                (post_tera_stab - current_stab) * is_move,
                (current_target_eff == 0).to(dtype=torch.float32) * is_move,
                ((current_target_eff > 0) & (current_target_eff < 1)).to(dtype=torch.float32) * is_move,
                (current_target_eff > 1).to(dtype=torch.float32) * is_move,
                target_eff_scaled * is_move,
                foe_best_scaled * is_move,
                foe_mean_scaled * is_move,
                current_power * is_move,
                tera_power_gain * is_move,
                live_foe_count.expand(-1, self.num_actions) / 2.0,
                target_flags[..., 4] * is_move,
                target_flags[..., 5] * is_move,
            ],
            dim=-1,
        )
        return features

    def _stab_multipliers(
        self,
        *,
        move_types: torch.Tensor,
        actor_types: torch.Tensor,
        actor_continuous: torch.Tensor,
        can_tera_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        move_types = move_types.to(dtype=torch.long)
        type_1 = actor_types[:, 0:1]
        type_2 = actor_types[:, 1:2]
        tera_type = actor_types[:, 2:3]
        original_match = (move_types == type_1) | ((type_2 != 0) & (move_types == type_2))
        tera_match = (tera_type != 0) & (move_types == tera_type)
        terastallized = actor_continuous[:, TERASTALLIZED_OFFSET:TERASTALLIZED_OFFSET + 1] > 0.5

        current = torch.ones_like(move_types, dtype=torch.float32)
        current = torch.where(original_match, torch.full_like(current, 1.5), current)
        terastallized_stab = torch.where(
            tera_match & original_match,
            torch.full_like(current, 2.0),
            torch.where(tera_match | original_match, torch.full_like(current, 1.5), current),
        )
        current = torch.where(terastallized, terastallized_stab, current)

        can_tera_now = can_tera_slot.view(-1, 1) & ~terastallized
        post_tera = torch.where(
            can_tera_now,
            torch.where(
                tera_match & original_match,
                torch.full_like(current, 2.0),
                torch.where(tera_match | original_match, torch.full_like(current, 1.5), torch.ones_like(current)),
            ),
            current,
        )
        return current, post_tera

    def _type_effectiveness(
        self,
        move_types: torch.Tensor,
        target_type_1: torch.Tensor,
        target_type_2: torch.Tensor,
    ) -> torch.Tensor:
        move_types = move_types.clamp(min=0, max=self._type_effectiveness_lookup.shape[0] - 1)
        target_type_1 = target_type_1.clamp(min=0, max=self._type_effectiveness_lookup.shape[1] - 1)
        target_type_2 = target_type_2.clamp(min=0, max=self._type_effectiveness_lookup.shape[1] - 1)
        multiplier_1 = self._type_effectiveness_lookup[move_types, target_type_1]
        multiplier_2 = self._type_effectiveness_lookup[move_types, target_type_2]
        multiplier_2 = torch.where(target_type_2 != 0, multiplier_2, torch.ones_like(multiplier_1))
        multiplier = multiplier_1 * multiplier_2
        return torch.where(move_types != 0, multiplier, torch.ones_like(multiplier))

    @staticmethod
    def _defensive_type_pair(
        types: torch.Tensor,
        continuous: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tera_type = types[..., 2]
        terastallized = continuous[..., TERASTALLIZED_OFFSET] > 0.5
        type_1 = torch.where(terastallized, tera_type, types[..., 0])
        type_2 = torch.where(terastallized, torch.zeros_like(types[..., 1]), types[..., 1])
        return type_1, type_2

    @staticmethod
    def _gather_team_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices.view(indices.shape[0], 1, *([1] * (values.ndim - 2))).expand(-1, 1, *values.shape[2:])
        return values.gather(1, gather_index).squeeze(1)

    @staticmethod
    def _gather_team_slots(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices.unsqueeze(-1).expand(-1, -1, values.shape[-1])
        return values.gather(1, gather_index)

    @staticmethod
    def _active_slot_present(
        battle_obs: dict[str, torch.Tensor],
        side_prefix: str,
        slot: int,
    ) -> torch.Tensor:
        key = f"{side_prefix}_active_present"
        if key not in battle_obs:
            batch_size = battle_obs[f"{side_prefix}_active_indices"].shape[0]
            return torch.ones(batch_size, dtype=torch.bool, device=battle_obs[f"{side_prefix}_active_indices"].device)
        return battle_obs[key][:, slot].to(dtype=torch.bool)

    @staticmethod
    def _ensure_long_buffer(tensor: torch.Tensor | None) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(1, dtype=torch.long)
        return tensor.to(dtype=torch.long)

    @staticmethod
    def _ensure_float_buffer(
        tensor: torch.Tensor | None,
        *,
        dims: int = 1,
        width: int = 1,
    ) -> torch.Tensor:
        if tensor is None:
            if dims == 1:
                return torch.zeros(1, dtype=torch.float32)
            if dims == 2:
                return torch.zeros((1, width), dtype=torch.float32)
            raise ValueError(f"Unsupported fallback tensor rank {dims}.")
        return tensor.to(dtype=torch.float32)

    @staticmethod
    def _build_action_metadata() -> dict[str, torch.Tensor]:
        action_ids = []
        kinds = []
        move_slots = []
        target_slots = []
        switch_slots = []
        is_switch = []
        is_move = []

        move_none_slot = NUM_MOVES
        target_none_slot = NUM_TARGETS
        switch_none_slot = MAX_TEAM_SLOTS
        kind_lookup = {
            "move": 0,
            "switch": 1,
            "pass": 2,
        }

        for action_idx in range(ACTIONS_PER_SLOT):
            kind, slot_idx, target_idx = decode_action(action_idx)
            action_ids.append(action_idx)
            kinds.append(kind_lookup[kind])
            move_slots.append(slot_idx if kind == "move" else move_none_slot)
            target_slots.append(target_idx if kind == "move" else (TARGET_NONE if kind == "pass" else target_none_slot))
            switch_slots.append(slot_idx if kind == "switch" else switch_none_slot)
            is_switch.append(kind == "switch")
            is_move.append(kind == "move")

        return {
            "action_ids": torch.tensor(action_ids, dtype=torch.long),
            "kind": torch.tensor(kinds, dtype=torch.long),
            "move_slot": torch.tensor(move_slots, dtype=torch.long),
            "target_slot": torch.tensor(target_slots, dtype=torch.long),
            "switch_slot": torch.tensor(switch_slots, dtype=torch.long),
            "is_switch": torch.tensor(is_switch, dtype=torch.bool),
            "is_move": torch.tensor(is_move, dtype=torch.bool),
        }


class TeamPreviewHead(nn.Module):
    """Canonical bring-set selection plus ordered lead-pair choice."""

    def __init__(self, pokemon_dim: int, context_dim: int, pick_size: int = 4):
        super().__init__()
        if pick_size < 2:
            raise ValueError("TeamPreviewHead requires pick_size >= 2 so it can choose a lead pair.")
        self.pick_size = pick_size
        self.opponent_attention = MaskedSetAttention(context_dim, pokemon_dim, 128)
        self.bring_score = nn.Sequential(
            nn.Linear(pokemon_dim * 2 + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.context_proj = nn.Linear(context_dim + pokemon_dim, context_dim)
        self.lead_context_proj = nn.Linear(context_dim + pokemon_dim, context_dim)
        self.lead_pair_score = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

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
        if team_mask is None:
            available = torch.ones(batch, team_size, dtype=torch.bool, device=context.device)
        else:
            available = ~team_mask.clone()
        if (available.sum(dim=1) < self.pick_size).any():
            raise ValueError(
                f"TeamPreviewHead needs at least {self.pick_size} available pokemon per batch item."
            )

        canonical_bring = None if picks is None else picks.sort(dim=1).values
        team_positions = torch.arange(team_size, device=context.device).view(1, team_size)

        bring_log_probs: list[torch.Tensor] = []
        bring_entropies: list[torch.Tensor] = []
        bring_picks: list[torch.Tensor] = []
        picked_sum = torch.zeros(batch, pdim, device=context.device)
        next_min_index = torch.zeros(batch, 1, dtype=torch.long, device=context.device)
        remaining = available.clone()

        for step in range(self.pick_size):
            ctx = self.context_proj(torch.cat([context, picked_sum], dim=-1))
            opponent_summary = self.opponent_attention(ctx, opponent_reps, opponent_mask)

            ctx_expanded = ctx.unsqueeze(1).expand(-1, team_size, -1)
            opponent_expanded = opponent_summary.unsqueeze(1).expand(-1, team_size, -1)
            bring_inputs = torch.cat([pokemon_reps, ctx_expanded, opponent_expanded], dim=-1)
            bring_scores = self.bring_score(bring_inputs).squeeze(-1)
            picks_left = self.pick_size - step
            suffix_available = torch.cumsum(remaining.flip(1).to(dtype=torch.int64), dim=1).flip(1)
            valid = remaining & (team_positions >= next_min_index) & (suffix_available >= picks_left)
            bring_scores = bring_scores.masked_fill(~valid, -1e8)
            dist = torch.distributions.Categorical(logits=bring_scores)

            if canonical_bring is None:
                chosen = bring_scores.argmax(dim=1) if greedy else dist.sample()
            else:
                chosen = canonical_bring[:, step]

            if not valid.gather(1, chosen.unsqueeze(1)).squeeze(1).all():
                raise RuntimeError("TeamPreviewHead received an invalid canonical bring target.")

            bring_picks.append(chosen)
            bring_log_probs.append(dist.log_prob(chosen))
            bring_entropies.append(dist.entropy())

            remaining.scatter_(1, chosen.unsqueeze(1), False)
            next_min_index = chosen.unsqueeze(1) + 1
            chosen_rep = pokemon_reps.gather(
                1,
                chosen.unsqueeze(1).unsqueeze(2).expand(-1, -1, pdim),
            ).squeeze(1)
            picked_sum = picked_sum + chosen_rep

        bring_picks_tensor = torch.stack(bring_picks, dim=1)
        bring_log_prob = torch.stack(bring_log_probs, dim=1).sum(dim=1)
        bring_entropy = torch.stack(bring_entropies, dim=1).sum(dim=1)

        bring_mask = torch.zeros(batch, team_size, dtype=torch.bool, device=context.device)
        bring_mask.scatter_(1, bring_picks_tensor, True)
        bring_summary = picked_sum / float(self.pick_size)
        lead_context = self.lead_context_proj(torch.cat([context, bring_summary], dim=-1))
        lead_opponent_summary = self.opponent_attention(lead_context, opponent_reps, opponent_mask)

        left = pokemon_reps.unsqueeze(2).expand(-1, team_size, team_size, -1)
        right = pokemon_reps.unsqueeze(1).expand(-1, team_size, team_size, -1)
        bring_expanded = bring_summary.unsqueeze(1).unsqueeze(2).expand(-1, team_size, team_size, -1)
        opponent_expanded = lead_opponent_summary.unsqueeze(1).unsqueeze(2).expand(-1, team_size, team_size, -1)
        lead_context_expanded = lead_context.unsqueeze(1).unsqueeze(2).expand(-1, team_size, team_size, -1)
        lead_pair_logits = self.lead_pair_score(
            torch.cat([left, right, bring_expanded, opponent_expanded, lead_context_expanded], dim=-1)
        ).squeeze(-1)

        invalid_pairs = torch.eye(team_size, dtype=torch.bool, device=context.device).unsqueeze(0)
        invalid_pairs = invalid_pairs | (~bring_mask).unsqueeze(1) | (~bring_mask).unsqueeze(2)
        if team_mask is not None:
            invalid_pairs = invalid_pairs | team_mask.unsqueeze(1) | team_mask.unsqueeze(2)
        lead_pair_logits = lead_pair_logits.masked_fill(invalid_pairs, -1e8).reshape(batch, team_size * team_size)
        lead_dist = torch.distributions.Categorical(logits=lead_pair_logits)

        if picks is None:
            chosen_pair = lead_pair_logits.argmax(dim=1) if greedy else lead_dist.sample()
        else:
            chosen_pair = picks[:, 0] * team_size + picks[:, 1]

        valid_pairs = (~invalid_pairs).reshape(batch, team_size * team_size)
        if not valid_pairs.gather(1, chosen_pair.unsqueeze(1)).squeeze(1).all():
            raise RuntimeError("TeamPreviewHead received an invalid lead pair target.")

        lead_log_prob = lead_dist.log_prob(chosen_pair)
        lead_entropy = lead_dist.entropy()
        lead_pair = torch.stack([chosen_pair // team_size, chosen_pair % team_size], dim=1)

        remaining_bring_mask = bring_mask.clone()
        remaining_bring_mask.scatter_(1, lead_pair, False)
        bench_tail = team_positions.expand(batch, -1).masked_fill(~remaining_bring_mask, team_size)
        bench_tail = bench_tail.sort(dim=1).values[:, : self.pick_size - 2]
        final_picks = torch.cat([lead_pair, bench_tail], dim=1)

        return {
            "bring_picks": bring_picks_tensor,
            "lead_pair": lead_pair,
            "picks": final_picks,
            "bring_log_prob": bring_log_prob,
            "lead_log_prob": lead_log_prob,
            "log_prob": bring_log_prob + lead_log_prob,
            "bring_entropy": bring_entropy,
            "lead_entropy": lead_entropy,
            "entropy": bring_entropy + lead_entropy,
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


class HiddenInfoHead(nn.Module):
    """Auxiliary heads that predict hidden opponent attributes from its team reps.

    These are pure training-signal heads: they only run during PPO update and
    do not participate in the rollout/action path. Labels come from the
    end-of-episode observation (the "eventually revealed" state), so a turn-1
    prediction is trained to anticipate what will be known by battle's end.
    """

    def __init__(
        self,
        pokemon_dim: int,
        num_items: int,
        num_abilities: int,
        num_types: int,
        num_moves: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.item_mlp = nn.Sequential(
            nn.Linear(pokemon_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_items),
        )
        self.ability_mlp = nn.Sequential(
            nn.Linear(pokemon_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_abilities),
        )
        self.tera_type_mlp = nn.Sequential(
            nn.Linear(pokemon_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_types),
        )
        self.move_mlp = nn.Sequential(
            nn.Linear(pokemon_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_moves),
        )

    def forward(self, opp_reps: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "item_logits": self.item_mlp(opp_reps),
            "ability_logits": self.ability_mlp(opp_reps),
            "tera_type_logits": self.tera_type_mlp(opp_reps),
            "move_logits": self.move_mlp(opp_reps),
        }


class OpponentActionHead(nn.Module):
    """Predicts the opponent's joint (a1, a2, tera) action for the current turn.

    Factorized as three independent categorical distributions for simplicity;
    the autoregressive form can be introduced in commit 5 if Option B's
    expectation over opp actions requires it. Labels come from the opponent's
    actually-chosen actions captured in the env.
    """

    def __init__(
        self,
        context_dim: int,
        pokemon_dim: int,
        num_actions: int,
        num_tera_choices: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 4, hidden_dim),
            nn.ReLU(),
        )
        self.a1_head = nn.Linear(hidden_dim, num_actions)
        self.a2_head = nn.Linear(hidden_dim, num_actions)
        self.tera_head = nn.Linear(hidden_dim, num_tera_choices)

    def forward(
        self,
        context: torch.Tensor,
        opp_active_reps: torch.Tensor,
        own_active_reps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        query = self.query(
            torch.cat(
                [
                    context,
                    opp_active_reps[:, 0],
                    opp_active_reps[:, 1],
                    own_active_reps[:, 0],
                    own_active_reps[:, 1],
                ],
                dim=-1,
            )
        )
        return {
            "opp_a1_logits": self.a1_head(query),
            "opp_a2_logits": self.a2_head(query),
            "opp_tera_logits": self.tera_head(query),
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
