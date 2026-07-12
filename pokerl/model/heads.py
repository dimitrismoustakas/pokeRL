"""Action head (autoregressive for doubles) and value head."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    MAX_TEAM_SLOTS,
    NUM_MOVES,
    NUM_TARGETS,
    NUM_TERA_CHOICES,
    TARGET_NONE,
    TERA_NONE,
    decode_action,
)
from ..env.observation import TERASTALLIZED_OFFSET


def _sample_categorical_logits(logits: torch.Tensor) -> torch.Tensor:
    """Sample from categorical logits with the Gumbel-max trick.

    If ``E`` is exponential(1), ``-log(E)`` is a standard Gumbel sample.
    This is distribution-equivalent to multinomial sampling from softmax
    probabilities while avoiding an unnecessary softmax.
    """
    sample_logits = logits.detach()
    exponential = torch.empty_like(sample_logits).exponential_()
    return (sample_logits - exponential.log()).argmax(dim=1)


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

        key_proj, values = self._project_items(items)
        return self._pool_projected(query, key_proj, values, mask)

    def _project_items(
        self,
        items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a set once so multiple queries can share the result."""
        return self.key_proj(items), self.value_proj(items)

    def _pool_projected(
        self,
        query: torch.Tensor,
        key_proj: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool already-projected items for one query."""
        query_proj = self.query_proj(query).unsqueeze(1)
        scores = (query_proj * key_proj).sum(dim=-1) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask, -1e8)

        weights = torch.softmax(scores, dim=1)
        if mask is not None:
            weights = weights.masked_fill(mask, 0.0)
        weights_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(
            weights_sum > 0, weights / weights_sum.clamp(min=1e-8), weights
        )

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
    _INITIAL_TERA_USE_PROB = 0.20

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
        self._static_action_features_cache: torch.Tensor | None = None
        self._static_action_features_cache_key: tuple[object, ...] | None = None

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
                context_dim
                + pokemon_dim * 4
                + hidden_dim * 2
                + self._MATCHUP_FEATURE_DIM * 2,
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
        # Factor Tera into an optimal-stopping-style gate (use now vs wait)
        # and a conditional slot choice. The resulting marginal is still the
        # exact three-choice distribution {none, slot 0, slot 1} consumed by
        # the environment and PPO.
        self.tera_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.tera_gate_score = nn.Linear(hidden_dim, 1)
        self.tera_slot_score = nn.Linear(hidden_dim, NUM_TERA_CHOICES - 1)
        nn.init.zeros_(self.tera_gate_score.weight)
        nn.init.constant_(
            self.tera_gate_score.bias,
            math.log(self._INITIAL_TERA_USE_PROB / (1.0 - self._INITIAL_TERA_USE_PROB)),
        )

        metadata = self._build_action_metadata()
        self.register_buffer("_action_ids", metadata["action_ids"], persistent=False)
        self.register_buffer("_action_kind", metadata["kind"], persistent=False)
        self.register_buffer(
            "_action_move_slot", metadata["move_slot"], persistent=False
        )
        self.register_buffer(
            "_action_target_slot", metadata["target_slot"], persistent=False
        )
        self.register_buffer(
            "_action_switch_slot", metadata["switch_slot"], persistent=False
        )
        self.register_buffer(
            "_action_is_switch", metadata["is_switch"], persistent=False
        )
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
            self._ensure_float_buffer(
                move_target_flags, dims=2, width=self._TARGET_FLAG_DIM
            ),
            persistent=False,
        )
        self.register_buffer(
            "_type_effectiveness_lookup",
            self._ensure_float_buffer(type_effectiveness, dims=2, width=1),
            persistent=False,
        )

    def _apply(self, fn):
        self._static_action_features_cache = None
        self._static_action_features_cache_key = None
        return super()._apply(fn)

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
        validate_actions: bool = True,
        include_policy_stats: bool = True,
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
        batch_size = context.shape[0]
        device = context.device
        joint_masks = masks["joint"]
        can_tera = masks["can_tera"]
        mask1 = joint_masks[:, FIRST_SLOT_MASK_ROW]
        action_features = self._static_action_features()
        matchup_context = self._matchup_context(battle_obs, batch_size, device)
        matchup_features_1 = self._candidate_action_features(
            matchup_context=matchup_context,
            actor_slot=0,
            can_tera_slot=can_tera[:, 0],
            batch_size=batch_size,
            device=device,
        )
        matchup_features_2 = self._candidate_action_features(
            matchup_context=matchup_context,
            actor_slot=1,
            can_tera_slot=can_tera[:, 1],
            batch_size=batch_size,
            device=device,
        )

        base_query1 = self.query1(torch.cat([context, rep1, rep2, opp1, opp2], dim=-1))
        reserve_item_projections = None
        if not torch.is_grad_enabled():
            reserve_item_projections = self.reserve_attention._project_items(
                opp_team_reps
            )
            reserve_summary1 = self.reserve_attention._pool_projected(
                base_query1,
                *reserve_item_projections,
                opp_reserve_mask,
            )
        else:
            reserve_summary1 = self.reserve_attention(
                base_query1,
                opp_team_reps,
                opp_reserve_mask,
            )
        query1 = self.query1_fuse(torch.cat([base_query1, reserve_summary1], dim=-1))
        logits1 = self._score_actions(
            query1,
            rep1,
            rep2,
            own_team_reps,
            opp_active_reps,
            matchup_features_1,
            action_features,
        )
        logits1 = logits1.masked_fill(~mask1, -1e8)
        a1, log_prob1, entropy1 = self._categorical_choice(
            logits1,
            actions=None if actions is None else actions[:, 0],
            greedy=greedy,
            include_stats=include_policy_stats,
        )

        # A sampled action comes directly from masked logits. Re-reading its
        # legality with a Python boolean would only add a CUDA-to-host barrier;
        # validation remains available for externally supplied PPO actions.
        validate_supplied_actions = validate_actions and actions is not None
        if (
            validate_supplied_actions
            and not mask1.gather(1, a1.unsqueeze(1)).squeeze(1).all()
        ):
            raise RuntimeError(
                "ActionHead received an illegal first-slot action for the current joint mask."
            )

        batch_indices = torch.arange(context.shape[0], device=context.device)
        mask2 = joint_masks[batch_indices, a1 + 1]
        if validate_supplied_actions and not mask2.any(dim=1).all():
            raise RuntimeError(
                "ActionHead received an empty conditional joint action mask for pokemon 2. "
                "This means build_joint_action_mask() produced an inconsistent legality table."
            )

        a1_emb = self.action_id_emb(a1)
        chosen_features_1 = matchup_features_1[batch_indices, a1]
        base_query2 = self.query2(
            torch.cat([context, rep2, rep1, opp1, opp2, a1_emb], dim=-1)
        )
        if reserve_item_projections is None:
            reserve_summary2 = self.reserve_attention(
                base_query2,
                opp_team_reps,
                opp_reserve_mask,
            )
        else:
            reserve_summary2 = self.reserve_attention._pool_projected(
                base_query2,
                *reserve_item_projections,
                opp_reserve_mask,
            )
        query2 = self.query2_fuse(torch.cat([base_query2, reserve_summary2], dim=-1))
        logits2 = self._score_actions(
            query2,
            rep2,
            rep1,
            own_team_reps,
            opp_active_reps,
            matchup_features_2,
            action_features,
        )
        logits2 = logits2.masked_fill(~mask2, -1e8)
        a2, log_prob2, entropy2 = self._categorical_choice(
            logits2,
            actions=None if actions is None else actions[:, 1],
            greedy=greedy,
            include_stats=include_policy_stats,
        )

        if (
            validate_supplied_actions
            and not mask2.gather(1, a2.unsqueeze(1)).squeeze(1).all()
        ):
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
                [
                    context,
                    rep1,
                    rep2,
                    opp1,
                    opp2,
                    a1_emb,
                    a2_emb,
                    chosen_features_1,
                    chosen_features_2,
                ],
                dim=-1,
            )
        )
        tera_hidden = self.tera_score(tera_query)
        tera_gate_logits = self.tera_gate_score(tera_hidden).squeeze(-1)
        tera_slot_logits = self.tera_slot_score(tera_hidden)
        (
            base_tera_log_probs,
            base_tera_probs,
            base_tera_entropy,
            base_tera_gate_entropy,
            base_tera_slot_entropy,
            base_tera_use_prob,
        ) = self._factorized_tera_distribution(
            tera_gate_logits,
            tera_slot_logits,
            tera_mask[:, 1:],
        )
        self._validate_exploration_probability(
            "tera_exploration_prob",
            tera_exploration_prob,
        )
        use_tera_behavior_policy = not greedy and tera_exploration_prob > 0.0
        if use_tera_behavior_policy:
            tera_log_probs, tera_probs = self._tera_behavior_distribution(
                base_tera_log_probs,
                tera_mask,
                exploration_prob=tera_exploration_prob,
            )
            tera_sampling_logits = tera_log_probs
            tera_entropy = (
                self._entropy_from_probs(tera_probs) if include_policy_stats else None
            )
        else:
            tera_log_probs = base_tera_log_probs
            tera_probs = base_tera_probs
            tera_sampling_logits = base_tera_log_probs
            tera_entropy = base_tera_entropy if include_policy_stats else None

        if actions is None:
            tera_choice = (
                base_tera_log_probs.argmax(dim=1)
                if greedy
                else _sample_categorical_logits(tera_sampling_logits)
            )
        else:
            tera_choice = actions[:, 2]

        if (
            validate_supplied_actions
            and not tera_mask.gather(1, tera_choice.unsqueeze(1)).squeeze(1).all()
        ):
            raise RuntimeError(
                "ActionHead received an illegal tera choice for the current joint action pair."
            )

        result = {
            "actions": torch.stack([a1, a2, tera_choice], dim=1),
        }
        if include_policy_stats:
            if (
                tera_log_probs is None
                or log_prob1 is None
                or log_prob2 is None
                or entropy1 is None
                or entropy2 is None
                or tera_entropy is None
            ):
                raise RuntimeError("ActionHead policy stats were not computed.")
            tera_log_prob = tera_log_probs.gather(1, tera_choice.unsqueeze(1)).squeeze(
                1
            )
            result["log_prob"] = log_prob1 + log_prob2 + tera_log_prob
            result["action_entropy"] = entropy1 + entropy2
            result["entropy"] = entropy1 + entropy2 + tera_entropy
            result["base_tera_entropy"] = base_tera_entropy
            result["base_tera_gate_entropy"] = base_tera_gate_entropy
            result["base_tera_slot_entropy"] = base_tera_slot_entropy
            result["base_tera_use_prob"] = base_tera_use_prob
            result["base_tera_none_prob"] = base_tera_probs[:, TERA_NONE]
            result["base_tera_has_real_choice"] = tera_mask.sum(dim=-1) >= 2
            result["base_tera_has_slot_choice"] = tera_mask[:, 1:].sum(dim=-1) >= 2
        return result

    @staticmethod
    def _factorized_tera_distribution(
        gate_logits: torch.Tensor,
        slot_logits: torch.Tensor,
        valid_slots: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build the exact {wait, slot 0, slot 1} marginal distribution.

        ``gate_logits`` parameterizes whether to use Tera now. Conditional on
        use, ``slot_logits`` chooses one of the currently eligible active
        Pokemon. The returned slot entropy is conditional on using Tera; total
        entropy is ``gate_entropy + use_probability * slot_entropy``.
        """
        if gate_logits.ndim != 1:
            raise ValueError("Tera gate logits must have shape (batch,).")
        if slot_logits.shape != valid_slots.shape or slot_logits.ndim != 2:
            raise ValueError(
                "Tera slot logits and validity mask must have the same 2D shape."
            )
        if slot_logits.shape != (gate_logits.shape[0], NUM_TERA_CHOICES - 1):
            raise ValueError(
                f"Tera slot logits must have shape (batch, {NUM_TERA_CHOICES - 1})."
            )

        has_real_choice = valid_slots.any(dim=-1)
        safe_valid_slots = valid_slots.clone()
        safe_valid_slots[~has_real_choice, 0] = True
        masked_slot_logits = slot_logits.masked_fill(~safe_valid_slots, -torch.inf)
        raw_slot_log_probs = torch.log_softmax(masked_slot_logits, dim=-1)
        raw_slot_probs = raw_slot_log_probs.exp()
        slot_probs = torch.where(valid_slots, raw_slot_probs, 0.0)
        slot_log_probs = torch.where(
            valid_slots,
            raw_slot_log_probs,
            torch.full_like(raw_slot_log_probs, -torch.inf),
        )

        gate_use_log_prob = F.logsigmoid(gate_logits)
        gate_wait_log_prob = F.logsigmoid(-gate_logits)
        gate_use_prob = gate_use_log_prob.exp()
        gate_wait_prob = gate_wait_log_prob.exp()

        use_prob = gate_use_prob * has_real_choice.to(gate_use_prob.dtype)
        wait_prob = torch.where(has_real_choice, gate_wait_prob, 1.0)
        real_log_probs = gate_use_log_prob.unsqueeze(-1) + slot_log_probs
        base_log_probs = torch.cat(
            (gate_wait_log_prob.unsqueeze(-1), real_log_probs), dim=-1
        )
        no_choice_log_probs = torch.full_like(base_log_probs, -torch.inf)
        no_choice_log_probs[:, TERA_NONE] = 0.0
        base_log_probs = torch.where(
            has_real_choice.unsqueeze(-1),
            base_log_probs,
            no_choice_log_probs,
        )
        base_probs = torch.cat(
            (wait_prob.unsqueeze(-1), use_prob.unsqueeze(-1) * slot_probs), dim=-1
        )

        gate_entropy = -(
            gate_wait_prob * gate_wait_log_prob
            + gate_use_prob * gate_use_log_prob
        )
        gate_entropy = gate_entropy * has_real_choice.to(gate_entropy.dtype)
        slot_entropy = -torch.where(
            valid_slots,
            slot_probs * raw_slot_log_probs,
            torch.zeros_like(slot_probs),
        ).sum(dim=-1)
        total_entropy = gate_entropy + use_prob * slot_entropy
        return (
            base_log_probs,
            base_probs,
            total_entropy,
            gate_entropy,
            slot_entropy,
            use_prob,
        )

    @staticmethod
    def _tera_behavior_distribution(
        base_logits: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        exploration_prob: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact factorized rollout-exploration probabilities.

        Epsilon independently mixes the Bernoulli use-now gate toward 0.5 and
        the conditional eligible-slot policy toward uniform. Reconstructing
        their exact three-choice marginal keeps rollout and PPO likelihoods
        identical without a timing-specific action override.
        """
        if base_logits.ndim != 2 or base_logits.shape != valid_mask.shape:
            raise ValueError(
                "Tera logits and validity mask must have the same 2D shape."
            )
        if base_logits.shape[1] != NUM_TERA_CHOICES:
            raise ValueError(
                f"Expected {NUM_TERA_CHOICES} tera choices, got {base_logits.shape[1]}."
            )

        masked_logits = base_logits.masked_fill(~valid_mask, -torch.inf)
        base_probs = torch.softmax(masked_logits, dim=-1)
        valid_slots = valid_mask[:, 1:]
        has_real_choice = valid_slots.any(dim=-1)
        epsilon = has_real_choice.to(base_probs.dtype) * exploration_prob
        behavior_use_prob = base_probs[:, 1:].sum(dim=-1) + epsilon * (
            0.5 - base_probs[:, 1:].sum(dim=-1)
        )

        safe_valid_slots = valid_slots.clone()
        safe_valid_slots[~has_real_choice, 0] = True
        base_slot_probs = torch.softmax(
            base_logits[:, 1:].masked_fill(~safe_valid_slots, -torch.inf),
            dim=-1,
        )
        base_slot_probs = torch.where(valid_slots, base_slot_probs, 0.0)
        uniform_slot_probs = valid_slots.to(base_probs.dtype) / valid_slots.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1)
        behavior_slot_probs = base_slot_probs + epsilon.unsqueeze(-1) * (
            uniform_slot_probs - base_slot_probs
        )
        behavior_probs = torch.cat(
            (
                (1.0 - behavior_use_prob).unsqueeze(-1),
                behavior_use_prob.unsqueeze(-1) * behavior_slot_probs,
            ),
            dim=-1,
        )
        behavior_probs = behavior_probs.masked_fill(~valid_mask, 0.0)
        safe_log_probs = behavior_probs.clamp_min(
            torch.finfo(behavior_probs.dtype).tiny
        ).log()
        behavior_log_probs = torch.where(
            behavior_probs > 0,
            safe_log_probs,
            torch.full_like(safe_log_probs, -torch.inf),
        )
        return behavior_log_probs, behavior_probs

    @staticmethod
    def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
        safe_log_probs = probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
        return -(probs * safe_log_probs).sum(dim=-1)

    @staticmethod
    def _validate_exploration_probability(name: str, value: float) -> None:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1], got {value}.")

    @staticmethod
    def _categorical_choice(
        logits: torch.Tensor,
        *,
        actions: torch.Tensor | None,
        greedy: bool,
        include_stats: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if include_stats:
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            if actions is None:
                selected = (
                    logits.argmax(dim=1)
                    if greedy
                    else _sample_categorical_logits(logits)
                )
            else:
                selected = actions
            selected_log_prob = log_probs.gather(1, selected.unsqueeze(1)).squeeze(1)
            entropy = -(probs * log_probs).sum(dim=-1)
            return selected, selected_log_prob, entropy

        if actions is None:
            if greedy:
                selected = logits.argmax(dim=1)
            else:
                selected = _sample_categorical_logits(logits)
        else:
            selected = actions
        return selected, None, None

    def _score_actions(
        self,
        query: torch.Tensor,
        actor_rep: torch.Tensor,
        partner_rep: torch.Tensor,
        own_team_reps: torch.Tensor,
        opp_active_reps: torch.Tensor,
        matchup_features: torch.Tensor,
        action_features: torch.Tensor,
    ) -> torch.Tensor:
        query_features = query.unsqueeze(1).expand(-1, self.num_actions, -1)
        action_features = action_features.unsqueeze(0).expand(query.shape[0], -1, -1)
        entity_features = self.entity_proj(
            self._dynamic_action_entities(
                actor_rep, partner_rep, own_team_reps, opp_active_reps
            )
        )
        matchup_emb = self.matchup_proj(matchup_features)
        return self.score(
            torch.cat(
                [query_features, action_features, entity_features, matchup_emb], dim=-1
            )
        ).squeeze(-1)

    def _static_action_features(self) -> torch.Tensor:
        parameters = (
            self.action_id_emb.weight,
            self.kind_emb.weight,
            self.move_slot_emb.weight,
            self.target_emb.weight,
            self.switch_slot_emb.weight,
        )
        cache_allowed = not torch.is_grad_enabled() or (
            not self.training
            and all(not parameter.requires_grad for parameter in parameters)
        )
        cache_key = (
            tuple(
                (
                    id(parameter),
                    parameter._version,
                    parameter.device,
                    parameter.dtype,
                )
                for parameter in parameters
            )
            if cache_allowed
            else None
        )
        if (
            cache_allowed
            and self._static_action_features_cache is not None
            and self._static_action_features_cache_key == cache_key
        ):
            return self._static_action_features_cache

        action_ids = self._action_ids
        kind = self._action_kind
        move_slot = self._action_move_slot
        target_slot = self._action_target_slot
        switch_slot = self._action_switch_slot
        features = (
            self.action_id_emb(action_ids)
            + self.kind_emb(kind)
            + self.move_slot_emb(move_slot)
            + self.target_emb(target_slot)
            + self.switch_slot_emb(switch_slot)
        )
        if cache_allowed:
            self._static_action_features_cache = features
            self._static_action_features_cache_key = cache_key
        return features

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
        target_indices = self._action_target_slot.view(1, self.num_actions, 1).expand(
            batch_size, -1, pokemon_dim
        )
        target_entities = target_bank.gather(1, target_indices)

        switch_bank = torch.cat([own_team_reps, zero_rep.unsqueeze(1)], dim=1)
        switch_indices = self._action_switch_slot.view(1, self.num_actions, 1).expand(
            batch_size, -1, pokemon_dim
        )
        switch_entities = switch_bank.gather(1, switch_indices)

        is_switch = self._action_is_switch.view(1, self.num_actions, 1).to(
            device=actor_rep.device
        )
        return torch.where(is_switch, switch_entities, target_entities)

    def _matchup_context(
        self,
        battle_obs: dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor] | None:
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
            return None

        own_active_indices = battle_obs["own_active_indices"]
        opp_active_indices = battle_obs["opp_active_indices"]
        own_types_active = self._gather_team_slots(
            battle_obs["own_types"], own_active_indices
        )
        own_cont_active = self._gather_team_slots(
            battle_obs["own_continuous"], own_active_indices
        )
        opp_types_active = self._gather_team_slots(
            battle_obs["opp_types"], opp_active_indices
        )
        opp_cont_active = self._gather_team_slots(
            battle_obs["opp_continuous"], opp_active_indices
        )
        opp_def_type_1, opp_def_type_2 = self._defensive_type_pair(
            opp_types_active, opp_cont_active
        )
        opp_present = battle_obs["opp_active_present"].to(dtype=torch.bool)

        action_is_move = self._action_is_move.view(1, self.num_actions)
        action_is_switch = self._action_is_switch.view(1, self.num_actions)
        action_target_slot = self._action_target_slot.view(1, self.num_actions)
        action_is_move_float = action_is_move.to(dtype=torch.float32)
        action_is_switch_float = action_is_switch.to(dtype=torch.float32)

        context = {
            "own_moves": battle_obs["own_moves"],
            "own_active_indices": own_active_indices,
            "own_types_active": own_types_active,
            "own_cont_active": own_cont_active,
            "opp_def_type_1": opp_def_type_1,
            "opp_def_type_2": opp_def_type_2,
            "opp_present": opp_present,
            "live_foe_count": opp_present.to(dtype=torch.float32).sum(
                dim=1, keepdim=True
            ),
            "move_slots": self._action_move_slot.clamp(max=NUM_MOVES - 1)
            .view(1, self.num_actions)
            .expand(batch_size, -1),
            "target_indices": action_target_slot.expand(batch_size, -1),
            "is_move_bool": action_is_move,
            "is_move": action_is_move_float.expand(batch_size, -1),
            "is_switch": action_is_switch_float.expand(batch_size, -1),
            "is_pass": (1.0 - action_is_move_float - action_is_switch_float).expand(
                batch_size, -1
            ),
            "no_target_move": action_is_move & (action_target_slot == TARGET_NONE),
            "zero_long": torch.zeros(batch_size, dtype=torch.long, device=device),
            "zero_bool": torch.zeros(batch_size, dtype=torch.bool, device=device),
        }
        if "own_active_present" in battle_obs:
            context["own_active_present"] = battle_obs["own_active_present"]
        return context

    def _candidate_action_features(
        self,
        *,
        matchup_context: dict[str, torch.Tensor] | None = None,
        battle_obs: dict[str, torch.Tensor] | None = None,
        actor_slot: int,
        can_tera_slot: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if matchup_context is None and battle_obs is not None:
            matchup_context = self._matchup_context(battle_obs, batch_size, device)
        if matchup_context is None:
            return torch.zeros(
                (batch_size, self.num_actions, self._MATCHUP_FEATURE_DIM),
                dtype=torch.float32,
                device=device,
            )

        own_active_indices = matchup_context["own_active_indices"]
        own_types_active = matchup_context["own_types_active"]
        own_cont_active = matchup_context["own_cont_active"]

        actor_indices = own_active_indices[:, actor_slot]
        actor_moves = self._gather_team_rows(
            matchup_context["own_moves"], actor_indices
        )
        actor_types = own_types_active[:, actor_slot]
        actor_continuous = own_cont_active[:, actor_slot]
        actor_present = self._active_slot_present_from_context(
            matchup_context, actor_slot
        )

        partner_slot = 1 - actor_slot
        partner_types = own_types_active[:, partner_slot]
        partner_continuous = own_cont_active[:, partner_slot]
        partner_present = self._active_slot_present_from_context(
            matchup_context, partner_slot
        )

        actor_def_type_1, actor_def_type_2 = self._defensive_type_pair(
            actor_types, actor_continuous
        )
        partner_def_type_1, partner_def_type_2 = self._defensive_type_pair(
            partner_types, partner_continuous
        )
        opp_def_type_1 = matchup_context["opp_def_type_1"]
        opp_def_type_2 = matchup_context["opp_def_type_2"]
        opp_present = matchup_context["opp_present"]

        action_move_ids = actor_moves.gather(1, matchup_context["move_slots"])
        action_move_ids = torch.where(
            matchup_context["is_move_bool"],
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
        is_move = matchup_context["is_move"]
        is_switch = matchup_context["is_switch"]
        is_pass = matchup_context["is_pass"]

        current_stab, post_tera_stab = self._stab_multipliers(
            move_types=move_types,
            actor_types=actor_types,
            actor_continuous=actor_continuous,
            can_tera_slot=can_tera_slot,
        )

        target_type_1_bank = torch.stack(
            [
                opp_def_type_1[:, 0],
                opp_def_type_1[:, 1],
                partner_def_type_1,
                actor_def_type_1,
                matchup_context["zero_long"],
                matchup_context["zero_long"],
            ],
            dim=1,
        )
        target_type_2_bank = torch.stack(
            [
                opp_def_type_2[:, 0],
                opp_def_type_2[:, 1],
                partner_def_type_2,
                actor_def_type_2,
                matchup_context["zero_long"],
                matchup_context["zero_long"],
            ],
            dim=1,
        )
        target_present_bank = torch.stack(
            [
                opp_present[:, 0],
                opp_present[:, 1],
                partner_present,
                actor_present,
                matchup_context["zero_bool"],
                matchup_context["zero_bool"],
            ],
            dim=1,
        )
        target_indices = matchup_context["target_indices"]
        target_type_1 = target_type_1_bank.gather(1, target_indices)
        target_type_2 = target_type_2_bank.gather(1, target_indices)
        target_present = target_present_bank.gather(1, target_indices)

        # All five effectiveness queries share the same move ids. Adding a
        # target axis lets one set of lookup/pointwise kernels evaluate them
        # without changing the per-element lookup or multiplication order.
        target_type_1_queries = torch.stack(
            [
                opp_def_type_1[:, 0:1].expand(-1, self.num_actions),
                opp_def_type_1[:, 1:2].expand(-1, self.num_actions),
                actor_def_type_1.unsqueeze(1).expand(-1, self.num_actions),
                partner_def_type_1.unsqueeze(1).expand(-1, self.num_actions),
                target_type_1,
            ],
            dim=-1,
        )
        target_type_2_queries = torch.stack(
            [
                opp_def_type_2[:, 0:1].expand(-1, self.num_actions),
                opp_def_type_2[:, 1:2].expand(-1, self.num_actions),
                actor_def_type_2.unsqueeze(1).expand(-1, self.num_actions),
                partner_def_type_2.unsqueeze(1).expand(-1, self.num_actions),
                target_type_2,
            ],
            dim=-1,
        )
        effectiveness = self._type_effectiveness(
            move_types.unsqueeze(-1),
            target_type_1_queries,
            target_type_2_queries,
        )
        opp1_eff, opp2_eff, self_eff, partner_eff, targeted_eff = effectiveness.unbind(
            dim=-1
        )
        targeted_eff = torch.where(
            target_present, targeted_eff, torch.ones_like(targeted_eff)
        )

        live_foe_count = matchup_context["live_foe_count"]
        foe_mean_eff = (
            opp1_eff * opp_present[:, 0:1].to(dtype=opp1_eff.dtype)
            + opp2_eff * opp_present[:, 1:2].to(dtype=opp2_eff.dtype)
        ) / live_foe_count.clamp(min=1.0)
        foe_mean_eff = torch.where(
            live_foe_count > 0, foe_mean_eff, torch.ones_like(foe_mean_eff)
        )
        foe_best_eff = torch.maximum(
            torch.where(opp_present[:, 0:1], opp1_eff, torch.zeros_like(opp1_eff)),
            torch.where(opp_present[:, 1:2], opp2_eff, torch.zeros_like(opp2_eff)),
        )
        foe_best_eff = torch.where(
            live_foe_count > 0, foe_best_eff, torch.ones_like(foe_best_eff)
        )

        no_target_move = matchup_context["no_target_move"]
        no_target_eff = torch.ones_like(targeted_eff)
        no_target_eff = torch.where(
            (target_flags[..., 0] > 0) | (target_flags[..., 1] > 0),
            foe_mean_eff,
            no_target_eff,
        )
        no_target_eff = torch.where(
            target_flags[..., 2] > 0, partner_eff, no_target_eff
        )
        no_target_eff = torch.where(target_flags[..., 3] > 0, self_eff, no_target_eff)
        current_target_eff = torch.where(no_target_move, no_target_eff, targeted_eff)

        current_power = base_power * current_stab * current_target_eff / 4.0
        tera_power_gain = (
            base_power * current_target_eff * (post_tera_stab - current_stab) / 4.0
        )
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
                ((current_target_eff > 0) & (current_target_eff < 1)).to(
                    dtype=torch.float32
                )
                * is_move,
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

    @staticmethod
    def _active_slot_present_from_context(
        matchup_context: dict[str, torch.Tensor],
        slot: int,
    ) -> torch.Tensor:
        if "own_active_present" not in matchup_context:
            batch_size = matchup_context["own_active_indices"].shape[0]
            device = matchup_context["own_active_indices"].device
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return matchup_context["own_active_present"][:, slot].to(dtype=torch.bool)

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
        original_match = (move_types == type_1) | (
            (type_2 != 0) & (move_types == type_2)
        )
        tera_match = (tera_type != 0) & (move_types == tera_type)
        terastallized = (
            actor_continuous[:, TERASTALLIZED_OFFSET : TERASTALLIZED_OFFSET + 1] > 0.5
        )

        current = torch.ones_like(move_types, dtype=torch.float32)
        current = torch.where(original_match, torch.full_like(current, 1.5), current)
        terastallized_stab = torch.where(
            tera_match & original_match,
            torch.full_like(current, 2.0),
            torch.where(
                tera_match | original_match, torch.full_like(current, 1.5), current
            ),
        )
        current = torch.where(terastallized, terastallized_stab, current)

        can_tera_now = can_tera_slot.view(-1, 1) & ~terastallized
        post_tera = torch.where(
            can_tera_now,
            torch.where(
                tera_match & original_match,
                torch.full_like(current, 2.0),
                torch.where(
                    tera_match | original_match,
                    torch.full_like(current, 1.5),
                    torch.ones_like(current),
                ),
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
        move_types = move_types.clamp(
            min=0, max=self._type_effectiveness_lookup.shape[0] - 1
        )
        target_type_1 = target_type_1.clamp(
            min=0, max=self._type_effectiveness_lookup.shape[1] - 1
        )
        target_type_2 = target_type_2.clamp(
            min=0, max=self._type_effectiveness_lookup.shape[1] - 1
        )
        multiplier_1 = self._type_effectiveness_lookup[move_types, target_type_1]
        multiplier_2 = self._type_effectiveness_lookup[move_types, target_type_2]
        multiplier_2 = torch.where(
            target_type_2 != 0, multiplier_2, torch.ones_like(multiplier_1)
        )
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
        type_2 = torch.where(
            terastallized, torch.zeros_like(types[..., 1]), types[..., 1]
        )
        return type_1, type_2

    @staticmethod
    def _gather_team_rows(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_index = indices.view(
            indices.shape[0], 1, *([1] * (values.ndim - 2))
        ).expand(-1, 1, *values.shape[2:])
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
            return torch.ones(
                batch_size,
                dtype=torch.bool,
                device=battle_obs[f"{side_prefix}_active_indices"].device,
            )
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
            target_slots.append(
                target_idx
                if kind == "move"
                else (TARGET_NONE if kind == "pass" else target_none_slot)
            )
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
            raise ValueError(
                "TeamPreviewHead requires pick_size >= 2 so it can choose a lead pair."
            )
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
        include_policy_stats: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch, team_size, pdim = pokemon_reps.shape
        if team_mask is None:
            available = torch.ones(
                batch, team_size, dtype=torch.bool, device=context.device
            )
        else:
            available = ~team_mask.clone()
        if (available.sum(dim=1) < self.pick_size).any():
            raise ValueError(
                f"TeamPreviewHead needs at least {self.pick_size} available pokemon per batch item."
            )

        canonical_bring = None if picks is None else picks.sort(dim=1).values
        team_positions = torch.arange(team_size, device=context.device).view(
            1, team_size
        )

        bring_log_probs: list[torch.Tensor] = []
        bring_entropies: list[torch.Tensor] = []
        bring_picks: list[torch.Tensor] = []
        picked_sum = torch.zeros(batch, pdim, device=context.device)
        next_min_index = torch.zeros(batch, 1, dtype=torch.long, device=context.device)
        remaining = available.clone()

        for step in range(self.pick_size):
            ctx = self.context_proj(torch.cat([context, picked_sum], dim=-1))
            opponent_summary = self.opponent_attention(
                ctx, opponent_reps, opponent_mask
            )

            ctx_expanded = ctx.unsqueeze(1).expand(-1, team_size, -1)
            opponent_expanded = opponent_summary.unsqueeze(1).expand(-1, team_size, -1)
            bring_inputs = torch.cat(
                [pokemon_reps, ctx_expanded, opponent_expanded], dim=-1
            )
            bring_scores = self.bring_score(bring_inputs).squeeze(-1)
            picks_left = self.pick_size - step
            suffix_available = torch.cumsum(
                remaining.flip(1).to(dtype=torch.int64), dim=1
            ).flip(1)
            valid = (
                remaining
                & (team_positions >= next_min_index)
                & (suffix_available >= picks_left)
            )
            bring_scores = bring_scores.masked_fill(~valid, -1e8)
            if include_policy_stats:
                bring_log_probs_step = torch.log_softmax(bring_scores, dim=1)
                bring_probs_step = bring_log_probs_step.exp()

            if canonical_bring is None:
                chosen = (
                    bring_scores.argmax(dim=1)
                    if greedy
                    else _sample_categorical_logits(bring_scores)
                )
            else:
                chosen = canonical_bring[:, step]

            # Sampled picks come from the masked scores above. Only external
            # training labels need a host-visible validity check.
            if (
                picks is not None
                and not valid.gather(1, chosen.unsqueeze(1)).squeeze(1).all()
            ):
                raise RuntimeError(
                    "TeamPreviewHead received an invalid canonical bring target."
                )

            bring_picks.append(chosen)
            if include_policy_stats:
                bring_log_probs.append(
                    bring_log_probs_step.gather(1, chosen.unsqueeze(1)).squeeze(1)
                )
                bring_entropies.append(
                    -(bring_probs_step * bring_log_probs_step).sum(dim=1)
                )

            remaining.scatter_(1, chosen.unsqueeze(1), False)
            next_min_index = chosen.unsqueeze(1) + 1
            chosen_rep = pokemon_reps.gather(
                1,
                chosen.unsqueeze(1).unsqueeze(2).expand(-1, -1, pdim),
            ).squeeze(1)
            picked_sum = picked_sum + chosen_rep

        bring_picks_tensor = torch.stack(bring_picks, dim=1)

        bring_mask = torch.zeros(
            batch, team_size, dtype=torch.bool, device=context.device
        )
        bring_mask.scatter_(1, bring_picks_tensor, True)
        bring_summary = picked_sum / float(self.pick_size)
        lead_context = self.lead_context_proj(
            torch.cat([context, bring_summary], dim=-1)
        )
        lead_opponent_summary = self.opponent_attention(
            lead_context, opponent_reps, opponent_mask
        )

        left = pokemon_reps.unsqueeze(2).expand(-1, team_size, team_size, -1)
        right = pokemon_reps.unsqueeze(1).expand(-1, team_size, team_size, -1)
        bring_expanded = (
            bring_summary.unsqueeze(1).unsqueeze(2).expand(-1, team_size, team_size, -1)
        )
        opponent_expanded = (
            lead_opponent_summary.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, team_size, team_size, -1)
        )
        lead_context_expanded = (
            lead_context.unsqueeze(1).unsqueeze(2).expand(-1, team_size, team_size, -1)
        )
        lead_pair_logits = self.lead_pair_score(
            torch.cat(
                [left, right, bring_expanded, opponent_expanded, lead_context_expanded],
                dim=-1,
            )
        ).squeeze(-1)

        invalid_pairs = torch.eye(
            team_size, dtype=torch.bool, device=context.device
        ).unsqueeze(0)
        invalid_pairs = (
            invalid_pairs | (~bring_mask).unsqueeze(1) | (~bring_mask).unsqueeze(2)
        )
        if team_mask is not None:
            invalid_pairs = (
                invalid_pairs | team_mask.unsqueeze(1) | team_mask.unsqueeze(2)
            )
        lead_pair_logits = lead_pair_logits.masked_fill(invalid_pairs, -1e8).reshape(
            batch, team_size * team_size
        )

        if picks is None:
            chosen_pair = (
                lead_pair_logits.argmax(dim=1)
                if greedy
                else _sample_categorical_logits(lead_pair_logits)
            )
        else:
            chosen_pair = picks[:, 0] * team_size + picks[:, 1]

        valid_pairs = (~invalid_pairs).reshape(batch, team_size * team_size)
        if (
            picks is not None
            and not valid_pairs.gather(1, chosen_pair.unsqueeze(1)).squeeze(1).all()
        ):
            raise RuntimeError("TeamPreviewHead received an invalid lead pair target.")

        lead_pair = torch.stack(
            [chosen_pair // team_size, chosen_pair % team_size], dim=1
        )

        remaining_bring_mask = bring_mask.clone()
        remaining_bring_mask.scatter_(1, lead_pair, False)
        bench_tail = team_positions.expand(batch, -1).masked_fill(
            ~remaining_bring_mask, team_size
        )
        bench_tail = bench_tail.sort(dim=1).values[:, : self.pick_size - 2]
        final_picks = torch.cat([lead_pair, bench_tail], dim=1)

        result = {
            "bring_picks": bring_picks_tensor,
            "lead_pair": lead_pair,
            "picks": final_picks,
        }
        if include_policy_stats:
            bring_log_prob = torch.stack(bring_log_probs, dim=1).sum(dim=1)
            bring_entropy = torch.stack(bring_entropies, dim=1).sum(dim=1)
            lead_log_probs = torch.log_softmax(lead_pair_logits, dim=1)
            lead_probs = lead_log_probs.exp()
            lead_log_prob = lead_log_probs.gather(1, chosen_pair.unsqueeze(1)).squeeze(
                1
            )
            lead_entropy = -(lead_probs * lead_log_probs).sum(dim=1)
            result.update(
                {
                    "bring_log_prob": bring_log_prob,
                    "lead_log_prob": lead_log_prob,
                    "log_prob": bring_log_prob + lead_log_prob,
                    "bring_entropy": bring_entropy,
                    "lead_entropy": lead_entropy,
                    "entropy": bring_entropy + lead_entropy,
                }
            )
        return result


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
        bring_logits = self.bring_score(
            torch.cat([pokemon_reps, context_expanded], dim=-1)
        ).squeeze(-1)

        left = pokemon_reps.unsqueeze(2).expand(-1, team_size, team_size, -1)
        right = pokemon_reps.unsqueeze(1).expand(-1, team_size, team_size, -1)
        pair_context = (
            context.unsqueeze(1).unsqueeze(2).expand(-1, team_size, team_size, -1)
        )
        lead_pair_logits = self.lead_pair_score(
            torch.cat([left, right, pair_context], dim=-1)
        ).squeeze(-1)

        invalid_pairs = torch.eye(
            team_size, dtype=torch.bool, device=context.device
        ).unsqueeze(0)
        if team_mask is not None:
            invalid_pairs = (
                invalid_pairs | team_mask.unsqueeze(1) | team_mask.unsqueeze(2)
            )
        lead_pair_logits = lead_pair_logits.masked_fill(invalid_pairs, -1e8).reshape(
            batch_size, team_size * team_size
        )

        return {
            "opponent_bring_logits": bring_logits,
            "opponent_lead_pair_logits": lead_pair_logits,
        }


class _ZeroInitializedLinear(nn.Linear):
    """Linear residual branch that does not consume the model RNG stream."""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class HiddenInfoHead(nn.Module):
    """Predict trainer-only opponent attributes from spatial and recurrent state.

    These heads run only during PPO updates and never expose their predictions
    or exact packed-team labels to policy inference. A zero-initialized
    residual adapter adds recurrent context to every opponent representation;
    its initial outputs therefore exactly match the spatial-only head while
    later updates can train temporal belief features through memory.
    """

    def __init__(
        self,
        pokemon_dim: int,
        context_dim: int,
        num_items: int,
        num_abilities: int,
        num_types: int,
        num_moves: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.context_adapter = _ZeroInitializedLinear(context_dim, pokemon_dim)
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

    def forward(
        self,
        opp_reps: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context is not None:
            if context.shape[:-1] != opp_reps.shape[:-2]:
                raise ValueError(
                    "Hidden-info context must match every opponent-representation "
                    "batch/sequence dimension."
                )
            opp_reps = opp_reps + self.context_adapter(context).unsqueeze(-2)
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
        self.own_bench_attention = MaskedSetAttention(
            context_dim, pokemon_dim, hidden_dim
        )
        self.opp_bench_attention = MaskedSetAttention(
            context_dim, pokemon_dim, hidden_dim
        )
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
