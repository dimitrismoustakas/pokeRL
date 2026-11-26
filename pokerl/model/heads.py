"""Action head (autoregressive for doubles) and value head."""

import itertools
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import MOVE_MECHANICS_DIM, MOVE_SCOPE_SCOPED, MOVE_SCOPE_TARGETED
from ..env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    MAX_TEAM_SLOTS,
    NUM_INTENT_KINDS,
    NUM_MOVES,
    NUM_TARGETS,
    NUM_TERA_CHOICES,
    SWITCH_OFFSET,
    TARGET_NONE,
    TERA_NONE,
    decode_action,
)
from ..env.observation import (
    ACTIVE_MECHANICS_FEATURE_DIM,
    BOOSTS_OFFSET,
    NUM_PUBLIC_TARGET_POSITIONS,
    TERASTALLIZED_OFFSET,
)
from ..field_transition import (
    FIELD_CHANGE_COMPONENT_DIM,
    PUBLIC_FIELD_DIM,
    TERRAIN_CLASS_SLICE,
    WEATHER_CLASS_SLICE,
    advanced_public_field_persistence,
    decode_public_field_transition,
)
from ..public_consequence import (
    MAX_PUBLIC_HIT_COUNT,
    MOVE_BRANCH_TARGET_PHASE,
    NUM_MOVE_BRANCHES,
    NUM_NONMOVE_REASONS,
    NUM_TARGET_INTERACTION_REASONS,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
)


def _sample_categorical_logits(
    logits: torch.Tensor,
    exponential: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample from categorical logits with the Gumbel-max trick.

    If ``E`` is exponential(1), ``-log(E)`` is a standard Gumbel sample.
    This is distribution-equivalent to multinomial sampling from softmax
    probabilities while avoiding an unnecessary softmax.
    """
    sample_logits = logits.detach()
    if exponential is None:
        exponential = torch.empty_like(sample_logits).exponential_()
    elif (
        exponential.shape != sample_logits.shape
        or exponential.dtype != sample_logits.dtype
        or exponential.device != sample_logits.device
    ):
        raise ValueError(
            "External categorical exponentials must exactly match the logits."
        )
    return (sample_logits - exponential.log()).argmax(dim=1)


class _PublicConsequenceState(NamedTuple):
    """Inputs retained only long enough to decode the selected action."""

    score_hidden: torch.Tensor
    public_target_reps: torch.Tensor
    target_mechanics: torch.Tensor
    all_logits: dict[str, torch.Tensor] | None
    field_persistence: torch.Tensor | None = None


class MaskedSetAttention(nn.Module):
    """Query-conditioned pooling over a masked set of pokemon representations.

    ``query`` may contain one query per batch item (``[B, D]``) or any number
    of structured queries (``[B, ..., D]``). The opponent keys and values are
    projected once, then shared by every query belonging to that batch item.
    """

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
        """Pool already-projected items for one or more structured queries."""
        return self._pool_projected_queries(
            self.query_proj(query),
            key_proj,
            values,
            mask,
        )

    def _pool_projected_queries(
        self,
        query_proj: torch.Tensor,
        key_proj: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool items when callers can form query projections without concat."""
        batch_size, _, _ = key_proj.shape
        query_shape = query_proj.shape
        flat_query_proj = query_proj.reshape(batch_size, -1, key_proj.shape[-1])
        scores = torch.bmm(flat_query_proj, key_proj.transpose(1, 2)) / self.scale

        if mask is not None:
            expanded_mask = mask.unsqueeze(1)
            scores = scores.masked_fill(expanded_mask, -torch.inf)
            # Avoid an all--infinity softmax. These rows represent an empty
            # public opponent set and must pool to exactly zero.
            all_items_masked = (mask.sum(dim=1) == mask.shape[1]).view(
                batch_size,
                1,
                1,
            )
            scores = scores.masked_fill(all_items_masked, 0.0)

        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights.masked_fill(expanded_mask, 0.0)

        summaries = torch.bmm(weights, values)
        return summaries.reshape(*query_shape[:-1], values.shape[-1])


class _GroupedEmbedding(nn.Embedding):
    """Indices in the same group share one trainable embedding."""

    def __init__(self, groups: list[int], embedding_dim: int):
        super().__init__(max(groups) + 1, embedding_dim)
        self.register_buffer("groups", torch.tensor(groups, dtype=torch.long))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return super().forward(self.groups[indices])


class ActionHead(nn.Module):
    """Autoregressive action selection for doubles.

    Each slot scores every fixed action against the live battle entities:
    the acting pokemon, its partner, both opponent actives, and the action's
    explicit target or switch destination. The second slot additionally sees
    the first slot's chosen action embedding so the joint decision remains
    coordinated. Terastallization is modeled as a third conditional decision
    over {no tera, tera slot 0, tera slot 1}.
    """

    _BASE_MATCHUP_FEATURE_DIM = 27
    _MATCHUP_FEATURE_DIM = _BASE_MATCHUP_FEATURE_DIM + ACTIVE_MECHANICS_FEATURE_DIM
    _TARGET_FLAG_DIM = 6
    _TARGET_MECHANICS_DIM = 6
    # Immediate public target outcomes are a small conditional prediction
    # problem.  Keeping their latent narrower than the policy candidate state
    # avoids expanding every move/target pair to the full action-head width.
    _MAX_TARGET_CONSEQUENCE_LATENT_DIM = 64
    _PUBLIC_CONSEQUENCE_ADAPTER_DIM = (
        1
        + NUM_NONMOVE_REASONS
        + NUM_MOVE_BRANCHES
        + NUM_PUBLIC_TARGET_POSITIONS
        * (2 + NUM_TARGET_INTERACTION_REASONS + MAX_PUBLIC_HIT_COUNT)
        + 2
        + PUBLIC_CONSEQUENCE_FIELD_DIM
    )
    _INITIAL_TERA_USE_PROB = 0.20

    def __init__(
        self,
        context_dim: int,
        pokemon_dim: int,
        hidden_dim: int = 256,
        *,
        stellar_type_index: int = -1,
        tera_blast_index: int = -1,
        move_base_power_lookup: torch.Tensor | None = None,
        move_accuracy_lookup: torch.Tensor | None = None,
        move_bypasses_accuracy_lookup: torch.Tensor | None = None,
        move_ignores_evasion_lookup: torch.Tensor | None = None,
        move_target_scope_lookup: torch.Tensor | None = None,
        move_multihit_lookup: torch.Tensor | None = None,
        move_priority_lookup: torch.Tensor | None = None,
        move_target_flags: torch.Tensor | None = None,
        move_mechanics_lookup: torch.Tensor | None = None,
        move_embedding_dim: int = 32,
        type_effectiveness: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_actions = ACTIONS_PER_SLOT
        self._stellar_type_index = stellar_type_index
        self._tera_blast_index = tera_blast_index
        if move_embedding_dim <= 0:
            raise ValueError("move_embedding_dim must be positive.")
        self.move_embedding_dim = move_embedding_dim

        # Moves share an action embedding for each target. Switches and pass
        # retain distinct embeddings. The offered move supplies its identity.
        action_groups = list(range(NUM_TARGETS)) * NUM_MOVES
        action_groups.extend(
            range(NUM_TARGETS, NUM_TARGETS + self.num_actions - SWITCH_OFFSET)
        )
        self.action_id_emb = _GroupedEmbedding(action_groups, hidden_dim)
        self.kind_emb = nn.Embedding(3, hidden_dim)
        self.move_slot_emb = _GroupedEmbedding([0] * NUM_MOVES + [1], hidden_dim)
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
        self.move_semantic_proj = nn.Sequential(
            nn.Linear(move_embedding_dim + MOVE_MECHANICS_DIM, hidden_dim),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Predict public consequences for every candidate from the same hidden
        # representation used by its raw policy score.  The compact adapter is
        # the only route from these predictions back into the policy logits.
        # Its inputs are detached so actor gradients cannot train the
        # consequence model into a convenient (but ungrounded) latent code.
        self.public_consequence_action_head = nn.Linear(
            hidden_dim,
            1 + NUM_NONMOVE_REASONS + NUM_MOVE_BRANCHES + 2,
        )
        self.public_consequence_field_change_head = nn.Linear(
            hidden_dim,
            FIELD_CHANGE_COMPONENT_DIM,
        )
        self.public_consequence_field_alternative_head = nn.Linear(
            hidden_dim,
            PUBLIC_CONSEQUENCE_FIELD_DIM,
        )
        # The untrained model starts at the known no-event transition instead
        # of inventing 33 unrelated field values.  The alternative values are
        # deliberately neutral until exact next-request labels train them.
        nn.init.zeros_(self.public_consequence_field_change_head.weight)
        nn.init.constant_(self.public_consequence_field_change_head.bias, -12.0)
        nn.init.zeros_(self.public_consequence_field_alternative_head.weight)
        nn.init.zeros_(self.public_consequence_field_alternative_head.bias)
        target_consequence_dim = min(
            hidden_dim, self._MAX_TARGET_CONSEQUENCE_LATENT_DIM
        )
        self.public_consequence_target_candidate_proj = nn.Linear(
            hidden_dim, target_consequence_dim
        )
        self.public_consequence_target_entity_proj = nn.Linear(
            pokemon_dim, target_consequence_dim
        )
        self.public_consequence_target_position_emb = nn.Embedding(
            NUM_PUBLIC_TARGET_POSITIONS,
            target_consequence_dim,
        )
        self.public_consequence_target_mechanics_proj = nn.Sequential(
            nn.Linear(self._TARGET_MECHANICS_DIM, target_consequence_dim),
            nn.ReLU(),
        )
        self.public_consequence_target_head = nn.Linear(
            target_consequence_dim,
            2 + NUM_TARGET_INTERACTION_REASONS + MAX_PUBLIC_HIT_COUNT,
        )
        self.public_consequence_policy_adapter = nn.Linear(
            hidden_dim,
            self._PUBLIC_CONSEQUENCE_ADAPTER_DIM,
            bias=False,
        )
        nn.init.zeros_(self.public_consequence_policy_adapter.weight)
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
            "_action_entity_slot",
            torch.where(
                metadata["is_switch"],
                NUM_TARGETS + metadata["switch_slot"],
                metadata["target_slot"],
            ),
            persistent=False,
        )
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
                move_base_power_lookup,
                move_accuracy_lookup,
                move_bypasses_accuracy_lookup,
                move_ignores_evasion_lookup,
                move_target_scope_lookup,
                move_multihit_lookup,
                move_priority_lookup,
                move_target_flags,
                type_effectiveness,
            )
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
            "_move_bypasses_accuracy_lookup",
            self._ensure_float_buffer(move_bypasses_accuracy_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_ignores_evasion_lookup",
            self._ensure_float_buffer(move_ignores_evasion_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_target_scope_lookup",
            self._ensure_long_buffer(move_target_scope_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_multihit_lookup",
            self._ensure_float_buffer(move_multihit_lookup),
            persistent=False,
        )
        self.register_buffer(
            "_move_priority_lookup",
            self._ensure_float_buffer(move_priority_lookup),
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
            "_move_mechanics_lookup",
            self._ensure_float_buffer(
                move_mechanics_lookup,
                dims=2,
                width=MOVE_MECHANICS_DIM,
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
        move_embedding: nn.Embedding | None = None,
        sampling_exponentials: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None = None,
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
        field_persistence = self._field_persistence(
            battle_obs,
            batch_size=batch_size,
            device=device,
            dtype=context.dtype,
        )
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
        move_semantics_1 = self._offered_move_semantics(
            matchup_context,
            actor_slot=0,
            move_embedding=move_embedding,
            batch_size=batch_size,
            device=device,
        )
        move_semantics_2 = self._offered_move_semantics(
            matchup_context,
            actor_slot=1,
            move_embedding=move_embedding,
            batch_size=batch_size,
            device=device,
        )
        public_target_reps = torch.stack([rep1, rep2, opp1, opp2], dim=1)
        target_mechanics_1 = self._candidate_public_target_mechanics(
            matchup_context,
            actor_slot=0,
            matchup_features=matchup_features_1,
        )
        target_mechanics_2 = self._candidate_public_target_mechanics(
            matchup_context,
            actor_slot=1,
            matchup_features=matchup_features_2,
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
        logits1, public_consequence_state_1 = self._score_actions(
            query1,
            rep1,
            rep2,
            own_team_reps,
            opp_active_reps,
            matchup_features_1,
            move_semantics_1,
            action_features,
            public_target_reps,
            target_mechanics_1,
            field_persistence,
            mask1,
        )
        logits1 = logits1.masked_fill(~mask1, -1e8)
        a1, log_prob1, entropy1 = self._categorical_choice(
            logits1,
            actions=None if actions is None else actions[:, 0],
            greedy=greedy,
            include_stats=include_policy_stats,
            sampling_exponential=(
                None if sampling_exponentials is None else sampling_exponentials[0]
            ),
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
        selected_move_semantics_1 = move_semantics_1[
            batch_indices, self._action_move_slot[a1]
        ]
        a1_conditioning = a1_emb + selected_move_semantics_1
        chosen_features_1 = matchup_features_1[batch_indices, a1]
        base_query2 = self.query2(
            torch.cat([context, rep2, rep1, opp1, opp2, a1_conditioning], dim=-1)
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
        logits2, public_consequence_state_2 = self._score_actions(
            query2,
            rep2,
            rep1,
            own_team_reps,
            opp_active_reps,
            matchup_features_2,
            move_semantics_2,
            action_features,
            public_target_reps,
            target_mechanics_2,
            field_persistence,
            mask2,
        )
        logits2 = logits2.masked_fill(~mask2, -1e8)
        a2, log_prob2, entropy2 = self._categorical_choice(
            logits2,
            actions=None if actions is None else actions[:, 1],
            greedy=greedy,
            include_stats=include_policy_stats,
            sampling_exponential=(
                None if sampling_exponentials is None else sampling_exponentials[1]
            ),
        )

        if (
            validate_supplied_actions
            and not mask2.gather(1, a2.unsqueeze(1)).squeeze(1).all()
        ):
            raise RuntimeError(
                "ActionHead received an illegal second-slot action for the current joint mask."
            )

        a2_emb = self.action_id_emb(a2)
        selected_move_semantics_2 = move_semantics_2[
            batch_indices, self._action_move_slot[a2]
        ]
        a2_conditioning = a2_emb + selected_move_semantics_2
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
                    a1_conditioning,
                    a2_conditioning,
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
                else _sample_categorical_logits(
                    tera_sampling_logits,
                    None if sampling_exponentials is None else sampling_exponentials[2],
                )
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
            public_consequence_1 = self._selected_public_consequence_logits(
                public_consequence_state_1,
                a1,
            )
            public_consequence_2 = self._selected_public_consequence_logits(
                public_consequence_state_2,
                a2,
            )
            for consequence_name in (
                "move_emitted_logits",
                "nonmove_reason_logits",
                "move_branch_logits",
                "route_logits",
                "interaction_logits",
                "interaction_reason_logits",
                "hit_count_logits",
                "cant_logits",
                "field_logits",
                "field_probs",
                "field_change_logits",
                "field_alternative_logits",
                "field_persistence",
            ):
                result[f"public_consequence_{consequence_name}"] = torch.stack(
                    [
                        public_consequence_1[consequence_name],
                        public_consequence_2[consequence_name],
                    ],
                    dim=1,
                )
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
            gate_wait_prob * gate_wait_log_prob + gate_use_prob * gate_use_log_prob
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
        sampling_exponential: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if include_stats:
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            if actions is None:
                selected = (
                    logits.argmax(dim=1)
                    if greedy
                    else _sample_categorical_logits(logits, sampling_exponential)
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
                selected = _sample_categorical_logits(logits, sampling_exponential)
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
        move_semantics: torch.Tensor,
        action_features: torch.Tensor,
        public_target_reps: torch.Tensor,
        target_mechanics: torch.Tensor,
        field_persistence: torch.Tensor | None = None,
        legal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, _PublicConsequenceState]:
        # The first score layer is linear across these five inputs. Project each
        # query, target and offered move once before gathering the 27 action rows.
        layer = self.score[0]
        w_query, w_action, w_entity, w_matchup, w_move = layer.weight.chunk(5, dim=1)
        entities = torch.cat(
            (
                opp_active_reps,
                partner_rep.unsqueeze(1),
                actor_rep.unsqueeze(1),
                torch.zeros_like(actor_rep).unsqueeze(1),
                own_team_reps,
            ),
            dim=1,
        )
        entity_scores = F.linear(self.entity_proj(entities), w_entity)
        move_scores = F.linear(move_semantics, w_move)
        score_hidden = self.score[1](
            F.linear(query, w_query, layer.bias).unsqueeze(1)
            + F.linear(action_features, w_action)
            + entity_scores.index_select(1, self._action_entity_slot)
            + F.linear(self.matchup_proj(matchup_features), w_matchup)
            + move_scores.index_select(1, self._action_move_slot)
        )
        raw_logits = self.score[2](score_hidden).squeeze(-1)

        # PPO already has each slot's legal mask. The adapter only changes
        # move scores, so decode the legal move candidates in the minibatch.
        # CUDA rollout graphs take the fixed-shape path.
        compact = torch.is_grad_enabled() and legal is not None
        all_logits = None
        with torch.no_grad():
            if compact:
                rows, moves = legal[:, :SWITCH_OFFSET].nonzero(as_tuple=True)
                hidden = score_hidden[rows, moves].unsqueeze(1)
                persistence = (
                    field_persistence[rows] if field_persistence is not None else None
                )
                decoded = self._public_consequence_action_logits(hidden, persistence)
                decoded.update(
                    self._public_consequence_target_logits(
                        hidden,
                        public_target_reps[rows],
                        target_mechanics[rows, moves].unsqueeze(1),
                    )
                )
                predictions = self._public_consequence_adapter_predictions(
                    decoded, matchup_features[rows, moves].unsqueeze(1)
                ).squeeze(1)
            else:
                all_logits = self._all_public_consequence_logits(
                    score_hidden,
                    public_target_reps,
                    target_mechanics,
                    field_persistence,
                )
                predictions = self._public_consequence_adapter_predictions(
                    all_logits, matchup_features
                )

        weights = self.public_consequence_policy_adapter(query)
        if compact:
            residual = (weights[rows] * predictions).sum(-1) / math.sqrt(
                predictions.shape[-1]
            )
            logits = raw_logits.index_put((rows, moves), residual, accumulate=True)
        else:
            residual = (weights.unsqueeze(1) * predictions).sum(-1) / math.sqrt(
                predictions.shape[-1]
            )
            logits = raw_logits + residual * self._action_is_move.to(
                residual.dtype
            ).unsqueeze(0)
        state = _PublicConsequenceState(
            score_hidden,
            public_target_reps,
            target_mechanics,
            None if torch.is_grad_enabled() else all_logits,
            field_persistence,
        )
        return logits, state

    @staticmethod
    def _field_persistence(
        battle_obs: dict[str, torch.Tensor] | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Advance public timers once per board, before scoring either actor."""
        if battle_obs is None or "field" not in battle_obs:
            field = torch.zeros(
                batch_size,
                PUBLIC_FIELD_DIM,
                device=device,
                dtype=dtype,
            )
            field[..., WEATHER_CLASS_SLICE.start] = 1.0
            field[..., TERRAIN_CLASS_SLICE.start] = 1.0
            return field

        encoded_field = battle_obs["field"]
        if (
            encoded_field.ndim != 2
            or encoded_field.shape[0] != batch_size
            or encoded_field.shape[-1] < PUBLIC_FIELD_DIM
        ):
            raise ValueError(
                "ActionHead battle_obs['field'] must have shape "
                f"({batch_size}, >= {PUBLIC_FIELD_DIM})."
            )
        return advanced_public_field_persistence(
            encoded_field[..., :PUBLIC_FIELD_DIM].to(device=device, dtype=dtype)
        )

    def _public_consequence_action_logits(
        self,
        score_hidden: torch.Tensor,
        field_persistence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        action_consequence_logits = self.public_consequence_action_head(score_hidden)
        (
            move_emitted_logits,
            nonmove_reason_logits,
            move_branch_logits,
            cant_logits,
        ) = action_consequence_logits.split(
            (
                1,
                NUM_NONMOVE_REASONS,
                NUM_MOVE_BRANCHES,
                2,
            ),
            dim=-1,
        )
        field_change_logits = self.public_consequence_field_change_head(score_hidden)
        field_alternative_logits = self.public_consequence_field_alternative_head(
            score_hidden
        )
        if field_persistence is None:
            field_persistence = self._field_persistence(
                None,
                batch_size=score_hidden.shape[0],
                device=score_hidden.device,
                dtype=score_hidden.dtype,
            )
        if score_hidden.ndim == 3:
            field_persistence = field_persistence.unsqueeze(1).expand_as(
                field_alternative_logits
            )
        field_transition = decode_public_field_transition(
            persistence=field_persistence,
            change_logits=field_change_logits,
            alternative_logits=field_alternative_logits,
        )
        move_emitted_logits = move_emitted_logits.squeeze(-1)
        return {
            "move_emitted_logits": move_emitted_logits,
            "nonmove_reason_logits": nonmove_reason_logits,
            "move_branch_logits": move_branch_logits,
            "cant_logits": cant_logits,
            "field_logits": field_transition.logits,
            "field_probs": field_transition.probabilities,
            "field_change_logits": field_change_logits,
            "field_alternative_logits": field_alternative_logits,
            "field_persistence": field_transition.persistence,
        }

    def _public_consequence_target_logits(
        self,
        move_score_hidden: torch.Tensor,
        public_target_reps: torch.Tensor,
        target_mechanics: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode move/target pairs with additive, amortized projections."""

        target_rep_features = self.public_consequence_target_entity_proj(
            public_target_reps
        ).unsqueeze(1)
        target_positions = self.public_consequence_target_position_emb.weight.view(
            1, 1, NUM_PUBLIC_TARGET_POSITIONS, -1
        )
        target_candidate_features = self.public_consequence_target_candidate_proj(
            move_score_hidden
        ).unsqueeze(2)
        target_mechanics_features = self.public_consequence_target_mechanics_proj(
            target_mechanics
        )
        target_hidden = F.relu(
            target_candidate_features
            + target_rep_features
            + target_positions
            + target_mechanics_features
        )
        target_consequence_logits = self.public_consequence_target_head(target_hidden)
        (
            route_logits,
            interaction_logits,
            interaction_reason_logits,
            hit_count_logits,
        ) = target_consequence_logits.split(
            (1, 1, NUM_TARGET_INTERACTION_REASONS, MAX_PUBLIC_HIT_COUNT),
            dim=-1,
        )
        route_logits = route_logits.squeeze(-1)
        interaction_logits = interaction_logits.squeeze(-1)
        return {
            "route_logits": route_logits,
            "interaction_logits": interaction_logits,
            "interaction_reason_logits": interaction_reason_logits,
            "hit_count_logits": hit_count_logits,
        }

    def _all_public_consequence_logits(
        self,
        score_hidden: torch.Tensor,
        public_target_reps: torch.Tensor,
        target_mechanics: torch.Tensor,
        field_persistence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        logits = self._public_consequence_action_logits(
            score_hidden,
            field_persistence,
        )
        move_target_logits = self._public_consequence_target_logits(
            score_hidden[:, :SWITCH_OFFSET],
            public_target_reps,
            target_mechanics,
        )
        for name, move_logits in move_target_logits.items():
            nonmove_shape = (
                move_logits.shape[0],
                self.num_actions - SWITCH_OFFSET,
                *move_logits.shape[2:],
            )
            logits[name] = torch.cat(
                [move_logits, move_logits.new_zeros(nonmove_shape)],
                dim=1,
            )
        return logits

    def _selected_public_consequence_logits(
        self,
        state: _PublicConsequenceState,
        selected_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_indices = torch.arange(
            selected_actions.shape[0], device=selected_actions.device
        )
        if state.all_logits is not None:
            return {
                name: value[batch_indices, selected_actions]
                for name, value in state.all_logits.items()
            }

        selected_hidden = state.score_hidden[batch_indices, selected_actions]
        logits = self._public_consequence_action_logits(
            selected_hidden,
            state.field_persistence,
        )
        move_indices = selected_actions.clamp(max=SWITCH_OFFSET - 1)
        selected_mechanics = state.target_mechanics[batch_indices, move_indices]
        target_logits = self._public_consequence_target_logits(
            selected_hidden.unsqueeze(1),
            state.public_target_reps,
            selected_mechanics.unsqueeze(1),
        )
        is_move = self._action_is_move[selected_actions]
        for name, value in target_logits.items():
            value = value.squeeze(1)
            mask_shape = (is_move.shape[0],) + (1,) * (value.ndim - 1)
            logits[name] = value * is_move.reshape(mask_shape).to(value.dtype)
        return logits

    def _public_consequence_adapter_predictions(
        self,
        logits: dict[str, torch.Tensor],
        matchup_features: torch.Tensor,
    ) -> torch.Tensor:
        move_emitted_logits = logits["move_emitted_logits"]
        nonmove_reason_logits = logits["nonmove_reason_logits"]
        move_branch_logits = logits["move_branch_logits"]
        route_logits = logits["route_logits"]
        interaction_logits = logits["interaction_logits"]
        interaction_reason_logits = logits["interaction_reason_logits"]
        hit_count_logits = logits["hit_count_logits"]

        move_probability = move_emitted_logits.sigmoid()
        nonmove_joint = (1.0 - move_probability).unsqueeze(-1) * (
            nonmove_reason_logits.softmax(dim=-1)
        )
        move_event_probabilities = torch.cat(
            [move_probability.unsqueeze(-1), nonmove_joint], dim=-1
        )
        move_branch_joint = move_probability.unsqueeze(-1) * move_branch_logits.softmax(
            dim=-1
        )
        target_phase_probability = move_branch_joint[
            ..., MOVE_BRANCH_TARGET_PHASE
        ].unsqueeze(-1)
        route_joint = target_phase_probability * route_logits.sigmoid()
        # A marker is an observed target interaction, not whole-move failure:
        # Clear Amulet can block Parting Shot's stat drop while its pivot still
        # succeeds.  The adapter receives grounded probabilities, never a
        # hand-written positive/negative valuation of that marker.
        marked_joint = route_joint * interaction_logits.sigmoid()
        unmarked_joint = route_joint - marked_joint
        interaction_reason_joint = marked_joint.unsqueeze(
            -1
        ) * interaction_reason_logits.softmax(dim=-1)
        multihit_candidate = matchup_features[..., 9].unsqueeze(-1).unsqueeze(-1)
        hit_count_joint = (
            unmarked_joint.unsqueeze(-1)
            * hit_count_logits.softmax(dim=-1)
            * multihit_candidate
        )
        target_predictions = torch.cat(
            [
                route_joint.unsqueeze(-1),
                unmarked_joint.unsqueeze(-1),
                interaction_reason_joint,
                hit_count_joint,
            ],
            dim=-1,
        ).flatten(start_dim=-2)
        return torch.cat(
            [
                move_event_probabilities,
                move_branch_joint,
                target_predictions,
                logits["cant_logits"].sigmoid(),
                logits["field_probs"],
            ],
            dim=-1,
        )

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
        # A replay can use new weights in the same model storage. Capture the
        # embeddings that produce these features with the rest of the forward.
        if cache_allowed and self._action_ids.is_cuda:
            cache_allowed = not torch.cuda.is_current_stream_capturing()
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
                    "own_request_moves",
                    "own_request_move_metadata",
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
            "own_request_moves": battle_obs["own_request_moves"],
            "own_request_move_metadata": battle_obs["own_request_move_metadata"],
            "own_active_indices": own_active_indices,
            "own_types_active": own_types_active,
            "own_cont_active": own_cont_active,
            "opp_cont_active": opp_cont_active,
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
        if "active_mechanics" in battle_obs:
            context["active_mechanics"] = battle_obs["active_mechanics"]
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

        own_types_active = matchup_context["own_types_active"]
        own_cont_active = matchup_context["own_cont_active"]

        actor_request_moves = matchup_context["own_request_moves"][:, actor_slot]
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

        action_move_ids = actor_request_moves.gather(1, matchup_context["move_slots"])
        action_move_ids = torch.where(
            matchup_context["is_move_bool"],
            action_move_ids,
            torch.zeros_like(action_move_ids),
        )
        action_move_ids = action_move_ids.clamp(
            max=self._move_base_power_lookup.shape[0] - 1
        )
        move_types, categories = self._gather_team_slots(
            matchup_context["own_request_move_metadata"][:, actor_slot],
            matchup_context["move_slots"],
        ).unbind(dim=-1)
        base_power = self._move_base_power_lookup[action_move_ids]
        accuracy = self._move_accuracy_lookup[action_move_ids]
        bypasses_accuracy = self._move_bypasses_accuracy_lookup[action_move_ids]
        ignores_evasion = self._move_ignores_evasion_lookup[action_move_ids]
        target_scopes = self._move_target_scope_lookup[action_move_ids]
        is_multihit = self._move_multihit_lookup[action_move_ids]
        priority = self._move_priority_lookup[action_move_ids]
        target_flags = self._move_target_flags[action_move_ids]
        is_move = matchup_context["is_move"]
        is_switch = matchup_context["is_switch"]
        is_pass = matchup_context["is_pass"]

        tera_blast = action_move_ids == self._tera_blast_index
        post_move_types = torch.where(tera_blast, actor_types[:, 2:3], move_types)
        current_stab, post_tera_stab = self._stab_multipliers(
            move_types=move_types,
            post_move_types=post_move_types,
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
        stellar_vs_foes = (
            1 + matchup_context["opp_cont_active"][:, :, TERASTALLIZED_OFFSET]
        )
        stellar_vs_self = 1 + actor_continuous[:, TERASTALLIZED_OFFSET]
        stellar_vs_partner = 1 + partner_continuous[:, TERASTALLIZED_OFFSET]
        neutral = torch.ones_like(stellar_vs_self)
        stellar_bank = torch.stack(
            [
                stellar_vs_foes[:, 0],
                stellar_vs_foes[:, 1],
                stellar_vs_partner,
                stellar_vs_self,
                neutral,
                neutral,
            ],
            dim=1,
        )
        stellar_vs_target = stellar_bank.gather(1, target_indices)

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
        stellar_move = move_types == self._stellar_type_index
        opp1_eff = torch.where(stellar_move, stellar_vs_foes[:, :1], opp1_eff)
        opp2_eff = torch.where(stellar_move, stellar_vs_foes[:, 1:], opp2_eff)
        self_eff = torch.where(stellar_move, stellar_vs_self[:, None], self_eff)
        partner_eff = torch.where(
            stellar_move, stellar_vs_partner[:, None], partner_eff
        )
        targeted_eff = torch.where(stellar_move, stellar_vs_target, targeted_eff)
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

        stellar_blast = tera_blast & (actor_types[:, 2:3] == self._stellar_type_index)
        base_power = torch.where(stellar_move & tera_blast, 100 / 250, base_power)
        current_power = base_power * current_stab * current_target_eff / 4.0
        post_eff = self._type_effectiveness(
            post_move_types, target_type_1, target_type_2
        )
        post_eff = torch.where(stellar_blast, stellar_vs_target, post_eff)
        post_eff = torch.where(
            tera_blast & can_tera_slot[:, None] & target_present,
            post_eff,
            current_target_eff,
        )
        post_power = torch.where(
            stellar_blast & can_tera_slot[:, None], 100 / 250, base_power
        )
        tera_power_gain = post_power * post_eff * post_tera_stab / 4.0 - current_power
        target_eff_scaled = current_target_eff / 4.0
        foe_best_scaled = foe_best_eff / 4.0
        foe_mean_scaled = foe_mean_eff / 4.0

        base_features = torch.stack(
            [
                is_move,
                is_switch,
                is_pass,
                base_power * is_move,
                accuracy * is_move,
                bypasses_accuracy * is_move,
                ignores_evasion * is_move,
                (target_scopes == MOVE_SCOPE_SCOPED).to(torch.float32) * is_move,
                (target_scopes == MOVE_SCOPE_TARGETED).to(torch.float32) * is_move,
                is_multihit * is_move,
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
        active_mechanics = matchup_context.get("active_mechanics")
        if active_mechanics is None:
            direct_mechanics = base_features.new_zeros(
                (batch_size, ACTIVE_MECHANICS_FEATURE_DIM)
            )
        else:
            direct_mechanics = active_mechanics[:, actor_slot].to(
                dtype=base_features.dtype
            )
        return torch.cat(
            [
                base_features,
                direct_mechanics.unsqueeze(1).expand(-1, self.num_actions, -1),
            ],
            dim=-1,
        )

    def _offered_move_semantics(
        self,
        matchup_context: dict[str, torch.Tensor] | None,
        *,
        actor_slot: int,
        move_embedding: nn.Embedding | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Embed the four offered moves and append a zero for switch and pass."""
        if matchup_context is None or move_embedding is None:
            semantics = self.move_semantic_proj[0].weight.new_zeros(
                (batch_size, NUM_MOVES, self.move_semantic_proj[0].in_features),
                device=device,
            )
        else:
            if move_embedding.embedding_dim != self.move_embedding_dim:
                raise ValueError(
                    "ActionHead move embedding width does not match its configured "
                    f"semantic width: embedding={move_embedding.embedding_dim}, "
                    f"configured={self.move_embedding_dim}."
                )
            move_ids = matchup_context["own_request_moves"][:, actor_slot]
            embedding_ids = move_ids.clamp(0, move_embedding.num_embeddings - 1)
            mechanics_ids = move_ids.clamp(0, self._move_mechanics_lookup.shape[0] - 1)
            semantics = torch.cat(
                (
                    move_embedding(embedding_ids),
                    self._move_mechanics_lookup[mechanics_ids],
                ),
                dim=-1,
            )
        return F.pad(self.move_semantic_proj(semantics), (0, 0, 0, 1))

    def _candidate_public_target_mechanics(
        self,
        matchup_context: dict[str, torch.Tensor] | None,
        *,
        actor_slot: int,
        matchup_features: torch.Tensor,
    ) -> torch.Tensor:
        """Expose source accuracy and each public target's evasion explicitly."""

        batch_size = matchup_features.shape[0]
        if matchup_context is None:
            return matchup_features.new_zeros(
                (
                    batch_size,
                    SWITCH_OFFSET,
                    NUM_PUBLIC_TARGET_POSITIONS,
                    self._TARGET_MECHANICS_DIM,
                )
            )
        own_continuous = matchup_context["own_cont_active"]
        opponent_continuous = matchup_context["opp_cont_active"]
        source_accuracy_stage = own_continuous[:, actor_slot, BOOSTS_OFFSET + 5]
        target_evasion_stage = torch.cat(
            [
                own_continuous[:, :, BOOSTS_OFFSET + 6],
                opponent_continuous[:, :, BOOSTS_OFFSET + 6],
            ],
            dim=1,
        )
        own_present = torch.stack(
            [
                self._active_slot_present_from_context(matchup_context, 0),
                self._active_slot_present_from_context(matchup_context, 1),
            ],
            dim=1,
        )
        target_present = torch.cat(
            [own_present, matchup_context["opp_present"]], dim=1
        ).to(matchup_features.dtype)
        target_shape = (
            batch_size,
            SWITCH_OFFSET,
            NUM_PUBLIC_TARGET_POSITIONS,
        )
        move_matchup_features = matchup_features[:, :SWITCH_OFFSET]
        return torch.stack(
            [
                move_matchup_features[..., 4].unsqueeze(-1).expand(target_shape),
                move_matchup_features[..., 5].unsqueeze(-1).expand(target_shape),
                move_matchup_features[..., 6].unsqueeze(-1).expand(target_shape),
                source_accuracy_stage[:, None, None].expand(target_shape),
                target_evasion_stage[:, None, :].expand(target_shape),
                target_present[:, None, :].expand(target_shape),
            ],
            dim=-1,
        )

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
        post_move_types: torch.Tensor,
        actor_types: torch.Tensor,
        actor_continuous: torch.Tensor,
        can_tera_slot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        move_types = move_types.to(dtype=torch.long)
        type_1 = actor_types[:, 0:1]
        type_2 = actor_types[:, 1:2]
        tera_type = actor_types[:, 2:3]
        original_match = ((type_1 != 0) & (move_types == type_1)) | (
            (type_2 != 0) & (move_types == type_2)
        )
        tera_match = (
            (tera_type != 0)
            & (move_types == tera_type)
            & (tera_type != self._stellar_type_index)
        )
        terastallized = (
            actor_continuous[:, TERASTALLIZED_OFFSET : TERASTALLIZED_OFFSET + 1] > 0.5
        )

        current = torch.ones_like(move_types, dtype=torch.float32)
        current = torch.where(original_match, torch.full_like(current, 1.5), current)
        current = torch.where(terastallized & tera_match, current + 0.5, current)
        post_original = ((type_1 != 0) & (post_move_types == type_1)) | (
            (type_2 != 0) & (post_move_types == type_2)
        )
        post_match = (tera_type != 0) & (post_move_types == tera_type)
        post_stab = 1.0 + 0.5 * (post_original.float() + post_match.float())
        post_stab = torch.where(
            tera_type == self._stellar_type_index,
            torch.where(post_original, 2.0, 4915 / 4096),
            post_stab,
        )
        post_tera = torch.where(
            can_tera_slot.view(-1, 1) & ~terastallized, post_stab, current
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

    def _defensive_type_pair(
        self,
        types: torch.Tensor,
        continuous: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tera_type = types[..., 2]
        # Stellar retains both ordinary defensive types.
        replaces_types = (continuous[..., TERASTALLIZED_OFFSET] > 0.5) & (
            tera_type != self._stellar_type_index
        )
        type_1 = torch.where(replaces_types, tera_type, types[..., 0])
        type_2 = torch.where(
            replaces_types, torch.zeros_like(types[..., 1]), types[..., 1]
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
    """Exact bring-subset and ordered-lead distribution.

    A preview action is one unordered bring subset followed by one ordered lead
    pair from that subset. All legal subsets and lead orders are represented
    explicitly, so roster indices cannot create a prior through a
    variable-branching sequential sampler. Subset scoring is symmetric over its
    members, while every member and ordered lead pair retains
    candidate-conditioned attention over the public opponent roster.

    Live sampled inference scores every bring subset in one vectorized pass,
    samples one, and scores only its lead orders. Policy-statistics and
    deterministic argmax paths score the complete joint action space so entropy
    and action selection are exact.
    """

    _MAX_EXACT_ACTIONS = 4096

    def __init__(
        self,
        pokemon_dim: int,
        context_dim: int,
        pick_size: int = 4,
        *,
        team_size: int = 6,
    ):
        super().__init__()
        for name, value in (("team_size", team_size), ("pick_size", pick_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"TeamPreviewHead {name} must be an integer.")
        if pick_size < 2:
            raise ValueError(
                "TeamPreviewHead requires pick_size >= 2 so it can choose a lead pair."
            )
        if team_size < pick_size:
            raise ValueError(
                "TeamPreviewHead requires team_size >= pick_size "
                f"(got team_size={team_size}, pick_size={pick_size})."
            )
        action_count = math.comb(team_size, pick_size) * pick_size * (pick_size - 1)
        if action_count > self._MAX_EXACT_ACTIONS:
            raise ValueError(
                "TeamPreviewHead exact action space is too large: "
                f"{action_count} actions exceeds the supported maximum of "
                f"{self._MAX_EXACT_ACTIONS}."
            )
        self.team_size = team_size
        self.pick_size = pick_size
        self.bring_opponent_attention = MaskedSetAttention(
            context_dim + pokemon_dim, pokemon_dim, 128
        )
        self.lead_opponent_attention = MaskedSetAttention(
            context_dim + pokemon_dim * 2, pokemon_dim, 128
        )
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

        bring_subsets = tuple(itertools.combinations(range(team_size), pick_size))
        lead_orders = tuple(itertools.permutations(range(pick_size), 2))
        joint_picks: list[tuple[int, ...]] = []
        for subset in bring_subsets:
            for left_position, right_position in lead_orders:
                lead_positions = {left_position, right_position}
                bench_tail = tuple(
                    slot
                    for position, slot in enumerate(subset)
                    if position not in lead_positions
                )
                joint_picks.append(
                    (
                        subset[left_position],
                        subset[right_position],
                        *bench_tail,
                    )
                )

        # These are architecture constants, not learned state. Keeping them
        # non-persistent preserves a compact state dict while .to(device)
        # still moves them once with the module.
        self.register_buffer(
            "_bring_subsets",
            torch.tensor(bring_subsets, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_lead_orders",
            torch.tensor(lead_orders, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_joint_picks",
            torch.tensor(joint_picks, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_joint_subset_indices",
            torch.arange(len(joint_picks), dtype=torch.long) // len(lead_orders),
            persistent=False,
        )

    @property
    def legal_action_count(self) -> int:
        return int(self._joint_picks.shape[0])

    @property
    def action_table(self) -> torch.Tensor:
        """Canonical action rows in joint-log-probability column order."""
        return self._joint_picks.detach().clone()

    def _score_bring_subsets(
        self,
        pokemon_reps: torch.Tensor,
        opponent_reps: torch.Tensor,
        context: torch.Tensor,
        opponent_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score every subset symmetrically and project opponent items once."""
        subset_members = pokemon_reps[:, self._bring_subsets]
        subset_sum = subset_members.sum(dim=2)
        bring_summary = subset_sum / float(self.pick_size)

        # Score every member as a candidate conditioned on the other members in
        # its proposed subset. Averaging the shared member score makes the subset
        # logit invariant to the members' roster positions and ordering.
        other_member_sum = subset_sum.unsqueeze(2) - subset_members
        context_width = context.shape[-1]
        pokemon_width = pokemon_reps.shape[-1]
        context_projection = self.context_proj
        member_context = F.linear(
            context[:, None, None, :],
            context_projection.weight[:, :context_width],
            context_projection.bias,
        ) + F.linear(
            other_member_sum,
            context_projection.weight[:, context_width:],
            None,
        )
        opponent_keys, opponent_values = self.bring_opponent_attention._project_items(
            opponent_reps
        )
        attention_query = self.bring_opponent_attention.query_proj
        projected_queries = F.linear(
            member_context,
            attention_query.weight[:, :context_width],
            attention_query.bias,
        ) + F.linear(
            subset_members,
            attention_query.weight[:, context_width:],
            None,
        )
        candidate_opponent_summaries = (
            self.bring_opponent_attention._pool_projected_queries(
                projected_queries,
                opponent_keys,
                opponent_values,
                opponent_mask,
            )
        )
        bring_hidden_layer = self.bring_score[0]
        bring_hidden = F.linear(
            subset_members,
            bring_hidden_layer.weight[:, :pokemon_width],
            bring_hidden_layer.bias,
        )
        bring_hidden = bring_hidden + F.linear(
            member_context,
            bring_hidden_layer.weight[
                :,
                pokemon_width : pokemon_width + context_width,
            ],
            None,
        )
        bring_hidden = bring_hidden + F.linear(
            candidate_opponent_summaries,
            bring_hidden_layer.weight[:, pokemon_width + context_width :],
            None,
        )
        member_scores = F.linear(
            F.relu(bring_hidden),
            self.bring_score[2].weight,
            self.bring_score[2].bias,
        ).squeeze(-1)
        subset_logits = member_scores.mean(dim=2)
        return subset_logits, subset_members, bring_summary

    def _score_lead_orders(
        self,
        subset_members: torch.Tensor,
        bring_summary: torch.Tensor,
        context: torch.Tensor,
        opponent_keys: torch.Tensor,
        opponent_values: torch.Tensor,
        opponent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score every ordered lead pair for one or more bring subsets."""
        left = subset_members.index_select(-2, self._lead_orders[:, 0])
        right = subset_members.index_select(-2, self._lead_orders[:, 1])
        context_width = context.shape[-1]
        pokemon_width = subset_members.shape[-1]
        lead_context_projection = self.lead_context_proj
        lead_context = F.linear(
            context,
            lead_context_projection.weight[:, :context_width],
            lead_context_projection.bias,
        ) + F.linear(
            bring_summary,
            lead_context_projection.weight[:, context_width:],
            None,
        )
        attention_query = self.lead_opponent_attention.query_proj
        projected_queries = F.linear(
            lead_context.unsqueeze(-2),
            attention_query.weight[:, :context_width],
            attention_query.bias,
        )
        projected_queries = projected_queries + F.linear(
            left,
            attention_query.weight[
                :,
                context_width : context_width + pokemon_width,
            ],
            None,
        )
        projected_queries = projected_queries + F.linear(
            right,
            attention_query.weight[:, context_width + pokemon_width :],
            None,
        )
        lead_pair_opponent_summaries = (
            self.lead_opponent_attention._pool_projected_queries(
                projected_queries,
                opponent_keys,
                opponent_values,
                opponent_mask,
            )
        )
        lead_hidden_layer = self.lead_pair_score[0]
        first = 0
        second = first + pokemon_width
        third = second + pokemon_width
        fourth = third + pokemon_width
        fifth = fourth + pokemon_width
        lead_hidden = F.linear(
            left,
            lead_hidden_layer.weight[:, first:second],
            lead_hidden_layer.bias,
        )
        lead_hidden = lead_hidden + F.linear(
            right,
            lead_hidden_layer.weight[:, second:third],
            None,
        )
        lead_hidden = lead_hidden + F.linear(
            bring_summary.unsqueeze(-2),
            lead_hidden_layer.weight[:, third:fourth],
            None,
        )
        lead_hidden = lead_hidden + F.linear(
            lead_pair_opponent_summaries,
            lead_hidden_layer.weight[:, fourth:fifth],
            None,
        )
        lead_hidden = lead_hidden + F.linear(
            lead_context.unsqueeze(-2),
            lead_hidden_layer.weight[:, fifth:],
            None,
        )
        return F.linear(
            F.relu(lead_hidden),
            self.lead_pair_score[2].weight,
            self.lead_pair_score[2].bias,
        ).squeeze(-1)

    def _score_all_lead_orders(
        self,
        pokemon_reps: torch.Tensor,
        bring_summaries: torch.Tensor,
        context: torch.Tensor,
        opponent_keys: torch.Tensor,
        opponent_values: torch.Tensor,
        opponent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Score the full joint table without re-projecting repeated members."""
        context_width = context.shape[-1]
        pokemon_width = pokemon_reps.shape[-1]
        left_slots = self._joint_picks[:, 0]
        right_slots = self._joint_picks[:, 1]
        subset_indices = self._joint_subset_indices

        lead_context_projection = self.lead_context_proj
        lead_contexts = F.linear(
            context.unsqueeze(1),
            lead_context_projection.weight[:, :context_width],
            lead_context_projection.bias,
        ) + F.linear(
            bring_summaries,
            lead_context_projection.weight[:, context_width:],
            None,
        )

        attention_query = self.lead_opponent_attention.query_proj
        projected_contexts = F.linear(
            lead_contexts,
            attention_query.weight[:, :context_width],
            attention_query.bias,
        )
        projected_left = F.linear(
            pokemon_reps,
            attention_query.weight[
                :,
                context_width : context_width + pokemon_width,
            ],
            None,
        )
        projected_right = F.linear(
            pokemon_reps,
            attention_query.weight[:, context_width + pokemon_width :],
            None,
        )
        projected_queries = (
            projected_contexts[:, subset_indices]
            + projected_left[:, left_slots]
            + projected_right[:, right_slots]
        )
        opponent_summaries = self.lead_opponent_attention._pool_projected_queries(
            projected_queries,
            opponent_keys,
            opponent_values,
            opponent_mask,
        )

        lead_hidden_layer = self.lead_pair_score[0]
        first = 0
        second = first + pokemon_width
        third = second + pokemon_width
        fourth = third + pokemon_width
        fifth = fourth + pokemon_width
        projected_left_hidden = F.linear(
            pokemon_reps,
            lead_hidden_layer.weight[:, first:second],
            lead_hidden_layer.bias,
        )
        projected_right_hidden = F.linear(
            pokemon_reps,
            lead_hidden_layer.weight[:, second:third],
            None,
        )
        projected_bring_hidden = F.linear(
            bring_summaries,
            lead_hidden_layer.weight[:, third:fourth],
            None,
        )
        projected_opponent_hidden = F.linear(
            opponent_summaries,
            lead_hidden_layer.weight[:, fourth:fifth],
            None,
        )
        projected_context_hidden = F.linear(
            lead_contexts,
            lead_hidden_layer.weight[:, fifth:],
            None,
        )
        lead_hidden = (
            projected_left_hidden[:, left_slots]
            + projected_right_hidden[:, right_slots]
            + projected_bring_hidden[:, subset_indices]
            + projected_opponent_hidden
            + projected_context_hidden[:, subset_indices]
        )
        logits = F.linear(
            F.relu(lead_hidden),
            self.lead_pair_score[2].weight,
            self.lead_pair_score[2].bias,
        ).squeeze(-1)
        return logits.reshape(
            pokemon_reps.shape[0],
            self._bring_subsets.shape[0],
            self._lead_orders.shape[0],
        )

    def _external_action_indices(
        self,
        picks: torch.Tensor,
        valid_subsets: torch.Tensor,
    ) -> torch.Tensor:
        """Map externally supplied pick rows to exact joint-action indices."""
        batch_size = valid_subsets.shape[0]
        if picks.shape != (batch_size, self.pick_size):
            raise ValueError(
                "TeamPreviewHead expected picks shape "
                f"{(batch_size, self.pick_size)}, got {tuple(picks.shape)}."
            )
        if picks.device != self._joint_picks.device:
            raise ValueError("TeamPreviewHead picks must share the model device.")
        if picks.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("TeamPreviewHead picks must contain integer indices.")

        normalized = torch.cat(
            [picks[:, :2], picks[:, 2:].sort(dim=1).values],
            dim=1,
        )
        matches = (
            normalized.unsqueeze(1).eq(self._joint_picks.unsqueeze(0)).sum(dim=-1)
            == self.pick_size
        )
        action_indices = matches.to(torch.int64).argmax(dim=1)
        valid_actions = (
            valid_subsets.unsqueeze(2)
            .expand(-1, -1, self._lead_orders.shape[0])
            .reshape(batch_size, -1)
        )
        selected_is_valid = valid_actions.gather(
            1, action_indices.unsqueeze(1)
        ).squeeze(1)
        if not ((matches.sum(dim=1) == 1) & selected_is_valid).all():
            raise RuntimeError(
                "TeamPreviewHead received an invalid canonical bring target."
            )
        return action_indices

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
        batch, team_size, _ = pokemon_reps.shape
        opponent_batch, opponent_size, _ = opponent_reps.shape
        if team_size != self.team_size:
            raise ValueError(
                "TeamPreviewHead input team size does not match its configured "
                f"action space: input={team_size}, configured={self.team_size}."
            )
        if opponent_batch != batch:
            raise ValueError(
                "TeamPreviewHead expected batch-aligned own and opponent "
                f"representations, got {batch} and {opponent_batch}."
            )
        if team_mask is not None and team_mask.shape != (batch, team_size):
            raise ValueError(
                "TeamPreviewHead expected team_mask shape "
                f"{(batch, team_size)}, got {tuple(team_mask.shape)}."
            )
        if opponent_mask is not None and opponent_mask.shape != (
            batch,
            opponent_size,
        ):
            raise ValueError(
                "TeamPreviewHead expected opponent_mask shape "
                f"{(batch, opponent_size)}, got {tuple(opponent_mask.shape)}."
            )
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

        bring_logits, subset_members, bring_summaries = self._score_bring_subsets(
            pokemon_reps,
            opponent_reps,
            context,
            opponent_mask,
        )
        valid_subsets = available[:, self._bring_subsets].sum(dim=2) == self.pick_size
        bring_logits = bring_logits.masked_fill(~valid_subsets, -torch.inf)
        lead_count = self._lead_orders.shape[0]
        batch_indices = torch.arange(batch, device=context.device)
        bring_log_probs_all: torch.Tensor | None = None
        lead_log_probs_all: torch.Tensor | None = None
        joint_log_probs: torch.Tensor | None = None
        sampled_log_prob: torch.Tensor | None = None

        if picks is not None:
            action_indices = self._external_action_indices(picks, valid_subsets)
            bring_indices = torch.div(
                action_indices,
                lead_count,
                rounding_mode="floor",
            )
            lead_indices = action_indices.remainder(lead_count)
        elif not include_policy_stats and not greedy:
            # Fast live path: enumerate 15 bring subsets, then only the 12 lead
            # orders belonging to the sampled subset.
            bring_indices = _sample_categorical_logits(bring_logits)
            selected_members = subset_members[batch_indices, bring_indices]
            selected_summaries = bring_summaries[batch_indices, bring_indices]
            lead_opponent_keys, lead_opponent_values = (
                self.lead_opponent_attention._project_items(opponent_reps)
            )
            selected_lead_logits = self._score_lead_orders(
                selected_members,
                selected_summaries,
                context,
                lead_opponent_keys,
                lead_opponent_values,
                opponent_mask,
            )
            lead_indices = _sample_categorical_logits(selected_lead_logits)
            action_indices = bring_indices * lead_count + lead_indices
            sampled_log_prob = torch.log_softmax(bring_logits, dim=1).gather(
                1, bring_indices.unsqueeze(1)
            ).squeeze(1) + torch.log_softmax(selected_lead_logits, dim=1).gather(
                1, lead_indices.unsqueeze(1)
            ).squeeze(1)
        else:
            # Exact statistics and deterministic argmax selection need every
            # conditional lead distribution in the small joint action space.
            lead_opponent_keys, lead_opponent_values = (
                self.lead_opponent_attention._project_items(opponent_reps)
            )
            all_lead_logits = self._score_all_lead_orders(
                pokemon_reps,
                bring_summaries,
                context,
                lead_opponent_keys,
                lead_opponent_values,
                opponent_mask,
            )
            bring_log_probs_all = torch.log_softmax(bring_logits, dim=1)
            lead_log_probs_all = torch.log_softmax(all_lead_logits, dim=2)
            joint_log_probs = bring_log_probs_all.unsqueeze(2) + lead_log_probs_all
            if greedy:
                action_indices = joint_log_probs.reshape(batch, -1).argmax(dim=1)
                bring_indices = torch.div(
                    action_indices,
                    lead_count,
                    rounding_mode="floor",
                )
                lead_indices = action_indices.remainder(lead_count)
            else:
                bring_indices = _sample_categorical_logits(bring_logits)
                selected_lead_logits = all_lead_logits[
                    batch_indices,
                    bring_indices,
                ]
                lead_indices = _sample_categorical_logits(selected_lead_logits)
                action_indices = bring_indices * lead_count + lead_indices

        final_picks = self._joint_picks[action_indices]
        bring_picks = self._bring_subsets[bring_indices]
        lead_pair = final_picks[:, :2]
        result = {
            "bring_picks": bring_picks,
            "lead_pair": lead_pair,
            "picks": final_picks,
        }
        if not include_policy_stats:
            if sampled_log_prob is not None:
                result["log_prob"] = sampled_log_prob
            return result

        # ``picks`` may have taken the external-action branch, in which case the
        # exact joint tensors have not been computed yet.
        if joint_log_probs is None:
            lead_opponent_keys, lead_opponent_values = (
                self.lead_opponent_attention._project_items(opponent_reps)
            )
            all_lead_logits = self._score_all_lead_orders(
                pokemon_reps,
                bring_summaries,
                context,
                lead_opponent_keys,
                lead_opponent_values,
                opponent_mask,
            )
            bring_log_probs_all = torch.log_softmax(bring_logits, dim=1)
            lead_log_probs_all = torch.log_softmax(all_lead_logits, dim=2)
            joint_log_probs = bring_log_probs_all.unsqueeze(2) + lead_log_probs_all

        assert bring_log_probs_all is not None
        assert lead_log_probs_all is not None
        bring_probs = bring_log_probs_all.exp()
        lead_probs = lead_log_probs_all.exp()
        finite_bring_log_probs = bring_log_probs_all.masked_fill(
            ~valid_subsets,
            0.0,
        )
        bring_entropy = -(bring_probs * finite_bring_log_probs).sum(dim=1)
        conditional_lead_entropies = -(lead_probs * lead_log_probs_all).sum(dim=2)
        lead_entropy = (bring_probs * conditional_lead_entropies).sum(dim=1)
        bring_log_prob = bring_log_probs_all.gather(
            1,
            bring_indices.unsqueeze(1),
        ).squeeze(1)
        lead_log_prob = lead_log_probs_all[
            batch_indices,
            bring_indices,
            lead_indices,
        ]
        result.update(
            {
                "bring_log_prob": bring_log_prob,
                "lead_log_prob": lead_log_prob,
                "log_prob": bring_log_prob + lead_log_prob,
                "bring_entropy": bring_entropy,
                "lead_entropy": lead_entropy,
                "entropy": bring_entropy + lead_entropy,
                "joint_log_probs": joint_log_probs.reshape(batch, -1),
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
    """Encode compact beliefs and decode trainer-only opponent attributes.

    The compact residual encoder consumes only public spatial and recurrent
    state and is therefore safe for policy inference. Full attribute decoders
    run only during PPO updates; neither their predictions nor exact packed-team
    labels enter the policy. The zero-initialized adapter initially preserves
    the spatial opponent representations exactly.
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
        self.policy_belief_adapter = _ZeroInitializedLinear(context_dim, pokemon_dim)
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

    def encode(
        self,
        opp_reps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Add the public recurrent belief residual to each opponent slot."""
        return self.encode_with_residual(
            opp_reps,
            self.context_residual(context),
        )

    def context_residual(self, context: torch.Tensor) -> torch.Tensor:
        """Project public recurrent context once for one or more slot groups."""
        return self.policy_belief_adapter(context)

    @staticmethod
    def encode_with_residual(
        opp_reps: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast one precomputed belief residual across opponent slots."""
        if residual.shape[:-1] != opp_reps.shape[:-2]:
            raise ValueError(
                "Hidden-info residual must match every opponent-representation "
                "batch/sequence dimension."
            )
        if residual.shape[-1] != opp_reps.shape[-1]:
            raise ValueError(
                "Hidden-info residual width must match opponent representations."
            )
        return opp_reps + residual.unsqueeze(-2)

    def decode(self, belief_reps: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode trainer-supervised attributes from compact belief features."""
        return {
            "item_logits": self.item_mlp(belief_reps),
            "ability_logits": self.ability_mlp(belief_reps),
            "tera_type_logits": self.tera_type_mlp(belief_reps),
            "move_logits": self.move_mlp(belief_reps),
        }

    def forward(
        self,
        opp_reps: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        belief_reps = opp_reps if context is None else self.encode(opp_reps, context)
        return self.decode(belief_reps)


class OpponentIntentHead(nn.Module):
    """Predict trainer-only semantic opponent decisions.

    Request-local action indices are deliberately absent. Each opponent slot
    predicts action kind, global move identity, semantic target, and switch
    species. Tera is factorized into use-now and conditional slot decisions.
    The head runs only during PPO updates; its labels and logits never enter
    observations or rollout inference.
    """

    def __init__(
        self,
        context_dim: int,
        pokemon_dim: int,
        num_moves: int,
        num_species: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.slot_query = nn.Sequential(
            nn.Linear(context_dim + pokemon_dim * 4, hidden_dim),
            nn.ReLU(),
        )
        self.opponent_team_attention = MaskedSetAttention(
            hidden_dim,
            pokemon_dim,
            hidden_dim,
        )
        self.slot_fuse = nn.Sequential(
            nn.Linear(hidden_dim + pokemon_dim, hidden_dim),
            nn.ReLU(),
        )
        self.kind_head = nn.Linear(hidden_dim, NUM_INTENT_KINDS)
        self.move_head = nn.Linear(hidden_dim, num_moves)
        self.target_head = nn.Linear(hidden_dim, NUM_TARGETS)
        self.switch_species_head = nn.Linear(hidden_dim, num_species)
        self.tera_query = nn.Sequential(
            nn.Linear(context_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.tera_use_head = nn.Linear(hidden_dim, 2)
        self.tera_slot_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        context: torch.Tensor,
        opp_team_reps: torch.Tensor,
        opp_active_reps: torch.Tensor,
        own_active_reps: torch.Tensor,
        opp_team_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if opp_active_reps.shape[-2] != 2 or own_active_reps.shape[-2] != 2:
            raise ValueError("Opponent intent requires two active slots per side.")
        context_slots = context.unsqueeze(-2).expand(-1, 2, -1)
        opponent_partner_reps = opp_active_reps.flip(dims=(-2,))
        own_first = own_active_reps[:, :1].expand(-1, 2, -1)
        own_second = own_active_reps[:, 1:].expand(-1, 2, -1)
        base_slot_query = self.slot_query(
            torch.cat(
                [
                    context_slots,
                    opp_active_reps,
                    opponent_partner_reps,
                    own_first,
                    own_second,
                ],
                dim=-1,
            )
        )
        opponent_team_summary = self.opponent_team_attention(
            base_slot_query,
            opp_team_reps,
            opp_team_mask,
        )
        slot_latent = self.slot_fuse(
            torch.cat([base_slot_query, opponent_team_summary], dim=-1)
        )
        tera_latent = self.tera_query(
            torch.cat([context, slot_latent.flatten(start_dim=1)], dim=-1)
        )
        return {
            "opp_intent_kind_logits": self.kind_head(slot_latent),
            "opp_intent_move_logits": self.move_head(slot_latent),
            "opp_intent_target_logits": self.target_head(slot_latent),
            "opp_intent_switch_species_logits": self.switch_species_head(slot_latent),
            "opp_intent_tera_use_logits": self.tera_use_head(tera_latent),
            "opp_intent_tera_slot_logits": self.tera_slot_score(slot_latent).squeeze(
                -1
            ),
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
