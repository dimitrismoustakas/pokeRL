"""Full agent: spatial encoder -> memory -> action/value heads."""

from collections.abc import Mapping

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..data import GameData
from ..env.observation import (
    CONTINUOUS_DIM,
    NUM_EVENT_KINDS,
    NUM_EVENT_POSITIONS,
    ROLE_BENCH,
)
from ..env.actions import (
    ACTIONS_PER_SLOT,
    NUM_INTENT_KINDS,
    NUM_TERA_CHOICES,
)
from .encoders import (
    FieldEncoder,
    PokemonEncoder,
    RecentEventEncoder,
    SpatialEncoder,
    TeamEncoder,
)
from .memory import build_memory
from .heads import (
    ActionHead,
    HiddenInfoHead,
    OpponentActionHead,
    OpponentPreviewHead,
    TeamPreviewHead,
    ValueHead,
)


def load_state_dict_compatible(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, tuple[str, ...]]:
    """Load the compatible subset of a checkpoint state dict into a module."""
    current_state = module.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for key, value in state_dict.items():
        current_value = current_state.get(key)
        if current_value is None or current_value.shape != value.shape:
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    result = module.load_state_dict(compatible_state, strict=False)
    return {
        "missing_keys": tuple(sorted(result.missing_keys)),
        "skipped_keys": tuple(sorted(skipped_keys)),
        "unexpected_keys": tuple(sorted(result.unexpected_keys)),
    }


class PokeAgent(nn.Module):
    """Complete agent for doubles battles.

    Combines spatial encoding (order-invariant team attention),
    temporal memory (pluggable LSTM/identity), and autoregressive
    action selection for doubles + team preview.
    """

    def __init__(self, config: ModelConfig, game_data: GameData):
        super().__init__()
        self.config = config
        self.game_data_signature = game_data.vocab_signature()

        # --- Spatial encoder ---
        pokemon_encoder = PokemonEncoder(
            num_species=game_data.num_species,
            num_moves=game_data.num_moves,
            num_abilities=game_data.num_abilities,
            num_items=game_data.num_items,
            num_types=game_data.num_types,
            continuous_dim=CONTINUOUS_DIM,
            species_dim=config.species_dim,
            move_dim=config.move_dim,
            ability_dim=config.ability_dim,
            item_dim=config.item_dim,
            type_dim=config.type_dim,
            hidden_dim=config.pokemon_hidden,
            out_dim=config.pokemon_out,
        )

        team_own = TeamEncoder(
            config.pokemon_out,
            config.team_attention_heads,
            config.team_attention_layers,
        )
        team_opp = TeamEncoder(
            config.pokemon_out,
            config.team_attention_heads,
            config.team_attention_layers,
        )
        field_encoder = FieldEncoder(config.field_out)
        recent_event_encoder = RecentEventEncoder(
            num_moves=game_data.num_moves,
            num_species=game_data.num_species,
            num_types=game_data.num_types,
            num_event_kinds=NUM_EVENT_KINDS,
            num_positions=NUM_EVENT_POSITIONS,
            num_intent_kinds=NUM_INTENT_KINDS,
            out_dim=config.recent_event_out,
        )

        self.spatial = SpatialEncoder(
            pokemon_encoder,
            team_own,
            team_opp,
            field_encoder,
            recent_event_encoder,
        )

        # --- Memory ---
        self.memory = build_memory(
            config.memory_type, self.spatial.context_dim, config.memory_hidden
        )
        memory_out = config.memory_hidden

        # --- Heads ---
        move_stats = game_data.move_stats_matrix()
        self.action_head = ActionHead(
            memory_out,
            config.pokemon_out,
            config.action_hidden,
            move_type_lookup=torch.from_numpy(game_data.move_type_indices()),
            move_base_power_lookup=torch.from_numpy(move_stats[:, 0]),
            move_accuracy_lookup=torch.from_numpy(move_stats[:, 1]),
            move_priority_lookup=torch.from_numpy(game_data.move_priority_vector()),
            move_category_lookup=torch.from_numpy(game_data.move_category_indices()),
            move_target_flags=torch.from_numpy(game_data.move_target_flags()),
            type_effectiveness=torch.from_numpy(game_data.type_effectiveness_matrix()),
        )
        self.team_preview_head = TeamPreviewHead(
            config.pokemon_out, memory_out, config.pick_size
        )
        self.opponent_preview_head = OpponentPreviewHead(config.pokemon_out, memory_out)
        self.value_head = ValueHead(memory_out, config.pokemon_out, config.value_hidden)
        self.preview_value_head = ValueHead(
            memory_out, config.pokemon_out, config.value_hidden
        )

        # --- Auxiliary heads (trained during PPO update only) ---
        self.hidden_info_head = HiddenInfoHead(
            pokemon_dim=config.pokemon_out,
            context_dim=memory_out,
            num_items=game_data.num_items,
            num_abilities=game_data.num_abilities,
            num_types=game_data.num_types,
            num_moves=game_data.num_moves,
        )
        self.opponent_action_head = OpponentActionHead(
            context_dim=memory_out,
            pokemon_dim=config.pokemon_out,
            num_actions=ACTIONS_PER_SLOT,
            num_tera_choices=NUM_TERA_CHOICES,
        )

        # Optional: stat-based embedding init
        if config.init_embeddings_from_stats:
            game_data.init_species_embedding(pokemon_encoder.species_emb)
            game_data.init_move_embedding(pokemon_encoder.move_emb)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden=None,
        actions: torch.Tensor | None = None,
        action_masks: dict[str, torch.Tensor] | None = None,
        tera_exploration_prob: float = 0.0,
        greedy: bool = False,
        validate_actions: bool = True,
        include_value: bool = True,
        include_policy_stats: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for battle turns.

        Args:
            obs: observation dict from ObservationBuilder
            hidden: LSTM hidden state (or None)
            actions: (batch, 3) actions to evaluate, or None to sample
            action_masks: dict with boolean legality tables:
                ``joint`` has shape (batch, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT)
                ``can_tera`` has shape (batch, 2)
            tera_exploration_prob: epsilon for factorized use-gate and eligible-slot
                exploration during rollout. PPO passes the same value when evaluating
                supplied actions so behavior-policy likelihood ratios stay exact.
            greedy: choose the highest-probability legal actions instead of
                sampling when actions are not supplied.
            validate_actions: validate caller-supplied actions against the
                masks. Internally sampled actions are already constrained by
                those masks and do not require a host-side validation barrier.
            include_value: whether to compute and return the value estimate.
                League opponents set this to false because they only need actions
                and recurrent hidden state.
            include_policy_stats: whether to compute log-probability and entropy.
                Frozen opponents set this to false because PPO does not consume
                their policy statistics.

        Returns dict with: actions, log_prob, entropy, hidden, and value when
        the corresponding include flags are true.
        """
        context, active_reps, own_reps, opp_reps, opp_active_reps = self.spatial(obs)
        context, new_hidden = self.memory(context, hidden)
        opp_bench_mask = self._bench_mask(obs.get("opp_mask"), obs["opp_roles"])

        action_out = self.action_head(
            context,
            active_reps,
            own_reps,
            opp_reps,
            opp_bench_mask,
            opp_active_reps,
            action_masks,
            actions,
            greedy=greedy,
            battle_obs=obs,
            tera_exploration_prob=tera_exploration_prob,
            validate_actions=validate_actions,
            include_policy_stats=include_policy_stats,
        )

        result = {
            **action_out,
            "hidden": new_hidden,
        }
        if include_value:
            own_bench_mask = self._bench_mask(obs.get("own_mask"), obs["own_roles"])
            result["value"] = self.value_head(
                context,
                own_reps,
                opp_reps,
                own_bench_mask,
                opp_bench_mask,
            )
        return result

    def team_preview(
        self,
        obs: dict[str, torch.Tensor],
        hidden=None,
        picks: torch.Tensor | None = None,
        greedy: bool = False,
        include_value: bool = True,
        include_policy_stats: bool = True,
        include_opponent_preview: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for team preview (selecting 4 from 6).

        Uses the full own-team representations from the spatial encoder so the
        preview head can score all six pokemon. The spatial encoder is called
        once; no second pass is needed.
        """
        # Team preview precedes every public event and submitted intent. Its
        # recent-event representation is therefore exactly zero; bypass the
        # recurrent event encoder instead of evaluating a padded 12-step GRU.
        context, _, own_reps, opp_reps, _ = self.spatial(
            obs,
            include_recent_events=False,
        )
        context, new_hidden = self.memory(context, hidden)

        preview_out = self.team_preview_head(
            own_reps,
            opp_reps,
            context,
            obs.get("own_mask"),
            obs.get("opp_mask"),
            picks,
            greedy,
            include_policy_stats=include_policy_stats,
        )
        result = {
            **preview_out,
            "hidden": new_hidden,
        }
        if include_opponent_preview:
            result.update(
                self.opponent_preview_head(opp_reps, context, obs.get("opp_mask"))
            )
        if include_value:
            result["preview_value"] = self.preview_value_head(
                context,
                own_reps,
                opp_reps,
                obs.get("own_mask"),
                obs.get("opp_mask"),
            )
        return result

    def get_value(self, obs: dict[str, torch.Tensor], hidden=None) -> torch.Tensor:
        """Just the value estimate, for GAE computation."""
        context, _, own_reps, opp_reps, _ = self.spatial(obs)
        context, _ = self.memory(context, hidden)
        own_bench_mask = self._bench_mask(obs.get("own_mask"), obs["own_roles"])
        opp_bench_mask = self._bench_mask(obs.get("opp_mask"), obs["opp_roles"])
        return self.value_head(
            context, own_reps, opp_reps, own_bench_mask, opp_bench_mask
        )

    def get_preview_value(
        self, obs: dict[str, torch.Tensor], hidden=None
    ) -> torch.Tensor:
        """Preview-only value estimate used as the team-preview baseline."""
        context, _, own_reps, opp_reps, _ = self.spatial(
            obs,
            include_recent_events=False,
        )
        context, _ = self.memory(context, hidden)
        return self.preview_value_head(
            context,
            own_reps,
            opp_reps,
            obs.get("own_mask"),
            obs.get("opp_mask"),
        )

    def initial_hidden(self, batch_size: int):
        return self.memory.initial_hidden(batch_size)

    @staticmethod
    def _bench_mask(mask: torch.Tensor | None, roles: torch.Tensor) -> torch.Tensor:
        bench_mask = roles != ROLE_BENCH
        if mask is None:
            return bench_mask
        return mask | bench_mask
