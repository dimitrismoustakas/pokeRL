"""Full agent: spatial encoder -> memory -> action/value heads."""

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..data import GameData
from ..env.observation import CONTINUOUS_DIM
from .encoders import PokemonEncoder, TeamEncoder, FieldEncoder, SpatialEncoder
from .memory import build_memory
from .heads import ActionHead, TeamPreviewHead, ValueHead


class PokeAgent(nn.Module):
    """Complete agent for doubles battles.

    Combines spatial encoding (order-invariant team attention),
    temporal memory (pluggable LSTM/identity), and autoregressive
    action selection for doubles + team preview.
    """

    def __init__(self, config: ModelConfig, game_data: GameData):
        super().__init__()
        self.config = config

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

        team_own = TeamEncoder(config.pokemon_out, config.team_attention_heads, config.team_attention_layers)
        team_opp = TeamEncoder(config.pokemon_out, config.team_attention_heads, config.team_attention_layers)
        field_encoder = FieldEncoder(config.field_out)

        self.spatial = SpatialEncoder(pokemon_encoder, team_own, team_opp, field_encoder)

        # --- Memory ---
        self.memory = build_memory(config.memory_type, self.spatial.context_dim, config.memory_hidden)
        memory_out = config.memory_hidden

        # --- Heads ---
        self.action_head = ActionHead(memory_out, config.pokemon_out, config.action_hidden)
        self.team_preview_head = TeamPreviewHead(config.pokemon_out, memory_out)
        self.value_head = ValueHead(memory_out, config.value_hidden)

        # Optional: stat-based embedding init
        if config.init_embeddings_from_stats:
            game_data.init_species_embedding(pokemon_encoder.species_emb)
            game_data.init_move_embedding(pokemon_encoder.move_emb)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden=None,
        actions: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for battle turns.

        Args:
            obs: observation dict from ObservationBuilder
            hidden: LSTM hidden state (or None)
            actions: (batch, 2) actions to evaluate, or None to sample
            action_masks: (batch, 2, ACTIONS_PER_SLOT) boolean masks

        Returns dict with: actions, log_prob, entropy, value, hidden
        """
        context, active_reps, _ = self.spatial(obs)
        context, new_hidden = self.memory(context, hidden)

        action_out = self.action_head(context, active_reps, action_masks, actions)
        value = self.value_head(context)

        return {
            **action_out,
            "value": value,
            "hidden": new_hidden,
        }

    def team_preview(
        self,
        obs: dict[str, torch.Tensor],
        hidden=None,
        picks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for team preview (selecting 4 from 6).

        Uses the full own-team representations from the spatial encoder so the
        preview head can score all six pokemon. The spatial encoder is called
        once; no second pass is needed.
        """
        context, _, own_reps = self.spatial(obs)
        context, new_hidden = self.memory(context, hidden)

        preview_out = self.team_preview_head(own_reps, context, picks)
        return {
            **preview_out,
            "hidden": new_hidden,
        }

    def get_value(self, obs: dict[str, torch.Tensor], hidden=None) -> torch.Tensor:
        """Just the value estimate, for GAE computation."""
        context, _, _ = self.spatial(obs)
        context, _ = self.memory(context, hidden)
        return self.value_head(context)

    def initial_hidden(self, batch_size: int):
        return self.memory.initial_hidden(batch_size)
