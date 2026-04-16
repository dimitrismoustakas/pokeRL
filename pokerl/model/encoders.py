"""Encoders for pokemon, teams (with order invariance), and field conditions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PokemonEncoder(nn.Module):
    """Encodes a single pokemon's features into a fixed-size representation.

    Shared across all pokemon (own team and opponent's). Handles both fully
    observed (own pokemon) and partially observed (opponent's) inputs -- the
    caller passes UNKNOWN_IDX for unobserved categorical features and zeros
    for unobserved continuous features.
    """

    def __init__(
        self,
        num_species: int,
        num_moves: int,
        num_abilities: int,
        num_items: int,
        num_types: int,
        continuous_dim: int,
        species_dim: int = 64,
        move_dim: int = 32,
        ability_dim: int = 32,
        item_dim: int = 32,
        type_dim: int = 16,
        hidden_dim: int = 256,
        out_dim: int = 128,
    ):
        super().__init__()
        self.species_emb = nn.Embedding(num_species, species_dim, padding_idx=0)
        self.move_emb = nn.Embedding(num_moves, move_dim, padding_idx=0)
        self.ability_emb = nn.Embedding(num_abilities, ability_dim, padding_idx=0)
        self.item_emb = nn.Embedding(num_items, item_dim, padding_idx=0)
        self.type_emb = nn.Embedding(num_types, type_dim, padding_idx=0)

        total_input = (
            species_dim
            + 4 * move_dim    # 4 moves
            + ability_dim
            + item_dim
            + 3 * type_dim    # type_1, type_2, tera_type
            + continuous_dim
        )

        self.mlp = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        species: torch.Tensor,       # (batch, )
        moves: torch.Tensor,         # (batch, 4)
        ability: torch.Tensor,       # (batch, )
        item: torch.Tensor,          # (batch, )
        types: torch.Tensor,         # (batch, 3)  — type_1, type_2, tera_type
        continuous: torch.Tensor,    # (batch, continuous_dim)
    ) -> torch.Tensor:               # (batch, out_dim)
        x = torch.cat([
            self.species_emb(species),
            self.move_emb(moves).flatten(-2),       # (batch, 4 * move_dim)
            self.ability_emb(ability),
            self.item_emb(item),
            self.type_emb(types).flatten(-2),       # (batch, 2 * type_dim)
            continuous,
        ], dim=-1)
        return self.mlp(x)


class TeamEncoder(nn.Module):
    """Permutation-equivariant encoding of a team of pokemon using self-attention.

    Takes per-pokemon representations and role embeddings (active_slot_1,
    active_slot_2, bench), applies self-attention so each pokemon's output
    depends on all teammates without relying on position.
    """

    NUM_ROLES = 3  # active_slot_1, active_slot_2, bench

    def __init__(self, pokemon_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.role_emb = nn.Embedding(self.NUM_ROLES, pokemon_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=pokemon_dim,
            nhead=num_heads,
            dim_feedforward=pokemon_dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        pokemon_reps: torch.Tensor,  # (batch, team_size, pokemon_dim)
        roles: torch.Tensor,         # (batch, team_size) values in {0, 1, 2}
        mask: torch.Tensor | None = None,  # (batch, team_size) True = padding
    ) -> torch.Tensor:               # (batch, team_size, pokemon_dim)
        x = pokemon_reps + self.role_emb(roles)
        return self.attention(x, src_key_padding_mask=mask)


class FieldEncoder(nn.Module):
    """Encodes field conditions (weather, terrain, hazards, screens, etc.)."""

    # Weather: 9 (8 types + none)
    # Terrain: 5 (4 types + none)
    # Side conditions per side: stealth_rock, spikes, toxic_spikes,
    #   sticky_web, reflect, light_screen, aurora_veil, tailwind = 8 each
    # Global: trick_room
    #
    # Total: 9 + 5 + 2*8 + 1 = 31
    INPUT_DIM = 31

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, out_dim),
            nn.ReLU(),
        )

    def forward(self, field_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(field_features)


class SpatialEncoder(nn.Module):
    """Full spatial encoder: both teams + field -> battle representation.

    The output has three components:
      - context: a flat vector for memory / value head
      - active_reps: (batch, 2, dim) representations of our two active pokemon,
        used by the action head to select per-pokemon actions
      - own_reps: (batch, team_size, dim) full attention-processed own team
        representations, used by the team preview head
    """

    def __init__(
        self,
        pokemon_encoder: PokemonEncoder,
        team_encoder_own: TeamEncoder,
        team_encoder_opp: TeamEncoder,
        field_encoder: FieldEncoder,
    ):
        super().__init__()
        self.pokemon_encoder = pokemon_encoder
        self.team_own = team_encoder_own
        self.team_opp = team_encoder_opp
        self.field_encoder = field_encoder

        pokemon_dim = team_encoder_own.role_emb.embedding_dim
        # context = mean(own_team) + mean(opp_team) + field
        self.context_dim = pokemon_dim * 2 + field_encoder.out_dim

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dictionary with keys for own team, opponent team, field.
                 See observation.py for the exact structure.

        Returns:
            context:     (batch, context_dim) flat battle representation
            active_reps: (batch, 2, pokemon_dim) our active pokemon representations
            own_reps:    (batch, team_size, pokemon_dim) full attention-processed own
                         team, for use by the team preview head
        """
        # Encode each pokemon (own side)
        own_reps = self._encode_team(obs, prefix="own")  # (batch, 6, pokemon_dim)
        own_reps = self.team_own(own_reps, obs["own_roles"], obs.get("own_mask"))

        # Encode each pokemon (opponent side)
        opp_reps = self._encode_team(obs, prefix="opp")  # (batch, 6, pokemon_dim)
        opp_mask = obs.get("opp_mask")
        opp_reps = self.team_opp(opp_reps, obs["opp_roles"], opp_mask)

        # Field
        field = self.field_encoder(obs["field"])

        # Pool teams into summary vectors (mask-aware mean)
        own_summary = self._masked_mean(own_reps, obs.get("own_mask"))
        opp_summary = self._masked_mean(opp_reps, opp_mask)

        context = torch.cat([own_summary, opp_summary, field], dim=-1)

        # Extract our active pokemon representations by index
        active_indices = obs["own_active_indices"]  # (batch, 2)
        active_reps = own_reps.gather(
            1, active_indices.unsqueeze(-1).expand(-1, -1, own_reps.shape[-1])
        )

        return context, active_reps, own_reps

    def _encode_team(self, obs: dict, prefix: str) -> torch.Tensor:
        """Encode all pokemon for one side."""
        batch = obs[f"{prefix}_species"].shape[0]
        team_size = obs[f"{prefix}_species"].shape[1]

        # Flatten team dimension for shared encoder
        def flat(key):
            t = obs[f"{prefix}_{key}"]
            return t.reshape(batch * team_size, *t.shape[2:])

        reps = self.pokemon_encoder(
            species=flat("species"),
            moves=flat("moves"),
            ability=flat("ability"),
            item=flat("item"),
            types=flat("types"),
            continuous=flat("continuous"),
        )
        return reps.reshape(batch, team_size, -1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        # mask: True = padding -> invert to get valid positions
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
