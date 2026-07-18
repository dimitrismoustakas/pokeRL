"""Encoders for pokemon, teams (with order invariance), and field conditions."""

import torch
import torch.nn as nn

from ..data import NUM_MOVE_SCOPES
from ..env.observation import (
    ACTIVE_MECHANICS_FEATURE_DIM,
    NUM_PUBLIC_EVENT_FLAGS,
    NUM_PUBLIC_TARGET_POSITIONS,
    NUM_TARGET_OUTCOME_FEATURES,
)


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
            + 4 * move_dim  # 4 moves
            + move_dim  # last revealed move
            + ability_dim
            + item_dim
            + 3 * type_dim  # type_1, type_2, tera_type
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
        species: torch.Tensor,  # (batch, )
        moves: torch.Tensor,  # (batch, 4)
        last_move: torch.Tensor,  # (batch, )
        ability: torch.Tensor,  # (batch, )
        item: torch.Tensor,  # (batch, )
        types: torch.Tensor,  # (batch, 3)  — type_1, type_2, tera_type
        continuous: torch.Tensor,  # (batch, continuous_dim)
    ) -> torch.Tensor:  # (batch, out_dim)
        x = torch.cat(
            [
                self.species_emb(species),
                self.move_emb(moves).flatten(-2),  # (batch, 4 * move_dim)
                self.move_emb(last_move),
                self.ability_emb(ability),
                self.item_emb(item),
                self.type_emb(types).flatten(-2),  # (batch, 2 * type_dim)
                continuous,
            ],
            dim=-1,
        )
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
        # PyTorch's nested-tensor path does not support pre-norm encoder layers
        # and otherwise emits a warning before disabling itself.
        self.attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(
        self,
        pokemon_reps: torch.Tensor,  # (batch, team_size, pokemon_dim)
        roles: torch.Tensor,  # (batch, team_size) values in {0, 1, 2}
        mask: torch.Tensor | None = None,  # (batch, team_size) True = padding
    ) -> torch.Tensor:  # (batch, team_size, pokemon_dim)
        x = pokemon_reps + self.role_emb(roles)
        return self.attention(x, src_key_padding_mask=mask)


class FieldEncoder(nn.Module):
    """Encodes field conditions (weather, terrain, hazards, screens, etc.)."""

    # Weather: 10 (8 types + none + timer age)
    # Terrain: 6 (4 types + none + timer age)
    # Side conditions per side: stealth_rock, spikes, toxic_spikes,
    #   sticky_web, reflect_age, light_screen_age, aurora_veil_age, tailwind_age = 8 each
    # Global: trick_room_age, own tera used, opponent tera used, normalized turn
    #
    # Total: 10 + 6 + 2*8 + 1 + 2 + 1 = 36
    INPUT_DIM = 36

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, out_dim),
            nn.ReLU(),
        )

    def forward(self, field_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(field_features)


class RecentEventEncoder(nn.Module):
    """Encode the last public resolution segment and our prior submitted intent."""

    def __init__(
        self,
        *,
        num_moves: int,
        num_species: int,
        num_types: int,
        num_event_kinds: int,
        num_positions: int,
        num_intent_kinds: int,
        out_dim: int = 64,
    ):
        super().__init__()
        if out_dim <= 0:
            raise ValueError(
                f"Recent-event output width must be positive, got {out_dim}."
            )
        self.out_dim = out_dim
        self.kind_emb = nn.Embedding(num_event_kinds, 8, padding_idx=0)
        self.position_emb = nn.Embedding(num_positions, 8, padding_idx=0)
        self.move_emb = nn.Embedding(num_moves, 16, padding_idx=0)
        self.move_scope_emb = nn.Embedding(NUM_MOVE_SCOPES, 4, padding_idx=0)
        self.species_emb = nn.Embedding(num_species, 16, padding_idx=0)
        self.type_emb = nn.Embedding(num_types, 8, padding_idx=0)
        self.intent_kind_emb = nn.Embedding(num_intent_kinds, 8, padding_idx=0)
        self.event_projection = nn.Sequential(
            nn.Linear(
                8
                + 8
                + 8
                + 16
                + 16
                + 4
                + 16
                + 8
                + NUM_PUBLIC_EVENT_FLAGS
                + NUM_PUBLIC_TARGET_POSITIONS * NUM_TARGET_OUTCOME_FEATURES
                + NUM_PUBLIC_TARGET_POSITIONS * 16,
                64,
            ),
            nn.ReLU(),
        )
        self.event_memory = nn.GRU(64, 64, batch_first=True)
        self.intent_projection = nn.Sequential(
            nn.Linear(8 + 16 + 8 + 16 + 1, 32),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(64 + 2 * 32 + 3, out_dim),
            nn.ReLU(),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        reference = obs["field"]
        batch_size = reference.shape[0]
        if "recent_event_kind" not in obs:
            return reference.new_zeros((batch_size, self.out_dim))

        source_moves = obs.get("recent_event_source_move")
        if source_moves is None:
            source_moves = torch.zeros_like(obs["recent_event_move"])
        move_scopes = obs.get("recent_event_move_scope")
        if move_scopes is None:
            move_scopes = torch.zeros_like(obs["recent_event_move"])
        activation_moves = obs.get("recent_event_target_activation_move")
        if activation_moves is None:
            activation_moves = torch.zeros(
                (
                    batch_size,
                    obs["recent_event_kind"].shape[1],
                    NUM_PUBLIC_TARGET_POSITIONS,
                ),
                dtype=torch.long,
                device=reference.device,
            )
        target_outcomes = obs.get("recent_event_target_outcomes")
        if target_outcomes is None:
            target_outcomes = reference.new_zeros(
                (
                    batch_size,
                    obs["recent_event_kind"].shape[1],
                    NUM_PUBLIC_TARGET_POSITIONS,
                    NUM_TARGET_OUTCOME_FEATURES,
                )
            )

        event_tokens = torch.cat(
            [
                self.kind_emb(obs["recent_event_kind"]),
                self.position_emb(obs["recent_event_actor"]),
                self.position_emb(obs["recent_event_target"]),
                self.move_emb(obs["recent_event_move"]),
                self.move_emb(source_moves),
                self.move_scope_emb(move_scopes),
                self.species_emb(obs["recent_event_species"]),
                self.type_emb(obs["recent_event_tera_type"]),
                obs["recent_event_flags"],
                target_outcomes.flatten(start_dim=-2),
                self.move_emb(activation_moves).flatten(start_dim=-2),
            ],
            dim=-1,
        )
        event_tokens = self.event_projection(event_tokens)
        event_outputs, _ = self.event_memory(event_tokens)
        lengths = (~obs["recent_event_mask"]).sum(dim=1)
        last_indices = (lengths - 1).clamp_min(0)
        event_summary = event_outputs[
            torch.arange(batch_size, device=reference.device), last_indices
        ]
        event_summary = event_summary * (lengths > 0).unsqueeze(-1)

        intent_tokens = torch.cat(
            [
                self.intent_kind_emb(obs["prev_own_kind"]),
                self.move_emb(obs["prev_own_move"]),
                self.position_emb(obs["prev_own_target"]),
                self.species_emb(obs["prev_own_switch_species"]),
                obs["prev_own_tera"].unsqueeze(-1).to(reference.dtype),
            ],
            dim=-1,
        )
        intent_tokens = self.intent_projection(intent_tokens)
        intent_tokens = intent_tokens * obs["prev_own_valid"].unsqueeze(-1)
        intent_summary = intent_tokens.flatten(1)
        encoded = self.output(
            torch.cat([event_summary, intent_summary, obs["recent_event_meta"]], dim=-1)
        )
        valid = (lengths > 0) | obs["prev_own_valid"].any(dim=1)
        return encoded * valid.unsqueeze(-1)


class ActiveMechanicsEncoder(nn.Module):
    """Encode exact public mechanics aligned to the four active positions."""

    NUM_PUBLIC_SLOTS = 4

    def __init__(self, *, move_dim: int, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.position_emb = nn.Embedding(self.NUM_PUBLIC_SLOTS, 8)
        self.slot_projection = nn.Sequential(
            nn.Linear(move_dim + ACTIVE_MECHANICS_FEATURE_DIM + 8, 48),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(self.NUM_PUBLIC_SLOTS * 48, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        move_embedding: nn.Embedding,
    ) -> torch.Tensor:
        reference = obs["field"]
        batch_size = reference.shape[0]
        features = obs.get("active_mechanics")
        moves = obs.get("active_mechanics_move")
        if features is None or moves is None:
            return reference.new_zeros((batch_size, self.out_dim))
        positions = torch.arange(
            self.NUM_PUBLIC_SLOTS,
            device=reference.device,
        ).view(1, self.NUM_PUBLIC_SLOTS)
        positions = positions.expand(batch_size, -1)
        slot_tokens = torch.cat(
            [
                move_embedding(moves),
                features,
                self.position_emb(positions),
            ],
            dim=-1,
        )
        return self.output(self.slot_projection(slot_tokens).flatten(start_dim=1))


class SpatialEncoder(nn.Module):
    """Full spatial encoder: both teams + field -> battle representation.

    The output has five components:
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
        recent_event_encoder: RecentEventEncoder,
        active_mechanics_encoder: ActiveMechanicsEncoder,
    ):
        super().__init__()
        self.pokemon_encoder = pokemon_encoder
        self.team_own = team_encoder_own
        self.team_opp = team_encoder_opp
        self.field_encoder = field_encoder
        self.recent_event_encoder = recent_event_encoder
        self.active_mechanics_encoder = active_mechanics_encoder

        pokemon_dim = team_encoder_own.role_emb.embedding_dim
        # context = mean(own_team) + mean(opp_team) + field + events +
        # exact mechanics aligned to the four public active positions.
        self.context_dim = (
            pokemon_dim * 2
            + field_encoder.out_dim
            + recent_event_encoder.out_dim
            + active_mechanics_encoder.out_dim
        )

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        *,
        include_recent_events: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: dictionary with keys for own team, opponent team, field.
                 See observation.py for the exact structure.
            include_recent_events: whether to encode transition history. Team
                preview happens before any public event or submitted intent, so
                that path can supply the same all-zero representation directly.

        Returns:
            context:     (batch, context_dim) flat battle representation
            active_reps:     (batch, 2, pokemon_dim) our active pokemon representations
            own_reps:        (batch, team_size, pokemon_dim) full attention-processed own
                             team, for use by the team preview head and switch scoring
            opp_reps:        (batch, team_size, pokemon_dim) full attention-processed
                             opponent team, for preview auxiliaries
            opp_active_reps: (batch, 2, pokemon_dim) opponent active pokemon representations
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
        if include_recent_events:
            recent_events = self.recent_event_encoder(obs)
        else:
            recent_events = field.new_zeros(
                (field.shape[0], self.recent_event_encoder.out_dim)
            )
        active_mechanics = self.active_mechanics_encoder(
            obs,
            self.pokemon_encoder.move_emb,
        )

        # Pool teams into summary vectors (mask-aware mean)
        own_summary = self._masked_mean(own_reps, obs.get("own_mask"))
        opp_summary = self._masked_mean(opp_reps, opp_mask)

        context = torch.cat(
            [own_summary, opp_summary, field, recent_events, active_mechanics],
            dim=-1,
        )

        # Extract our active pokemon representations by index
        active_indices = obs["own_active_indices"]  # (batch, 2)
        active_reps = self._gather_slots(
            own_reps,
            active_indices,
            obs.get("own_active_present"),
        )

        opp_active_reps = self._gather_slots(
            opp_reps,
            obs["opp_active_indices"],
            obs.get("opp_active_present"),
        )

        return context, active_reps, own_reps, opp_reps, opp_active_reps

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
            last_move=flat("last_move"),
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

    @staticmethod
    def _gather_slots(
        reps: torch.Tensor,
        indices: torch.Tensor,
        present: torch.Tensor | None,
    ) -> torch.Tensor:
        gathered = reps.gather(1, indices.unsqueeze(-1).expand(-1, -1, reps.shape[-1]))
        if present is None:
            return gathered
        return gathered * present.unsqueeze(-1).to(gathered.dtype)
