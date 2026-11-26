import math
from dataclasses import dataclass


REGULATION_F = "gen9vgc2026regf"


@dataclass
class ModelConfig:
    # Embedding dimensions
    species_dim: int = 64
    move_dim: int = 32
    ability_dim: int = 32
    item_dim: int = 32
    type_dim: int = 16

    # Architecture
    pokemon_hidden: int = 256
    pokemon_out: int = 128
    team_attention_heads: int = 4
    team_attention_layers: int = 2
    field_out: int = 64
    recent_event_out: int = 64

    # Memory
    memory_type: str = "lstm"  # "lstm" or "identity"
    memory_hidden: int = 256

    # Heads
    action_hidden: int = 256
    value_hidden: int = 256
    team_size: int = 6
    pick_size: int = 4

    # Stat-based embedding init
    init_embeddings_from_stats: bool = True


@dataclass
class TrainingConfig:
    num_envs: int = 16

    # PPO
    optimizer_name: str = "adam"
    lr: float = 2.5e-4
    weight_decay: float = 0.0
    # Optimize undiscounted win probability. GAE lambda still controls the
    # battle critic's bias/variance tradeoff.
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    # Entropy coefficients for the factorized base (deployment) Tera policy.
    # Gate and conditional-slot entropy are separate so slot diversity cannot
    # create pressure to use Tera now.
    tera_gate_entropy_coef: float = 0.01
    tera_slot_entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    team_preview_value_coef: float = 0.5
    team_preview_bring_coef: float = 0.1
    team_preview_lead_coef: float = 0.05
    # Auxiliary prediction losses do not modify rewards, and every target is
    # trainer-only. Compact public-state beliefs and detached public-consequence
    # probabilities may still condition the deployed policy.
    aux_hidden_info_coef: float = 0.25
    aux_opponent_intent_coef: float = 0.25
    aux_public_consequence_coef: float = 0.1
    aux_damage_coef: float = 0.0
    diagnose_memory_gradients: bool = False

    # Batching
    rollout_steps: int = 128
    num_epochs: int = 4
    num_minibatches: int = 4
    bptt_chunk_steps: int | None = None

    # Training
    total_timesteps: int = 10_000_000
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"
    seed: int = 0

    # Factorized Tera exploration mixes the use-now gate toward Bernoulli(0.5)
    # and the conditional eligible-slot distribution toward uniform. PPO uses
    # the exact reconstructed three-choice behavior likelihood. Exploration
    # reaches its end value at this learner update. Fresh training must choose
    # the absolute window explicitly; strict resume restores it from checkpoint.
    tera_exploration_start: float = 0.15
    tera_exploration_end: float = 0.0
    tera_exploration_decay_updates: int | None = None

    # League. Every snapshot remains in the on-disk archive. ``league_size``
    # bounds only the number of frozen models resident in each inference
    # service; it no longer limits archive membership.
    league_size: int = 12
    league_update_interval: int = 10
    league_refresh_interval: int = 1
    # Online current-policy results form an exponentially discounted payoff
    # estimate for each archived opponent. PFSP focuses on opponents the live
    # policy struggles against while retaining a uniform coverage component.
    league_payoff_half_life_updates: float = 20.0
    league_sampler: str = "random"
    league_pfsp_exponent: float = 2.0
    league_pfsp_uniform_fraction: float = 0.1
    # Collapse the many small frozen-opponent CUDA launches into one replay.
    # The graph path is inference-only and falls back to eager execution on
    # non-CUDA devices or when graph capture is unavailable.
    league_cuda_graphs: bool = True
    # Production batches usually contain far fewer rows than one full opponent
    # group.  A smaller fixed graph avoids evaluating padded observations, while
    # four resident entries cover the two current and two draining checkpoints
    # that commonly coexist across an update boundary.
    league_cuda_graph_batch_size: int = 16
    league_cuda_graph_cache_size: int = 4
    # ``action`` reuses the frozen battle-policy service/device for preview.
    league_preview_device: str = "action"

    def __post_init__(self) -> None:
        if self.optimizer_name not in {"adam", "adamw"}:
            raise ValueError(
                "optimizer_name must be 'adam' or 'adamw' "
                f"(got {self.optimizer_name!r})."
            )
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and positive (got {self.lr}).")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError(
                "weight_decay must be finite and non-negative "
                f"(got {self.weight_decay})."
            )
        if self.optimizer_name == "adam" and self.weight_decay != 0.0:
            raise ValueError(
                "weight_decay must be zero for the Adam baseline; use "
                "optimizer_name='adamw' for decoupled weight decay."
            )
        if not 0 <= self.seed < 2**32:
            raise ValueError(f"seed must be in [0, 2**32) (got {self.seed}).")
        if self.league_size <= 0:
            raise ValueError(f"league_size must be positive (got {self.league_size}).")
        for name, value in (
            ("league_cuda_graph_batch_size", self.league_cuda_graph_batch_size),
            ("league_cuda_graph_cache_size", self.league_cuda_graph_cache_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer (got {value}).")
        if self.league_preview_device not in {"action", "cpu"}:
            raise ValueError(
                "league_preview_device must be 'action' or 'cpu' "
                f"(got {self.league_preview_device!r})."
            )
        if self.league_update_interval <= 0:
            raise ValueError(
                "league_update_interval must be positive "
                f"(got {self.league_update_interval})."
            )
        if self.league_sampler not in ("random", "balanced"):
            raise ValueError(f"Unknown league sampler: {self.league_sampler!r}.")
        if self.league_refresh_interval <= 0:
            raise ValueError(
                "league_refresh_interval must be positive "
                f"(got {self.league_refresh_interval})."
            )
        for name, value in (
            ("league_payoff_half_life_updates", self.league_payoff_half_life_updates),
            ("league_pfsp_exponent", self.league_pfsp_exponent),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive (got {value}).")
        if (
            not math.isfinite(self.league_pfsp_uniform_fraction)
            or not 0.0 <= self.league_pfsp_uniform_fraction <= 1.0
        ):
            raise ValueError(
                "league_pfsp_uniform_fraction must be finite and within [0, 1] "
                f"(got {self.league_pfsp_uniform_fraction})."
            )
        for name, value in (
            ("tera_gate_entropy_coef", self.tera_gate_entropy_coef),
            ("tera_slot_entropy_coef", self.tera_slot_entropy_coef),
            ("aux_hidden_info_coef", self.aux_hidden_info_coef),
            ("aux_opponent_intent_coef", self.aux_opponent_intent_coef),
            ("aux_damage_coef", self.aux_damage_coef),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative (got {value})."
                )
        if (
            not math.isfinite(self.aux_public_consequence_coef)
            or self.aux_public_consequence_coef <= 0.0
        ):
            raise ValueError(
                "aux_public_consequence_coef must be finite and positive because "
                "the policy architecture consumes its grounded predictions "
                f"(got {self.aux_public_consequence_coef})."
            )
        for name, value in (
            ("tera_exploration_start", self.tera_exploration_start),
            ("tera_exploration_end", self.tera_exploration_end),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1] (got {value}).")
        if self.tera_exploration_decay_updates is not None and (
            isinstance(self.tera_exploration_decay_updates, bool)
            or not isinstance(self.tera_exploration_decay_updates, int)
            or self.tera_exploration_decay_updates < 0
        ):
            raise ValueError(
                "tera_exploration_decay_updates must be a non-negative integer "
                f"or None (got {self.tera_exploration_decay_updates})."
            )
