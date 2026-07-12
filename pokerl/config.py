from dataclasses import dataclass


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

    # Memory
    memory_type: str = "lstm"  # "lstm" or "identity"
    memory_hidden: int = 256

    # Heads
    action_hidden: int = 256
    value_hidden: int = 256
    pick_size: int = (
        4  # how many pokemon to select at team preview (must match EnvConfig.pick_size)
    )

    # Stat-based embedding init
    init_embeddings_from_stats: bool = True


@dataclass
class EnvConfig:
    battle_format: str = "gen9randomdoublesbattle"
    num_envs: int = 16
    team_size: int = 6
    pick_size: int = 4  # how many pokemon to bring from team_size
    showdown_dir: str | None = None
    worker_count: int | None = None
    battles_per_worker: int | None = None
    runtime_loop_count: int | None = None
    battle_loop_count: int | None = None
    battlestream_startup_timeout: float = 30.0
    reset_concurrency: int | None = None
    opponent_group: int = 0


@dataclass
class TrainingConfig:
    # PPO
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    team_preview_value_coef: float = 0.5
    team_preview_bring_coef: float = 0.1
    team_preview_lead_coef: float = 0.05
    team_preview_max_samples: int = 1024  # uniform current-rollout subsample cap
    # Auxiliary prediction losses. These do not modify the reward or the
    # action/value path; they add dense gradient to the shared encoder by
    # predicting opponent hidden info (end-of-episode labels) and the
    # opponent's observed action this turn.
    aux_hidden_info_coef: float = 0.25
    aux_opp_action_coef: float = 0.25
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

    # Tera exploration: with probability epsilon on each turn that has a real
    # tera choice (>= 2 valid options), override the policy's tera sample with a
    # uniform draw over valid options. Breaks the "tera always turn 1" attractor
    # by forcing the policy to observe outcomes from saving tera. Rollout and
    # PPO log-probabilities both use the exact resulting behavior distribution.
    # Linear decay from start to end across tera_exploration_decay_updates PPO
    # updates.
    tera_exploration_start: float = 0.15
    tera_exploration_end: float = 0.02
    tera_exploration_decay_updates: int = 2000
    # Targeted anti-collapse exploration: on early turns, if the policy samples
    # tera now, override that third action to no-tera with this decayed
    # probability. This directly creates delayed-tera rollouts without changing
    # the action/value model or reward.
    tera_force_delay_start: float = 0.40
    tera_force_delay_end: float = 0.03
    tera_force_delay_decay_updates: int = 3000
    tera_force_delay_max_turn: int = 3

    # League
    league_size: int = 12
    league_update_interval: int = 10
    league_refresh_interval: int = 1
    league_coalesce_ms: float = 5.0

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**32:
            raise ValueError(f"seed must be in [0, 2**32) (got {self.seed}).")
        if self.league_size <= 0:
            raise ValueError(f"league_size must be positive (got {self.league_size}).")
        if self.league_update_interval <= 0:
            raise ValueError(
                "league_update_interval must be positive "
                f"(got {self.league_update_interval})."
            )
        if self.league_refresh_interval <= 0:
            raise ValueError(
                "league_refresh_interval must be positive "
                f"(got {self.league_refresh_interval})."
            )
        if self.team_preview_max_samples <= 0:
            raise ValueError(
                "team_preview_max_samples must be positive "
                f"(got {self.team_preview_max_samples})."
            )
        for name, value in (
            ("tera_exploration_start", self.tera_exploration_start),
            ("tera_exploration_end", self.tera_exploration_end),
            ("tera_force_delay_start", self.tera_force_delay_start),
            ("tera_force_delay_end", self.tera_force_delay_end),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1] (got {value}).")
        if self.tera_exploration_decay_updates < 0:
            raise ValueError(
                "tera_exploration_decay_updates must be >= 0 "
                f"(got {self.tera_exploration_decay_updates}); "
                "use 0 to skip decay and apply tera_exploration_end immediately."
            )
        if self.tera_force_delay_decay_updates < 0:
            raise ValueError(
                "tera_force_delay_decay_updates must be >= 0 "
                f"(got {self.tera_force_delay_decay_updates}); "
                "use 0 to skip decay and apply tera_force_delay_end immediately."
            )
        if self.tera_force_delay_max_turn < 0:
            raise ValueError(
                "tera_force_delay_max_turn must be >= 0 "
                f"(got {self.tera_force_delay_max_turn}); "
                "use 0 to disable early-turn force-delay exploration."
            )
