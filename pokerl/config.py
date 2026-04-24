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
    pick_size: int = 4  # how many pokemon to select at team preview (must match EnvConfig.pick_size)

    # Stat-based embedding init
    init_embeddings_from_stats: bool = True


@dataclass
class EnvConfig:
    battle_format: str = "gen9randomdoublesbattle"
    server_url: str = "localhost:8000"
    num_envs: int = 16
    team_size: int = 6
    pick_size: int = 4  # how many pokemon to bring from team_size


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
    team_preview_update_interval: int = 8  # flush the preview accumulator every N PPO updates
    team_preview_max_samples: int = 1024    # cap the accumulator; oldest entries are dropped first

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

    # Tera exploration: with probability epsilon on each turn that has a real
    # tera choice (>= 2 valid options), override the policy's tera sample with a
    # uniform draw over valid options. Breaks the "tera always turn 1" attractor
    # by forcing the policy to observe outcomes from saving tera. Log-probs are
    # stored under the policy (not behavior), so PPO's clip absorbs the mild
    # off-policy bias for small epsilon. Linear decay from start to end across
    # tera_exploration_decay_updates PPO updates.
    tera_exploration_start: float = 0.15
    tera_exploration_end: float = 0.02
    tera_exploration_decay_updates: int = 2000

    # League
    league_size: int = 12
    league_update_interval: int = 10
    league_coalesce_ms: float = 10.0

    # Anchor evaluation harness (see LEAGUE_IMPROVEMENTS.md §A): permanent
    # frozen checkpoints kept outside the rolling league pool so we can track
    # whether the live agent keeps beating its earlier selves, or has drifted.
    anchor_eval_enabled: bool = True
    # Fractions of total_timesteps at which to snapshot an anchor. Auto-scales
    # to any run length. The doc's absolute 100k/500k/1M/2M defaults would
    # never fire in short runs, so the harness uses fractions instead.
    anchor_fractions: tuple[float, ...] = (0.05, 0.20, 0.40, 0.60)
    # Units are PPO updates.
    anchor_eval_interval_updates: int = 50
    anchor_eval_games_per_anchor: int = 30
    anchor_eval_num_envs: int = 16

    def __post_init__(self) -> None:
        if self.tera_exploration_decay_updates < 0:
            raise ValueError(
                "tera_exploration_decay_updates must be >= 0 "
                f"(got {self.tera_exploration_decay_updates}); "
                "use 0 to skip decay and apply tera_exploration_end immediately."
            )
        if self.anchor_eval_enabled:
            if not self.anchor_fractions:
                raise ValueError("anchor_fractions must contain at least one value when anchor_eval_enabled.")
            for fraction in self.anchor_fractions:
                if not 0.0 < fraction < 1.0:
                    raise ValueError(
                        f"anchor_fractions must be in (0, 1); got {fraction}."
                    )
            if sorted(self.anchor_fractions) != list(self.anchor_fractions):
                raise ValueError("anchor_fractions must be sorted ascending.")
            if self.anchor_eval_interval_updates <= 0:
                raise ValueError("anchor_eval_interval_updates must be > 0.")
            if self.anchor_eval_games_per_anchor <= 0:
                raise ValueError("anchor_eval_games_per_anchor must be > 0.")
            if self.anchor_eval_num_envs <= 0:
                raise ValueError("anchor_eval_num_envs must be > 0.")
