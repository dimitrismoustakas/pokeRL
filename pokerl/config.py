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
    # Auxiliary prediction losses. These do not modify the reward or the
    # action/value path; they add dense gradient to the shared encoder by
    # predicting opponent hidden info (end-of-episode labels) and the
    # opponent's observed action this turn.
    aux_hidden_info_coef: float = 0.25
    aux_opp_action_coef: float = 0.25

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
    league_coalesce_ms: float = 10.0

    def __post_init__(self) -> None:
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
