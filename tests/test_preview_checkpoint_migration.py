from __future__ import annotations

from dataclasses import asdict

import torch

from pokerl.config import ModelConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.model.agent import PokeAgent
from pokerl.training.preview_migration import (
    _architecture_signature,
    migrate_v3_preview_checkpoint,
)
from pokerl.training.ppo import PPOTrainer


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        species_dim=4,
        move_dim=4,
        ability_dim=4,
        item_dim=4,
        type_dim=4,
        pokemon_hidden=16,
        pokemon_out=8,
        team_attention_heads=1,
        team_attention_layers=1,
        field_out=8,
        recent_event_out=8,
        memory_hidden=8,
        action_hidden=16,
        value_hidden=16,
        team_size=4,
        pick_size=2,
        init_embeddings_from_stats=False,
    )


def _write_declared_v3_checkpoint(
    path,
    *,
    game_data: GameData,
    config: ModelConfig,
) -> dict[str, torch.Tensor]:
    agent = PokeAgent(config, game_data)
    with torch.no_grad():
        for name, parameter in agent.named_parameters():
            if name.startswith("team_preview_head."):
                parameter.fill_(7.0)
            else:
                parameter.fill_(0.125)
    trainer = PPOTrainer(agent, TrainingConfig(seed=3), torch.device("cpu"))
    first_parameter = next(iter(agent.parameters()))
    trainer.optimizer.state[first_parameter] = {
        "step": torch.tensor(9.0),
        "exp_avg": torch.ones_like(first_parameter),
        "exp_avg_sq": torch.full_like(first_parameter, 2.0),
    }
    trainer.global_step = 12_345
    trainer.completed_updates = 37
    trainer.save(str(path), training_state_id="source-v3-state")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_config = asdict(config)
    source_config.pop("team_size")
    architecture = _architecture_signature(
        source_config,
        architecture_version=3,
    )
    metadata = checkpoint["model_metadata"]
    metadata["model_config"]["values"] = source_config
    metadata["architecture_signature"] = architecture
    torch.save(checkpoint, path)
    return checkpoint["agent"]


def test_preview_checkpoint_migration_is_explicit_uniform_and_fresh(tmp_path) -> None:
    game_data = GameData(gen=9)
    config = _small_model_config()
    source = tmp_path / "source-v3.pt"
    output = tmp_path / "migrated-v4.pt"
    source_state = _write_declared_v3_checkpoint(
        source,
        game_data=game_data,
        config=config,
    )

    provenance = migrate_v3_preview_checkpoint(
        source_path=source,
        output_path=output,
        game_data=game_data,
        team_size=config.team_size,
        seed=11,
    )

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 0
    assert checkpoint["completed_updates"] == 0
    assert checkpoint["optimizer"]["state"] == {}
    assert checkpoint["initialization_provenance"] == provenance
    assert provenance["source_training_state_id"] == "source-v3-state"
    assert provenance["source_global_step"] == 12_345
    assert provenance["source_completed_updates"] == 37
    assert provenance["reinitialized_prefixes"] == ["preview_value_head."]
    assert provenance["zeroed_policy_output_tensors"] == [
        "team_preview_head.bring_score.2.weight",
        "team_preview_head.bring_score.2.bias",
        "team_preview_head.lead_pair_score.2.weight",
        "team_preview_head.lead_pair_score.2.bias",
    ]

    for key, migrated_tensor in checkpoint["agent"].items():
        if (
            key.startswith(tuple(provenance["reinitialized_prefixes"]))
            or key in (provenance["zeroed_policy_output_tensors"])
        ):
            continue
        assert torch.equal(migrated_tensor, source_state[key])
    assert torch.equal(
        checkpoint["agent"]["team_preview_head.context_proj.weight"],
        source_state["team_preview_head.context_proj.weight"],
    )
    assert (
        torch.count_nonzero(
            checkpoint["agent"]["team_preview_head.bring_score.2.weight"]
        )
        == 0
    )
    assert (
        torch.count_nonzero(
            checkpoint["agent"]["team_preview_head.lead_pair_score.2.weight"]
        )
        == 0
    )
    assert not torch.equal(
        checkpoint["agent"]["preview_value_head.mlp.0.weight"],
        source_state["preview_value_head.mlp.0.weight"],
    )
    assert torch.equal(
        checkpoint["agent"]["opponent_preview_head.bring_score.0.weight"],
        source_state["opponent_preview_head.bring_score.0.weight"],
    )

    target_agent = PokeAgent(config, game_data)
    target_trainer = PPOTrainer(
        target_agent,
        TrainingConfig(seed=11),
        torch.device("cpu"),
    )
    target_trainer.load(str(output))
    result = target_agent.team_preview_head(
        torch.randn(2, config.team_size, config.pokemon_out),
        torch.randn(2, config.team_size, config.pokemon_out),
        torch.randn(2, config.memory_hidden),
        greedy=True,
    )
    expected = torch.full_like(
        result["joint_log_probs"],
        -torch.log(
            torch.tensor(float(target_agent.team_preview_head.legal_action_count))
        ),
    )
    torch.testing.assert_close(
        result["joint_log_probs"],
        expected,
        atol=1e-6,
        rtol=1e-6,
    )

    fresh_agent = PokeAgent(config, game_data)
    fresh_trainer = PPOTrainer(
        fresh_agent,
        TrainingConfig(seed=19),
        torch.device("cpu"),
    )
    fresh_trainer.initialize_from_trainer_weights(str(output))
    assert fresh_trainer.global_step == 0
    assert fresh_trainer.completed_updates == 0
    assert fresh_trainer.optimizer.state == {}
    assert (
        fresh_trainer.initialized_from_training_state_id
        == provenance["target_training_state_id"]
    )
    assert fresh_trainer.initialization_provenance == provenance

    descendant = tmp_path / "descendant.pt"
    fresh_trainer.save(str(descendant))
    descendant_checkpoint = torch.load(
        descendant,
        map_location="cpu",
        weights_only=False,
    )
    assert descendant_checkpoint["initialization_provenance"] == provenance
    descendant_agent = PokeAgent(config, game_data)
    descendant_trainer = PPOTrainer(
        descendant_agent,
        TrainingConfig(seed=23),
        torch.device("cpu"),
    )
    descendant_trainer.load(str(descendant))
    assert descendant_trainer.initialization_provenance == provenance
