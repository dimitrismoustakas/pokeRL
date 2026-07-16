import json

import pytest
import torch
import torch.nn as nn

from pokerl.model.checkpoint import model_checkpoint_metadata
from pokerl.training.population import PolicyRole, PopulationRevision
from pokerl.training.ppo import TRAINER_CHECKPOINT_FORMAT_VERSION
from scripts.bootstrap_population import build_initial_population
from scripts.plan_exploiter_phase import plan_exploiter


def _trainer_checkpoint(path, *, signature: str = "2" * 64) -> None:
    agent = nn.Linear(2, 2)
    agent.config = {"width": 2}
    agent.game_data_signature = "0" * 64
    agent.architecture_signature = "1" * 64
    agent.model_signature = signature
    torch.save(
        {
            "checkpoint_type": "trainer",
            "format_version": TRAINER_CHECKPOINT_FORMAT_VERSION,
            "model_metadata": model_checkpoint_metadata(agent),
            "agent": agent.state_dict(),
            "training_state_id": "state-main-v3",
        },
        path,
    )


def _team_pool(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    team_path = tmp_path / "team.txt"
    team_path.write_text(
        "Pikachu @ Light Ball\nAbility: Static\nLevel: 50\n- Thunderbolt\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "gen9vgc2026regf",
                "purpose": "training",
                "allow_training": True,
                "curated": [{"title": "Test", "file": str(team_path)}],
                "generated": [],
                "pool": [{"title": "Test", "file": str(team_path)}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_bootstrap_population_pins_checkpoint_and_team_pool(tmp_path) -> None:
    checkpoint = tmp_path / "main.pt"
    _trainer_checkpoint(checkpoint)
    output = tmp_path / "population" / "revision.json"

    revision = build_initial_population(
        checkpoint_path=checkpoint,
        output_path=output,
        team_pool_manifest=_team_pool(tmp_path),
        team_pool_subset="curated",
        policy_id="main-v3-u20",
        lineage_id="main-v3",
        battle_format="gen9vgc2026regf",
    )

    assert revision.evaluation_action_mode == "sample"
    assert revision.model_signature == "2" * 64
    assert revision.policies[0].role is PolicyRole.MAIN
    assert revision.policies[0].source_training_state_id == "state-main-v3"
    assert PopulationRevision.load(output) == revision


def test_plan_main_and_singleton_league_exploiters_are_runnable(tmp_path) -> None:
    checkpoint = tmp_path / "main.pt"
    _trainer_checkpoint(checkpoint)
    population_path = tmp_path / "population" / "revision.json"
    build_initial_population(
        checkpoint_path=checkpoint,
        output_path=population_path,
        team_pool_manifest=_team_pool(tmp_path),
        team_pool_subset="curated",
        policy_id="main-v3-u20",
        lineage_id="main-v3",
        battle_format="gen9vgc2026regf",
    )

    main_phase, main_certificate = plan_exploiter(
        population_path=population_path,
        output_path=tmp_path / "phases" / "main-exploiter.json",
        role="main",
        phase_id="main-exploiter-v3-1",
        learner_policy_id="main-exploiter-v3-1-final",
        initializer_policy_id="main-v3-u20",
        target_main_policy_id="main-v3-u20",
    )
    league_phase, league_certificate = plan_exploiter(
        population_path=population_path,
        output_path=tmp_path / "phases" / "league-exploiter.json",
        role="league",
        phase_id="league-exploiter-v3-1",
        learner_policy_id="league-exploiter-v3-1-final",
        initializer_policy_id="main-v3-u20",
    )

    assert main_certificate["target_distribution"] == {"main-v3-u20": 1.0}
    assert league_certificate["target_distribution"] == {"main-v3-u20": 1.0}
    assert league_certificate["certified_exploitability_gap"] == pytest.approx(0.0)
    assert main_phase.resolve_target_population_path(
        tmp_path / "phases"
    ) == population_path.resolve()
    assert league_phase.resolve_target_population_path(
        tmp_path / "phases"
    ) == population_path.resolve()


def test_plan_main_exploiter_requires_an_explicit_main_target(tmp_path) -> None:
    checkpoint = tmp_path / "main.pt"
    _trainer_checkpoint(checkpoint)
    population_path = tmp_path / "population.json"
    build_initial_population(
        checkpoint_path=checkpoint,
        output_path=population_path,
        team_pool_manifest=_team_pool(tmp_path),
        team_pool_subset="curated",
        policy_id="main",
        lineage_id="main",
        battle_format="gen9vgc2026regf",
    )

    with pytest.raises(ValueError, match="target-main-policy-id"):
        plan_exploiter(
            population_path=population_path,
            output_path=tmp_path / "phase.json",
            role="main",
            phase_id="missing-target",
            learner_policy_id="exploit",
            initializer_policy_id="main",
        )


def test_bootstrap_rejects_noncurrent_or_incomplete_trainer_checkpoint(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "main.pt"
    _trainer_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format_version"] = TRAINER_CHECKPOINT_FORMAT_VERSION - 1
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="unsupported trainer format_version"):
        build_initial_population(
            checkpoint_path=checkpoint,
            output_path=tmp_path / "population.json",
            team_pool_manifest=_team_pool(tmp_path),
            team_pool_subset="curated",
            policy_id="main",
            lineage_id="main",
            battle_format="gen9vgc2026regf",
        )

    _trainer_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_metadata"].pop("architecture_signature")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="architecture_signature"):
        build_initial_population(
            checkpoint_path=checkpoint,
            output_path=tmp_path / "population.json",
            team_pool_manifest=_team_pool(tmp_path),
            team_pool_subset="curated",
            policy_id="main",
            lineage_id="main",
            battle_format="gen9vgc2026regf",
        )

    _trainer_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_metadata"]["model_config"] = {"type": "builtins.dict"}
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="complete model_config metadata"):
        build_initial_population(
            checkpoint_path=checkpoint,
            output_path=tmp_path / "population.json",
            team_pool_manifest=_team_pool(tmp_path),
            team_pool_subset="curated",
            policy_id="main",
            lineage_id="main",
            battle_format="gen9vgc2026regf",
        )
