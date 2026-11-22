"""Create the first immutable population revision from a trained main policy.

This is intentionally a one-policy operation.  Later policies must be admitted
with frozen cross-play evidence through ``build_population_revision.py``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from pokerl.team_pool import load_team_pool
from pokerl.model.checkpoint import validate_declared_model_checkpoint_metadata
from pokerl.training.ppo import TRAINER_CHECKPOINT_FORMAT_VERSION
from pokerl.training.population import (
    CheckpointRef,
    PolicyMember,
    PolicyRole,
    PopulationRevision,
)


def checkpoint_identity(path: str | Path) -> tuple[str, str]:
    """Return the complete model signature and training-state id."""
    source = Path(path).expanduser().resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Main checkpoint must contain a trainer checkpoint mapping.")
    if checkpoint.get("checkpoint_type") != "trainer":
        raise ValueError("Main checkpoint must be a versioned trainer checkpoint.")
    if checkpoint.get("format_version") != TRAINER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Main checkpoint has unsupported trainer format_version="
            f"{checkpoint.get('format_version')!r}; expected "
            f"{TRAINER_CHECKPOINT_FORMAT_VERSION}."
        )
    missing = sorted(
        {"agent", "model_metadata", "training_state_id"}.difference(checkpoint)
    )
    if missing:
        raise ValueError(
            "Main checkpoint is missing required key(s): " + ", ".join(missing)
        )
    metadata = checkpoint.get("model_metadata")
    _, metadata = validate_declared_model_checkpoint_metadata(
        path=str(source),
        checkpoint_state=checkpoint["agent"],
        checkpoint_metadata=metadata,
    )
    if not isinstance(metadata.get("model_type"), str) or not metadata["model_type"]:
        raise ValueError("Main checkpoint model_type must be a non-empty string.")
    model_config = metadata.get("model_config")
    if (
        not isinstance(model_config, Mapping)
        or set(model_config) != {"type", "values"}
        or not isinstance(model_config.get("type"), str)
        or not model_config["type"]
        or not isinstance(model_config.get("values"), Mapping)
    ):
        raise ValueError("Main checkpoint must contain complete model_config metadata.")
    for name in (
        "game_data_signature",
        "architecture_signature",
        "model_signature",
    ):
        signature = metadata.get(name)
        if (
            not isinstance(signature, str)
            or len(signature) != 64
            or any(character not in "0123456789abcdef" for character in signature)
        ):
            raise ValueError(
                f"Main checkpoint {name} must be a lowercase SHA-256 digest."
            )
    model_signature = metadata.get("model_signature")
    training_state_id = checkpoint.get("training_state_id")
    assert isinstance(model_signature, str)
    if not isinstance(training_state_id, str) or not training_state_id:
        raise ValueError("Main checkpoint is missing a valid training_state_id.")
    return model_signature, training_state_id


def build_initial_population(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    team_pool_manifest: str | Path,
    team_pool_subset: str,
    policy_id: str,
    lineage_id: str,
    battle_format: str,
) -> PopulationRevision:
    """Build and save a sampled one-main population revision."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    model_signature, training_state_id = checkpoint_identity(checkpoint)
    team_pool = load_team_pool(
        team_pool_manifest,
        subset=team_pool_subset,
        usage="training",
    )
    if team_pool.format_id != battle_format:
        raise ValueError(
            "Team-pool format does not match --battle-format: "
            f"pool={team_pool.format_id!r}, requested={battle_format!r}."
        )
    member = PolicyMember(
        policy_id=policy_id,
        role=PolicyRole.MAIN,
        checkpoint=CheckpointRef.from_file(
            checkpoint,
            relative_to=output.parent,
        ),
        lineage_id=lineage_id,
        parent_policy_id=None,
        generation=0,
        source_training_state_id=training_state_id,
    )
    revision = PopulationRevision.create(
        parent_revision_id=None,
        battle_format=battle_format,
        team_pool_signature=team_pool.signature(),
        model_signature=model_signature,
        evaluation_action_mode="sample",
        policies=(member,),
    )
    revision.save(output)
    return revision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap an immutable sampled population with one trained main "
            "policy. No unevaluated child policy is admitted by this command."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--team-pool-manifest", required=True)
    parser.add_argument("--team-pool-subset", default="curated")
    parser.add_argument("--battle-format", default="gen9vgc2026regf")
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = build_initial_population(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        team_pool_manifest=args.team_pool_manifest,
        team_pool_subset=args.team_pool_subset,
        policy_id=args.policy_id,
        lineage_id=args.lineage_id,
        battle_format=args.battle_format,
    )
    print(
        json.dumps(
            {
                "revision_id": revision.revision_id,
                "policy_id": revision.policies[0].policy_id,
                "checkpoint_sha256": revision.policies[0].checkpoint.sha256,
                "team_pool_signature": revision.team_pool_signature,
                "model_signature": revision.model_signature,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
