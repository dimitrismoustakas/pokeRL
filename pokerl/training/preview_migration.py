"""Convert version-3 weights into a fresh exact-preview initialization."""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from ..config import ModelConfig, TrainingConfig
from ..data import GameData
from ..model.agent import PokeAgent
from ..model.checkpoint import validate_declared_model_checkpoint_metadata
from ..model.signature import MODEL_ARCHITECTURE_VERSION
from .ppo import PPOTrainer


PREVIEW_MIGRATION_FORMAT_VERSION = 1
LEGACY_ARCHITECTURE_VERSION = 3
TARGET_ARCHITECTURE_VERSION = 4
LEGACY_TRAINER_CHECKPOINT_FORMAT_VERSION = 7
_ZEROED_POLICY_OUTPUT_TENSORS = (
    "team_preview_head.bring_score.2.weight",
    "team_preview_head.bring_score.2.bias",
    "team_preview_head.lead_pair_score.2.weight",
    "team_preview_head.lead_pair_score.2.bias",
)
_RESET_PREFIXES = ("preview_value_head.",)
_MODEL_CONFIG_TYPE = "pokerl.config.ModelConfig"
_MODEL_TYPE = "pokerl.model.agent.PokeAgent"


def _architecture_signature(
    model_config: Mapping[str, object],
    *,
    architecture_version: int,
) -> str:
    payload = {
        "architecture_version": architecture_version,
        "agent_type": _MODEL_TYPE,
        "model_config": dict(model_config),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_legacy_source(
    source: Path,
    game_data: GameData,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object], Mapping[str, object]]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Preview migration source must be a checkpoint mapping.")
    if checkpoint.get("checkpoint_type") != "trainer":
        raise ValueError("Preview migration requires checkpoint_type='trainer'.")
    if checkpoint.get("format_version") != LEGACY_TRAINER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Preview migration source has unsupported trainer format_version="
            f"{checkpoint.get('format_version')!r}; expected "
            f"{LEGACY_TRAINER_CHECKPOINT_FORMAT_VERSION}."
        )
    missing = sorted(
        {
            "agent",
            "model_metadata",
            "global_step",
            "completed_updates",
            "training_state_id",
        }.difference(checkpoint)
    )
    if missing:
        raise ValueError(
            "Preview migration source is missing required key(s): " + ", ".join(missing)
        )
    source_state, metadata = validate_declared_model_checkpoint_metadata(
        path=str(source),
        checkpoint_state=checkpoint["agent"],
        checkpoint_metadata=checkpoint["model_metadata"],
    )
    if metadata.get("model_type") != _MODEL_TYPE:
        raise ValueError("Preview migration source is not a PokeAgent checkpoint.")
    config_metadata = metadata.get("model_config")
    if (
        not isinstance(config_metadata, Mapping)
        or config_metadata.get("type") != _MODEL_CONFIG_TYPE
        or not isinstance(config_metadata.get("values"), Mapping)
    ):
        raise ValueError(
            "Preview migration source must declare a complete ModelConfig."
        )
    source_config = dict(config_metadata["values"])
    expected_fields = set(ModelConfig.__dataclass_fields__) - {"team_size"}
    if set(source_config) != expected_fields:
        raise ValueError(
            "Preview migration accepts only the exact version-3 ModelConfig "
            "schema without team_size."
        )

    expected_architecture = _architecture_signature(
        source_config,
        architecture_version=LEGACY_ARCHITECTURE_VERSION,
    )
    if metadata.get("architecture_signature") != expected_architecture:
        raise ValueError(
            "Preview migration source does not have the expected version-3 "
            "architecture signature."
        )
    game_data_signature = game_data.vocab_signature()
    if metadata.get("game_data_signature") != game_data_signature:
        raise ValueError(
            "Preview migration source game-data vocabulary does not match the "
            "requested GameData."
        )
    training_state_id = checkpoint.get("training_state_id")
    if not isinstance(training_state_id, str) or not training_state_id:
        raise ValueError(
            "Preview migration source training_state_id must be non-empty."
        )
    for counter_name in ("global_step", "completed_updates"):
        counter = checkpoint.get(counter_name)
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError(
                f"Preview migration source {counter_name} must be a "
                "non-negative integer."
            )
    return source_state, metadata, checkpoint


def _zero_preview_policy_outputs(agent: PokeAgent) -> None:
    """Start the new exact preview policy at a uniform distribution."""
    with torch.no_grad():
        for output_layer in (
            agent.team_preview_head.bring_score[2],
            agent.team_preview_head.lead_pair_score[2],
        ):
            output_layer.weight.zero_()
            output_layer.bias.zero_()


def migrate_v3_preview_checkpoint(
    *,
    source_path: str | Path,
    output_path: str | Path,
    game_data: GameData,
    team_size: int,
    seed: int,
) -> dict[str, object]:
    """Convert compatible version-3 weights into a fresh version-4 learner."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if MODEL_ARCHITECTURE_VERSION != TARGET_ARCHITECTURE_VERSION:
        raise RuntimeError(
            "This conversion supports target architecture version "
            f"{TARGET_ARCHITECTURE_VERSION}, but the current code declares "
            f"{MODEL_ARCHITECTURE_VERSION}."
        )
    if source == output:
        raise ValueError("Preview migration source and output must differ.")
    if output.exists():
        raise FileExistsError("Preview migration refuses to overwrite its output.")
    if isinstance(team_size, bool) or not isinstance(team_size, int):
        raise ValueError("Preview migration team_size must be an integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Preview migration seed must be in [0, 2**32).")

    source_state, source_metadata, source_checkpoint = _load_legacy_source(
        source,
        game_data,
    )
    source_config_metadata = source_metadata["model_config"]
    assert isinstance(source_config_metadata, Mapping)
    source_config_values = source_config_metadata["values"]
    assert isinstance(source_config_values, Mapping)
    target_config_values = dict(source_config_values)
    target_config_values["team_size"] = team_size
    target_config = ModelConfig(**target_config_values)
    if target_config.team_size < target_config.pick_size:
        raise ValueError(
            "Preview migration requires team_size >= the source pick_size."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = PokeAgent(target_config, game_data)
    _zero_preview_policy_outputs(agent)
    target_state = agent.state_dict()
    if set(source_state) != set(target_state):
        raise ValueError(
            "Version-3 and version-4 model tensor schemas differ; this migration "
            "only supports the reviewed preview-semantics change."
        )

    copied_keys: list[str] = []
    reinitialized_keys: list[str] = []
    with torch.no_grad():
        for key, target_tensor in target_state.items():
            source_tensor = source_state[key]
            if source_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"Preview migration tensor shape mismatch for {key!r}."
                )
            if source_tensor.dtype != target_tensor.dtype:
                raise ValueError(
                    f"Preview migration tensor dtype mismatch for {key!r}."
                )
            if key in _ZEROED_POLICY_OUTPUT_TENSORS or key.startswith(_RESET_PREFIXES):
                reinitialized_keys.append(key)
                continue
            target_tensor.copy_(source_tensor)
            copied_keys.append(key)

    target_training_state_id = uuid.uuid4().hex
    provenance = {
        "kind": "v3_to_v4_preview_warm_start",
        "format_version": PREVIEW_MIGRATION_FORMAT_VERSION,
        "source_checkpoint": str(source),
        "source_training_state_id": source_checkpoint["training_state_id"],
        "source_global_step": source_checkpoint["global_step"],
        "source_completed_updates": source_checkpoint["completed_updates"],
        "source_architecture_signature": source_metadata["architecture_signature"],
        "target_training_state_id": target_training_state_id,
        "target_architecture_signature": agent.architecture_signature,
        "target_model_signature": agent.model_signature,
        "team_size": team_size,
        "pick_size": target_config.pick_size,
        "seed": seed,
        "copied_tensor_count": len(copied_keys),
        "reinitialized_tensor_count": len(reinitialized_keys),
        "reinitialized_prefixes": list(_RESET_PREFIXES),
        "zeroed_policy_output_tensors": list(_ZEROED_POLICY_OUTPUT_TENSORS),
        "optimizer_state_imported": False,
        "training_counters_imported": False,
    }
    trainer = PPOTrainer(
        agent,
        TrainingConfig(seed=seed),
        torch.device("cpu"),
    )
    trainer.initialization_provenance = provenance

    output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(
        str(output),
        training_state_id=target_training_state_id,
    )
    return provenance
