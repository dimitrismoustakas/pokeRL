"""Shared model-state metadata for strict checkpoint compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

import torch
import torch.nn as nn


MODEL_CHECKPOINT_METADATA_FIELDS = frozenset(
    {
        "model_type",
        "model_config",
        "game_data_signature",
        "architecture_signature",
        "model_signature",
        "state_schema",
    }
)
LEAGUE_MODEL_CHECKPOINT_FORMAT_VERSION = 1


def _qualified_type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _state_dict_schema(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    schema: dict[str, dict[str, object]] = {}
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Model state entry {key!r} must be a tensor, got "
                f"{type(value).__name__}."
            )
        schema[key] = {
            "shape": tuple(int(dimension) for dimension in value.shape),
            "dtype": str(value.dtype),
            "layout": str(value.layout),
        }
    return schema


def _model_config_metadata(agent: nn.Module) -> dict[str, object] | None:
    config = getattr(agent, "config", None)
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        return {
            "type": _qualified_type_name(config),
            "values": asdict(config),
        }
    if isinstance(config, Mapping):
        return {
            "type": _qualified_type_name(config),
            "values": dict(config),
        }
    return None


def model_checkpoint_metadata(agent: nn.Module) -> dict[str, object]:
    """Describe the exact architecture, vocabulary, and tensor-state schema."""
    return {
        "model_type": _qualified_type_name(agent),
        "model_config": _model_config_metadata(agent),
        "game_data_signature": getattr(agent, "game_data_signature", None),
        "architecture_signature": getattr(agent, "architecture_signature", None),
        "model_signature": getattr(agent, "model_signature", None),
        "state_schema": _state_dict_schema(agent.state_dict()),
    }


def validate_declared_model_checkpoint_metadata(
    *,
    path: str,
    checkpoint_state: object,
    checkpoint_metadata: object,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
    """Validate a checkpoint's state against its own complete declaration."""
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError(f"Checkpoint {path!r} agent state must be a mapping.")
    if not isinstance(checkpoint_metadata, Mapping):
        raise ValueError(f"Checkpoint {path!r} model metadata must be a mapping.")
    missing_metadata = sorted(
        MODEL_CHECKPOINT_METADATA_FIELDS.difference(checkpoint_metadata)
    )
    if missing_metadata:
        raise ValueError(
            f"Checkpoint {path!r} model metadata is missing key(s): "
            + ", ".join(missing_metadata)
        )

    declared_schema = checkpoint_metadata["state_schema"]
    if not isinstance(declared_schema, Mapping):
        raise ValueError(f"Checkpoint {path!r} model state schema must be a mapping.")
    actual_checkpoint_schema = _state_dict_schema(checkpoint_state)
    if actual_checkpoint_schema != dict(declared_schema):
        raise ValueError(
            f"Checkpoint {path!r} agent state does not match its declared "
            "model-state schema (missing, unexpected, shape, dtype, or layout mismatch)."
        )
    return checkpoint_state, checkpoint_metadata


def validate_model_checkpoint_metadata(
    *,
    path: str,
    checkpoint_state: object,
    checkpoint_metadata: object,
    current_agent: nn.Module,
) -> Mapping[str, torch.Tensor]:
    """Require an exact model contract before loading checkpoint tensors."""
    checkpoint_state, checkpoint_metadata = validate_declared_model_checkpoint_metadata(
        path=path,
        checkpoint_state=checkpoint_state,
        checkpoint_metadata=checkpoint_metadata,
    )
    declared_schema = checkpoint_metadata["state_schema"]
    assert isinstance(declared_schema, Mapping)

    current_metadata = model_checkpoint_metadata(current_agent)
    if checkpoint_metadata["model_type"] != current_metadata["model_type"]:
        raise ValueError(
            f"Checkpoint {path!r} model type mismatch: "
            f"checkpoint={checkpoint_metadata['model_type']!r}, "
            f"current={current_metadata['model_type']!r}."
        )
    if checkpoint_metadata["model_config"] != current_metadata["model_config"]:
        raise ValueError(
            f"Checkpoint {path!r} model config mismatch; loading requires "
            "the exact architecture configuration used to create the checkpoint."
        )
    if (
        checkpoint_metadata["game_data_signature"]
        != current_metadata["game_data_signature"]
    ):
        raise ValueError(
            f"Checkpoint {path!r} game-data vocabulary mismatch; loading requires "
            "the exact categorical index mapping used by the checkpoint."
        )
    if (
        checkpoint_metadata["architecture_signature"]
        != current_metadata["architecture_signature"]
    ):
        raise ValueError(f"Checkpoint {path!r} architecture signature mismatch.")
    if checkpoint_metadata["model_signature"] != current_metadata["model_signature"]:
        raise ValueError(f"Checkpoint {path!r} model signature mismatch.")

    current_schema = current_metadata["state_schema"]
    if declared_schema != current_schema:
        checkpoint_keys = set(declared_schema)
        current_keys = set(current_schema)
        missing_keys = sorted(current_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - current_keys)
        incompatible_keys = sorted(
            key
            for key in checkpoint_keys & current_keys
            if declared_schema[key] != current_schema[key]
        )
        raise ValueError(
            f"Checkpoint {path!r} model-state schema mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}, "
            f"incompatible={incompatible_keys}."
        )
    return checkpoint_state
