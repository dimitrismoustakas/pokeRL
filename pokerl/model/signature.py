"""Content fingerprints for model architecture and categorical vocabulary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from ..config import ModelConfig


MODEL_ARCHITECTURE_VERSION = 6


def architecture_signature(config: ModelConfig) -> str:
    """Fingerprint every declared architectural choice for population safety."""
    payload = {
        "architecture_version": MODEL_ARCHITECTURE_VERSION,
        "agent_type": "pokerl.model.agent.PokeAgent",
        "model_config": asdict(config),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_signature(config: ModelConfig, game_data_signature: str) -> str:
    """Fingerprint the exact architecture/vocabulary compatibility domain."""
    payload = {
        "architecture_signature": architecture_signature(config),
        "game_data_signature": game_data_signature,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
