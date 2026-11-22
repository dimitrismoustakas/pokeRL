from dataclasses import replace

from pokerl.config import ModelConfig
from pokerl.model.signature import architecture_signature, model_signature


def test_architecture_signature_is_stable_and_config_sensitive() -> None:
    config = ModelConfig()

    assert architecture_signature(config) == architecture_signature(config)
    assert architecture_signature(replace(config, memory_hidden=384)) != (
        architecture_signature(config)
    )


def test_model_signature_covers_architecture_and_vocabulary() -> None:
    config = ModelConfig()
    baseline = model_signature(config, "vocab-a")

    assert len(baseline) == 64
    assert model_signature(config, "vocab-b") != baseline
    assert model_signature(replace(config, move_dim=48), "vocab-a") != baseline
