import pytest

from scripts.train import _close_training_resources


def test_close_training_resources_attempts_every_close() -> None:
    calls: list[str] = []

    class _Resource:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

    with pytest.raises(ExceptionGroup, match="training resources"):
        _close_training_resources(
            ("first", _Resource("first", fail=True)),
            ("second", _Resource("second")),
            ("third", _Resource("third", fail=True)),
        )

    assert calls == ["first", "second", "third"]
