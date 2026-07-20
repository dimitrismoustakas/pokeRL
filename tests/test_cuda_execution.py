from threading import Event, Thread

import torch

from pokerl.training.cuda_execution import cuda_model_execution


def test_cuda_execution_synchronizes_only_outermost_section(monkeypatch) -> None:
    synchronizations: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronizations.append(str(device)),
    )

    device = torch.device("cuda")
    with cuda_model_execution(device):
        with cuda_model_execution(device):
            pass

    assert synchronizations == ["cuda"]


def test_cuda_execution_hands_off_only_after_synchronization(monkeypatch) -> None:
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    events: list[str] = []

    def synchronize(_device) -> None:
        events.append("sync")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    device = torch.device("cuda")

    def first() -> None:
        with cuda_model_execution(device):
            events.append("first")
            first_entered.set()
            assert release_first.wait(timeout=1.0)

    def second() -> None:
        assert first_entered.wait(timeout=1.0)
        with cuda_model_execution(device):
            events.append("second")
            second_entered.set()

    first_thread = Thread(target=first)
    second_thread = Thread(target=second)
    first_thread.start()
    second_thread.start()

    assert first_entered.wait(timeout=1.0)
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert events == ["first", "sync", "second", "sync"]


def test_cpu_execution_does_not_synchronize(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda _device: (_ for _ in ()).throw(AssertionError("unexpected sync")),
    )

    with cuda_model_execution(torch.device("cpu")):
        pass
