import copy

import pytest
import torch

from pokerl.model.memory import (
    IdentityMemory,
    LSTMMemory,
    _build_packed_lstm_plan,
)


def _stepwise_unroll(
    memory: LSTMMemory,
    inputs: torch.Tensor,
    dones: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
    chunk_steps: int | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    outputs = []
    num_steps, num_envs, _ = inputs.shape
    for step in range(num_steps):
        output, hidden = memory(inputs[step], hidden)
        outputs.append(output)
        not_done = (
            (~dones[step])
            .to(
                device=output.device,
                dtype=output.dtype,
            )
            .view(1, num_envs, 1)
        )
        hidden = tuple(state * not_done for state in hidden)
        if (
            chunk_steps is not None
            and chunk_steps > 0
            and (step + 1) % chunk_steps == 0
            and step + 1 < num_steps
        ):
            hidden = tuple(state.detach() for state in hidden)
    return torch.stack(outputs), hidden


DONE_PATTERNS = (
    torch.zeros((7, 4), dtype=torch.bool),
    torch.ones((7, 4), dtype=torch.bool),
    torch.tensor(
        [
            [False, False, True, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [True, False, False, True],
            [False, False, False, False],
            [False, True, False, True],
        ]
    ),
)


@pytest.mark.parametrize("dones", DONE_PATTERNS)
@pytest.mark.parametrize("chunk_steps", [None, 0, 1, 3, 7, 20])
def test_packed_lstm_unroll_matches_stepwise_outputs_and_gradients(
    dones: torch.Tensor,
    chunk_steps: int | None,
) -> None:
    torch.manual_seed(7)
    reference = LSTMMemory(input_dim=5, hidden_dim=3).double()
    packed = copy.deepcopy(reference)
    reference_inputs = torch.randn(7, 4, 5, dtype=torch.float64, requires_grad=True)
    packed_inputs = reference_inputs.detach().clone().requires_grad_(True)
    reference_initial_hidden = (
        torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True),
        torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True),
    )
    packed_initial_hidden = tuple(
        state.detach().clone().requires_grad_(True)
        for state in reference_initial_hidden
    )

    reference_output, reference_hidden = _stepwise_unroll(
        reference,
        reference_inputs,
        dones,
        reference_initial_hidden,
        chunk_steps,
    )
    packed_output, packed_hidden = packed.unroll(
        packed_inputs,
        dones,
        packed_initial_hidden,
        bptt_chunk_steps=chunk_steps,
    )

    torch.testing.assert_close(packed_output, reference_output, atol=1e-12, rtol=1e-12)
    for actual, expected in zip(packed_hidden, reference_hidden, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    weights = torch.randn_like(reference_output)
    (reference_output * weights).sum().backward()
    (packed_output * weights).sum().backward()
    torch.testing.assert_close(
        packed_inputs.grad,
        reference_inputs.grad,
        atol=1e-12,
        rtol=1e-12,
    )
    for actual, expected in zip(
        packed_initial_hidden,
        reference_initial_hidden,
        strict=True,
    ):
        torch.testing.assert_close(
            actual.grad,
            expected.grad,
            atol=1e-12,
            rtol=1e-12,
        )
    for packed_parameter, reference_parameter in zip(
        packed.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            packed_parameter.grad,
            reference_parameter.grad,
            atol=1e-11,
            rtol=1e-11,
        )


def test_packed_lstm_plan_covers_every_token_once() -> None:
    dones = DONE_PATTERNS[-1]
    plan = _build_packed_lstm_plan(dones)

    assert torch.equal(
        plan.gather_indices.sort().values,
        torch.arange(dones.numel()),
    )
    assert torch.all(plan.batch_sizes[:-1] >= plan.batch_sizes[1:])
    assert plan.initial_state_indices.shape[0] == plan.batch_sizes[0]
    assert plan.final_state_indices.shape == (dones.shape[1],)


@pytest.mark.parametrize(
    ("dones", "chunk_steps", "loss_step"),
    [
        (torch.tensor([[False], [True], [False], [False]]), None, 2),
        (torch.zeros((4, 1), dtype=torch.bool), 2, 2),
    ],
)
def test_packed_lstm_stops_gradients_at_resets_and_tbptt_boundaries(
    dones: torch.Tensor,
    chunk_steps: int | None,
    loss_step: int,
) -> None:
    memory = LSTMMemory(input_dim=2, hidden_dim=3).double()
    inputs = torch.randn(4, 1, 2, dtype=torch.float64, requires_grad=True)

    outputs, _ = memory.unroll(
        inputs,
        dones,
        bptt_chunk_steps=chunk_steps,
    )
    outputs[loss_step].sum().backward()

    assert torch.count_nonzero(inputs.grad[:loss_step]) == 0
    assert torch.count_nonzero(inputs.grad[loss_step]) > 0


def test_identity_memory_unroll_matches_flat_projection() -> None:
    memory = IdentityMemory(input_dim=3, hidden_dim=4)
    inputs = torch.randn(5, 2, 3)
    dones = torch.randint(0, 2, (5, 2), dtype=torch.bool)

    outputs, hidden = memory.unroll(inputs, dones, bptt_chunk_steps=2)

    torch.testing.assert_close(outputs, memory.proj(inputs))
    assert hidden is None


def test_lstm_unroll_rejects_empty_sequences() -> None:
    memory = LSTMMemory(input_dim=3, hidden_dim=4)

    with pytest.raises(ValueError, match="at least one step"):
        memory.unroll(
            torch.empty((0, 2, 3)),
            torch.empty((0, 2), dtype=torch.bool),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_packed_lstm_cuda_has_numerical_parity_with_stepwise() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    reference = LSTMMemory(input_dim=5, hidden_dim=4).to(device)
    packed = copy.deepcopy(reference)
    reference.lstm.flatten_parameters()
    packed.lstm.flatten_parameters()
    dones = DONE_PATTERNS[-1].to(device="cpu")
    reference_inputs = torch.randn(
        7,
        4,
        5,
        device=device,
        requires_grad=True,
    )
    packed_inputs = reference_inputs.detach().clone().requires_grad_(True)
    reference_initial = (
        torch.randn(1, 4, 4, device=device),
        torch.randn(1, 4, 4, device=device),
    )

    expected, _ = _stepwise_unroll(
        reference,
        reference_inputs,
        dones,
        reference_initial,
        None,
    )
    actual, _ = packed.unroll(
        packed_inputs,
        dones,
        reference_initial,
    )
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        packed_inputs.grad,
        reference_inputs.grad,
        atol=2e-5,
        rtol=2e-5,
    )
    for actual_parameter, expected_parameter in zip(
        packed.parameters(),
        reference.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=1e-4,
            rtol=2e-5,
        )
