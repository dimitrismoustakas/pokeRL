"""Pluggable memory modules for temporal reasoning over battle history."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence


@dataclass(frozen=True)
class _PackedLSTMPlan:
    """CPU-side indexing plan for one reset-aware packed LSTM window."""

    gather_indices: torch.Tensor
    batch_sizes: torch.Tensor
    initial_state_indices: torch.Tensor
    final_state_indices: torch.Tensor


def _build_packed_lstm_plan(dones: torch.Tensor) -> _PackedLSTMPlan:
    """Split each environment after terminals and sort fragments by length."""
    if dones.device.type != "cpu" or dones.dtype != torch.bool:
        raise ValueError("Packed LSTM dones must be a CPU boolean tensor.")
    if dones.ndim != 2 or dones.shape[0] <= 0 or dones.shape[1] <= 0:
        raise ValueError("Packed LSTM dones must have shape (steps, envs).")

    num_steps, num_envs = dones.shape
    terminal = dones.numpy().T.copy()
    terminal[:, -1] = True
    env, last_step = np.nonzero(terminal)
    ends = last_step + 1
    starts = np.zeros_like(ends)
    starts[1:] = np.where(env[1:] == env[:-1], ends[:-1], 0)
    lengths = ends - starts

    # Equal-length sequences retain environment and start-time order.
    order = np.argsort(-lengths, kind="stable")
    env, starts, lengths = env[order], starts[order], lengths[order]
    time = np.arange(lengths[0])
    batch_sizes = np.searchsorted(-lengths, -time)

    # PackedSequence concatenates one row of active fragments at each time.
    row_starts = np.cumsum(batch_sizes) - batch_sizes
    packed_time = np.repeat(time, batch_sizes)
    fragment = np.arange(num_steps * num_envs) - row_starts[packed_time]
    gather_indices = (starts[fragment] + packed_time) * num_envs + env[fragment]
    initial_state_indices = np.where(starts == 0, env, num_envs)
    final_fragments = np.flatnonzero(ends[order] == num_steps)
    final_state_indices = np.empty(num_envs, dtype=np.int64)
    final_state_indices[env[final_fragments]] = final_fragments

    return _PackedLSTMPlan(
        gather_indices=torch.from_numpy(gather_indices),
        batch_sizes=torch.from_numpy(batch_sizes),
        initial_state_indices=torch.from_numpy(initial_state_indices),
        final_state_indices=torch.from_numpy(final_state_indices),
    )


class IdentityMemory(nn.Module):
    """No-op memory: just projects the observation. For feedforward baseline."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self, x: torch.Tensor, hidden: None = None
    ) -> tuple[torch.Tensor, None]:
        return self.proj(x), None

    def initial_hidden(self, batch_size: int) -> None:
        return None

    def unroll(
        self,
        x: torch.Tensor,
        dones: torch.Tensor,
        hidden: None = None,
        bptt_chunk_steps: int | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Project a complete time-major sequence in one batched call."""
        del dones, hidden, bptt_chunk_steps
        return self.proj(x), None


class LSTMMemory(nn.Module):
    """LSTM history fused with a short path from the exact current state.

    The recurrent state carries partial-observability evidence, while the
    residual projection prevents current public mechanics from having to
    survive an LSTM gate before the policy can use them.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.current_proj = nn.Linear(input_dim, hidden_dim)
        # RMS normalization keeps scale controlled without forcing every
        # context vector's feature sum to zero (which needlessly removes one
        # direction from downstream linear heads).
        self.fusion_norm = nn.RMSNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def _fuse_current(
        self,
        recurrent: torch.Tensor,
        current: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion_norm(recurrent + self.current_proj(current))

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: (batch, input_dim) for single step, or (batch, seq, input_dim) for sequence
        was_2d = x.dim() == 2
        if was_2d:
            x = x.unsqueeze(1)

        output, hidden = self.lstm(x, hidden)
        output = self._fuse_current(output, x)

        if was_2d:
            output = output.squeeze(1)
        return output, hidden

    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def unroll(
        self,
        x: torch.Tensor,
        dones: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        bptt_chunk_steps: int | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run a time-major sequence with reset-aware packed LSTM calls.

        Episode fragments are independent packed sequences. Explicit truncated
        BPTT retains the same global window boundaries as the stepwise path.
        """
        if x.ndim != 3:
            raise ValueError(
                "LSTM unroll input must have shape (steps, envs, features)."
            )
        num_steps, num_envs, _ = x.shape
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError("LSTM unroll requires at least one step and environment.")
        if dones.shape != (num_steps, num_envs):
            raise ValueError("LSTM dones must have shape (steps, envs).")
        if dones.device.type != "cpu":
            dones = dones.detach().to(device="cpu", dtype=torch.bool)
        elif dones.dtype != torch.bool:
            dones = dones.to(dtype=torch.bool)

        chunk_steps = (
            num_steps
            if bptt_chunk_steps is None or bptt_chunk_steps <= 0
            else min(bptt_chunk_steps, num_steps)
        )
        outputs: list[torch.Tensor] = []
        next_hidden = hidden

        for window_start in range(0, num_steps, chunk_steps):
            window_end = min(window_start + chunk_steps, num_steps)
            window_x = x[window_start:window_end]
            window_dones = dones[window_start:window_end]
            plan = _build_packed_lstm_plan(window_dones)
            gather_indices = plan.gather_indices.to(device=x.device)
            packed_input = PackedSequence(
                window_x.reshape(-1, window_x.shape[-1]).index_select(
                    0,
                    gather_indices,
                ),
                plan.batch_sizes,
            )

            if next_hidden is None:
                packed_output, packed_hidden = self.lstm(packed_input)
            else:
                zero_state = next_hidden[0].new_zeros((1, 1, self.hidden_dim))
                initial_state_indices = plan.initial_state_indices.to(device=x.device)
                packed_initial = tuple(
                    torch.cat((state, zero_state), dim=1).index_select(
                        1,
                        initial_state_indices,
                    )
                    for state in next_hidden
                )
                packed_output, packed_hidden = self.lstm(
                    packed_input,
                    packed_initial,
                )

            flat_output = packed_output.data.new_zeros(
                (window_x.shape[0] * num_envs, self.hidden_dim)
            ).index_copy(0, gather_indices, packed_output.data)
            recurrent_output = flat_output.reshape(
                window_x.shape[0], num_envs, self.hidden_dim
            )
            outputs.append(self._fuse_current(recurrent_output, window_x))

            final_state_indices = plan.final_state_indices.to(device=x.device)
            next_hidden = tuple(
                state.index_select(1, final_state_indices) for state in packed_hidden
            )
            not_done = (
                (~window_dones[-1])
                .to(
                    device=x.device,
                    dtype=x.dtype,
                )
                .view(1, num_envs, 1)
            )
            next_hidden = tuple(state * not_done for state in next_hidden)
            if window_end < num_steps:
                next_hidden = tuple(state.detach() for state in next_hidden)

        return torch.cat(outputs, dim=0), next_hidden


def build_memory(memory_type: str, input_dim: int, hidden_dim: int) -> nn.Module:
    if memory_type == "lstm":
        return LSTMMemory(input_dim, hidden_dim)
    elif memory_type == "identity":
        return IdentityMemory(input_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
