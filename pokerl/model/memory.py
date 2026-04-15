"""Pluggable memory modules for temporal reasoning over battle history."""

import torch
import torch.nn as nn


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


class LSTMMemory(nn.Module):
    """LSTM memory: maintains hidden state across turns within a battle."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

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

        if was_2d:
            output = output.squeeze(1)
        return output, hidden

    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h, c)


def build_memory(memory_type: str, input_dim: int, hidden_dim: int) -> nn.Module:
    if memory_type == "lstm":
        return LSTMMemory(input_dim, hidden_dim)
    elif memory_type == "identity":
        return IdentityMemory(input_dim, hidden_dim)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
