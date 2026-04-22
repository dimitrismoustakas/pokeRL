"""Vectorized CFR+ solver for the per-turn stage-game matrix.

Used by Option D (``ModelConfig.belief_mode="nash"``) to replace the
belief-weighted Q integration with a proper zero-sum Nash mixed strategy on
the 27x27 matrix defined by ``Q(s, our_action, opp_action)``. At this matrix
size CFR+ converges in well under 200 iterations; we keep the iteration count
modest because the full flat batch (``T x E_mb`` states) solves together.

The solver assumes ``Q`` is the row player's payoff and the game is zero-sum
(the column player's payoff is ``-Q``). In Pokemon the final reward is
strictly +1 / -1, so the game is zero-sum at the terminal level; for the
stage-game approximation we inherit that property by training ``Q`` to
predict the GAE return (which itself is a bootstrapped zero-sum quantity).

Known simplifications:
- Point-estimate belief. A proper POMG treatment (DeepStack-style) would
  operate on ranges; here we solve the stage game at the currently-observed
  state without a range distribution.
- Q is trained under GAE-regressed semantics, not Nash-Q. The Nash we solve
  for is therefore "Nash of the learned Q", not "Nash of the true game".
- Opponent action legality is not masked (we don't have the opponent's
  legal-action mask from our side). CFR naturally downweights dominated
  column actions given a well-trained Q.
"""

import torch


def solve_matrix_game_cfr_plus(
    payoff: torch.Tensor,
    row_legal: torch.Tensor,
    num_iters: int = 200,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a batched zero-sum matrix game via CFR+ regret matching.

    Args:
        payoff: ``(batch, A, A)`` — row player's payoff when row picks ``i``
            and column picks ``j``.
        row_legal: ``(batch, A)`` bool mask of legal row actions. Illegal row
            actions are pinned to zero probability.
        num_iters: CFR+ iterations.
        eps: small constant used for safe division.

    Returns:
        row_mix: ``(batch, A)`` — row player's average Nash mixed strategy,
            supported on legal row actions.
        col_mix: ``(batch, A)`` — column player's average Nash mixed strategy.
    """
    if payoff.dim() != 3 or payoff.shape[-1] != payoff.shape[-2]:
        raise ValueError(f"payoff must have shape (batch, A, A), got {tuple(payoff.shape)}")
    if row_legal.shape != payoff.shape[:2]:
        raise ValueError(
            f"row_legal must have shape (batch, A), got {tuple(row_legal.shape)} for payoff {tuple(payoff.shape)}"
        )

    batch_size, num_actions, _ = payoff.shape
    device = payoff.device
    dtype = payoff.dtype

    row_legal_f = row_legal.to(dtype)
    row_legal_count = row_legal_f.sum(-1, keepdim=True).clamp(min=1.0)
    uniform_row = row_legal_f / row_legal_count
    uniform_col = torch.full(
        (batch_size, num_actions), 1.0 / num_actions, device=device, dtype=dtype
    )

    row_regret = torch.zeros((batch_size, num_actions), device=device, dtype=dtype)
    col_regret = torch.zeros((batch_size, num_actions), device=device, dtype=dtype)
    row_mix_sum = torch.zeros((batch_size, num_actions), device=device, dtype=dtype)
    col_mix_sum = torch.zeros((batch_size, num_actions), device=device, dtype=dtype)

    row_mix = uniform_row.clone()
    col_mix = uniform_col.clone()

    for _ in range(num_iters):
        row_ev = torch.bmm(payoff, col_mix.unsqueeze(-1)).squeeze(-1)
        # Row-given-col-plays-j payoff: needed to compute column player's EV in
        # the zero-sum frame (col's payoff is -row's).
        row_payoff_per_col = torch.bmm(payoff.transpose(1, 2), row_mix.unsqueeze(-1)).squeeze(-1)
        col_ev = -row_payoff_per_col

        row_ev_mean = (row_mix * row_ev).sum(-1, keepdim=True)
        col_ev_mean = (col_mix * col_ev).sum(-1, keepdim=True)

        row_regret = torch.clamp(row_regret + (row_ev - row_ev_mean), min=0.0) * row_legal_f
        col_regret = torch.clamp(col_regret + (col_ev - col_ev_mean), min=0.0)

        row_regret_sum = row_regret.sum(-1, keepdim=True)
        col_regret_sum = col_regret.sum(-1, keepdim=True)
        row_mix = torch.where(
            row_regret_sum > 0,
            row_regret / row_regret_sum.clamp(min=eps),
            uniform_row,
        )
        col_mix = torch.where(
            col_regret_sum > 0,
            col_regret / col_regret_sum.clamp(min=eps),
            uniform_col,
        )

        row_mix_sum = row_mix_sum + row_mix
        col_mix_sum = col_mix_sum + col_mix

    avg_row = row_mix_sum / num_iters
    avg_col = col_mix_sum / num_iters

    # Enforce row legality on the averaged strategy; renormalize. Illegal
    # columns are not filtered (see module docstring).
    avg_row = avg_row * row_legal_f
    avg_row = avg_row / avg_row.sum(-1, keepdim=True).clamp(min=eps)
    return avg_row, avg_col
