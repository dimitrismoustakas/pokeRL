# Repository working rules

## Required checks

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check pokerl scripts tests
& .\.venv\Scripts\ruff.exe format --check pokerl scripts tests
```

Run focused tests while developing, then the relevant broader suite before
handoff.

## Engineering principles

- Fail loudly. Do not hide invalid state behind silent fallbacks.
- Fix root causes; do not add bypasses or compatibility branches without a
  demonstrated use case.
- Prefer the smallest implementation that solves a demonstrated problem. Do not
  add speculative abstractions, sidecars, or defensive checks without a clear
  failure mode and payoff.
- Keep one clear production path. Put retired experiments in a pinned,
  reproducible archive rather than runtime compatibility branches.
- Separate simulation, environment, model, training, evaluation, and
  orchestration concerns.
- Preserve sparse terminal rewards. Public mechanics and diagnostics may be
  observations or auxiliary targets, but not hand-shaped reward.
- Treat private simulator data as trainer-only labels. Inference receives only
  information legitimately available to a player.
- Preserve user changes and unrelated worktree state.
- Speak like a working ML engineer: use ordinary established terms, define
  necessary jargon, and do not invent process labels.
- Keep documentation current and compact. Replace
  `docs/INDEPENDENT_REVIEW.md`; do not append a chronological project diary.
