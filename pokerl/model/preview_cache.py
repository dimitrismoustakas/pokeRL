"""Reuse public team-preview encodings within a single policy version."""

import numpy as np
import torch

from pokerl.model.agent import PokeAgent


_SIDE_FIELDS = (
    "species",
    "moves",
    "last_move",
    "ability",
    "item",
    "types",
    "continuous",
    "roles",
    "mask",
)
_SIDE_COLUMNS = {
    side: tuple(f"{side}_{key}" for key in _SIDE_FIELDS) for side in ("own", "opp")
}
_CONTEXT_COLUMNS = ("field", "active_mechanics", "active_mechanics_move")
PREVIEW_COLUMNS = (*_SIDE_COLUMNS["own"], *_SIDE_COLUMNS["opp"], *_CONTEXT_COLUMNS)


class TeamPreviewCache:
    def __init__(self):
        self._teams = {"own": {}, "opp": {}}

    def _encode_side(self, agent, device, batch, side):
        columns = _SIDE_COLUMNS[side]
        cache = self._teams[side]
        # Cache identity includes only values visible to this player.
        key_bytes = np.concatenate(
            [
                np.ascontiguousarray(batch[key])
                .view(np.uint8)
                .reshape(len(batch[key]), -1)
                for key in columns
            ],
            axis=1,
        )
        keys = [row.tobytes() for row in key_bytes]
        missing = {}
        for i, key in enumerate(keys):
            if key not in cache:
                missing.setdefault(key, i)
        if missing:
            indices = list(missing.values())
            obs = {
                key: torch.from_numpy(batch[key][indices]).to(device) for key in columns
            }
            reps, summaries = agent.spatial.encode_side(obs, side)
            cache.update(
                zip(
                    missing,
                    zip(reps.unbind(), summaries.unbind(), strict=True),
                    strict=True,
                )
            )
        return torch.stack([cache[key][0] for key in keys]), torch.stack(
            [cache[key][1] for key in keys]
        )

    @torch.inference_mode()
    def predict(
        self,
        agent: PokeAgent,
        device: torch.device,
        observations: dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        include_value: bool,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        if agent.training:
            raise RuntimeError(
                "Team preview encodings require a policy in evaluation mode."
            )
        batch = {key: observations[key][indices] for key in PREVIEW_COLUMNS}
        own, own_summary = self._encode_side(agent, device, batch, "own")
        foe, foe_summary = self._encode_side(agent, device, batch, "opp")
        obs = {
            key: torch.from_numpy(batch[key]).to(device)
            for key in (*_CONTEXT_COLUMNS, "own_mask", "opp_mask")
        }
        context = agent.spatial.battle_context(
            obs, own_summary, foe_summary, include_recent_events=False
        )
        return agent.team_preview_from_encodings(
            obs,
            context,
            own,
            foe,
            greedy=greedy,
            include_value=include_value,
            include_policy_stats=False,
            include_opponent_preview=False,
        )
