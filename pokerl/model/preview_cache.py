"""Reuse public team-preview encodings within a single policy version."""

from collections.abc import Sequence

import numpy as np
import torch

from pokerl.model.agent import PokeAgent


class TeamPreviewCache:
    def __init__(self):
        fields = (
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
        self._columns = {
            side: tuple(f"{side}_{key}" for key in fields) for side in ("own", "opp")
        }
        self._teams = {"own": {}, "opp": {}}

    def _encode_side(self, agent, device, rows, side):
        columns = self._columns[side]
        cache = self._teams[side]
        # Only player-visible inputs identify a team. The other side's stat
        # spread or held item must never enter its public preview encoding.
        keys = [b"".join(row[key].tobytes() for key in columns) for row in rows]
        missing = {}
        for key, row in zip(keys, rows, strict=True):
            if key not in cache:
                missing.setdefault(key, row)
        if missing:
            obs = {
                key: torch.from_numpy(
                    np.stack([row[key] for row in missing.values()])
                ).to(device)
                for key in columns
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
        rows: Sequence[dict[str, np.ndarray]],
        *,
        include_value: bool,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        if agent.training:
            raise RuntimeError(
                "Team preview encodings require a policy in evaluation mode."
            )
        own, own_summary = self._encode_side(agent, device, rows, "own")
        foe, foe_summary = self._encode_side(agent, device, rows, "opp")
        columns = (
            "field",
            "active_mechanics",
            "active_mechanics_move",
            "own_mask",
            "opp_mask",
        )
        obs = {
            key: torch.from_numpy(np.stack([row[key] for row in rows])).to(device)
            for key in columns
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
