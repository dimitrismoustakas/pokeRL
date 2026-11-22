"""Team preview decisions and PPO episode-start samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def _copy_array_mapping(
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in values.items()}


@dataclass(frozen=True)
class TeamPreviewEpisodeStart:
    """Immutable trainer record for a preview action at episode start."""

    obs: Mapping[str, np.ndarray]
    picks: np.ndarray
    log_prob: float
    value: float
    policy_version: int
    opponent_picks: np.ndarray
    opponent_aux_labels: Mapping[str, np.ndarray] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "obs", _copy_array_mapping(self.obs))
        object.__setattr__(
            self,
            "picks",
            np.array(self.picks, dtype=np.int64, copy=True),
        )
        object.__setattr__(self, "log_prob", float(self.log_prob))
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(
            self,
            "opponent_picks",
            np.array(self.opponent_picks, dtype=np.int64, copy=True),
        )
        if self.opponent_aux_labels is not None:
            object.__setattr__(
                self,
                "opponent_aux_labels",
                _copy_array_mapping(self.opponent_aux_labels),
            )
