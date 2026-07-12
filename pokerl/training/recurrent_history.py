"""Compact ongoing-episode histories for refreshing recurrent inference state."""

from dataclasses import dataclass

import torch


@dataclass
class RecurrentObservationHistory:
    """Device-resident public observations since each environment's last done."""

    obs: dict[str, torch.Tensor]
    lengths: torch.Tensor  # CPU int64, shape (num_envs,)
    num_envs: int

    @classmethod
    def empty(cls, num_envs: int) -> "RecurrentObservationHistory":
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        return cls(
            obs={},
            lengths=torch.zeros(num_envs, dtype=torch.int64),
            num_envs=num_envs,
        )

    def extend(
        self,
        obs_seq: dict[str, torch.Tensor],
        dones: torch.Tensor,
    ) -> "RecurrentObservationHistory":
        """Append one rollout and discard every prefix ending in a terminal."""
        if not obs_seq:
            raise ValueError("obs_seq must not be empty.")
        num_steps, num_envs = dones.shape
        if num_envs != self.num_envs:
            raise ValueError(
                f"History has {self.num_envs} envs, rollout has {num_envs}."
            )
        if any(value.shape[:2] != (num_steps, num_envs) for value in obs_seq.values()):
            raise ValueError("Every observation tensor must start with (steps, envs).")
        if self.obs and self.obs.keys() != obs_seq.keys():
            raise ValueError("Observation keys changed while an episode was ongoing.")

        done_rows = dones.detach().to(device="cpu", dtype=torch.bool).tolist()
        starts = [0] * num_envs
        carry_lengths = [0] * num_envs
        for env_index in range(num_envs):
            last_done = -1
            for step, done_row in enumerate(done_rows):
                if done_row[env_index]:
                    last_done = step
            if last_done >= 0:
                starts[env_index] = last_done + 1
            else:
                carry_lengths[env_index] = int(self.lengths[env_index])

        new_lengths = torch.tensor(
            [
                carry_lengths[env_index] + num_steps - starts[env_index]
                for env_index in range(num_envs)
            ],
            dtype=torch.int64,
        )
        max_length = int(new_lengths.max())

        previous_dest_times: list[int] = []
        previous_source_times: list[int] = []
        previous_envs: list[int] = []
        current_dest_times: list[int] = []
        current_source_times: list[int] = []
        current_envs: list[int] = []
        for env_index in range(num_envs):
            carry_length = carry_lengths[env_index]
            previous_dest_times.extend(range(carry_length))
            previous_source_times.extend(range(carry_length))
            previous_envs.extend([env_index] * carry_length)

            start = starts[env_index]
            current_length = num_steps - start
            current_dest_times.extend(
                range(carry_length, carry_length + current_length)
            )
            current_source_times.extend(range(start, num_steps))
            current_envs.extend([env_index] * current_length)

        device = next(iter(obs_seq.values())).device

        def indices(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=device)

        previous_dest = indices(previous_dest_times)
        previous_source = indices(previous_source_times)
        previous_env = indices(previous_envs)
        current_dest = indices(current_dest_times)
        current_source = indices(current_source_times)
        current_env = indices(current_envs)

        next_obs: dict[str, torch.Tensor] = {}
        for key, rollout_value in obs_seq.items():
            value = torch.empty(
                (max_length, num_envs, *rollout_value.shape[2:]),
                dtype=rollout_value.dtype,
                device=rollout_value.device,
            )
            if previous_dest.numel() > 0:
                value[previous_dest, previous_env] = self.obs[key][
                    previous_source,
                    previous_env,
                ]
            if current_dest.numel() > 0:
                value[current_dest, current_env] = rollout_value[
                    current_source,
                    current_env,
                ].detach()
            next_obs[key] = value

        return RecurrentObservationHistory(
            obs=next_obs,
            lengths=new_lengths,
            num_envs=num_envs,
        )

    @torch.no_grad()
    def recompute_hidden(self, agent) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Replay ongoing episode tails under the agent's current weights."""
        hidden = agent.initial_hidden(self.num_envs)
        if hidden is None or not self.obs:
            return hidden
        max_length = int(self.lengths.max())
        if max_length == 0:
            return hidden

        valid_times: list[int] = []
        valid_envs: list[int] = []
        for env_index, length in enumerate(self.lengths.tolist()):
            valid_times.extend(range(length))
            valid_envs.extend([env_index] * length)

        device = next(iter(self.obs.values())).device
        time_indices = torch.tensor(valid_times, dtype=torch.int64, device=device)
        env_indices = torch.tensor(valid_envs, dtype=torch.int64, device=device)
        flat_obs = {
            key: value[time_indices, env_indices] for key, value in self.obs.items()
        }
        context, _, _, _, _ = agent.spatial(flat_obs)
        context_seq = context.new_zeros((max_length, self.num_envs, context.shape[-1]))
        context_seq[time_indices, env_indices] = context

        lengths_device = self.lengths.to(device=device)
        for step in range(max_length):
            _, next_hidden = agent.memory(context_seq[step], hidden)
            active = (lengths_device > step).view(1, self.num_envs, 1)
            hidden = tuple(
                torch.where(active, next_state, state)
                for next_state, state in zip(next_hidden, hidden, strict=True)
            )

        return tuple(state.detach() for state in hidden)

    @property
    def mean_length(self) -> float:
        return float(self.lengths.to(torch.float32).mean())

    @property
    def max_length(self) -> int:
        return int(self.lengths.max())
